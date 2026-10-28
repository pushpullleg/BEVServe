# M6 plan: CPU preprocessing optimization

Stage A (investigation) of M6. Goal: a measurable, reproducible reduction of the CPU-bound request
preparation time with **bit-identical model inputs**, so the golden-output contract holds by
construction. Baseline: M5 (`docs/m5-report.md`, image `sha256:4dfd2292b36f…`, commit 5312d9c).

All Stage A measurements are CPU-only (host Python, same numpy 1.24.4 / Pillow 12.3.0 /
opencv-python-headless 4.8.1.78 as the image), on the 20 nuScenes val frames M5 benchmarked,
read locally and never written (timings only). Tool: `tools/profile_preprocess.py`.

## 1. Code path (verified against the code)

| Step | Where | Thread | M5 server time, C = 1 |
| --- | --- | --- | --- |
| Multipart parse | `api._read_form` (Starlette) | event loop | parse ~1–2 ms |
| Part checks, calibration JSON, per-image magic + Pillow header parse, exact 1600x900 | `api._collect_parts`, `_parse_calibration`, `_check_images` → `imaging.inspect_header`, `check_size` | event loop | validate ~0.4–2 ms |
| Admission (bounded, before any decode) | `executor.try_admit` | event loop | — |
| JPEG decode, 6 images **serially** | `api._decode_and_build` → `imaging.decode_rgb` (Pillow, libjpeg-turbo) | one anyio worker thread | decode 14.2 ms |
| Per camera, **serially**: resize 1600x900 → 704x396 (Pillow BICUBIC), crop rows 140–396, rotate(0) | `preprocess.build_inputs` → `transform_image` | same worker thread | preprocess 74.8 ms (all steps in this and the next two rows) |
| Per camera: uint8 → float32, RGB→BGR swap, subtract mean, multiply 1/std (cv2, in place), HWC → CHW copy | `preprocess.normalize` | same | (included above) |
| Stack 6 planes, calibration matrices, shape/dtype check | `build_inputs`, `calibration.rig_matrices`, `ModelInputs.check` | same | (included above) |
| Single GPU slot: `torch.from_numpy(...).to(cuda)` for the 7 tensors | `executor` (1-worker pool) → `BEVDetDetector.detect_native` | slot thread | h2d 0.8 ms |
| BEVDet `forward_test` unchanged (backbone, LSS view transform with `bev_pool_v2`, BEV encoder, CenterHead, decode, NMS), results to host | `detect_native` | slot thread | forward 11.3 ms, gpu 12.4 ms |
| Box conversion (bottom → geometric centre, ego@CAM_FRONT_LEFT), filtering, JSON | `boxes.detections_from_arrays`, `api._boxes`, response | event loop | serialize 0.4–1.5 ms |

There is no PyTorch work on the CPU side of preprocessing: the 7 arrays are numpy until
`detect_native`, and no batching or tensor packing exists beyond `np.stack`.

## 2. Where the ~75 ms goes (CPU profile, real frames, per camera, median of 240)

| Operation | ms / camera | × 6 |
| --- | --- | --- |
| **Pillow BICUBIC resize 1600x900 → 704x396** | **9.07** | **54.4** |
| crop, rotate(0) (Pillow copies) | 0.04 | 0.3 |
| `np.array(image)`, `.copy()`, `astype(float32)` | 0.13 | 0.8 |
| cv2 `cvtColor` / `subtract` / `multiply` (64-thread OpenCV pool) | 1.03 | 6.2 |
| HWC → CHW `ascontiguousarray` | 0.68 | 4.0 |
| sum of steps | 10.94 | 65.6 |
| production `build_inputs` (adds `np.stack`, calibration, checks) | | 72.4 |
| JPEG decode (real 144 KB average; synthetic JPEGs are 211 KB and not representative) | 2.5 | 20.7 in the production loop |

Findings:

1. The resize is 75% of preprocessing. It is a fixed-cost separable convolution (content does not
   matter), so it cannot be reduced without changing the resampling arithmetic, but it **releases
   the GIL** (as does JPEG decode), so the six cameras can run in parallel threads.
2. Intermediate copies cost < 1 ms per camera in total; removing them is not worth parity risk.
3. OpenCV's default thread pool (64 threads on this host) **slows** the small per-image cv2 calls:
   with `cv2.setNumThreads(1)` the cv2 steps drop from 1.03 to 0.38 ms per camera and the full
   decode + build path from 105 to 93 ms, bit-identical on all 20 frames. Under 8 concurrent
   requests the serial path's CPU throughput rises 26% (64.5 → 81.4 req/s) with one OpenCV thread
   (oversubscription: 8 requests × 64-thread pool on 64 hardware threads).
4. Synchronization points: one anyio worker thread per request does decode + preprocessing; the
   single GPU slot serializes inference; admission (max_in_flight 5) is held from before decode
   until inference finishes, so preprocessing time directly limits admitted throughput.

## 3. Independence and invariants

The six per-camera pipelines (decode → resize/crop/rotate → normalize) share no state: each reads
one JPEG and writes one (3, 256, 704) plane. Shared geometry (post_rot, post_tran, bda) is
constant, and the calibration matrices depend only on the request JSON; neither depends on pixels.
Global camera order matters only when the planes are stacked (index 0 = CAM_FRONT_LEFT defines the
output frame), so a parallel version must place results by camera index, never by completion.

Parity invariants (all must hold bit for bit):

- camera set and order `CAMERA_ORDER`; plane i comes from part `CAMERA_ORDER[i]`;
- `imgs` (6, 3, 256, 704) float32 C-contiguous, values byte-identical (includes the BGR channel
  order the network was trained on); same for `sensor2egos`, `ego2globals`, `intrins`,
  `post_rots`, `post_trans`, `bda` (shapes and dtypes per `EXPECTED_SHAPES`);
- decode semantics: non-RGB modes converted to RGB, EXIF orientation ignored;
- error behaviour: the same status, code and camera name for a bad image (the first failing camera
  in `CAMERA_ORDER`, as today);
- model outputs: boxes, scores, labels, frame `ego@CAM_FRONT_LEFT`, identical; checkpoint sha256
  `8b67e60d…3a13443`; config sha256 unchanged.

Because the candidate below calls the same functions on the same inputs, identical inputs imply
identical model outputs given the M3-established run-to-run repeatability; the GPU checks confirm
rather than carry the argument.

## 4. Candidates (evidence-ranked)

Experiments: scratch scripts calling the production functions (not committed); timings are CPU-only
on this host and must be confirmed in the container on the GPU.

| # | Candidate | Measured CPU effect | Parity | Complexity | Concurrency / resource notes | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Per-camera decode + transform + normalize in a bounded, shared thread pool, with OpenCV single-threaded** | decode + planes 105 → 15.3 ms at C = 1 (6 workers); C = 8 CPU-only throughput 64.5 → 124.6 (pool 12) / 182.0 (pool 24) req/s | bit-identical on 20 real frames (same functions; order by index) | ~40 lines + setting + tests | a pool of 6 **lowers** C = 8 throughput (71.6 req/s): pool must be shared and sized ≥ 12; more busy threads may slow the GPU slot thread at C = 8 (unmeasured) | **attempt first** |
| 2 | `cv2.setNumThreads(1)` alone | 105 → 93 ms at C = 1; +26% CPU throughput at C = 8 | bit-identical on 20 frames (elementwise ops) | 1 line | reduces thread oversubscription | included in #1; also a fallback if #1 regresses at C = 8 |
| 3 | Fuse normalization (exact 3×256-entry lookup table built from the original cv2 ops, written straight into a preallocated (6,3,256,704) buffer) | ≈ 0.5–1 ms per camera after #2 | provable (only 768 possible input values per channel), still needs proof | moderate | none | defer: ≤ 6 ms per request, not worth the proof burden now |
| 4 | Crop-aware resize (`Image.resize(box=...)`, resample only the 582 source rows that survive the crop) | 8.8 → 5.7 ms per camera | **not bit-identical**: max 1-level pixel difference on 34 images (coefficient rounding) | low | — | **rejected** |
| 5 | Other resamplers (cv2 `INTER_CUBIC`/`INTER_AREA`, Pillow-SIMD, JPEG DCT-scaled `draft()`, GPU decode/resize) | not measured | different arithmetic | — | GPU variants also load the single GPU slot | **rejected** (would change model inputs) |

Projected effect of #1 at C = 1: server decode + preprocess 89 ms → ~15–20 ms, i.e. client p50
107.7 → roughly 35–45 ms (−55% to −65%). The 15% target is therefore conservative. At C = 8, M5's
throughput (42.2 req/s) is close to the GPU-slot ceiling (1000 / 22.7 ms ≈ 44 req/s; GPU-slot time
was inflated from 12.4 ms by CPU contention), so faster preprocessing cannot by itself raise C = 8
throughput much; whether it helps (less contention) or hurts (more threads competing with the slot
thread) is the main open question and is measured, not assumed. Admission will also behave
differently (shorter hold time → fewer `queue_full` at the same load); the contract (only 200 or
503 with `Retry-After: 1`, recovery) must still hold.

## 5. Recommended first optimization

Candidate #1, as one change:

- `BEVSERVE_PREPROCESS_THREADS` (startup setting): `0` = today's serial path exactly (including
  OpenCV's default threads), `N > 0` = shared process-wide pool of N threads (proposed default 12,
  bounded 1–32) running decode → transform → normalize per camera, OpenCV set to one thread.
- Results are collected in `CAMERA_ORDER`; the first failing camera in that order raises the same
  `ImageRejectedError` as today; the pool is created at service start and shut down at exit.
- No change to `transform_image`, `normalize`, `decode_rgb`, geometry, calibration or the model.

The setting makes the before/after comparison run on **the same image** (only an environment
variable differs), which removes rebuild confounds; the M5 image is kept as an external reference.

## 6. Parity validation

1. CPU, local: `build_inputs` arrays from the serial and pooled paths byte-identical (all 7 arrays,
   shape, dtype, C-contiguity) on synthetic inputs (unit tests) and on **all 6,019 val frames**
   (private tool, no GPU, ~15 min CPU); error-precedence tests for corrupt images in several
   cameras; pool sizes 1, 6, 12, 32 and OpenCV threads 1/default.
2. GPU, private test: `detect_native` outputs identical for serial vs pooled inputs.
3. Container (operator): M4 checks on the new image including the 10-frame HTTP golden
   (byte-identical), in both settings.

## 7. Benchmark design

Lesson from M5: per-container CPU state shifts latency by up to ~14 ms, so trials, not requests,
are the replicates. Design: one new image; arms A = `PREPROCESS_THREADS=0`, B = `12`; 12 fresh
containers in ABBA blocks (6 pairs); each container: 20 warm-up, 100 requests at C = 1, 200 at
C = 8, with the resource sampler. Per trial: client p50 (C = 1), throughput and p95 (C = 8), error
counts, RSS and GPU memory peaks, CPU and GPU utilization. Analysis: paired log-ratios B/A with a
95% t interval over 6 pairs (the existing ABBA tooling, extended to ratios).

Power at C = 1: M5's paired SD (8.8 ms ≈ 8%) with 6 pairs gives a ±9% half-width; an expected
−55% effect would be established far from the −15% bound. At C = 8 the between-container variance
is unknown (M5 ran C = 8 once), so the non-inferiority margins below may come out inconclusive; if
so, that is reported, not tuned.

Also: admission (C = 8 and 16, max_in_flight 5, both client behaviours) and the 100-request soak
on arm B, as in M5; the M5 image is run once at C = 1 as a cross-check of the baseline.

## 8. Provisional acceptance criteria

| Criterion | Pass condition |
| --- | --- |
| Input parity | all 7 arrays byte-identical, serial vs pooled, on all 6,019 val frames and synthetic tests |
| Output parity | private GPU test identical; 10-frame HTTP golden byte-identical in both arms |
| C = 1 latency | 95% CI upper bound of the paired p50 ratio B/A ≤ 0.85 (≥ 15% faster) |
| C = 8 throughput | 95% CI lower bound of the paired throughput ratio ≥ 0.95 |
| C = 8 p95 | 95% CI upper bound of the paired p95 ratio ≤ 1.10 |
| Errors | zero failed requests in all levels; admission returns only 200 / 503 `queue_full` with `Retry-After: 1`; recovery 5/5 |
| Memory | RSS and GPU memory peaks within +10% of arm A |
| Trade-offs | CPU and GPU utilization of both arms reported |

A criterion whose interval is too wide is `insufficient_data`, never a pass.

## 9. Resource budget and stages

- Stage B (local, no GPU, no Docker): implementation, unit tests, CPU-only 6,019-frame input
  parity (~15 min of a few cores), lint/types.
- Stage C (operator, GPU): rebuild a new image tag (only the source layer changes; M5 image and
  tag kept), M4 checks on it; private GPU test.
- Stage D (operator, GPU): pilot (bounded) then the 12-trial benchmark (~8 min), admission and soak
  pilot (~4 min). One inference at a time; CPU up to ~25 of 64 hardware threads.

## 10. Risks

- C = 8 regression from extra busy threads competing with the GPU-slot thread (main risk; measured,
  with candidate #2 alone as the fallback).
- CPU-only projections may not transfer to the container (cgroup limits, different core
  placement); only the container benchmark counts.
- Admission dynamics change; documented and re-verified, limit unchanged.
- Thread pool lifecycle on fatal exit (exit code 70) and shutdown; covered by the M4 fault and
  shutdown checks.

## 11. Stage B results (implementation and CPU-only evidence)

Implementation: `src/bevserve/camera_pool.py` (process-wide pool owned by `Service`, created only
when `BEVSERVE_PREPROCESS_THREADS` > 0; default 0 = the M5 serial path, unchanged, including
OpenCV's default threads). `preprocess.build_inputs` was split into `camera_plane` (per camera) and
`assemble_inputs` (stack + calibration); both paths call these same functions, so no arithmetic is
duplicated. Compose passes the setting through (default 0); `tools/m4_container_checks.sh` adds a
golden-parity check with the pool enabled (`PARALLEL_THREADS`, default 12).

Input parity, all 6,019 val frames (`tools/m6_input_parity.py`): reference = `preprocess.py`
from commit 5312d9c loaded verbatim, with OpenCV at its default 64 threads; compared with the
current serial path and the pooled path (pool size rotating 6 / 12 / 24, 2,006–2,007 frames
each, OpenCV single-threaded): **12,038 comparisons of all seven arrays (dtype, shape, strides,
C-contiguity, bytes), 0 mismatches.** 20.5 min, about 5.4 cores on average, peak RSS 1.6 GiB, one
frame in memory at a time; nothing image-derived was written.

Tests: 65 new public tests (`tests/unit/test_camera_pool.py`, `tests/api/test_parallel_preprocess.py`)
cover all seven arrays for pool sizes 1/6/12/24/32 and JPEG, PNG, L, RGBA and P inputs; camera
order under reversed completion; repeated calls; 8 concurrent requests; first-failing-camera
precedence with 1–6 corrupt cameras (same HTTP error body as serial); decode > calibration >
per-camera error precedence; no task outliving a failed request; lifecycle (lazy workers, bounded
count, joined on shutdown, idempotent, OpenCV setting restored after the last pool); setting
bounds. A private GPU test (`test_parallel_preprocessing_gives_identical_outputs`) is added, not
yet run.

CPU-only benchmark of decode + preprocessing (`tools/m6_cpu_bench.py`; 3 interleaved repeats per
arm, fresh process each; median, range in brackets). **Preliminary: no API latency claim until the
GPU-container benchmark.**

| Arm | C = 1 p50 ms | C = 1 cores | C = 8 p50 ms | C = 8 p95 ms | C = 8 req/s | C = 8 cores | peak threads | peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serial (M5) | 101.7 [101.5, 103.4] | 5.09 | 112.3 [112.0, 118.2] | 124.6 | 68.3 [64.4, 69.5] | 26.1 | 200 | 1,748 |
| pool 6 | 16.0 [15.8, 16.0] | 5.04 | 112.5 [110.6, 112.7] | 119.5 | 70.8 [70.8, 71.6] | 6.0 | 143 | 1,576 |
| pool 12 | 16.3 [15.7, 16.6] | 5.03 | 61.6 [61.3, 62.1] | 70.2 | 127.3 [124.9, 128.6] | 11.4 | 149 | 1,647 |
| pool 24 | 16.0 [15.7, 16.1] | 4.98 | 49.4 [48.0, 49.7] | 60.5 | 160.6 [159.6, 165.5] | 16.2 | 161 | 1,854 |

Observations: at C = 1 every pool size cuts decode + preprocessing by about 84%. The serial path
uses about 5 cores at C = 1 and 26 at C = 8 for its work: OpenCV's 64-thread pool spends CPU
(much of it system time) on tiny per-image operations, so per request the pooled path uses about
4–6x less CPU time. A pool of 6 does not help at C = 8 (all requests share 6 workers), as
predicted in Stage A; 12 and 24 do. RSS includes about 1.4 GiB of dataset index loaded by the
benchmark itself; arm differences are within about 0.3 GiB. The production default stays 0;
pool size is chosen only after the GPU benchmark (Stage D), which also decides whether the C = 8
GPU-slot contention improves or worsens.

## 12. Stage C result and Stage D harness

Stage C: private GPU tests 14/14; `tools/m4_container_checks.sh` on the new image
`bevserve:m6-20261027-224240` (id `sha256:8d1a445473a106c8d55df04bfe7da2da92e4826fb8d678650b65dd4667d580fb`):
27/27 PASS, including the 10-frame golden parity serial and with 12 preprocessing threads (container
recreation and runtime `BEVSERVE_PREPROCESS_THREADS` verified), fault recovery and shutdown.

Stage D harness: `tools/m5_bench.sh` stage `ab` (analysis `tools/bench/ab.py`, criteria
`tools/bench/criteria.py`).

- Arms: A = `BEVSERVE_PREPROCESS_THREADS=0`, B = `AB_THREADS` (12). Same image, checkpoint, 20 val
  frames, host and settings otherwise (`max_in_flight` 16 so that C = 8 is not limited by admission,
  Server-Timing on in both arms for the queue/GPU split). The image must pass the verification gate:
  id equal to `EXPECT_IMAGE_ID`, Stage C run all PASS including the parallel golden and recreation
  checks for `AB_THREADS`, and its bevserve package byte-identical to the work tree.
- Trials: `AB_TRIALS` fresh containers in ABBA order (pilot 4, full 12 = 6 pairs). Every service
  start verifies a new container id and reads `BEVSERVE_PREPROCESS_THREADS` from the running
  container's `Config.Env`; a mismatch aborts the run. Per trial: C = 1 (20 / 100 requests) and
  C = 8 (40 / 200), each after `AB_WARMUP` (5 / 20) excluded warm-up requests, then the 10-frame
  golden parity in that container; one resource sampler (0.5 s) per trial.
- Metrics per trial: client p50/p95/p99 and successful throughput at C = 1 and C = 8;
  **preprocessing** = median of response-body `decode + preprocess` (wall time of the whole
  decode-and-build step, comparable across modes); Server-Timing `gpu` and `queue` medians;
  mean process CPU % and GPU utilization per phase (samples within the phase window);
  **peak memory** = maximum of the 0.5 s samples over the whole trial (warm-up, both phases and
  golden check) of the serving process's RSS (`/proc/<pid>/status` VmRSS) and of its GPU memory
  (`nvidia-smi` compute-apps entry of the server's process tree; includes the CUDA caching
  allocator; a sampled peak can miss sub-0.5 s spikes).
- Statistics: pairs (4b+1, 4b+2), (4b+4, 4b+3); estimate = geometric mean of B/A; 95% t interval on
  log ratios (df = pairs − 1). Pass = the whole interval on the right side of the threshold, fail =
  the estimate on the wrong side, otherwise `insufficient_data` (never a pass).
- Criteria: counted in both modes: fresh container with the intended setting per trial, every
  request ok, golden parity in every container, harness validity, verified cleanup. Counted in
  full mode only (pilot: reported): C = 1 p50 ratio UCB <= 0.85; C = 8 throughput ratio
  LCB >= 0.95; C = 8 p95 ratio UCB <= 1.10; peak RSS and peak GPU memory ratio UCB <= 1.10.
  Preprocessing, GPU/queue, CPU and GPU-utilization ratios are reported as information.
- Gate: `MODE=full` refuses to start unless `PILOT_DIR` is a `MODE=pilot` `ab` run on the same image
  id and `AB_THREADS` with every counted criterion passing.
- Precision: with M5's between-container spread (paired SD about 8% of latency), 6 pairs give
  about ±9% at C = 1, far from the 0.85 bound if the CPU-only projection holds. C = 8 variance is
  unknown and the throughput bound (0.95) is close to the expected ratio (C = 8 is near the GPU-slot
  ceiling), so that criterion may come out `insufficient_data`; it would be reported as such.
- M5 results are context only (different image and harness), never the causal comparison.

## 13. Stage D and E

Stage D (pilot `m5-20261027-234730`, full run `m5-20261027-235104`, private): all counted criteria
passed; results, independent verification and limitations in [m6-report.md](m6-report.md). The
production default stays `BEVSERVE_PREPROCESS_THREADS=0`.
