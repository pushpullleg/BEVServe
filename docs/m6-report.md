# M6 report: parallel six-camera preprocessing

M6 removes the CPU bottleneck M5 found in request preparation, without changing any model input:
the six cameras' JPEG decode and preprocessing now can run on a bounded thread pool
(`BEVSERVE_PREPROCESS_THREADS`), producing byte-identical model inputs and outputs. In a controlled
A/B on the same verified image, enabling 12 threads cut concurrency-1 client median latency to
**0.302x** (95% CI 0.298–0.305) and raised concurrency-8 throughput to **1.173x** (1.159–1.187),
with no failed requests and golden parity in every container. Plan and stage records:
[m6-plan.md](m6-plan.md). Raw results are kept privately, outside git.

## 1. Problem (from M5)

At concurrency 1, M5 measured 104 ms of server time per request, of which 89 ms was CPU: JPEG decode
14.2 ms and preprocessing 74.8 ms, against 12.4 ms on the GPU ([m5-report.md](m5-report.md)). The
M6 profile (`tools/profile_preprocess.py`, real frames) located the cost: Pillow's bicubic resize
1600x900 → 704x396 takes 9.07 ms per camera, about 75% of preprocessing, and the six cameras ran
one after another in a single thread. OpenCV's default 64-thread pool also made the small
normalization calls slower and burned CPU under concurrency.

## 2. Implementation

- `src/bevserve/camera_pool.py`: a process-wide `ThreadPoolExecutor` owned by the service, created
  only when `BEVSERVE_PREPROCESS_THREADS` is 1–32 (default 0 = the M5 serial path, unchanged,
  including OpenCV's default threading). Each camera task runs the existing
  `imaging.decode_rgb` and `preprocess.camera_plane` on its own JPEG bytes; the planes are placed by
  camera index and assembled by the same `preprocess.assemble_inputs` the serial path uses.
- Pillow releases the GIL during decode and resize, so the threads run in parallel. OpenCV is set
  to one thread while a pool exists (restored after the last pool), avoiding N x 64 threads.
- Errors keep the serial precedence: first camera (in camera order) whose decode failed, then
  calibration, then per-camera preprocessing; every task of a request finishes before an error is
  raised. Work is bounded by the pool size and by admission (at most 6 x `max_in_flight` queued
  camera tasks). Workers start lazily and are joined at shutdown.
- With the pool enabled, `timing_ms.decode` is the time until the last camera is decoded and
  `timing_ms.preprocess` the remainder; the stages overlap. Their sum is the decode-and-build wall
  time in both modes and is what M6 compares.

## 3. Why inputs and outputs are byte-identical

The pooled path calls the same decode, resize, crop, normalize and assembly functions on the same
bytes; it adds only scheduling, and no camera task shares mutable state. Camera order is fixed by
index. Identical model inputs give identical outputs because inference is deterministic run to run
(M3: 20 repeats bit-identical). Each link was also tested directly:

| Stage | Evidence |
| --- | --- |
| B: input parity, full validation set | 6,019 val frames, 12,038 comparisons of all seven input arrays (dtype, shape, strides, C-contiguity, bytes) against `preprocess.py` from commit 5312d9c (M5) run with OpenCV's default threads: 0 mismatches. The current serial path was compared on all 6,019 frames; the pooled path on all 6,019 with the pool size rotating per frame: 6 threads on 2,007 frames, 12 on 2,006, 24 on 2,006 |
| B: public tests | 65 tests: all seven arrays for pool sizes 1/6/12/24/32 and JPEG/PNG/L/RGBA/P inputs, camera order under reversed completion, repeated and concurrent requests, error precedence with 1–6 corrupt cameras (identical HTTP error body), lifecycle |
| C1: private GPU tests | 14/14, including bit-identical model outputs for serial vs pooled inputs (pool sizes 6, 12, 24) |
| C2: container checks | 27/27 on the M6 image, including the 10-frame HTTP golden parity in serial mode and with 12 threads (container recreation and runtime `BEVSERVE_PREPROCESS_THREADS` verified from the running container), fault recovery and shutdown |
| D: benchmark | golden parity (10 real frames, byte-identical boxes) in all 12 benchmark containers, both arms: 120/120 frame comparisons |

## 4. Benchmark design (Stage D)

- **Image**: `bevserve:m6-20261027-224240`,
  `sha256:8d1a445473a106c8d55df04bfe7da2da92e4826fb8d678650b65dd4667d580fb`; its bevserve package
  was verified byte-identical to the benchmarked source (`5312d9c` + the M6 changes) before the
  run. Checkpoint sha256 `8b67e60d…3a13443`. Host: RTX PRO 6000 Blackwell (driver 570.211.01),
  Threadripper PRO 7975WX (64 threads), Linux 6.14.0-37.
- **Arms**: A = `BEVSERVE_PREPROCESS_THREADS=0` (serial), B = `12`. Everything else identical:
  same image, checkpoint, 20 nuScenes val frames cycled, host, `max_in_flight` 16 (so C = 8 is not
  limited by admission), Server-Timing on in both arms (for the queue/GPU split).
- **Replicates**: 12 fresh containers in ABBA order (A B B A, three times); pairs (1,2), (4,3),
  (5,6), (8,7), (9,10), (12,11) = **6 independent pairs**. The 3,600 HTTP requests are not
  replicates. Every start verified a new container id and read the thread setting from the running
  container (observed 0,12,12,0,0,12,12,0,0,12,12,0).
- **Per container**: C = 1, 100 requests after 20 warm-up; C = 8, 200 requests after 20 warm-up;
  then the 10-frame golden comparison; one resource sampler (0.5 s) for the whole trial.
- **Statistics**: for each metric the estimate is the geometric mean of the six paired ratios B/A,
  with a 95% Student t interval on the log ratios (t(0.975, 5) = 2.571). A criterion passes only if
  the whole interval is on the right side of its threshold.
- **Gate**: the full run could start only after a passing pilot (4 containers) on the same image.
- **Verification**: all ratios and intervals below were recomputed from the raw per-request records
  and sample files with independent code (nearest-rank percentiles, scipy's t quantile, pairing
  read from the containers' settings) and agree with the harness to < 1e-4.

## 5. Results

Medians over the six trials of each arm (per-trial values in the private report):

| Metric | A: serial | B: 12 threads |
| --- | --- | --- |
| C = 1 client p50 / p95 / p99 (ms) | 109.1 / 114.0 / 116.5 | 32.9 / 36.9 / 38.1 |
| C = 1 throughput (req/s) | 9.11 | 29.68 |
| C = 1 decode + preprocess (ms, response body) | 90.6 | 15.8 |
| C = 8 client p50 / p95 / p99 (ms) | 184.9 / 194.1 / 197.9 | 157.5 / 164.6 / 168.3 |
| C = 8 throughput (req/s) | 42.06 | 49.48 |
| C = 8 decode + preprocess (ms) | 117.2 | 23.1 |
| C = 8 queue for the GPU slot / GPU time / forward (ms, Server-Timing p50) | 36.7 / 22.4 / 19.2 | 108.2 / 19.3 / 16.3 |
| C = 1 / C = 8 server CPU (mean %, 100 = one core) | 424 / 1,221 | 339 / 576 |
| C = 8 GPU utilization (mean %) | 18.2 | 18.9 |
| peak server RSS / GPU memory (MiB) | 2,734 / 1,316 | 2,618 / 1,316 |

Arm A reproduces M5's baseline (M5: 107.7 ms at C = 1, 42.23 req/s at C = 8) on a different image
and harness; this is context only, the causal comparison is A vs B.

### Acceptance (counted, full run)

| Criterion | Estimate B/A | 95% CI | Threshold | Result |
| --- | --- | --- | --- | --- |
| C = 1 client p50 ratio | 0.3016 | [0.2985, 0.3047] | upper bound <= 0.85 | **pass** |
| C = 8 throughput ratio | 1.1730 | [1.1591, 1.1870] | lower bound >= 0.95 | **pass** |
| C = 8 client p95 ratio | 0.8488 | [0.8290, 0.8691] | upper bound <= 1.10 | **pass** |
| Peak server RSS ratio | 0.9585 | [0.9552, 0.9619] | upper bound <= 1.10 | **pass** |
| Peak server GPU memory ratio | 1.0000 | [1.0000, 1.0000] | upper bound <= 1.10 | **pass** (see limitations) |
| Every trial in a fresh container with the intended setting | 12/12 | | | **pass** |
| Every request succeeded | 3,600/3,600 | | | **pass** |
| Golden parity in every container | 12/12 containers, 120/120 frames | | | **pass** |
| Harness validity, samplers, verified cleanup | 24/24 runs, 12/12 samplers, 12/12 cleanups | | | **pass** |

Information only (not criteria): C = 1 decode + preprocess 0.173 [0.170, 0.176]; C = 8 decode +
preprocess 0.197 [0.194, 0.200]; C = 1 throughput 3.259 [3.226, 3.292]; C = 8 p50 0.854 [0.843,
0.865]; C = 8 GPU time 0.862 [0.849, 0.875]; C = 8 queue time 2.961 [2.897, 3.026]; server CPU
0.798 [0.786, 0.810] at C = 1 and 0.470 [0.459, 0.481] at C = 8; C = 8 GPU utilization 1.019
[0.976, 1.064].

## 6. Trade-offs and interpretation

- **C = 1**: decode + preprocessing fell from 90.6 to 15.8 ms; the server now spends about 31 ms
  per request, of which 12.6 ms is the GPU. Client latency fell 70%, throughput rose 3.3x.
- **C = 8**: preprocessing is no longer the limit; the single GPU slot is. Requests now wait
  longer for it (queue 36.7 → 108.2 ms), so latency improves less (−15%) than at C = 1. GPU time per
  request fell from 22.4 to 19.3 ms, consistent with less CPU contention for the inference thread,
  and the throughput gain (+17%) matches that: 1000 / 19.3 ms ≈ 52 req/s is the new ceiling
  (observed 49.5). Further C = 8 gains would need GPU-side work (batching, more GPU slots), which is
  out of M6 scope.
- **CPU**: the parallel path uses less CPU per request, not more: server CPU fell 20% at C = 1
  (while serving 3.3x the requests) and 53% at C = 8. Stage B's CPU-only measurements point to
  OpenCV's default thread pool in the serial path (much of its CPU time is system time); this run
  did not attribute CPU time to threads.
- **Memory**: peak RSS 4% lower; peak GPU memory unchanged.

## 7. Limitations

- One host and one GPU; client and server share the host.
- 20 frames cycled; repeated JPEG bytes and images may be warmer in caches than in production.
- Short windows: about 3.4 s (B) to 11 s (A) per C = 1 phase and 4–5 s per C = 8 phase; 6 pairs.
  The intervals are narrow because between-container variation was small in this run, which M5
  showed is not guaranteed.
- Server-Timing was on in both arms. Its own overhead is **not established** (M5: +1.12 ms,
  95% CI −8.11 to +10.35 ms); it is identical across arms but the absolute numbers include it.
- `max_in_flight` was 16 for the A/B. Admission with the production limit (5), the soak and the
  1,000-request soak were not re-run with parallel preprocessing; the production default stays
  serial (`BEVSERVE_PREPROCESS_THREADS=0`) until they are.
- Memory peaks are 0.5 s samples (short spikes can be missed); GPU memory is nvidia-smi's
  per-process figure (whole MiB, includes the CUDA caching allocator). Every trial of both arms
  peaked at exactly 1,316 MiB, so that interval is degenerate: it shows no increase at this
  resolution, not a precisely estimated ratio.
- Full-validation parity covers the inputs (all 6,019 frames, each pool size on one third); output
  parity on real frames covers the 10 golden frames over HTTP and synthetic frames on the GPU.
- Only 12 threads was benchmarked on the GPU; 6 threads is expected to help less under load
  (CPU-only Stage B).

## 8. Reproduction

1. Build and verify: `IMAGE=bevserve:<tag> PARALLEL_THREADS=12 ... bash tools/m4_container_checks.sh`
   (all checks must pass, including the parallel golden check).
2. Input parity (private, CPU): `nice -n 19 ionice -c3 python tools/m6_input_parity.py
   --bevdet-root <BEVDet> --out outputs/m6-parity-<id>`.
3. Pilot, then full A/B (operator, `sudo docker`):
   `MODE=pilot STAGES=ab IMAGE=... EXPECT_IMAGE_ID=sha256:... STAGE_A_DIR=outputs/m4-<run> ...
   bash tools/m5_bench.sh`, then the same with `MODE=full PILOT_DIR=outputs/m5-<pilot run>`.
   Criteria: `tools/bench/criteria.py`; analysis: `tools/bench/ab.py`.

## 9. Conclusion

Parallel six-camera preprocessing removes the dominant CPU cost of a request while keeping every
model input byte-identical. On the verified M6 image, with 12 threads, single-request latency fell
from 109 to 33 ms (ratio 0.30), loaded throughput rose from 42 to 49 req/s (ratio 1.17) and loaded
p95 fell 15%, using less CPU and no more memory; all 3,600 requests succeeded and golden parity
held in every container. The bottleneck at concurrency 8 is now the single GPU slot. The
optimization is opt-in (`BEVSERVE_PREPROCESS_THREADS=12`); serial remains the default until
admission at the production limit and the soak are verified in parallel mode.
