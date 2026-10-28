# BEVServe

HTTP inference service for camera-only 3D object detection with
[BEVDet](https://github.com/HuangJunJie2017/BEVDet): six camera images plus calibration in, 3D
boxes out.

> **Status: milestone M7.** The service serves the released BEVDet-R50 checkpoint on a local GPU
> (`BEVSERVE_DETECTOR=bevdet`), or a **FakeDetector** with fixed synthetic boxes (default).
> Verified locally: over all 6,019 nuScenes val frames BEVServe's input path and adapter reproduce
> the reference evaluation (mAP 0.3086, NDS 0.3941, identical to every printed digit), and HTTP
> responses on 200 frames are identical to BEVDet's own evaluation path. The GPU container (M4) was
> benchmarked on one workstation (M5), and an opt-in parallel preprocessing mode was verified and
> benchmarked against it (M6, below). A reliability review (M7) kept serial preprocessing as the
> default (decision: conditional; see below).

## What M1 contains

- `POST /v1/detect`: multipart request with six images named by camera and a `calibration` JSON
  field; strict validation; JSON response with boxes, model identity and stage timings.
- `GET /healthz` (liveness), `GET /readyz` (readiness with reason), `GET /v1/model`.
- Preprocessing that reproduces BEVDet's test-time pipeline (resize 0.44, crop to 704x256,
  BGR mean/std normalization, calibration matrices). On synthetic inputs it is bit-identical to
  BEVDet's own code (`tools/reference_check_bevdet.py`).
- One GPU execution slot with bounded admission and safe deadlines; init-failure vs fatal-failure
  handling; loopback-only binding by default; upload limits checked before decoding.
- Errors as `application/problem+json` (RFC 9457) with a request ID, never a stack trace.

## Setup

Python 3.10.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Test

```bash
.venv/bin/python -m pytest -q          # synthetic unit and API tests, CPU only
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src tests tools
.venv/bin/mypy                          # strict
.venv/bin/python tools/check_restricted.py
# T1 CPU smoke test with random weights (needs torch + mmcv-full + BEVDet's mmdet3d):
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/smoke
```

Optional, local only: compare preprocessing with a BEVDet dev3.0 checkout and its environment
(CPU, synthetic data):

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 BEVSERVE_BEVDET_ROOT=/path/to/BEVDet \
  /path/to/bevdet-env/bin/python tools/reference_check_bevdet.py
```

## Demo

```bash
.venv/bin/bevserve &                                  # serves on 127.0.0.1:8000
.venv/bin/python tools/demo.py --url http://127.0.0.1:8000
kill %1
```

The server refuses to bind a non-loopback address unless it is listed in
`BEVSERVE_ALLOWED_BIND_HOSTS` (JSON list).

## Real model (M2, M3)

Needs a BEVDet dev3.0 checkout, its CUDA environment and an operator-supplied checkpoint with a
trusted sha256; see [docs/env-m2.md](docs/env-m2.md). The checkpoint is read once, its hash is
checked on exactly the bytes that are loaded, and it is loaded with `torch.load(weights_only=True)`;
a wrong hash keeps the service up but not ready (`/readyz` reason `checkpoint_hash_mismatch`).
Private checks, never run in public CI:

```bash
BEVSERVE_PRIVATE_GPU=1 BEVSERVE_BEVDET_CONFIG=... BEVSERVE_CHECKPOINT=... \
  BEVSERVE_CHECKPOINT_SHA256=... python -m pytest -q tests/private
cd /path/to/BEVDet && python /path/to/bevserve/tools/compare_direct.py \
  --checkpoint checkpoints/.../bevdet-r50-cbgs.pth --select scenes:50 --url http://127.0.0.1:8000
cd /path/to/BEVDet && python /path/to/bevserve/tools/full_val_parity.py --out /tmp/fullval
```

## Containers (M4)

`docker/Dockerfile.gpu` (multi-stage, pinned sources and base-image digests, non-root, no weights
or data in any layer), `docker/compose.yaml` (GPU, read-only checkpoint mount, host port on
127.0.0.1 only, restart on process exit) and `docker/Dockerfile.cpu` for the CPU smoke test. Host
prerequisites and their rollback: [docs/host-setup.md](docs/host-setup.md); deployment details and
offline verification: [docs/deploy.md](docs/deploy.md). The CPU smoke stack builds from public
sources without Docker: `python3.10 -m venv .venv-cpu && PYTHON=.venv-cpu/bin/python bash
docker/build_cpu_stack.sh`.

## Performance (M5)

Measured on one RTX PRO 6000 Blackwell workstation, client on the same host, 20 nuScenes val
frames (six camera JPEGs each), closed-loop load; full method, caveats and raw-result provenance
in [docs/m5-report.md](docs/m5-report.md).

| Concurrency | Throughput | p50 | p95 |
| --- | --- | --- | --- |
| 1 | 9.2 req/s | 107.7 ms | 112.1 ms |
| 8 | 42.2 req/s | 185.7 ms | 195.7 ms |

- At concurrency 1 the server spends 89 of 104 ms on CPU JPEG decode and preprocessing and 12 ms
  on the GPU; GPU utilization stayed below 40%.
- All 1,200 requests across concurrency 1–8 succeeded. Under saturation with the production
  admission limit, only `503 queue_full` with `Retry-After: 1` was returned, and the service
  recovered.
- A 100-request soak passed its memory and latency-drift limits, and 10-frame golden parity held
  afterwards.
- Cold start to ready took 4.6–4.8 s.
- Not established: the overhead of the optional `Server-Timing` instrumentation (+1.1 ms measured,
  95% CI −8.1 to +10.4 ms, against a 3.1 ms limit).

## Parallel preprocessing (M6)

M5 showed that most of a request's time was spent decoding and resizing the six camera images one
after another on the CPU. Setting `BEVSERVE_PREPROCESS_THREADS=12` processes the six cameras on a
shared, bounded thread pool. It runs the same decode, resize, crop and normalize code, so model
inputs are byte-identical to the serial path: verified on all 6,019 nuScenes val frames, and by
golden output parity in every benchmark container. Full evidence:
[docs/m6-report.md](docs/m6-report.md).

Controlled A/B on the same image (`sha256:8d1a4454…80fb`; 12 fresh containers, ABBA order, 6
pairs; ratio parallel / serial with 95% CI):

| Metric | Serial | 12 threads | Ratio [95% CI] |
| --- | --- | --- | --- |
| p50 latency, concurrency 1 | 109.1 ms | 32.9 ms | 0.302 [0.298, 0.305] |
| Throughput, concurrency 8 | 42.1 req/s | 49.5 req/s | 1.173 [1.159, 1.187] |
| p95 latency, concurrency 8 | 194.1 ms | 164.6 ms | 0.849 [0.829, 0.869] |

All 3,600 measured requests succeeded. Server CPU use fell 20% at concurrency 1 and 53% at
concurrency 8, and peak memory did not increase. At concurrency 8 the single GPU slot is now the limit.

The default stays `BEVSERVE_PREPROCESS_THREADS=0` (serial, as verified in M5), because admission
at the production limit and the soak have not yet been re-run in parallel mode. To enable it, set
the variable for the container, for example in the Compose env file:

```bash
BEVSERVE_PREPROCESS_THREADS=12   # 0 = serial (default); 1-32 = parallel; 12 is the benchmarked value
```

## Reliability under overload (M7)

Production admission limit 5, both preprocessing modes, verified M6 image
([docs/m7-report.md](docs/m7-report.md)):

- Fault recovery (exit 70, restart, recovery) and clean shutdown also hold with 12 threads.
- Under overload the service returns only 200 or `503 queue_full` with `Retry-After: 1` and
  `/readyz` stays healthy throughout; after the overload all five admission slots are free
  again. Every successful response matched the serial reference, and the 10 golden frames stayed
  byte-identical.
- Limitation: when clients ignore `Retry-After` and offer about 5–7× capacity (~250 attempts/s),
  latency of admitted requests rises: p99 270 ms serial and 254 ms with 12 threads, against
  baselines of 198 and 110 ms (ratios 1.37× and 2.31×; one container per arm, no statistical
  claim). The 12-thread mode is lower in absolute terms but misses the M7 criterion (at most 2× its
  own baseline); the original M7 overload pilot also failed it. Sustained-load reliability has not
  been measured.
- Decision **conditional**: `BEVSERVE_PREPROCESS_THREADS=0` stays the default; 12 threads remains a
  verified opt-in. An ingress-protection change (cheap rejection before parsing the upload) is
  proposed, not implemented.

## API summary

Request parts: `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, `CAM_BACK`,
`CAM_BACK_RIGHT` (JPEG or PNG, exactly 1600x900) and `calibration`:

```json
{
  "schema_version": "1",
  "cameras": {
    "CAM_FRONT": {
      "intrinsic": [[1250, 0, 800], [0, 1250, 450], [0, 0, 1]],
      "sensor2ego": {"rotation_wxyz": [0.5, -0.5, 0.5, -0.5], "translation_m": [1.7, 0.0, 1.5]}
    }
  },
  "options": {"score_threshold": 0.1, "max_boxes": 500}
}
```

(all six cameras are required; `ego2global` is optional, for all six or none). Boxes are returned
in `ego@CAM_FRONT_LEFT` with `center` = geometric center; see [docs/frames.md](docs/frames.md).

| Status | Meaning |
| --- | --- |
| 400 | malformed multipart or calibration JSON |
| 413 | body, image part or calibration over its limit |
| 415 | not multipart; image part not JPEG/PNG |
| 422 | wrong camera set, image size or calibration value |
| 503 | not ready, or admission full (`Retry-After`) |
| 504 | deadline passed; a running job keeps the slot until it finishes |
| 500 | inference failure; a fatal failure also makes the process exit (code 70) for restart |

## Configuration

Environment variables `BEVSERVE_*` (see `src/bevserve/settings.py`): `HOST` (default
`127.0.0.1`), `PORT`, `ALLOWED_BIND_HOSTS`, `MAX_REQUEST_BYTES` (50 MiB), `MAX_IMAGE_BYTES`
(8 MiB), `MAX_CALIBRATION_BYTES` (64 KiB), `MAX_IN_FLIGHT` (5), `DEADLINE_S` (10),
`STALL_AFTER_S` (30), `FATAL_EXIT_GRACE_S`, `FATAL_EXIT_CODE`, `SERVER_TIMING` (false; adds a
`Server-Timing` response header for benchmarks, body unchanged). `PREPROCESS_THREADS` (0 = serial decode and
preprocessing, the default; 1–32 = the six cameras on a shared thread pool with byte-identical
model inputs; 12 was verified and benchmarked in M6).

## Data and licensing

Code: Apache-2.0 (see `LICENSE`, `NOTICE`). This repository contains **no** dataset files, no
model weights and no values derived from nuScenes; tests use generated images and a made-up camera
rig. nuScenes is CC BY-NC-SA 4.0 plus its Terms of Use; obtain it and BEVDet checkpoints from
their owners. `tools/check_restricted.py` runs in CI and blocks weights, dataset files, large
files and private paths.
