# BEVServe

HTTP inference service for camera-only 3D object detection with
[BEVDet](https://github.com/HuangJunJie2017/BEVDet): six camera images plus calibration in, 3D
boxes out.

> **Status: milestone M2.** The service serves the released BEVDet-R50 checkpoint on a local GPU
> (`BEVSERVE_DETECTOR=bevdet`), or a **FakeDetector** with fixed synthetic boxes (default). On three
> nuScenes val samples its boxes are identical to BEVDet's own evaluation path. Full-validation
> accuracy parity (M3) and benchmarks (M5) are not done yet; no accuracy or latency claims are made.

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

## Real model (M2)

Needs a BEVDet dev3.0 checkout, its CUDA environment and an operator-supplied checkpoint with a
trusted sha256; see [docs/env-m2.md](docs/env-m2.md). The checkpoint is read once, its hash is
checked on exactly the bytes that are loaded, and it is loaded with `torch.load(weights_only=True)`;
a wrong hash keeps the service up but not ready (`/readyz` reason `checkpoint_hash_mismatch`).
Private checks, never run in public CI:

```bash
BEVSERVE_PRIVATE_GPU=1 BEVSERVE_BEVDET_CONFIG=... BEVSERVE_CHECKPOINT=... \
  BEVSERVE_CHECKPOINT_SHA256=... python -m pytest -q tests/private
cd /path/to/BEVDet && python /path/to/bevserve/tools/m2_compare_direct.py \
  --checkpoint checkpoints/.../bevdet-r50-cbgs.pth --index 0 --url http://127.0.0.1:8000
```

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
`STALL_AFTER_S` (30), `FATAL_EXIT_GRACE_S`, `FATAL_EXIT_CODE`.

## Data and licensing

Code: Apache-2.0 (see `LICENSE`, `NOTICE`). This repository contains **no** dataset files, no
model weights and no values derived from nuScenes; tests use generated images and a made-up camera
rig. nuScenes is CC BY-NC-SA 4.0 plus its Terms of Use; obtain it and BEVDet checkpoints from
their owners. `tools/check_restricted.py` runs in CI and blocks weights, dataset files, large
files and private paths.
