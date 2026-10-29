# BEVServe

An HTTP inference service for camera-only 3D object detection with
[BEVDet](https://github.com/HuangJunJie2017/BEVDet)-R50. A request carries six camera images plus
their calibration; the response holds 3D boxes in the ego frame.

BEVServe is an engineering project about serving one research model correctly and observably.
The focus is reproducing the model's outputs exactly, defining failure behaviour explicitly, and
measuring performance under controlled conditions. It is **not**:
- a multi-GPU or multi-host serving system;
- a batching server;
- a production deployment;
- suitable for safety-relevant use.

All GPU results come from one workstation.

**What you can run:**
- **Without GPU, model or dataset:** the full API, a FakeDetector, the test suite and a demo.
- **With your own GPU and legally obtained assets:** the real model in a hardened container, plus
  the verification tools.

[docs/reproducibility.md](docs/reproducibility.md) states exactly what runs where.

## What it does

- **Endpoints:**
  - `POST /v1/detect`: a multipart request with six 1600×900 JPEG/PNG images named by camera,
    plus a `calibration` JSON field;
  - `GET /healthz` (liveness), `GET /readyz` (readiness with a reason), `GET /v1/model`.
- **Validation and errors:**
  - strict validation with upload limits, checked before decoding;
  - errors as `application/problem+json` (RFC 9457) with a request ID, never a stack trace.
- **Preprocessing:** reproduces BEVDet's test-time pipeline (resize, crop to 704×256,
  normalization, calibration matrices).
  - An opt-in thread pool (`BEVSERVE_PREPROCESS_THREADS=12`) processes the six cameras in
    parallel, with byte-identical model inputs.
  - The default stays serial.
- **Execution:**
  - one GPU execution slot with bounded admission (`max_in_flight`, default 5) and deadlines;
  - when admission is full, the response is 503 `queue_full` with `Retry-After: 1`.
- **Failure handling:**
  - an initialization failure (wrong checkpoint hash, no GPU, missing CUDA op) keeps the service
    up and not ready;
  - a fatal runtime error makes the process exit with code 70, so the container supervisor
    restarts it.
- **Model and container:**
  - the checkpoint is verified by sha256 on exactly the bytes that are loaded, then loaded with
    `torch.load(weights_only=True)`;
  - binds to loopback only unless the host is allow-listed;
  - the GPU container is hardened: non-root, read-only root filesystem, all capabilities dropped,
    read-only checkpoint mount, host port on 127.0.0.1.

Components, request lifecycle and design trade-offs: [docs/architecture.md](docs/architecture.md).

## Quick start without a GPU (FakeDetector)

Linux and a CPython 3.10 interpreter are required, because dependencies are pinned exactly.

Python 3.10 is not the default on current distributions (Ubuntu 24.04 ships 3.12). Install one
separately, for example with pyenv, uv, conda or a distribution package. Then use its path in
place of `python3.10` below.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q                       # CPU only, synthetic data
.venv/bin/bevserve &                                # FakeDetector on 127.0.0.1:8000
.venv/bin/python tools/demo.py --url http://127.0.0.1:8000
kill %1
```

The FakeDetector returns fixed synthetic boxes and identifies itself as `fake-detector` in every
response. It exercises the real API, validation, admission and error paths, **not** the model.
Interactive API docs are at `http://127.0.0.1:8000/docs`; the multipart body of `/v1/detect` is
described below rather than in the generated schema.

## Results

All figures below were measured on one workstation:
- an NVIDIA RTX PRO 6000 Blackwell GPU and an AMD Threadripper PRO 7975WX CPU;
- the client ran on the same host;
- checkpoint BEVDet-R50-CBGS, sha256 `8b67e60d…3443`.

Full methods and limitations are in the linked reports.

### Correctness

| Result | Conditions | Evidence |
| --- | --- | --- |
| Over all **6,019** nuScenes val frames, BEVServe's input path and box conversion give **mAP 0.3086, NDS 0.3941**, equal to the author's earlier reference evaluation of the same checkpoint with BEVDet's own test path | in-process adapter with BEVDet's own evaluator (not over HTTP); run once; BEVServe's result is preserved, the reference-run artifact is not | [verification-m3.md](docs/verification-m3.md) |
| HTTP responses equal BEVDet's own evaluation path on **50 + 200** val frames: every box field differs by exactly 0.0, and every preprocessing array matches | same checkpoint and host | [verification-m3.md](docs/verification-m3.md) |
| One frame sent 20 times gives bit-identical outputs, over HTTP and in process | one frame | [verification-m3.md](docs/verification-m3.md) |
| Containerized service: 10 golden frames, 1,209/1,209 boxes byte-identical to the M3 host results | M4 image | [verification-m4.md](docs/verification-m4.md) |
| Parallel preprocessing produces byte-identical model inputs | all 6,019 val frames (CPU check) | [m6-report.md](docs/m6-report.md) |

### Performance

These runs used **20 nuScenes val frames** (six JPEGs each, about 860 KB per request), cycled by a
closed-loop client. They also used `max_in_flight` **16**, so that concurrency 8 is not limited by
admission, and Server-Timing **on** in every run — neither is the production configuration. The
overhead of Server-Timing itself was **not established**: +1.12 ms, 95% CI −8.11 to +10.35 ms.

| Result | Conditions | Evidence |
| --- | --- | --- |
| Serial baseline: concurrency 1 → 9.23 req/s, p50 107.7 ms; concurrency 8 → 42.23 req/s, p50 185.7 ms | 300 requests per level, one run | [m5-report.md](docs/m5-report.md) |
| At concurrency 1, 89 of 104 ms of server time is CPU decode and preprocessing; the GPU takes 12.4 ms | Server-Timing medians | [m5-report.md](docs/m5-report.md) |
| 12-thread preprocessing vs serial: concurrency-1 p50 **0.302×** (95% CI 0.298–0.305), concurrency-8 throughput **1.173×** (1.159–1.187), concurrency-8 p95 0.849× (0.829–0.869) | 12 fresh containers in ABBA order (6 pairs), same image | [m6-report.md](docs/m6-report.md) |

### Reliability (production admission limit 5)

| Result | Conditions | Evidence |
| --- | --- | --- |
| Fault recovery (exit 70 → restart → ready) and clean shutdown, in both preprocessing modes | container suite | [m7-report.md](docs/m7-report.md) |
| Under overload, responses are only 200 or 503 `queue_full` with `Retry-After: 1`; `/readyz` stays healthy; all admission slots are free afterwards; every 200 equals the serial reference | both modes | [m7-report.md](docs/m7-report.md) |
| **Limitation.** With clients that ignore `Retry-After` at about 5–7× capacity, admitted-request p99 rises 1.37× (serial) and 2.31× (12 threads) over its own baseline. The 12-thread mode failed the pre-registered ≤ 2× criterion. Decision: **conditional**; serial stays the default | one container per arm, descriptive only | [m7-report.md](docs/m7-report.md) |
| Rejecting early (before parsing the upload) was tested in one pre-registered A/B (pilot + 4 ABBA pairs). The unmodified arm reproduced the > 2× tail in only 1 of 4 containers, so the reproduction criterion **P0 failed**; P1–P6 passed. Effectiveness was **not established**; the change was **not adopted (Option A)** and its code was removed | ~4 s bursts, small tail samples | [m8-report.md](docs/m8-report.md) |

## Limitations

- **One host, one GPU type:**
  - only sm_120 (Blackwell) is built and validated;
  - the benchmark client shared the host with the server;
  - there are no multi-GPU, batching or horizontal-scaling results.
- **Sustained load not measured:** the long soak and the M7 sustained stages were never run (the
  M5 soak was 100 requests).
- **Overload tail:** admitted-request latency rises under overload from clients that ignore
  `Retry-After` (see above). Clients should honour `Retry-After`.
- **Known defect:** a client disconnect during upload raises an unhandled `ClientDisconnect`,
  logged server-side with a traceback. No resources leak; the defect is documented in
  [m7-report.md](docs/m7-report.md).
- **Workstation-bound tooling:**
  - the container verification script requires the workstation's exact kernel
    (`6.14.0-37-generic`);
  - the benchmark and verification tools need nuScenes, a host model environment and
    `sudo docker`.
- **External build input:** the GPU image builds BEVDet from a pinned commit on the author's
  public fork; see [docs/reproducibility.md](docs/reproducibility.md).
- **CI not yet run:** workflows exist (`.github/workflows/`), but this repository has not yet been
  pushed to a remote, so **no CI run has been observed**. The T0 checks were run locally (guard,
  lint, format, types, tests).

## API summary

Request parts: `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, `CAM_BACK`,
`CAM_BACK_RIGHT` (JPEG or PNG, exactly 1600×900), and `calibration`:

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

All six cameras are required. `ego2global` is optional (for all six cameras or none). Boxes are
returned in `ego@CAM_FRONT_LEFT`, with `center` as the geometric center; see
[docs/frames.md](docs/frames.md).

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

Environment variables `BEVSERVE_*` (see `src/bevserve/settings.py`):
- `HOST` (default `127.0.0.1`), `PORT`, `ALLOWED_BIND_HOSTS`;
- limits: `MAX_REQUEST_BYTES` (50 MiB), `MAX_IMAGE_BYTES` (8 MiB), `MAX_CALIBRATION_BYTES`
  (64 KiB);
- admission and timing: `MAX_IN_FLIGHT` (5), `DEADLINE_S` (10), `STALL_AFTER_S` (30);
- fatal exit: `FATAL_EXIT_GRACE_S`, `FATAL_EXIT_CODE` (70);
- `SERVER_TIMING` (false; adds a `Server-Timing` header for benchmarks, body unchanged);
- `PREPROCESS_THREADS`: 0 = serial, the default; 1–32 = thread pool; 12 is the verified and
  benchmarked value;
- model: `DETECTOR` (`fake` by default, or `bevdet`), with `BEVDET_CONFIG`, `CHECKPOINT`,
  `CHECKPOINT_SHA256` and optionally `CHECKPOINT_ALLOWED_DIRS` for the real model. Setup:
  [docs/env-m2.md](docs/env-m2.md) (host) or [docs/deploy.md](docs/deploy.md) (container).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/bevserve/` | the service (API, validation, preprocessing, executor, lifecycle, detectors, self-check) |
| `tests/` | unit, API and fake-Docker orchestration tests (public); `smoke/` (CPU, random weights); `private/` (GPU, never run in public CI) |
| `docker/` | GPU and CPU-smoke Dockerfiles, Compose service, pinned model-stack requirements, one mmcv patch |
| `tools/` | demo, parity and repeatability tools, container verification, benchmark harness, restricted-content guard |
| `configs/` | inference-only copy of BEVDet's model config (for the CPU smoke test) |
| `docs/` | architecture, reproducibility, deployment, verification records, milestone plans and reports |

## Documentation

| Topic | Documents |
| --- | --- |
| Design | [architecture.md](docs/architecture.md), [frames.md](docs/frames.md) |
| Reproduction and deployment | [reproducibility.md](docs/reproducibility.md), [deploy.md](docs/deploy.md), [host-setup.md](docs/host-setup.md), [env-m2.md](docs/env-m2.md) |
| Verification records | [verification-m3.md](docs/verification-m3.md), [verification-m4.md](docs/verification-m4.md), [m9-report.md](docs/m9-report.md) (release-preparation and fresh-clone verification) |
| Performance and reliability | [m5-report.md](docs/m5-report.md), [m6-report.md](docs/m6-report.md), [m7-report.md](docs/m7-report.md), [m8-report.md](docs/m8-report.md) |
| Plans (pre-registered criteria) | [m5-plan.md](docs/m5-plan.md), [m6-plan.md](docs/m6-plan.md), [m7-plan.md](docs/m7-plan.md), [m8-plan.md](docs/m8-plan.md), [m9-plan.md](docs/m9-plan.md) |

## Data and licensing

Code is Apache-2.0 (see `LICENSE` and `NOTICE`). This repository contains **no** dataset files, no
model weights and no files derived from nuScenes; tests use generated images and a made-up camera
rig.
- **Data and weights:** nuScenes is licensed CC BY-NC-SA 4.0 plus its Terms of Use. Obtain it, and
  the BEVDet checkpoint, from their owners and under their terms.
- **Container images:** no images are published. They would bundle third-party components
  (NVIDIA CUDA/cuDNN base images, mmcv's compiled operators, a few of which carry NVIDIA's own
  license), and their redistribution terms have not been verified. Build the image yourself from
  the pinned sources.
- **Restricted-content guard:** `tools/check_restricted.py` blocks weights, dataset files, large
  files and private paths. The CI workflow is configured to run it, and it was run locally during
  development.
