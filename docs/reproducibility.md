# Reproducibility

What can be run, with what, and what was actually verified. Levels L0 and L1 need no GPU, model or
dataset. Levels L2–L4 need a GPU and assets that you must obtain yourself; they were verified on
one workstation only.

| Level | What runs | Needs | Verified |
| --- | --- | --- | --- |
| **L0** | install, restricted-content guard, lint, format, type check, public test suite, FakeDetector server and demo, `/docs` | Linux, a CPython 3.10 interpreter (not the default on current distributions), `curl`, `ss`, `setsid`; network access for pip | yes, on the workstation; a fresh clone is the M9 Stage C gate |
| **L1** | CPU smoke test: real BEVDet model code with **random weights** and a pure-PyTorch `bev_pool_v2`, synthetic input | L0, plus building mmcv-full 1.7.2 from source (slow) and fetching BEVDet `a0142cc` | stack built locally; CI workflow never run |
| **L2** | real inference with your checkpoint, on a host or in the GPU container | NVIDIA GPU (only sm_120 built and validated), Docker + NVIDIA Container Toolkit (container) or the BEVDet model environment (host), checkpoint + trusted sha256 | one workstation |
| **L3** | container verification suite and benchmark procedures (M4–M8) | L2, plus nuScenes val, a host model environment, `sudo docker`, the exact workstation kernel (M4 script) | one workstation; not portable without changes |
| **L4** | numerical parity against BEVDet's own evaluation | L2, plus nuScenes, a BEVDet dev3.0 checkout and its environment | once, 2026-10-27 ([verification-m3.md](verification-m3.md)) |

## L0: CPU only, no assets

`python3.10` below means any CPython 3.10 interpreter. If none is on your `PATH`, install one (for
example with pyenv, uv, conda or a distribution package) and use its path. The M9 fresh-clone check
used a conda-provided CPython 3.10.21 as the base of a new, isolated virtual environment.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python tools/check_restricted.py
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src tests tools
.venv/bin/mypy
.venv/bin/python -m pytest -q
```

**Expected test result:** 313 passed and 4 skipped. The skips are the 3 private GPU test files and
the T1 smoke test, which needs torch. The run takes about 6–10 minutes; the fake-Docker
orchestration tests start local servers and write temporary run directories under `outputs/`,
which they remove.

**API exploration** (FakeDetector, synthetic boxes):

```bash
.venv/bin/bevserve &                          # 127.0.0.1:8000
.venv/bin/python tools/demo.py --url http://127.0.0.1:8000
curl -s http://127.0.0.1:8000/v1/model        # "name": "fake-detector"
kill %1
```

The demo sends one valid request and three invalid ones (5 cameras, wrong image size, bad
quaternion), so you see 200 and three 422 problem responses. `tools/demo.py` imports test helpers
from `tests/`, so it needs a source checkout.

**Notes:**
- **CI:** these are the same steps as `.github/workflows/ci.yml`. That workflow has never run,
  because the repository has not been pushed.
- **Python version:** dependencies are pinned exactly and require Python 3.10. numpy, Pillow and
  OpenCV match the model environment, because they change pixel values.

## L1: CPU smoke test of the real model code

```bash
python3.10 -m venv .venv-cpu && PYTHON=.venv-cpu/bin/python bash docker/build_cpu_stack.sh
CUDA_VISIBLE_DEVICES= .venv-cpu/bin/python -m pytest -q tests/smoke
# or, with Docker:
docker build -f docker/Dockerfile.cpu -t bevserve:cpu-smoke . && docker run --rm bevserve:cpu-smoke
```

**What the build does:**
- installs PyTorch 2.7.0+cpu;
- builds mmcv-full 1.7.2 (`4c01b026` plus `docker/patches/…`) with CPU ops only;
- fetches BEVDet's Python sources.

**What the test does:** builds the model from `configs/bevdet-r50-cbgs-inference.py` with random
weights, substitutes a pure-PyTorch `bev_pool_v2` (`src/bevserve/model/cpu_ops.py`), and compares
head statistics with `tests/smoke/golden_head_stats.json`.

This tests that the code path runs, not the model's accuracy.

## L2: real inference

**Assets you must obtain:**
- the BEVDet-R50-CBGS checkpoint, from the BEVDet project, under its terms;
- its sha256, which you must trust independently — BEVServe refuses to become ready if the hash
  differs.

**Container** ([deploy.md](deploy.md)):

```bash
docker build -f docker/Dockerfile.gpu -t bevserve:gpu .         # first full build: 1,360 s on the test workstation
cp docker/.env.example docker/.env                               # checkpoint dir, file, sha256
docker compose -f docker/compose.yaml --env-file docker/.env up -d
curl http://127.0.0.1:8000/readyz
```

**Host:** a BEVDet dev3.0 model environment with the service packages added; see
[env-m2.md](env-m2.md). Set `BEVSERVE_DETECTOR=bevdet`, `BEVSERVE_BEVDET_CONFIG`,
`BEVSERVE_CHECKPOINT` and `BEVSERVE_CHECKPOINT_SHA256`.

**Constraints:**

| Constraint | Detail |
| --- | --- |
| GPU architecture | The image is built for `CUDA_ARCH_LIST=12.0` (Blackwell, sm_120) only. Other GPUs need a rebuild with a different list (see `docker/Dockerfile.gpu`); they have **not** been validated |
| CUDA and driver | CUDA 12.8 base images; the host driver must support CUDA 12.8 (the test host ran 570.211.01) |
| Pinned BEVDet source | The build fetches BEVDet commit `a0142cc` (dev3.0 plus a Blackwell compatibility change) from the author's public fork, `github.com/pushpullleg/BEVDet`, where it currently sits on work-package branches. If that commit is no longer reachable, the GPU image and the L1 stack do not build |
| Host setup | The test workstation's Docker and NVIDIA toolkit installation is recorded in [host-setup.md](host-setup.md) as an example, not a requirement |

## L3: verification suite and benchmarks (workstation-bound)

**Container verification:** `tools/m4_container_checks.sh`.

It builds a new image and runs build provenance, layer scans, offline self-check, Compose
deployment, real-frame parity, fault restart, initialization failures and clean shutdown. It
requires:
- `BEVDET_ROOT` (a BEVDet checkout with `data/nuscenes/bevdetv3-trainval_infos_val.pkl`);
- `CKPT_DIR`, `CKPT_SHA256`;
- `GOLDEN_DIR`, holding:
  - `selfcheck.json` from `python -m bevserve.selfcheck --record`;
  - `http_golden.json` from `tools/golden_http.py record`, both recorded in a validated host
    environment;
- `HOST_PY`, a host Python with `httpx`, `mmcv` and `numpy`;
- `sudo docker` (or `DOCKER_CMD`), `docker compose`, `buildx` and the NVIDIA runtime.

**It stops unless the kernel is exactly `6.14.0-37-generic`** — the test workstation's pinned
kernel, a safety check for that host. On any other host the check fails by design; running it
elsewhere needs that check changed, which this repository does not do.

**Benchmarks:** `tools/m5_bench.sh` (M5–M8 procedures; header comment and plans `docs/m5-plan.md`
to `docs/m8-plan.md`). It needs a verified image from the suite above, the same inputs, and
nuScenes frames. Its results depend on the host. The published numbers are from one workstation
with the client on the same host. The M8 `m8-ab` stage was removed with the experiment (Option A).

## L4: numerical parity

Run from a BEVDet dev3.0 checkout with nuScenes prepared and the model environment active:

| Tool | Checks |
| --- | --- |
| `tools/reference_check_bevdet.py` | preprocessing vs BEVDet's own pipeline, synthetic inputs, CPU |
| `tools/compare_direct.py --select scenes:50` / `even:200` | HTTP responses vs BEVDet's `single_gpu_test` path, frame by frame |
| `tools/full_val_parity.py` | all 6,019 val frames through BEVServe's input path and adapter, scored by BEVDet's evaluator |
| `tools/repeatability.py --index 0 --repeats 20` | bit-identical repeats over HTTP and in process |
| `tools/m6_input_parity.py` | serial vs pooled preprocessing inputs on all val frames |

The 2026-10-27 results are in [verification-m3.md](verification-m3.md). Expect bit-identical
results only with the same checkpoint, code, library versions and GPU type; other environments
can differ in floating point.

## Known failure modes

| Symptom | Cause | Fix or status |
| --- | --- | --- |
| `/readyz` reports `model_stack_missing` in a fresh Ubuntu-based image | `python3-lib2to3` / `python3-distutils` missing (mmcv → yapf → lib2to3; torch extensions → distutils) | installed in the GPU image; a build-time import guard checks the whole model path |
| `/readyz` reports `checkpoint_hash_mismatch` | wrong file or wrong trusted hash | by design: the service stays up and not ready |
| `/readyz` reports `cuda_unavailable` | no GPU visible to the container (`--gpus` / NVIDIA toolkit) | host configuration |
| Image starts but CUDA kernels fail on another GPU | built for sm_120 only | rebuild with your architecture added (unvalidated) |
| `pip install` fails | Python other than 3.10 | use Python 3.10 |
| `pip check` warning about OpenCV in a model environment | `opencv-python` (model stack) vs `opencv-python-headless` (BEVServe) | expected; same version and code ([env-m2.md](env-m2.md)) |
| M4 script exits at "kernel is not the pinned 6.14.0-37-generic" | workstation-specific safety check | see L3 |
