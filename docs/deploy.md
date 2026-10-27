# Container deployment (M4)

## Images

| Image | Dockerfile | Purpose |
| --- | --- | --- |
| `bevserve:gpu` | `docker/Dockerfile.gpu` | Real BEVDet-R50 service on one NVIDIA GPU |
| `bevserve:cpu-smoke` | `docker/Dockerfile.cpu` | Public T1 smoke test, CPU only, random weights |

Pinned inputs (the build is reproducible from public sources):

| Input | Pin |
| --- | --- |
| CUDA base (build) | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3…2a85` |
| CUDA base (runtime) | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2934e…02bb` |
| CPU base | `ubuntu:22.04@sha256:e0f80c82…c9b` |
| PyTorch | 2.7.0+cu128 / 2.7.0+cpu (PyTorch index, installed without dependencies) |
| mmcv-full | open-mmlab/mmcv `4c01b026` (tag v1.7.2) + `docker/patches/mmcv-1.7.2-torch27-get-stream.patch` |
| BEVDet (mmdet3d) | public fork `a0142cc` (dev3.0 + Blackwell compatibility); model code and `bevdet-r50-cbgs.py` byte-identical to the M3-validated tree |
| Python packages | `docker/requirements-model.txt`: 104 exact pins from the M3-validated environment, installed with `--no-deps` |
| GPU architectures | `CUDA_ARCH_LIST=12.0` (Blackwell, validated). Other architectures must be added at build time and are not validated |

The GPU image is multi-stage: the build stage compiles mmcv and BEVDet's `bev_pool_v2`; the
runtime stage holds only `/opt/venv`, BEVDet's configs and BEVServe, runs as uid 10001, and
records `pip freeze` and `pip check` in `/opt/venv/`. No checkpoint, dataset or private file is in
the build context (`.dockerignore` is a whitelist), and no image layer contains one.

## Serving

```bash
cp docker/.env.example docker/.env      # set checkpoint dir, file and trusted sha256
docker compose -f docker/compose.yaml --env-file docker/.env up -d
curl http://127.0.0.1:8000/readyz
```

- **Networking.** Inside its network namespace the server binds `0.0.0.0` so that Docker's port
  forwarding can reach it; this is allowed only because the image allow-lists it
  (`BEVSERVE_ALLOWED_BIND_HOSTS`). On the host the port is published on `127.0.0.1` only. Run
  outside a container, the server still refuses any non-loopback address that is not
  allow-listed.
- **Weights.** Mounted read-only at `/models`; the path and the trusted sha256 come from the
  operator. The checkpoint must resolve inside `/models`, its hash is checked on the exact bytes
  loaded, and it is loaded with `torch.load(weights_only=True)`.
- **Hardening.** Read-only root filesystem with a 1 GiB `/tmp` tmpfs, all capabilities dropped,
  `no-new-privileges`, non-root user, memory and PID limits.
- **Restarts.** `restart: on-failure:5`. Docker restarts a container when its process exits, not
  when its health check fails. A fatal CUDA error makes BEVServe exit with code 70, so it is
  restarted; an initialization failure (wrong hash, no GPU, missing op) keeps the process up and
  not ready, so it does not crash-loop. Requests in flight or queued when the process exits are
  lost: there is no durable queue.
- **Health.** The Compose health check calls `/readyz`. It reports state; it does not restart.

## Offline verification (not the serving setup)

`--network none` is for verification runs only; a serving container needs its published port.

```bash
docker run --rm --network none --gpus device=0 -v /path/to/ckpt-dir:/models:ro \
  -e BEVSERVE_CHECKPOINT_SHA256=<sha> -v /path/to/golden:/golden:ro \
  --entrypoint python bevserve:gpu -m bevserve.selfcheck --golden /golden/selfcheck.json
```

`bevserve.selfcheck` loads and verifies the model, checks that the network is unreachable, runs a
fixed seeded input twice and compares boxes, scores, labels and a sha256 of the raw heatmap with a
golden file recorded in the validated host environment.

## Verification script

`tools/m4_container_checks.sh` runs every M4 container check with `sudo docker` (no docker-group
membership needed): build and provenance, offline self-check, Compose startup and readiness,
loopback-only publication, real-frame parity against the M3 golden, fatal exit and supervised
restart (`python -m bevserve.faultinject`, test-only), wrong hash, missing GPU, missing custom op,
and clean shutdown. Results go to `outputs/m4/` (git-ignored).
