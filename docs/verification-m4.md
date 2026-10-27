# M4 verification record (2026-10-27)

Containerized BEVServe on one RTX PRO 6000 Blackwell (sm_120), driver 570.211.01, kernel
6.14.0-37 (unchanged), Docker 29.1.3 with NVIDIA Container Toolkit 1.18.1, run with `sudo docker`
by the operator. Raw logs and outputs on nuScenes data stay local (`outputs/`, not committed).

## Verified image

| Item | Value |
| --- | --- |
| Image | `bevserve:m4-20261027-112843`, id `sha256:2f0ffba3a83ebef6a45bfab5fb3f05afe1df1c7e98a700cf29dea7286e9c12ed` |
| Size | 7,292,001,866 bytes (6.8 GiB), 13 layers |
| Build | 542 s (build stage from cache; first full build 1,360 s) |
| Stack | torch 2.7.0+cu128, CUDA 12.8, mmcv-full 1.7.2 @ 4c01b026 (+patch), BEVDet @ a0142cc, sm_120 |
| Checkpoint | operator-mounted, sha256 `8b67e60d…3443` (not in the image) |
| Golden | M3-validated host responses, 10 val frames, 1,209 boxes; file sha256 `603c18ec…` (unchanged) |

## Results

| Check | Result |
| --- | --- |
| Inputs, prerequisites, GPU in container | pass |
| Build; dependency manifest; non-root (uid 10001) | pass |
| Layer scan, runtime image (13 layers, 51,572 files) and build stage (20 layers, 60,821 files) | pass, no findings |
| Offline self-check, `--network none` | pass: ops load, checkpoint identity, raw heatmap sha256 bit-identical to the host, network unreachable |
| Startup to ready | 4 s |
| Restart policy | `on-failure:5` |
| Publication | `127.0.0.1:8000` only; host listener loopback only |
| Model identity | pass |
| **Real-frame HTTP parity vs M3** | **pass (targeted retest): 10/10 frames HTTP 200, 1,209/1,209 boxes byte-identical, identity, frame and units equal** |
| Fatal error | call 2 -> 500 `inference_fatal`; process exit code 70 (Docker `die` event); restart count 0 -> 1; ready again in 4 s |
| Wrong checkpoint hash / missing GPU / missing custom op (running service) | up, not ready with `checkpoint_hash_mismatch` / `cuda_unavailable` / `model_stack_missing`, 0 restarts |
| Clean shutdown | exit 0 in 1 s |

Observed while serving (one sample, not a benchmark): container memory 1.6 GiB of a 16 GiB
limit, GPU memory 1,228 MiB.

## Failures on the way, causes and fixes

| Run | Failure | Root cause (evidence) | Fix |
| --- | --- | --- | --- |
| 20261027-101109 | offline self-check; startup-ready | Runtime stage lacked Ubuntu's `python3-lib2to3` and `python3-distutils` (split out of the standard library). mmcv `Config` -> yapf -> `lib2to3`; mmcv `parrots_wrapper` -> `torch.utils.cpp_extension` -> setuptools -> `distutils.core`. Init reported `model_stack_missing`, so `/readyz` never turned 200. Confirmed on the preserved image by `tools/m4_diagnose_lib2to3.sh` | Install both packages in the runtime stage; build-time guard importing the whole model path (both compiled extensions included) |
| 20261027-112843 | http-golden-parity | Harness bug: a relative `GOLDEN_DIR` stopped resolving after the parity step's `cd` into the BEVDet checkout; zero frames were compared (`FileNotFoundError`, empty result) | All path inputs made absolute after validation; `tools/m4_parity_retest.sh` reruns parity on the built image |
| 20261027-115741 (retest) | none | | parity pass on the same image tag, checkpoint and golden |

The image id of the retest was not recorded by that version of the retest script (the tag is unique
to run 20261027-112843); the script now records it.
