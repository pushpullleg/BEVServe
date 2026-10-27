# M2 environment record

The real detector runs in a dedicated conda environment, **`bevserve-m2`**, created on 2026-10-27
as an offline clone of the working BEVDet model environment (`conda create --clone ... --offline`).
The original environment was not modified (its `pip freeze` hash was identical before and after).

## Model stack (inherited unchanged from the clone source)

| Package | Version |
| --- | --- |
| Python | 3.10 |
| torch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 (CUDA 12.8, cuDNN 9.7.1) |
| mmcv-full | 1.7.2, built from source for sm_120, torch-2.7 stream patch |
| mmdet / mmsegmentation | 2.28.2 / 0.30.0 |
| mmdet3d | 1.0.0rc4, BEVDet dev3.0, editable install from the BEVDet checkout (read-only use) |
| numpy / Pillow / opencv-python | 1.24.4 / 12.3.0 / 4.8.1.78 |

Custom CUDA extensions: `mmcv._ext` and `mmdet3d/ops/bev_pool_v2/bev_pool_v2_ext` (sm_120 only, no
PTX). Both load and run in the clone.

## Added to the clone (service layer only)

Installed with exact pins after a `pip install --dry-run` showed that no model-stack package would
change:

`fastapi==0.162.0 starlette==1.7.0 pydantic==2.14.1 pydantic-settings==2.15.0
python-multipart==0.0.32 anyio==4.15.1 uvicorn==0.54.0 httpx==0.28.1 pytest==9.1.1`, plus their
pure-Python dependencies (pydantic_core 2.50.1, annotated-types 0.8.0, annotated-doc 0.0.5,
typing-inspection 0.4.4, httpcore 1.0.9, h11 0.16.0, idna 3.20, certifi 2026.10.23, click 8.5.0,
exceptiongroup 1.3.1, iniconfig 2.3.1, pluggy 1.7.0, Pygments 2.21.0, python-dotenv 1.2.4,
tomli 2.5.0, opentelemetry-api 1.45.1), and `bevserve` itself (`pip install --no-deps -e .`).

`bevserve` declares `opencv-python-headless==4.8.1.78`; the clone keeps the model stack's
`opencv-python==4.8.1.78` instead (same version and cv2 code, and two cv2 distributions must not be
installed together), so `pip check` reports that one expected difference. All other `pip check`
messages are inherited from the source environment.

## Run

```bash
conda activate bevserve-m2
export BEVSERVE_DETECTOR=bevdet
export BEVSERVE_BEVDET_CONFIG=/path/to/BEVDet/configs/bevdet/bevdet-r50-cbgs.py
export BEVSERVE_CHECKPOINT=/path/to/bevdet-r50-cbgs.pth
export BEVSERVE_CHECKPOINT_SHA256=<trusted sha256 of that file>
export BEVSERVE_CHECKPOINT_ALLOWED_DIRS='["/path/to/checkpoints"]'   # optional
bevserve                                                              # 127.0.0.1:8000
```
