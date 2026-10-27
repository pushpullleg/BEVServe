"""BEVDet-R50 detector (M2): the only module that imports torch, mmcv and mmdet3d.

Load sequence (mirrors the thesis evaluation path, with safer checkpoint handling):
1. resolve the operator-configured config file; read the checkpoint once and verify its sha256
   against the operator-configured trusted value (bevserve.checkpoint);
2. Config.fromfile -> compat_cfg -> pretrained=None, train_cfg=None -> build_model;
3. torch.load(weights_only=True) of the verified bytes, strict key match, mmcv load_state_dict;
4. eval mode, parameters frozen, moved to the configured CUDA device;
5. warm-up passes on a synthetic input.
Steps 2-3 run with network download helpers disabled, so a config that tries to fetch weights
fails instead of downloading.

Inference runs BEVDet's own forward_test unchanged with img_metas = [[{box_type_3d}]], the only
metadata the CenterHead decoder reads.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bevserve.boxes import detections_from_arrays
from bevserve.checkpoint import read_verified_checkpoint, resolve_operator_file, sha256_file
from bevserve.constants import (
    CAMERA_ORDER,
    CLASS_NAMES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    SRC_HEIGHT,
    SRC_WIDTH,
)
from bevserve.detector import (
    Detection,
    DetectorError,
    DetectorInfo,
    DetectorInitError,
    FatalDetectorError,
)
from bevserve.preprocess import ModelInputs, bda_matrix, image_post_transform
from bevserve.settings import Settings

log = logging.getLogger("bevserve.model")
MODEL_NAME = "bevdet-r50-cbgs"

# CUDA failures after which the context is unusable. Out-of-memory is deliberately excluded:
# PyTorch recovers from it, so it is a per-request error.
_FATAL_CUDA = re.compile(
    r"illegal memory access|device-side assert|unspecified launch failure|misaligned address"
    r"|illegal instruction|uncorrectable ECC|cudaErrorLaunch|CUDA driver error"
    r"|CUBLAS_STATUS_EXECUTION_FAILED|CUDNN_STATUS_EXECUTION_FAILED|CUDNN_STATUS_INTERNAL_ERROR",
    re.IGNORECASE,
)


def classify_runtime_error(exc: BaseException) -> type[DetectorError] | type[FatalDetectorError]:
    """FatalDetectorError for unrecoverable CUDA errors, DetectorError for everything else."""
    message = str(exc)
    if "out of memory" in message.lower():
        return DetectorError
    return FatalDetectorError if _FATAL_CUDA.search(message) else DetectorError


def warmup_inputs() -> ModelInputs:
    """Synthetic, data-free input with valid geometry (identity extrinsics)."""
    n = len(CAMERA_ORDER)
    k = np.array(
        [[1250.0, 0.0, SRC_WIDTH / 2], [0.0, 1250.0, SRC_HEIGHT / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rot, tran = image_post_transform()
    eye = np.eye(4, dtype=np.float32)
    return ModelInputs(
        imgs=np.zeros((n, 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32),
        sensor2egos=np.stack([eye] * n),
        ego2globals=np.stack([eye] * n),
        intrins=np.stack([k] * n),
        post_rots=np.stack([rot] * n),
        post_trans=np.stack([tran] * n),
        bda=bda_matrix(),
    )


@contextlib.contextmanager
def _downloads_disabled() -> Iterator[None]:
    import mmcv.runner.checkpoint as mmcv_ckpt
    import torch.hub

    def refuse(*_: object, **__: object) -> None:
        raise DetectorInitError("network_download_attempted", "model build tried to download")

    targets: list[tuple[Any, str]] = [
        (torch.hub, "load_state_dict_from_url"),
        (torch.hub, "download_url_to_file"),
        (mmcv_ckpt, "load_from_http"),
        (mmcv_ckpt, "load_url"),
    ]
    saved = [(obj, name, getattr(obj, name)) for obj, name in targets if hasattr(obj, name)]
    try:
        for obj, name, _ in saved:
            setattr(obj, name, refuse)
        yield
    finally:
        for obj, name, original in saved:
            setattr(obj, name, original)


class BEVDetDetector:
    def __init__(self, model: Any, device: Any, info: DetectorInfo, box_type: Any) -> None:
        import torch

        self._torch = torch
        self._model = model
        self._device = device
        self._info = info
        self._meta = {"box_type_3d": box_type}

    @property
    def info(self) -> DetectorInfo:
        return self._info

    def detect_native(self, inputs: ModelInputs) -> tuple[Any, Any, Any]:
        """BEVDet's own output: (N, 9) boxes with bottom-center z, scores, labels (numpy)."""
        inputs.check()
        torch = self._torch
        try:
            with torch.no_grad():
                tensors = [
                    torch.from_numpy(a).unsqueeze(0).to(self._device) for a in inputs.as_tuple()
                ]
                result = self._model.forward_test(
                    points=[None], img_metas=[[self._meta]], img_inputs=[tensors], rescale=True
                )
                pts = result[0]["pts_bbox"]
                boxes = pts["boxes_3d"].tensor.detach().cpu().numpy()
                scores = pts["scores_3d"].detach().cpu().numpy()
                labels = pts["labels_3d"].detach().cpu().numpy()
        except Exception as exc:
            kind = classify_runtime_error(exc)
            raise kind(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        return boxes, scores, labels

    def detect(self, inputs: ModelInputs) -> Sequence[Detection]:
        return detections_from_arrays(*self.detect_native(inputs))


def _check_config(cfg: Any) -> None:
    """The preprocessing and output contract assume this exact model configuration."""
    checks = {
        "cams": tuple(cfg.data_config["cams"]) == CAMERA_ORDER,
        "Ncams": cfg.data_config["Ncams"] == len(CAMERA_ORDER),
        "input_size": tuple(cfg.data_config["input_size"]) == (INPUT_HEIGHT, INPUT_WIDTH),
        "src_size": tuple(cfg.data_config["src_size"]) == (SRC_HEIGHT, SRC_WIDTH),
        "resize_test": float(cfg.data_config.get("resize_test", 0.0)) == 0.0,
        "crop_h": tuple(cfg.data_config.get("crop_h", (0.0, 0.0))) == (0.0, 0.0),
        "class_names": tuple(cfg.class_names) == CLASS_NAMES,
        "model_type": cfg.model.type == "BEVDet",
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise DetectorInitError("config_mismatch", f"config differs from the contract: {failed}")


def build_unloaded_model(config_path: Path) -> Any:
    """Config -> contract check -> BEVDet model with no weights loaded (as in the thesis path).

    compat_cfg only touches data/runner settings and is applied when the config has them.
    Call inside _downloads_disabled() unless the caller controls network access itself.
    """
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(str(config_path))
    if "data" in cfg:
        try:
            from mmdet.utils import compat_cfg
        except ImportError:
            from mmdet3d.utils import compat_cfg
        cfg = compat_cfg(cfg)
    _check_config(cfg)
    cfg.model.pretrained = None
    if "img_backbone" in cfg.model:
        cfg.model.img_backbone.pretrained = None
    cfg.model.train_cfg = None
    return build_model(cfg.model, test_cfg=cfg.get("test_cfg"))


def load_bevdet(settings: Settings) -> BEVDetDetector:
    """Build, verify, load and warm up BEVDet-R50. Raises DetectorInitError with a reason."""
    try:
        import torch
        from mmcv.runner.checkpoint import load_state_dict
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from mmdet3d.ops.bev_pool_v2 import bev_pool  # noqa: F401  (compiled op present)
    except ImportError as exc:
        raise DetectorInitError("model_stack_missing", f"cannot import {exc.name}") from exc

    if not torch.cuda.is_available():
        raise DetectorInitError("cuda_unavailable", "no CUDA device visible")
    device = torch.device(settings.device)
    if device.type != "cuda" or (device.index or 0) >= torch.cuda.device_count():
        raise DetectorInitError("device_invalid", f"device {settings.device!r} is not available")

    config_path = resolve_operator_file(settings.bevdet_config, "config")
    data, ckpt_sha = read_verified_checkpoint(
        settings.checkpoint,
        settings.checkpoint_sha256,
        settings.checkpoint_allowed_dirs,
        settings.checkpoint_max_bytes,
    )

    with _downloads_disabled():
        model = build_unloaded_model(config_path)
        try:
            checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        except Exception as exc:  # pickle / unsafe-global errors: never retry unsafely
            raise DetectorInitError(
                "checkpoint_unsafe_or_corrupt",
                f"weights_only load refused the checkpoint ({type(exc).__name__})",
            ) from exc
    del data
    state = checkpoint.get("state_dict", checkpoint)
    state = {re.sub(r"^module\.", "", k): v for k, v in state.items()}
    expected = set(model.state_dict())
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise DetectorInitError(
            "checkpoint_mismatch",
            f"{len(missing)} missing and {len(unexpected)} unexpected keys",
        )
    load_state_dict(model, state, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    info = DetectorInfo(
        name=MODEL_NAME,
        checkpoint_sha256=ckpt_sha,
        config_sha256=sha256_file(config_path),
        class_names=CLASS_NAMES,
    )
    detector = BEVDetDetector(model, device, info, LiDARInstance3DBoxes)

    inputs = warmup_inputs()
    try:
        for i in range(settings.warmup_runs):
            started = time.perf_counter()
            detector.detect(inputs)
            torch.cuda.synchronize(device)
            log.info("warm-up %d: %.1f ms", i + 1, (time.perf_counter() - started) * 1000)
    except (DetectorError, FatalDetectorError) as exc:
        raise DetectorInitError("warmup_failed", str(exc)[:200]) from exc
    log.info(
        "BEVDet ready on %s: checkpoint %s..., peak allocated %.0f MiB",
        device,
        ckpt_sha[:12],
        torch.cuda.max_memory_allocated(device) / 2**20,
    )
    return detector
