"""T1: public CPU smoke test of the real model code path, with random weights.

Needs torch, mmcv-full and BEVDet's mmdet3d importable; no dataset, no checkpoint, no GPU.
Skipped where the model stack is missing (the default .venv and public CI until a CPU model image
exists). bev_pool_v2 has no CPU kernel, so the pure-PyTorch version in bevserve.model.cpu_ops is
patched into the view transformer for the duration of the test, in this process only.

    CUDA_VISIBLE_DEVICES= python -m pytest -q tests/smoke
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mmdet3d")

# A CPU-only stack has no compiled bev_pool_v2 extension (BEVDet cannot build it without CUDA).
# Register an empty placeholder so BEVDet's modules import; the test replaces bev_pool_v2 itself
# with the pure-PyTorch version below, so the placeholder is never called.
_EXT = "mmdet3d.ops.bev_pool_v2.bev_pool_v2_ext"
try:
    importlib.import_module(_EXT)
except ImportError:
    sys.modules[_EXT] = types.ModuleType(_EXT)

from tests.synthetic import synthetic_image, synthetic_rig  # noqa: E402

from bevserve.constants import CAMERA_ORDER, CLASS_NAMES  # noqa: E402
from bevserve.detector import DetectorInfo  # noqa: E402
from bevserve.model.bevdet_detector import BEVDetDetector, build_unloaded_model  # noqa: E402
from bevserve.model.cpu_ops import cpu_bev_pool_v2  # noqa: E402
from bevserve.preprocess import ModelInputs, build_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/bevdet-r50-cbgs-inference.py"
GOLDEN = Path(__file__).with_name("golden_head_stats.json")
HEAD_KEYS = ("reg", "height", "dim", "rot", "vel", "heatmap")


@pytest.fixture(scope="module")
def cpu_model() -> Any:
    import mmdet3d.models.necks.view_transformer as vt
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    original = vt.bev_pool_v2
    vt.bev_pool_v2 = cpu_bev_pool_v2
    try:
        torch.manual_seed(0)
        model = build_unloaded_model(CONFIG).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        info = DetectorInfo(name="bevdet-r50-random", checkpoint_sha256=None, config_sha256=None)
        yield model, BEVDetDetector(model, torch.device("cpu"), info, LiDARInstance3DBoxes)
    finally:
        vt.bev_pool_v2 = original


@pytest.fixture(scope="module")
def inputs() -> ModelInputs:
    images = {n: synthetic_image(10 + i) for i, n in enumerate(CAMERA_ORDER)}
    return build_inputs(images, synthetic_rig(with_ego_poses=True))


def head_stats(model: Any, inputs: ModelInputs) -> dict[str, list[float]]:
    tensors = [torch.from_numpy(a).unsqueeze(0) for a in inputs.as_tuple()]
    with torch.no_grad():
        feats, _, _ = model.extract_feat(points=None, img=tensors, img_metas=None)
        task = model.pts_bbox_head(feats)[0][0]
    out = {}
    for key in HEAD_KEYS:
        t = task[key].double()
        out[key] = [list(t.shape), float(t.mean()), float(t.std()), float(t.abs().sum())]
    return out


def test_cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        pytest.skip("run with CUDA_VISIBLE_DEVICES= to prove the CPU path")
    assert not torch.cuda.is_available()


def test_forward_and_decode_run_on_cpu(cpu_model: Any, inputs: ModelInputs) -> None:
    model, detector = cpu_model
    assert next(model.parameters()).device.type == "cpu"
    boxes, scores, labels = detector.detect_native(inputs)
    assert boxes.ndim == 2 and boxes.shape[1] == 9 and len(boxes) == len(scores) == len(labels)
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    assert ((labels >= 0) & (labels < len(CLASS_NAMES))).all()
    assert (scores >= 0.1).all() and len(scores) <= 500


def test_cpu_path_is_deterministic(cpu_model: Any, inputs: ModelInputs) -> None:
    _, detector = cpu_model
    a, b = detector.detect_native(inputs), detector.detect_native(inputs)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_head_outputs_match_golden(cpu_model: Any, inputs: ModelInputs) -> None:
    """Random weights (seed 0) + synthetic input: any change to the model code path, config or
    preprocessing shows up here. After a reviewed change, regenerate with
    BEVSERVE_UPDATE_GOLDEN=1."""
    model, _ = cpu_model
    stats = head_stats(model, inputs)
    if os.environ.get("BEVSERVE_UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(stats, indent=2) + "\n")
    golden = json.loads(GOLDEN.read_text())
    for key in HEAD_KEYS:
        assert stats[key][0] == golden[key][0], key
        np.testing.assert_allclose(
            stats[key][1:], golden[key][1:], rtol=1e-4, atol=1e-6, err_msg=key
        )
