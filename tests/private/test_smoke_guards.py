"""Private guards for the public CPU smoke test (T1): its config copy and its CPU op stand-in
must match the real ones. Needs the BEVDet checkout (via BEVSERVE_BEVDET_CONFIG) and a GPU."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

if os.environ.get("BEVSERVE_PRIVATE_GPU") != "1":
    pytest.skip("private GPU tests: set BEVSERVE_PRIVATE_GPU=1", allow_module_level=True)
torch = pytest.importorskip("torch")
pytest.importorskip("mmdet3d")

ROOT = Path(__file__).resolve().parents[2]


def test_inference_config_copy_matches_upstream() -> None:
    from mmcv import Config

    upstream = Config.fromfile(os.environ["BEVSERVE_BEVDET_CONFIG"])
    copy = Config.fromfile(str(ROOT / "configs/bevdet-r50-cbgs-inference.py"))
    for key in ("model", "data_config", "grid_config", "class_names", "point_cloud_range"):
        assert copy[key] == upstream[key], key


def test_cpu_bev_pool_matches_cuda_kernel() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2

    from bevserve.model.cpu_ops import cpu_bev_pool_v2

    g = torch.Generator().manual_seed(0)
    b, n, d, h, w, c = 1, 6, 59, 16, 44, 64
    z, y, x = 1, 128, 128
    depth = torch.rand(b, n, d, h, w, generator=g)
    feat = torch.rand(b, n, h, w, c, generator=g)
    points = 200_000
    ranks_depth = torch.randint(0, depth.numel(), (points,), generator=g, dtype=torch.int32)
    ranks_feat = torch.randint(0, feat.numel() // c, (points,), generator=g, dtype=torch.int32)
    ranks_bev = torch.randint(0, z * y * x, (points,), generator=g, dtype=torch.int32)
    order = ranks_bev.argsort()
    ranks_depth, ranks_feat, ranks_bev = ranks_depth[order], ranks_feat[order], ranks_bev[order]
    kept = torch.ones(points, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    starts = torch.where(kept)[0].int()
    lengths = torch.diff(torch.cat([starts, torch.tensor([points], dtype=torch.int32)])).int()
    shape = (b, z, y, x, c)
    cpu = cpu_bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev, shape, starts, lengths)
    gpu = bev_pool_v2(
        depth.cuda(),
        feat.cuda(),
        ranks_depth.cuda(),
        ranks_feat.cuda(),
        ranks_bev.cuda(),
        shape,
        starts.cuda(),
        lengths.cuda(),
    ).cpu()
    assert cpu.shape == gpu.shape == (b, c, z, y, x)
    torch.testing.assert_close(cpu, gpu, rtol=1e-5, atol=1e-5)
