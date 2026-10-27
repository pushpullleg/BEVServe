"""Pure-PyTorch stand-in for BEVDet's CUDA-only bev_pool_v2 forward, for CPU tests (T1).

Semantics of mmdet3d/ops/bev_pool_v2 (forward): out[b, z, y, x, c] = sum over points p with
ranks_bev[p] == that voxel of depth.flat[ranks_depth[p]] * feat.flat_rows[ranks_feat[p], c];
returned as (B, C, Z, Y, X). interval_starts/lengths only group equal ranks_bev for the CUDA
kernel; a scatter-add over every point gives the same sums (up to float summation order).

Used only by tests, by patching the name in the view transformer at test time. It is never used
by the service and never modifies BEVDet's code on disk.
"""

from __future__ import annotations

from typing import Any


def cpu_bev_pool_v2(
    depth: Any,
    feat: Any,
    ranks_depth: Any,
    ranks_feat: Any,
    ranks_bev: Any,
    bev_feat_shape: Any,
    interval_starts: Any,
    interval_lengths: Any,
) -> Any:
    b, z, y, x, c = (int(v) for v in bev_feat_shape)
    weights = depth.reshape(-1)[ranks_depth.long()]
    rows = feat.reshape(-1, c)[ranks_feat.long()]
    out = feat.new_zeros(b * z * y * x, c)
    out.index_add_(0, ranks_bev.long(), weights[:, None] * rows)
    return out.view(b, z, y, x, c).permute(0, 4, 1, 2, 3).contiguous()
