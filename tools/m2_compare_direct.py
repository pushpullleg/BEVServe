"""M2 check: one real nuScenes val sample through the API vs BEVDet's own evaluation path.

Private tool: needs a local BEVDet dev3.0 checkout with nuScenes prepared, the model environment,
a GPU, and a running BEVServe (detector=bevdet). Nothing restricted is written to the repository:
results go to --out (default: a temporary directory).

    cd /path/to/BEVDet   # info files use paths relative to the BEVDet root
    PYTHONDONTWRITEBYTECODE=1 python /path/to/bevserve/tools/m2_compare_direct.py \\
        --checkpoint checkpoints/.../bevdet-r50-cbgs.pth --index 0 --url http://127.0.0.1:8766

Reference path = single_gpu_test's per-sample call: dataset pipeline -> collate ->
MMDataParallel(model)(return_loss=False, rescale=True, **data), with mmcv.load_checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bevserve.constants import CAMERA_ORDER
from bevserve.imaging import decode_rgb
from bevserve.preprocess import build_inputs
from bevserve.schemas import CalibrationRequest

NAMES = ("imgs", "sensor2egos", "ego2globals", "intrins", "post_rots", "post_trans", "bda")


def reference(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[np.ndarray]]:
    cfg = Config.fromfile(args.config)
    try:
        from mmdet.utils import compat_cfg
    except ImportError:
        from mmdet3d.utils import compat_cfg
    cfg = compat_cfg(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])
    sample = dataset[args.index]
    data = collate([sample], samples_per_gpu=1)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]["pts_bbox"]
    img_inputs = [t[0].numpy() for t in data["img_inputs"][0]]
    out = {
        "boxes": result["boxes_3d"].tensor.numpy(),
        "scores": result["scores_3d"].numpy(),
        "labels": result["labels_3d"].numpy(),
    }
    return out, dataset.data_infos[args.index], img_inputs


def request_parts(info: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    images, cameras = {}, {}
    for name in CAMERA_ORDER:
        cam = info["cams"][name]
        images[name] = Path(cam["data_path"]).read_bytes()
        cameras[name] = {
            "intrinsic": np.asarray(cam["cam_intrinsic"]).tolist(),
            "sensor2ego": {
                "rotation_wxyz": list(cam["sensor2ego_rotation"]),
                "translation_m": list(cam["sensor2ego_translation"]),
            },
            "ego2global": {
                "rotation_wxyz": list(cam["ego2global_rotation"]),
                "translation_m": list(cam["ego2global_translation"]),
            },
        }
    return images, {"schema_version": "1", "cameras": cameras}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bevdet/bevdet-r50-cbgs.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ann-file", default="data/nuscenes/bevdetv3-trainval_infos_val.pkl")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--url", default="http://127.0.0.1:8766")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out_dir = Path(args.out or tempfile.mkdtemp(prefix="bevserve-m2-"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ref, info, ref_inputs = reference(args)
    images, calibration = request_parts(info)

    # 1. Preprocessing on real images: BEVServe vs the dataset pipeline.
    rig = CalibrationRequest.model_validate(calibration).to_rig()
    ours = build_inputs({n: decode_rgb(images[n]) for n in CAMERA_ORDER}, rig).as_tuple()
    pre = {
        n: float(np.abs(r.astype(np.float64) - o.astype(np.float64)).max())
        for n, r, o in zip(NAMES, ref_inputs, ours, strict=True)
    }

    # 2. The API on the same sample.
    files = [(n, (f"{n}.jpg", images[n], "image/jpeg")) for n in CAMERA_ORDER]
    resp = httpx.post(
        f"{args.url}/v1/detect",
        files=files,
        data={"calibration": json.dumps(calibration)},
        timeout=60,
    )
    resp.raise_for_status()
    api = resp.json()

    # 3. Compare boxes (both sorted by descending score; API center z = bottom z + h/2).
    order = np.argsort(-ref["scores"], kind="stable")
    r_boxes, r_scores, r_labels = ref["boxes"][order], ref["scores"][order], ref["labels"][order]
    a = api["boxes"]
    a_boxes = np.array(
        [[*b["center"], *b["size"], b["yaw"], *b["velocity"]] for b in a], dtype=np.float64
    ).reshape(-1, 9)
    if len(a_boxes):
        a_boxes[:, 2] -= 0.5 * a_boxes[:, 5]
    a_scores = np.array([b["score"] for b in a])
    a_labels = np.array([b["label_id"] for b in a])
    same_count = len(a) == len(r_scores)
    report: dict[str, Any] = {
        "index": args.index,
        "preprocess_max_abs_diff": pre,
        "box_count": {"api": len(a), "reference": len(r_scores)},
        "frame": api["frame"],
        "units": api["units"],
        "model_version": resp.headers.get("x-model-version"),
        "timing_ms": api["timing_ms"],
    }
    if same_count and len(a):
        report.update(
            labels_equal=bool((a_labels == r_labels).all()),
            max_abs_diff={
                "score": float(np.abs(a_scores - r_scores).max()),
                "center_xy_m": float(np.abs(a_boxes[:, :2] - r_boxes[:, :2]).max()),
                "z_bottom_m": float(np.abs(a_boxes[:, 2] - r_boxes[:, 2]).max()),
                "size_m": float(np.abs(a_boxes[:, 3:6] - r_boxes[:, 3:6]).max()),
                "yaw_rad": float(np.abs(a_boxes[:, 6] - r_boxes[:, 6]).max()),
                "velocity_mps": float(np.abs(a_boxes[:, 7:9] - r_boxes[:, 7:9]).max()),
            },
            top_box=a[0],
        )
    (out_dir / f"m2_compare_{args.index}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "top_box"}, indent=2))
    print("top API box:", json.dumps(report.get("top_box")))
    print("saved to", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
