"""Frame-by-frame comparison: BEVServe (preprocessing + HTTP API) vs BEVDet's own evaluation path.

Private tool: needs a local BEVDet dev3.0 checkout with nuScenes prepared, the model environment,
a GPU, and a running BEVServe (detector=bevdet). Results go to --out (default: a temp dir);
nothing restricted is written to the repository.

    cd /path/to/BEVDet   # info files use paths relative to the BEVDet root
    PYTHONDONTWRITEBYTECODE=1 python /path/to/bevserve/tools/compare_direct.py \\
        --checkpoint checkpoints/.../bevdet-r50-cbgs.pth --select scenes:50 \\
        --url http://127.0.0.1:8766 --out /tmp/cmp

Reference path = single_gpu_test's per-sample call: dataset pipeline -> collate ->
MMDataParallel(model)(return_loss=False, rescale=True, **data), with mmcv.load_checkpoint.
--select: "scenes:N" = middle frame of N scenes spread over the val scene list; "even:N" =
N frames evenly spaced over the val set; or a comma-separated index list.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
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
from bevserve.calibration import rig_matrices
from bevserve.constants import CAMERA_ORDER, CLASS_NAMES
from bevserve.imaging import decode_rgb
from bevserve.preprocess import build_inputs
from bevserve.schemas import CalibrationRequest

NAMES = ("imgs", "sensor2egos", "ego2globals", "intrins", "post_rots", "post_trans", "bda")
FIELDS = ("score", "center_xy_m", "z_bottom_m", "size_m", "yaw_rad", "velocity_mps")


def select(spec: str, infos: list[dict[str, Any]]) -> list[int]:
    if spec.startswith("even:"):
        n = int(spec.split(":")[1])
        return sorted({round(i * (len(infos) - 1) / (n - 1)) for i in range(n)})
    if spec.startswith("scenes:"):
        n = int(spec.split(":")[1])
        scenes: dict[str, list[int]] = {}
        for idx, info in enumerate(infos):
            scenes.setdefault(info["scene_token"], []).append(idx)
        ordered = list(scenes.values())
        picks = [ordered[round(i * (len(ordered) - 1) / (n - 1))] for i in range(n)]
        return [frames[len(frames) // 2] for frames in picks]
    return [int(x) for x in spec.split(",")]


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


def api_native(boxes: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """API boxes back to BEVDet's native layout (bottom-center z)."""
    arr = np.array(
        [[*b["center"], *b["size"], b["yaw"], *b["velocity"]] for b in boxes], dtype=np.float64
    ).reshape(-1, 9)
    arr[:, 2] -= 0.5 * arr[:, 5]
    return arr, np.array([b["score"] for b in boxes]), np.array([b["label_id"] for b in boxes])


def field_diffs(a: np.ndarray, r: np.ndarray, sa: np.ndarray, sr: np.ndarray) -> dict[str, float]:
    if not len(a):
        return dict.fromkeys(FIELDS, 0.0)
    d = np.abs(a - r)
    return {
        "score": float(np.abs(sa - sr).max()),
        "center_xy_m": float(d[:, :2].max()),
        "z_bottom_m": float(d[:, 2].max()),
        "size_m": float(d[:, 3:6].max()),
        "yaw_rad": float(d[:, 6].max()),
        "velocity_mps": float(d[:, 7:9].max()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bevdet/bevdet-r50-cbgs.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ann-file", default="data/nuscenes/bevdetv3-trainval_infos_val.pkl")
    p.add_argument("--select", default="scenes:50")
    p.add_argument("--url", default="http://127.0.0.1:8766")
    p.add_argument("--no-preprocess", action="store_true", help="skip the tensor comparison")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out_dir = Path(args.out or tempfile.mkdtemp(prefix="bevserve-cmp-"))
    out_dir.mkdir(parents=True, exist_ok=True)

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
    assert tuple(dataset.CLASSES) == CLASS_NAMES
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])

    indices = select(args.select, dataset.data_infos)
    rows: list[dict[str, Any]] = []
    client = httpx.Client(base_url=args.url, timeout=60)
    model_info = client.get("/v1/model").json()
    started = time.time()
    for n, idx in enumerate(indices):
        info = dataset.data_infos[idx]
        data = collate([dataset[idx]], samples_per_gpu=1)
        with torch.no_grad():
            ref = model(return_loss=False, rescale=True, **data)[0]["pts_bbox"]
        r_boxes = ref["boxes_3d"].tensor.numpy().astype(np.float64)
        r_scores = ref["scores_3d"].numpy().astype(np.float64)
        r_labels = ref["labels_3d"].numpy()
        order = np.argsort(-r_scores, kind="stable")
        r_boxes, r_scores, r_labels = r_boxes[order], r_scores[order], r_labels[order]

        images, calibration = request_parts(info)
        row: dict[str, Any] = {"index": idx, "scene": info["scene_token"][:8]}
        rig = CalibrationRequest.model_validate(calibration).to_rig()
        _, e2g, _ = rig_matrices(rig)
        row["ego_pose_spread_m"] = float(np.ptp(e2g[:, :3, 3], axis=0).max())
        if not args.no_preprocess:
            ours = build_inputs({c: decode_rgb(images[c]) for c in CAMERA_ORDER}, rig).as_tuple()
            ref_inputs = [t[0].numpy() for t in data["img_inputs"][0]]
            row["preprocess"] = {
                name: float(np.abs(r.astype(np.float64) - o.astype(np.float64)).max())
                for name, r, o in zip(NAMES, ref_inputs, ours, strict=True)
            }

        rid = f"cmp-{idx}"
        files = [(c, (f"{c}.jpg", images[c], "image/jpeg")) for c in CAMERA_ORDER]
        resp = client.post(
            "/v1/detect",
            files=files,
            data={"calibration": json.dumps(calibration)},
            headers={"X-Request-ID": rid},
        )
        row["status"] = resp.status_code
        if resp.status_code == 200:
            body = resp.json()
            a_boxes, a_scores, a_labels = api_native(body["boxes"])
            row.update(
                api_count=len(a_scores),
                ref_count=len(r_scores),
                labels_equal=bool(len(a_labels) == len(r_labels) and (a_labels == r_labels).all()),
                identity_ok=(
                    body["request_id"] == rid == resp.headers.get("x-request-id")
                    and body["model"]["checkpoint_sha256"] == model_info["checkpoint_sha256"]
                    and body["frame"] == "ego@CAM_FRONT_LEFT"
                    and body["units"] == {"length": "m", "angle": "rad", "velocity": "m/s"}
                ),
            )
            if len(a_scores) == len(r_scores):
                row["max_abs_diff"] = field_diffs(a_boxes, r_boxes, a_scores, r_scores)
        rows.append(row)
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(indices)} frames, {time.time() - started:.0f}s", flush=True)

    (out_dir / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ok = [r for r in rows if r["status"] == 200 and r.get("max_abs_diff") is not None]
    summary: dict[str, Any] = {
        "frames": len(rows),
        "distinct_scenes": len({r["scene"] for r in rows}),
        "http_200": sum(r["status"] == 200 for r in rows),
        "count_equal": sum(r.get("api_count") == r.get("ref_count") for r in rows),
        "labels_equal": sum(bool(r.get("labels_equal")) for r in rows),
        "identity_ok": sum(bool(r.get("identity_ok")) for r in rows),
        "total_boxes": sum(r.get("ref_count", 0) for r in rows),
        "max_abs_diff": {f: max(r["max_abs_diff"][f] for r in ok) for f in FIELDS} if ok else None,
        "ego_pose_spread_m": [
            min(r["ego_pose_spread_m"] for r in rows),
            max(r["ego_pose_spread_m"] for r in rows),
        ],
        "seconds": round(time.time() - started, 1),
    }
    if not args.no_preprocess:
        summary["preprocess_max_abs_diff"] = {
            n: max(r["preprocess"][n] for r in rows) for n in NAMES
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
