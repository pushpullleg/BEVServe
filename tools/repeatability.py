"""Repeatability of real inference: the same frame N times, over HTTP and in-process.

Private tool (GPU, local nuScenes, running BEVServe for the HTTP part). Run from the BEVDet
root with the BEVSERVE_* model settings exported:

    python /path/to/bevserve/tools/repeatability.py --index 0 --repeats 20

Reports box-count / label changes and the max difference per field, and whether outputs are
bit-identical. Also repeats with other frames interleaved, to catch state carried over between
calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import mmcv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_direct import request_parts
from full_val_parity import prepare

from bevserve.constants import CAMERA_ORDER
from bevserve.model.bevdet_detector import load_bevdet
from bevserve.settings import Settings


def spread(runs: list[np.ndarray]) -> float:
    return float(max(np.abs(r.astype(np.float64) - runs[0]).max(initial=0.0) for r in runs))


def summarize(runs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    counts = {len(s) for _, s, _ in runs}
    same_shape = len(counts) == 1
    out: dict[str, Any] = {
        "runs": len(runs),
        "box_counts": sorted(counts),
        "labels_identical": same_shape and all(np.array_equal(lab, runs[0][2]) for *_, lab in runs),
        "bit_identical": same_shape
        and all(
            np.array_equal(b, runs[0][0]) and np.array_equal(s, runs[0][1]) for b, s, _ in runs
        ),
    }
    if same_shape:
        b = [r[0] for r in runs]
        out["max_abs_diff"] = {
            "score": spread([r[1] for r in runs]),
            "center_xy": spread([x[:, :2] for x in b]),
            "z": spread([x[:, 2] for x in b]),
            "size": spread([x[:, 3:6] for x in b]),
            "yaw": spread([x[:, 6] for x in b]),
            "velocity": spread([x[:, 7:9] for x in b]),
        }
    return out


def http_arrays(body: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.array(
        [[*b["center"], *b["size"], b["yaw"], *b["velocity"]] for b in body["boxes"]]
    ).reshape(-1, 9)
    return (
        boxes,
        np.array([b["score"] for b in body["boxes"]]),
        np.array([b["label_id"] for b in body["boxes"]]),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ann-file", default="data/nuscenes/bevdetv3-trainval_infos_val.pkl")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--url", default="http://127.0.0.1:8766")
    args = p.parse_args()
    # Same order as NuScenesDataset.load_annotations, so indices match the other tools.
    infos = sorted(mmcv.load(args.ann_file)["infos"], key=lambda e: e["timestamp"])
    report: dict[str, Any] = {"index": args.index}

    images, calib = request_parts(infos[args.index])
    files = [(c, (f"{c}.jpg", images[c], "image/jpeg")) for c in CAMERA_ORDER]
    with httpx.Client(base_url=args.url, timeout=60) as client:
        bodies = [
            client.post("/v1/detect", files=files, data={"calibration": json.dumps(calib)}).json()
            for _ in range(args.repeats)
        ]
    report["http"] = summarize([http_arrays(b) for b in bodies])
    report["http"]["json_boxes_identical"] = len({json.dumps(b["boxes"]) for b in bodies}) == 1

    detector = load_bevdet(Settings(detector="bevdet"))
    target = prepare(infos[args.index])
    report["in_process"] = summarize([detector.detect_native(target) for _ in range(args.repeats)])
    others = [prepare(infos[i]) for i in (1000, 3000, 5000)]
    interleaved = []
    for k in range(args.repeats):
        detector.detect_native(others[k % 3])
        interleaved.append(detector.detect_native(target))
    report["in_process_interleaved"] = summarize(interleaved)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
