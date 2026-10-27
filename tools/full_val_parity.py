"""Full nuScenes val parity: BEVServe's in-process adapter + BEVDet's own evaluator.

Private tool (GPU, local nuScenes, BEVDet checkout, trusted checkpoint). For every val frame:
raw JPEG bytes + calibration from the info file -> the API's own input path (header checks,
decode, CalibrationRequest validation, build_inputs) -> load_bevdet's BEVDetDetector.detect_native
-> the API conversion (detections_from_arrays, geometric-center z) -> verified inverse conversion
(z - h/2, asserted bit-identical to the native output) -> LiDARInstance3DBoxes exactly as
CenterHead builds them -> NuScenesDataset.evaluate with the config's evaluation settings.

The evaluator is used unchanged, including its own frame handling (_format_bbox applies the
CAM_FRONT ego pose); BEVServe does not correct it.

    cd /path/to/BEVDet
    PYTHONDONTWRITEBYTECODE=1 BEVSERVE_DETECTOR=bevdet BEVSERVE_BEVDET_CONFIG=... \\
      BEVSERVE_CHECKPOINT=... BEVSERVE_CHECKPOINT_SHA256=... \\
      python /path/to/bevserve/tools/full_val_parity.py --out /tmp/fullval
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmcv import Config
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_direct import request_parts

from bevserve.boxes import detections_from_arrays
from bevserve.constants import CAMERA_ORDER, CLASS_NAMES
from bevserve.imaging import check_size, decode_rgb, inspect_header
from bevserve.model.bevdet_detector import load_bevdet
from bevserve.preprocess import ModelInputs, build_inputs
from bevserve.schemas import CalibrationRequest
from bevserve.settings import Settings


def prepare(info: dict[str, Any]) -> ModelInputs:
    """The API's input path for one frame, minus HTTP."""
    images, calibration = request_parts(info)
    for name in CAMERA_ORDER:
        check_size(inspect_header(images[name]))
    rig = CalibrationRequest.model_validate(calibration).to_rig()
    return build_inputs({n: decode_rgb(images[n]) for n in CAMERA_ORDER}, rig)


def prefetch(infos: list[dict[str, Any]], workers: int, depth: int) -> Iterator[ModelInputs]:
    with cf.ThreadPoolExecutor(workers) as pool:
        pending: list[cf.Future[ModelInputs]] = []
        nxt = 0
        while nxt < len(infos) or pending:
            while nxt < len(infos) and len(pending) < depth:
                pending.append(pool.submit(prepare, infos[nxt]))
                nxt += 1
            yield pending.pop(0).result()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bevdet/bevdet-r50-cbgs.py")
    p.add_argument("--ann-file", default="data/nuscenes/bevdetv3-trainval_infos_val.pkl")
    p.add_argument("--limit", type=int, default=None, help="first N frames only (dry run)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    assert tuple(dataset.CLASSES) == CLASS_NAMES
    infos = dataset.data_infos[: args.limit] if args.limit else dataset.data_infos

    t_load = time.time()
    detector = load_bevdet(Settings(detector="bevdet"))
    load_s = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()

    results: list[dict[str, Any]] = []
    roundtrip_mismatch = 0
    gpu_s = 0.0
    started = time.time()
    for n, inputs in enumerate(prefetch(infos, args.workers, depth=4 * args.workers)):
        t0 = time.perf_counter()
        boxes, scores, labels = detector.detect_native(inputs)
        gpu_s += time.perf_counter() - t0
        dets = detections_from_arrays(boxes, scores, labels)  # what the API returns
        back = np.array(
            [[*d.center, *d.size, d.yaw, *d.velocity] for d in dets], dtype=np.float64
        ).reshape(-1, 9)
        back[:, 2] -= 0.5 * back[:, 5]
        back32 = back.astype(np.float32)
        if not np.array_equal(back32, boxes.astype(np.float32)):
            roundtrip_mismatch += 1
        results.append(
            {
                "pts_bbox": {
                    "boxes_3d": LiDARInstance3DBoxes(torch.from_numpy(back32), 9),
                    "scores_3d": torch.tensor([d.score for d in dets], dtype=torch.float32),
                    "labels_3d": torch.tensor([d.label_id for d in dets], dtype=torch.int64),
                }
            }
        )
        if (n + 1) % 500 == 0:
            print(f"{n + 1}/{len(infos)} frames, {time.time() - started:.0f}s", flush=True)
    infer_s = time.time() - started

    eval_kwargs = dict(cfg.get("evaluation", {}))
    for key in ("interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"):
        eval_kwargs.pop(key, None)
    eval_kwargs["metric"] = ["bbox"]
    t_eval = time.time()
    metrics: dict[str, Any] = {}
    if not args.limit:  # nuScenes evaluation needs predictions for every val sample
        metrics = dataset.evaluate(results, jsonfile_prefix=str(out / "results"), **eval_kwargs)
    eval_s = time.time() - t_eval

    def pick(key: str) -> float:
        for k in (f"pts_bbox_NuScenes/{key}", f"NuScenes/{key}", key):
            if k in metrics:
                return float(metrics[k])
        return float("nan")

    summary = {
        "frames": len(results),
        "mAP": pick("mAP"),
        "NDS": pick("NDS"),
        "boxes": int(sum(len(r["pts_bbox"]["scores_3d"]) for r in results)),
        "roundtrip_mismatch_frames": roundtrip_mismatch,
        "seconds": {
            "load": round(load_s, 1),
            "inference_loop": round(infer_s, 1),
            "gpu_detect_total": round(gpu_s, 1),
            "evaluate": round(eval_s, 1),
        },
        "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "checkpoint_sha256": detector.info.checkpoint_sha256,
        "config_sha256": detector.info.config_sha256,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
