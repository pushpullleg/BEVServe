"""CPU-only profile of BEVServe's decode + preprocessing path (M6 Stage A; no GPU, no network).

    python tools/profile_preprocess.py --inputs synthetic:6 [--repeats 30] [--cv-threads N]
    python tools/profile_preprocess.py --inputs nuscenes:even:20 --bevdet-root /path/to/BEVDet

Times every step of bevserve.imaging.decode_rgb and bevserve.preprocess.build_inputs on the same
JPEG bytes the service receives, per camera, plus the two production functions end to end.
nuScenes inputs are read locally and never written; output holds timings only (JSON on stdout).
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bevserve.constants import (  # noqa: E402
    CAMERA_ORDER,
    CROP_BOX,
    NORM_MEAN,
    NORM_STD,
    RESIZED_HEIGHT,
    RESIZED_WIDTH,
)
from bevserve.imaging import decode_rgb  # noqa: E402
from bevserve.preprocess import build_inputs  # noqa: E402


def jpeg_sets(spec: str, bevdet_root: str | None) -> list[tuple[dict[str, bytes], dict]]:
    """[(camera -> JPEG bytes, calibration payload)] exactly as load.py would send them."""
    kind, _, rest = spec.partition(":")
    if kind == "synthetic":
        from tests.synthetic import calibration_payload, synthetic_image
        from tests.synthetic import encode as enc

        return [
            ({n: enc(synthetic_image(100 * k + i)) for i, n in enumerate(CAMERA_ORDER)},
             calibration_payload())
            for k in range(int(rest or 1))
        ]  # fmt: skip
    import os

    import mmcv
    from compare_direct import request_parts, select

    cwd = os.getcwd()
    os.chdir(bevdet_root)
    try:
        infos = sorted(
            mmcv.load("data/nuscenes/bevdetv3-trainval_infos_val.pkl")["infos"],
            key=lambda e: e["timestamp"],
        )
        return [request_parts(infos[i]) for i in select(rest or "even:20", infos)]
    finally:
        os.chdir(cwd)


def steps(data: bytes) -> dict[str, float]:
    """Each operation of decode_rgb + transform_image + normalize, timed separately (ms)."""
    t: dict[str, float] = {}

    def lap(name: str, t0: float) -> float:
        now = time.perf_counter()
        t[name] = (now - t0) * 1000
        return now

    t0 = time.perf_counter()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        img = im if im.mode == "RGB" else im.convert("RGB")
    t0 = lap("decode", t0)
    resized = img.resize((RESIZED_WIDTH, RESIZED_HEIGHT))
    t0 = lap("resize_bicubic", t0)
    cropped = resized.crop(CROP_BOX)
    t0 = lap("crop", t0)
    rotated = cropped.rotate(0)
    t0 = lap("rotate0", t0)
    a = np.array(rotated)
    t0 = lap("to_numpy", t0)
    a = a.copy()
    t0 = lap("copy", t0)
    a = a.astype(np.float32)
    t0 = lap("astype_f32", t0)
    mean = np.float64(np.array(NORM_MEAN, dtype=np.float32).reshape(1, -1))
    stdinv = 1 / np.float64(np.array(NORM_STD, dtype=np.float32).reshape(1, -1))
    cv2.cvtColor(a, cv2.COLOR_BGR2RGB, a)
    t0 = lap("cvtColor", t0)
    cv2.subtract(a, mean, a)
    t0 = lap("subtract", t0)
    cv2.multiply(a, stdinv, a)
    t0 = lap("multiply", t0)
    np.ascontiguousarray(a.transpose(2, 0, 1))
    lap("chw_contiguous", t0)
    return t


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default="synthetic:6")
    p.add_argument("--bevdet-root")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--cv-threads", type=int, default=None)
    a = p.parse_args()
    if a.cv_threads is not None:
        cv2.setNumThreads(a.cv_threads)
    sets = jpeg_sets(a.inputs, a.bevdet_root)
    from bevserve.schemas import CalibrationRequest

    per_step: dict[str, list[float]] = {}
    decode6, build6 = [], []
    for r in range(a.repeats):
        jpegs, calib = sets[r % len(sets)]
        for data in jpegs.values():
            for k, v in steps(data).items():
                per_step.setdefault(k, []).append(v)
        rig = CalibrationRequest.model_validate(calib).to_rig()
        t0 = time.perf_counter()
        images = {n: decode_rgb(jpegs[n]) for n in CAMERA_ORDER}
        t1 = time.perf_counter()
        build_inputs(images, rig)
        t2 = time.perf_counter()
        decode6.append((t1 - t0) * 1000)
        build6.append((t2 - t1) * 1000)
    med = {k: round(statistics.median(v), 3) for k, v in per_step.items()}
    pre = [k for k in med if k != "decode"]
    print(json.dumps({
        "inputs": a.inputs, "repeats": a.repeats, "cv2_threads": cv2.getNumThreads(),
        "per_camera_median_ms": med,
        "per_camera_preprocess_sum_ms": round(sum(med[k] for k in pre), 3),
        "production_decode_6cam_median_ms": round(statistics.median(decode6), 2),
        "production_build_inputs_6cam_median_ms": round(statistics.median(build6), 2),
        "mean_jpeg_bytes": int(np.mean([len(b) for j, _ in sets for b in j.values()])),
    }, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
