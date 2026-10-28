"""M6 input parity over the nuScenes val set: parallel preprocessing vs serial vs the M5 code.

Private tool (local nuScenes, CPU only, no GPU, no network). Run from anywhere:

    nice -n 19 ionice -c3 python tools/m6_input_parity.py --bevdet-root /path/to/BEVDet \
        --out outputs/m6-parity-<id> [--limit N] [--pool-sizes 6,12,24]

For every val frame (sorted by timestamp, as M3/M5), the request bytes and calibration are built
exactly as the HTTP client sends them, then the seven model-input arrays are produced by
  A. the M5 implementation: preprocess.py loaded verbatim from commit 5312d9c (`git show`),
     with decode_rgb (imaging.py is unchanged since M5);
  B. the current serial API path (api._decode_and_build);
  C. the parallel path (camera_pool.CameraPool.build), pool size rotating over --pool-sizes,
with OpenCV at its default thread count for A and B (as in M5) and one thread for C,
and compared byte for byte (dtype, shape, C-contiguity, strides, bytes). One frame is in memory at
a time; the dataset is only read. Nothing derived from the images is written except counts,
mismatch locations and a SHA-256 over all compared bytes. Progress: <out>/progress.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

M5_COMMIT = "5312d9c"
ARRAYS = ("imgs", "sensor2egos", "ego2globals", "intrins", "post_rots", "post_trans", "bda")


def m5_preprocess() -> types.ModuleType:
    source = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{M5_COMMIT}:src/bevserve/preprocess.py"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    module = types.ModuleType("m5_preprocess")
    module.__file__ = f"{M5_COMMIT}:src/bevserve/preprocess.py"
    sys.modules["m5_preprocess"] = module  # dataclass resolution needs the module registered
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def differences(a: Any, b: Any) -> list[str]:
    out = []
    for name in ARRAYS:
        x, y = getattr(a, name), getattr(b, name)
        if (x.dtype, x.shape, x.strides, x.flags.c_contiguous) != (
            y.dtype, y.shape, y.strides, y.flags.c_contiguous
        ) or x.tobytes() != y.tobytes():  # fmt: skip
            out.append(name)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bevdet-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--pool-sizes", default="6,12,24")
    a = p.parse_args()
    os.umask(0o077)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)

    import cv2
    import mmcv
    from compare_direct import request_parts

    from bevserve.api import _decode_and_build
    from bevserve.camera_pool import CameraPool
    from bevserve.constants import CAMERA_ORDER
    from bevserve.imaging import decode_rgb
    from bevserve.schemas import CalibrationRequest

    m5 = m5_preprocess()
    cv_default = cv2.getNumThreads()  # M5's OpenCV setting (default pool)
    sizes = [int(s) for s in a.pool_sizes.split(",")]
    pools = {n: CameraPool(n) for n in sizes}  # OpenCV runs single-threaded while pools exist
    os.chdir(a.bevdet_root)
    infos = sorted(
        mmcv.load("data/nuscenes/bevdetv3-trainval_infos_val.pkl")["infos"],
        key=lambda e: e["timestamp"],
    )
    total = len(infos) if a.limit is None else min(a.limit, len(infos))
    digest = hashlib.sha256()
    mismatches: list[dict[str, Any]] = []
    per_size = dict.fromkeys(sizes, 0)
    started = time.time()
    for k in range(total):
        raw, calib = request_parts(infos[k])
        calibration = CalibrationRequest.model_validate(calib)
        cv2.setNumThreads(cv_default)  # A and B exactly as in M5 production
        images = {n: decode_rgb(raw[n]) for n in CAMERA_ORDER}
        ref = m5.build_inputs(images, calibration.to_rig())  # A
        del images
        serial, _, _ = _decode_and_build(raw, calibration)  # B
        cv2.setNumThreads(1)  # C as configured by CameraPool
        size = sizes[k % len(sizes)]
        parallel, _, _ = pools[size].build(raw, calibration.to_rig)  # C
        per_size[size] += 1
        for label, other in (("serial", serial), ("parallel", parallel)):
            if diff := differences(ref, other):
                mismatches.append({"frame": k, "path": label, "pool": size, "arrays": diff})
        for name in ARRAYS:
            digest.update(getattr(ref, name).tobytes())
        if (k + 1) % 250 == 0 or k + 1 == total:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            (out / "progress.json").write_text(json.dumps({
                "frames_done": k + 1, "frames_total": total, "mismatches": len(mismatches),
                "elapsed_s": round(time.time() - started), "max_rss_mib": usage.ru_maxrss // 1024,
            }))  # fmt: skip
    for pool in pools.values():
        pool.shutdown()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "pass": not mismatches and total > 0,
        "frames": total,
        "comparisons": 2 * total,
        "frames_per_pool_size": per_size,
        "reference": f"preprocess.py at {M5_COMMIT} + decode_rgb",
        "opencv_threads": {"reference_and_serial": cv_default, "parallel": 1},
        "arrays": list(ARRAYS),
        "mismatches": mismatches[:100],
        "mismatch_count": len(mismatches),
        "reference_sha256": digest.hexdigest(),
        "elapsed_s": round(time.time() - started),
        "cpu_user_s": round(usage.ru_utime),
        "cpu_system_s": round(usage.ru_stime),
        "max_rss_mib": usage.ru_maxrss // 1024,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
