"""Compare BEVServe preprocessing with BEVDet's own test pipeline, on synthetic data, CPU only.

Not part of public CI: it needs a local BEVDet dev3.0 checkout and its Python environment
(torch, mmcv-full, mmdet3d). It reads no dataset and no checkpoint.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \\
    BEVSERVE_BEVDET_ROOT=/path/to/BEVDet \\
    /path/to/bevdet-env/bin/python tools/reference_check_bevdet.py

Exit code 0 = images bit-identical and matrices within 1e-6 (sensor-to-key-ego within 1e-5).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent))

ROOT = os.environ.get("BEVSERVE_BEVDET_ROOT")
if not ROOT:
    sys.exit("set BEVSERVE_BEVDET_ROOT to a BEVDet dev3.0 checkout")
if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
    sys.exit("run with CUDA_VISIBLE_DEVICES= (empty): this check must not touch a GPU")
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet3d.datasets.pipelines.loading import (  # noqa: E402
    BEVAug,
    LoadAnnotations,
    PrepareImageInputs,
)
from mmdet3d.models.detectors.bevdet import BEVDet  # noqa: E402
from tests.synthetic import encode, synthetic_image, synthetic_rig  # noqa: E402

from bevserve.calibration import sensor_to_key_ego  # noqa: E402
from bevserve.constants import CAMERA_ORDER  # noqa: E402
from bevserve.imaging import decode_rgb  # noqa: E402
from bevserve.preprocess import build_inputs  # noqa: E402

assert not torch.cuda.is_available(), "GPU visible; refusing to run"

NAMES = ("imgs", "sensor2egos", "ego2globals", "intrins", "post_rots", "post_trans", "bda")


def main() -> int:
    cfg = Config.fromfile(os.path.join(ROOT, "configs/bevdet/bevdet-r50-cbgs.py"))
    rig = synthetic_rig(with_ego_poses=True)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        jpeg: dict[str, bytes] = {}
        cams: dict[str, dict[str, object]] = {}
        for i, name in enumerate(CAMERA_ORDER):
            data = encode(synthetic_image(100 + i))
            jpeg[name] = data
            path = Path(tmp) / f"{name}.jpg"
            path.write_bytes(data)
            cam = rig.cameras[name]
            assert cam.ego2global is not None
            cams[name] = {
                "data_path": str(path),
                "cam_intrinsic": [list(r) for r in cam.intrinsic],
                "sensor2ego_rotation": list(cam.sensor2ego.rotation_wxyz),
                "sensor2ego_translation": list(cam.sensor2ego.translation_m),
                "ego2global_rotation": list(cam.ego2global.rotation_wxyz),
                "ego2global_translation": list(cam.ego2global.translation_m),
            }
        results: dict[str, object] = {"curr": {"cams": cams}, "ann_infos": ([], [])}
        results = PrepareImageInputs(data_config=cfg.data_config)(results)
        results = LoadAnnotations()(results)
        results = BEVAug(cfg.bda_aug_conf, cfg.class_names, is_train=False)(results)
        reference = [t.numpy() for t in results["img_inputs"]]

        ours = build_inputs({n: decode_rgb(jpeg[n]) for n in CAMERA_ORDER}, rig).as_tuple()
        for name, ref, mine in zip(NAMES, reference, ours, strict=True):
            diff = float(np.abs(ref.astype(np.float64) - mine.astype(np.float64)).max())
            limit = 0.0 if name == "imgs" else 1e-6
            ok = ref.shape == mine.shape and ref.dtype == mine.dtype and diff <= limit
            failures += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {name:12s} shape {mine.shape} max|diff| {diff:.3g}")

        batched = [torch.from_numpy(a).unsqueeze(0) for a in reference]
        model_s2k = BEVDet.prepare_inputs(None, batched)[1][0].numpy()
        mine_s2k = sensor_to_key_ego(ours[1], ours[2])
        diff = float(np.abs(model_s2k.astype(np.float64) - mine_s2k).max())
        ok = diff <= 1e-5
        failures += not ok
        print(
            f"{'OK  ' if ok else 'FAIL'} sensor2keyego max|diff| {diff:.3g} (BEVDet.prepare_inputs)"
        )
    print("PASS" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
