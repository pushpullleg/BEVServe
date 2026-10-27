"""Synthetic fixtures. Nothing here is taken from nuScenes or any other dataset.

The rig is a made-up six-camera layout with plausible geometry: cameras on a ring around the
vehicle, yawed outward, with a generic 1600x900 pinhole intrinsic. Images are generated from a
seeded random generator.
"""

from __future__ import annotations

import io
import json
import math
from typing import Any

import numpy as np
from PIL import Image

from bevserve.calibration import CameraCalibration, Pose, RigCalibration
from bevserve.constants import CAMERA_ORDER, SRC_HEIGHT, SRC_WIDTH

# Yaw of each camera's optical axis in the ego frame (deg, counter-clockwise from x forward).
_YAW_DEG = {
    "CAM_FRONT_LEFT": 55.0,
    "CAM_FRONT": 0.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_BACK_LEFT": 110.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_RIGHT": -110.0,
}
INTRINSIC = ((1250.0, 0.0, 800.0), (0.0, 1250.0, 450.0), (0.0, 0.0, 1.0))


def _quat_wxyz_from_matrix(r: np.ndarray) -> tuple[float, float, float, float]:
    from pyquaternion import Quaternion

    q = Quaternion(matrix=r)
    return (float(q.w), float(q.x), float(q.y), float(q.z))


def camera_to_ego_rotation(yaw_deg: float) -> np.ndarray:
    """Rotation of a camera (x right, y down, z forward) looking along ego yaw `yaw_deg`."""
    yaw = math.radians(yaw_deg)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    down = -up
    return np.stack([right, down, forward], axis=1)  # columns: camera axes in ego coordinates


def synthetic_sensor2ego(name: str) -> Pose:
    yaw = _YAW_DEG[name]
    r = camera_to_ego_rotation(yaw)
    t = (1.0 * math.cos(math.radians(yaw)) + 1.2, 0.8 * math.sin(math.radians(yaw)), 1.6)
    return Pose(rotation_wxyz=_quat_wxyz_from_matrix(r), translation_m=t)


def yaw_pose(yaw_deg: float, translation: tuple[float, float, float]) -> Pose:
    half = math.radians(yaw_deg) / 2
    return Pose(rotation_wxyz=(math.cos(half), 0.0, 0.0, math.sin(half)), translation_m=translation)


def synthetic_rig(with_ego_poses: bool = False) -> RigCalibration:
    cams: dict[str, CameraCalibration] = {}
    for i, name in enumerate(CAMERA_ORDER):
        ego = (
            yaw_pose(30.0 + 0.4 * i, (100.0 + 0.3 * i, 50.0 - 0.2 * i, 0.0))
            if with_ego_poses
            else None
        )
        cams[name] = CameraCalibration(
            intrinsic=INTRINSIC, sensor2ego=synthetic_sensor2ego(name), ego2global=ego
        )
    return RigCalibration(cameras=cams)


def calibration_payload(with_ego_poses: bool = False) -> dict[str, Any]:
    rig = synthetic_rig(with_ego_poses)
    cameras: dict[str, Any] = {}
    for name, cam in rig.cameras.items():
        entry: dict[str, Any] = {
            "intrinsic": [list(row) for row in cam.intrinsic],
            "sensor2ego": {
                "rotation_wxyz": list(cam.sensor2ego.rotation_wxyz),
                "translation_m": list(cam.sensor2ego.translation_m),
            },
        }
        if cam.ego2global is not None:
            entry["ego2global"] = {
                "rotation_wxyz": list(cam.ego2global.rotation_wxyz),
                "translation_m": list(cam.ego2global.translation_m),
            }
        cameras[name] = entry
    return {"schema_version": "1", "cameras": cameras}


def synthetic_image(seed: int, width: int = SRC_WIDTH, height: int = SRC_HEIGHT) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height // 30 + 1, width // 30 + 1, 3), dtype=np.uint8)
    small = Image.fromarray(base, "RGB")
    return small.resize((width, height))


def encode(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=90) if fmt == "JPEG" else image.save(buffer, format=fmt)
    return buffer.getvalue()


def multipart_files(
    images: dict[str, bytes] | None = None, calibration: dict[str, Any] | None = None
) -> tuple[list[tuple[str, tuple[str, bytes, str]]], dict[str, str]]:
    """(files, data) for httpx: six image parts + the calibration form field."""
    if images is None:
        images = {name: encode(synthetic_image(i)) for i, name in enumerate(CAMERA_ORDER)}
    files = [(name, (f"{name}.jpg", data, "image/jpeg")) for name, data in images.items()]
    data = {"calibration": json.dumps(calibration or calibration_payload())}
    return files, data
