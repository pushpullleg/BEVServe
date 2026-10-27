"""Camera calibration and coordinate-frame transforms.

Frames (see docs/frames.md):

- camera i: OpenCV convention, x right, y down, z forward (nuScenes camera frame).
- ego i: nuScenes ego-vehicle frame (x forward, y left, z up) at the time camera i fired.
- key ego: ego 0, i.e. the ego frame at the CAM_FRONT_LEFT capture (CAMERA_ORDER[0]).
  BEVDet predicts every box in this frame.
- global: nuScenes world frame.

The 4x4 builders reproduce PrepareImageInputs.get_sensor_transforms (pyquaternion rotation,
float32 storage) and sensor_to_key_ego reproduces BEVDet.prepare_inputs (inverse and products
in float64, result cast to float32).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

from bevserve.constants import CAMERA_ORDER

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Pose:
    """Rigid transform child -> parent: rotation quaternion (w, x, y, z) + translation (m)."""

    rotation_wxyz: tuple[float, float, float, float]
    translation_m: tuple[float, float, float]


@dataclass(frozen=True)
class CameraCalibration:
    intrinsic: tuple[tuple[float, float, float], ...]  # 3x3, for the 1600x900 source image
    sensor2ego: Pose  # camera -> ego
    ego2global: Pose | None  # ego (at this camera's capture) -> global; None = identity


@dataclass(frozen=True)
class RigCalibration:
    cameras: Mapping[str, CameraCalibration]

    def ordered(self) -> list[CameraCalibration]:
        """Cameras in canonical model order, independent of how the request listed them."""
        missing = [name for name in CAMERA_ORDER if name not in self.cameras]
        if missing or len(self.cameras) != len(CAMERA_ORDER):
            raise ValueError(f"rig must contain exactly {CAMERA_ORDER}; missing {missing}")
        return [self.cameras[name] for name in CAMERA_ORDER]


def pose_matrix(pose: Pose) -> FloatArray:
    """4x4 float32 matrix of a pose, built exactly as get_sensor_transforms builds it."""
    w, x, y, z = pose.rotation_wxyz
    rotation = np.asarray(Quaternion(w, x, y, z).rotation_matrix, dtype=np.float32)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[3, 3] = 1
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(pose.translation_m, dtype=np.float32)
    return matrix


def identity_pose_matrix() -> FloatArray:
    return np.eye(4, dtype=np.float32)


def rig_matrices(rig: RigCalibration) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(sensor2egos, ego2globals, intrinsics) in canonical order: (6,4,4), (6,4,4), (6,3,3)."""
    cams = rig.ordered()
    sensor2egos = np.stack([pose_matrix(c.sensor2ego) for c in cams])
    ego2globals = np.stack(
        [
            pose_matrix(c.ego2global) if c.ego2global is not None else identity_pose_matrix()
            for c in cams
        ]
    )
    intrinsics = np.stack([np.asarray(c.intrinsic, dtype=np.float32) for c in cams])
    return sensor2egos, ego2globals, intrinsics


def sensor_to_key_ego(
    sensor2egos: Sequence[FloatArray] | FloatArray,
    ego2globals: Sequence[FloatArray] | FloatArray,
) -> FloatArray:
    """camera i -> key ego, as BEVDet.prepare_inputs computes it.

    T(key ego <- cam i) = inv(T(global <- ego 0)) @ T(global <- ego i) @ T(ego i <- cam i)
    """
    s2e = np.asarray(sensor2egos, dtype=np.float32).astype(np.float64)
    e2g = np.asarray(ego2globals, dtype=np.float32).astype(np.float64)
    if s2e.shape != e2g.shape or s2e.shape[-2:] != (4, 4):
        raise ValueError(f"expected matching (N,4,4) inputs, got {s2e.shape} and {e2g.shape}")
    global2keyego = np.linalg.inv(e2g[0])
    result: FloatArray = (global2keyego @ e2g @ s2e).astype(np.float32)
    return result
