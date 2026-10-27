"""Coordinate-frame tests (review correction 4): deliberately different camera poses."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bevserve.calibration import (
    CameraCalibration,
    Pose,
    RigCalibration,
    pose_matrix,
    rig_matrices,
    sensor_to_key_ego,
)
from bevserve.constants import CAMERA_ORDER, KEY_CAMERA
from tests.synthetic import INTRINSIC, synthetic_rig, synthetic_sensor2ego, yaw_pose


def _homog(p: np.ndarray) -> np.ndarray:
    return np.append(p, 1.0)


def test_key_camera_is_front_left() -> None:
    assert KEY_CAMERA == "CAM_FRONT_LEFT" == CAMERA_ORDER[0]


def test_pose_matrix_rotates_x_to_y_for_90_deg_yaw() -> None:
    m = pose_matrix(yaw_pose(90.0, (1.0, 2.0, 3.0)))
    assert m.dtype == np.float32
    np.testing.assert_allclose(
        m @ _homog(np.array([1.0, 0.0, 0.0])), [1.0, 3.0, 3.0, 1.0], atol=1e-6
    )
    np.testing.assert_array_equal(m[3], [0, 0, 0, 1])


def test_pose_matrix_normalizes_slightly_non_unit_quaternion() -> None:
    q = yaw_pose(37.0, (0.0, 0.0, 0.0)).rotation_wxyz
    scaled = Pose(rotation_wxyz=tuple(1.0005 * v for v in q), translation_m=(0.0, 0.0, 0.0))  # type: ignore[arg-type]
    np.testing.assert_allclose(
        pose_matrix(scaled), pose_matrix(Pose(q, (0.0, 0.0, 0.0))), atol=1e-6
    )


def test_world_point_is_consistent_through_every_camera_with_different_ego_poses() -> None:
    """Each camera fired at a different ego pose. A fixed world point, observed from any camera
    and mapped with sensor_to_key_ego, must land on the same key-ego coordinates."""
    rig = synthetic_rig(with_ego_poses=True)
    s2e, e2g, _ = rig_matrices(rig)
    # The ego poses really differ, otherwise this test would prove nothing.
    assert not np.allclose(e2g[0], e2g[3])
    to_key = sensor_to_key_ego(s2e, e2g)

    world_point = _homog(np.array([112.0, 47.5, 0.8]))
    expected_in_key_ego = np.linalg.inv(e2g[0].astype(np.float64)) @ world_point
    for i in range(len(CAMERA_ORDER)):
        cam_to_global = e2g[i].astype(np.float64) @ s2e[i].astype(np.float64)
        point_in_cam = np.linalg.inv(cam_to_global) @ world_point
        np.testing.assert_allclose(to_key[i] @ point_in_cam, expected_in_key_ego, atol=2e-4)


def test_key_frame_follows_canonical_order_not_request_order() -> None:
    rig = synthetic_rig(with_ego_poses=True)
    reversed_rig = RigCalibration(cameras=dict(reversed(list(rig.cameras.items()))))
    assert next(iter(reversed_rig.cameras)) == "CAM_BACK_RIGHT"
    s2e, e2g, _ = rig_matrices(reversed_rig)
    to_key = sensor_to_key_ego(s2e, e2g)
    # Camera 0 is CAM_FRONT_LEFT, and its own ego pose cancels: key ego = its ego frame.
    np.testing.assert_allclose(
        to_key[0], pose_matrix(rig.cameras[KEY_CAMERA].sensor2ego), atol=1e-5
    )
    np.testing.assert_array_equal(to_key, sensor_to_key_ego(*rig_matrices(rig)[:2]))


def test_identity_ego_poses_reduce_to_sensor2ego() -> None:
    s2e, e2g, _ = rig_matrices(synthetic_rig(with_ego_poses=False))
    np.testing.assert_array_equal(e2g, np.broadcast_to(np.eye(4, dtype=np.float32), e2g.shape))
    np.testing.assert_allclose(sensor_to_key_ego(s2e, e2g), s2e, atol=1e-7)


def test_moving_the_whole_rig_in_global_does_not_change_key_ego_transforms() -> None:
    s2e, e2g, _ = rig_matrices(synthetic_rig(with_ego_poses=True))
    shift = pose_matrix(yaw_pose(-20.0, (-30.0, 12.0, 0.5))).astype(np.float64)
    moved = (shift @ e2g.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(
        sensor_to_key_ego(s2e, moved), sensor_to_key_ego(s2e, e2g), atol=2e-4
    )


def test_synthetic_camera_looks_along_its_yaw() -> None:
    m = pose_matrix(synthetic_sensor2ego("CAM_FRONT_LEFT"))
    optical_axis_in_ego = m[:3, :3] @ np.array([0.0, 0.0, 1.0])
    yaw = math.radians(55.0)
    np.testing.assert_allclose(optical_axis_in_ego, [math.cos(yaw), math.sin(yaw), 0.0], atol=1e-6)


def test_rig_requires_all_six_cameras() -> None:
    cam = CameraCalibration(INTRINSIC, synthetic_sensor2ego("CAM_FRONT"), None)
    with pytest.raises(ValueError, match="exactly"):
        RigCalibration(cameras={"CAM_FRONT": cam}).ordered()


def test_sensor_to_key_ego_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        sensor_to_key_ego(np.zeros((6, 4, 4), np.float32), np.zeros((5, 4, 4), np.float32))
