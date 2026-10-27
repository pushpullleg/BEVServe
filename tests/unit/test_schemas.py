"""One test per calibration rule in the design's validation table."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from bevserve.calibration import rig_matrices
from bevserve.constants import CAMERA_ORDER
from bevserve.schemas import CalibrationRequest
from tests.synthetic import calibration_payload


def _codes(payload: dict[str, Any]) -> set[str]:
    with pytest.raises(ValidationError) as err:
        CalibrationRequest.model_validate(payload)
    return {e["type"] for e in err.value.errors()}


def _mutate(path: list[str | int], value: Any, with_ego: bool = False) -> dict[str, Any]:
    payload = copy.deepcopy(calibration_payload(with_ego))
    node: Any = payload
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return payload


def test_valid_payload_round_trips_to_rig() -> None:
    req = CalibrationRequest.model_validate(calibration_payload(with_ego_poses=True))
    s2e, e2g, k = rig_matrices(req.to_rig())
    assert s2e.shape == e2g.shape == (6, 4, 4) and k.shape == (6, 3, 3)
    assert req.options.score_threshold == 0.1 and req.options.max_boxes == 500


def test_missing_camera() -> None:
    payload = calibration_payload()
    del payload["cameras"]["CAM_BACK"]
    assert _codes(payload) == {"camera_set"}


def test_unknown_camera() -> None:
    payload = calibration_payload()
    payload["cameras"]["CAM_ROOF"] = payload["cameras"]["CAM_BACK"]
    assert _codes(payload) == {"camera_set"}


def test_partial_ego_poses_rejected() -> None:
    payload = calibration_payload(with_ego_poses=True)
    del payload["cameras"]["CAM_FRONT"]["ego2global"]
    assert _codes(payload) == {"ego_pose_partial"}


@pytest.mark.parametrize(
    "k",
    [
        [[50.0, 0, 800], [0, 1250, 450], [0, 0, 1]],  # focal too small
        [[1250, 0, 1700], [0, 1250, 450], [0, 0, 1]],  # principal point outside
        [[1250, 3, 800], [0, 1250, 450], [0, 0, 1]],  # skew
        [[1250, 0, 800], [0, 1250, 450], [0, 0, 2]],  # bottom row
    ],
    ids=["focal", "principal-point", "skew", "bottom-row"],
)
def test_intrinsic_rules(k: list[list[float]]) -> None:
    assert _codes(_mutate(["cameras", "CAM_FRONT", "intrinsic"], k)) == {"intrinsic"}


def test_intrinsic_must_be_3x3() -> None:
    assert _codes(_mutate(["cameras", "CAM_FRONT", "intrinsic"], [[1, 0, 0], [0, 1, 0]])) == {
        "missing"
    }


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_values_rejected(bad: float) -> None:
    payload = _mutate(["cameras", "CAM_FRONT", "sensor2ego", "translation_m"], [bad, 0.0, 1.0])
    assert _codes(payload) == {"finite_number"}


def test_non_unit_quaternion_rejected() -> None:
    payload = _mutate(["cameras", "CAM_FRONT", "sensor2ego", "rotation_wxyz"], [0.9, 0.0, 0.0, 0.0])
    assert _codes(payload) == {"rotation"}


def test_camera_far_from_vehicle_rejected() -> None:
    payload = _mutate(["cameras", "CAM_FRONT", "sensor2ego", "translation_m"], [12.0, 0.0, 1.0])
    assert _codes(payload) == {"translation"}


def test_ego_pose_may_be_far_away() -> None:
    payload = _mutate(
        ["cameras", "CAM_FRONT", "ego2global", "translation_m"],
        [9000.0, -4000.0, 0.0],
        with_ego=True,
    )
    CalibrationRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("options", "code"),
    [
        ({"score_threshold": 0.05}, "greater_than_equal"),
        ({"score_threshold": 1.5}, "less_than_equal"),
        ({"max_boxes": 0}, "greater_than_equal"),
        ({"max_boxes": 501}, "less_than_equal"),
        ({"nms": False}, "extra_forbidden"),
    ],
)
def test_options_bounds(options: dict[str, Any], code: str) -> None:
    payload = calibration_payload()
    payload["options"] = options
    assert _codes(payload) == {code}


def test_schema_version_required() -> None:
    payload = calibration_payload()
    payload["schema_version"] = "2"
    assert _codes(payload) == {"literal_error"}


def test_extra_fields_forbidden() -> None:
    payload = calibration_payload()
    payload["cameras"][CAMERA_ORDER[0]]["distortion"] = [0.0] * 5
    assert _codes(payload) == {"extra_forbidden"}
