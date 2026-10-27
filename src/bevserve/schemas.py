"""Request and response schemas (API v1, calibration schema_version "1")."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from bevserve.calibration import CameraCalibration, Pose, RigCalibration
from bevserve.constants import (
    CAMERA_ORDER,
    MODEL_MAX_BOXES,
    MODEL_SCORE_THRESHOLD,
    SRC_HEIGHT,
    SRC_WIDTH,
)

_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

QUATERNION_NORM_TOLERANCE = 1e-3
MAX_SENSOR_TRANSLATION_M = 10.0
FOCAL_RANGE = (100.0, 10000.0)

Vec3 = tuple[float, float, float]


class PoseIn(BaseModel):
    """child -> parent transform. Quaternion order is (w, x, y, z), translation in meters."""

    model_config = _STRICT
    rotation_wxyz: tuple[float, float, float, float]
    translation_m: Vec3

    @field_validator("rotation_wxyz")
    @classmethod
    def _unit_quaternion(cls, q: tuple[float, float, float, float]) -> tuple[float, ...]:
        norm = math.sqrt(sum(v * v for v in q))
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            raise PydanticCustomError(
                "rotation",
                "quaternion norm is {norm:.6f}; expected 1 within {tol}",
                {"norm": norm, "tol": QUATERNION_NORM_TOLERANCE},
            )
        return q

    def to_pose(self) -> Pose:
        return Pose(rotation_wxyz=self.rotation_wxyz, translation_m=self.translation_m)


class CameraCalibrationIn(BaseModel):
    model_config = _STRICT
    intrinsic: tuple[Vec3, Vec3, Vec3] = Field(
        description=f"3x3 pinhole matrix for the {SRC_WIDTH}x{SRC_HEIGHT} source image"
    )
    sensor2ego: PoseIn = Field(description="camera -> ego vehicle")
    ego2global: PoseIn | None = Field(
        default=None, description="ego (at this camera's capture time) -> global; all or none"
    )

    @field_validator("intrinsic")
    @classmethod
    def _pinhole(cls, k: tuple[Vec3, Vec3, Vec3]) -> tuple[Vec3, Vec3, Vec3]:
        (fx, skew, cx), (k10, fy, cy), bottom = k
        lo, hi = FOCAL_RANGE
        problems = []
        if not (lo <= fx <= hi and lo <= fy <= hi):
            problems.append(f"fx, fy must be in [{lo:g}, {hi:g}]")
        if not (0 < cx < SRC_WIDTH and 0 < cy < SRC_HEIGHT):
            problems.append(f"principal point must lie inside {SRC_WIDTH}x{SRC_HEIGHT}")
        if skew != 0 or k10 != 0:
            problems.append("skew terms must be 0")
        if bottom != (0.0, 0.0, 1.0):
            problems.append("bottom row must be [0, 0, 1]")
        if problems:
            raise PydanticCustomError("intrinsic", "; ".join(problems))
        return k

    @field_validator("sensor2ego")
    @classmethod
    def _mounted_on_vehicle(cls, pose: PoseIn) -> PoseIn:
        if math.sqrt(sum(v * v for v in pose.translation_m)) >= MAX_SENSOR_TRANSLATION_M:
            raise PydanticCustomError(
                "translation",
                "camera is {limit} m or more from the ego origin",
                {"limit": MAX_SENSOR_TRANSLATION_M},
            )
        return pose


class DetectOptions(BaseModel):
    model_config = _STRICT
    score_threshold: float = Field(default=MODEL_SCORE_THRESHOLD, ge=MODEL_SCORE_THRESHOLD, le=1.0)
    max_boxes: int = Field(default=MODEL_MAX_BOXES, ge=1, le=MODEL_MAX_BOXES)


class CalibrationRequest(BaseModel):
    model_config = _STRICT
    schema_version: Literal["1"]
    cameras: dict[str, CameraCalibrationIn]
    options: DetectOptions = DetectOptions()

    @model_validator(mode="after")
    def _rig(self) -> CalibrationRequest:
        names = set(self.cameras)
        if names != set(CAMERA_ORDER):
            raise PydanticCustomError(
                "camera_set",
                "missing {missing}; unexpected {extra}",
                {
                    "missing": sorted(set(CAMERA_ORDER) - names),
                    "extra": sorted(names - set(CAMERA_ORDER)),
                },
            )
        with_pose = [n for n, c in self.cameras.items() if c.ego2global is not None]
        if with_pose and len(with_pose) != len(CAMERA_ORDER):
            raise PydanticCustomError(
                "ego_pose_partial",
                "ego2global given for {n} of 6 cameras; give all or none",
                {"n": len(with_pose)},
            )
        return self

    def to_rig(self) -> RigCalibration:
        return RigCalibration(
            cameras={
                name: CameraCalibration(
                    intrinsic=cam.intrinsic,
                    sensor2ego=cam.sensor2ego.to_pose(),
                    ego2global=cam.ego2global.to_pose() if cam.ego2global else None,
                )
                for name, cam in self.cameras.items()
            }
        )


# ---------------------------------------------------------------- responses


class Box3D(BaseModel):
    label: str
    label_id: int
    score: float
    center: Vec3 = Field(
        description="geometric box center (m) in the output frame; BEVDet stores the bottom"
        " center internally, z here = bottom z + z_size / 2"
    )
    size: Vec3 = Field(description="(x_size, y_size, z_size) in m; nuScenes wlh = (y, x, z)")
    yaw: float = Field(description="rad, about z of the output frame")
    velocity: tuple[float, float] = Field(description="(vx, vy) in m/s")


class ModelInfo(BaseModel):
    name: str
    checkpoint_sha256: str | None
    config_sha256: str | None
    service_version: str
    git_commit: str | None
    class_names: list[str]
    frame: str
    camera_order: list[str]
    source_size: tuple[int, int]


class Timing(BaseModel):
    decode: float
    preprocess: float
    inference: float
    total: float


class Units(BaseModel):
    length: Literal["m"] = "m"
    angle: Literal["rad"] = "rad"
    velocity: Literal["m/s"] = "m/s"


class DetectResponse(BaseModel):
    request_id: str
    model: ModelInfo
    frame: str
    units: Units = Units()
    boxes: list[Box3D]
    timing_ms: Timing


class FieldError(BaseModel):
    field: str
    code: str
    message: str


class Problem(BaseModel):
    """RFC 9457 problem details."""

    type: str
    title: str
    status: int
    detail: str
    request_id: str | None = None
    errors: list[FieldError] = []


class Readiness(BaseModel):
    status: Literal["ready", "initializing", "init_failed", "fatal", "inference_stalled"]
    reason: str | None = None
    detail: str | None = None
