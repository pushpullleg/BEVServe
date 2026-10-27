"""Request-time reimplementation of BEVDet's test-time input pipeline.

Reproduces, for one 6-camera sample, what PrepareImageInputs (test mode) followed by BEVAug
(test mode) produce in BEVDet dev3.0 (mmdet3d/datasets/pipelines/loading.py), without file I/O,
LiDAR points or annotations. Output arrays are numpy; the M2 adapter converts them to torch.

Fidelity notes (each is tested):
- Geometry: PIL resize to 704x396 with Pillow's default filter (BICUBIC), crop rows 140..396
  (the top is removed), no flip; rotate(0) is applied as in the original.
- Normalization: mmcv.imnormalize(to_rgb=True) is applied to an RGB array, so its BGR->RGB swap
  actually turns RGB into BGR. The network was trained on that order; it is kept on purpose.
- Matrices: post_rot = diag(0.44, 0.44, 1), post_tran = (0, -140, 0), bda = identity, built
  with the same float32 operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from bevserve.calibration import RigCalibration, rig_matrices
from bevserve.constants import (
    CAMERA_ORDER,
    CROP_BOX,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NORM_MEAN,
    NORM_STD,
    RESIZE,
    RESIZED_HEIGHT,
    RESIZED_WIDTH,
    SRC_HEIGHT,
    SRC_WIDTH,
)

FloatArray = npt.NDArray[np.float32]

N_CAMS = len(CAMERA_ORDER)
EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "imgs": (N_CAMS, 3, INPUT_HEIGHT, INPUT_WIDTH),
    "sensor2egos": (N_CAMS, 4, 4),
    "ego2globals": (N_CAMS, 4, 4),
    "intrins": (N_CAMS, 3, 3),
    "post_rots": (N_CAMS, 3, 3),
    "post_trans": (N_CAMS, 3),
    "bda": (4, 4),
}


@dataclass(frozen=True)
class ModelInputs:
    """The 7 tensors of BEVDet's img_inputs for one sample (no batch dimension), in order."""

    imgs: FloatArray
    sensor2egos: FloatArray
    ego2globals: FloatArray
    intrins: FloatArray
    post_rots: FloatArray
    post_trans: FloatArray
    bda: FloatArray

    def as_tuple(self) -> tuple[FloatArray, ...]:
        return (
            self.imgs,
            self.sensor2egos,
            self.ego2globals,
            self.intrins,
            self.post_rots,
            self.post_trans,
            self.bda,
        )

    def check(self) -> None:
        for name, shape in EXPECTED_SHAPES.items():
            array = getattr(self, name)
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(
                    f"{name}: got {array.shape} {array.dtype}, expected {shape} float32"
                )


def transform_image(image: Image.Image) -> Image.Image:
    """img_transform_core with the test-time augmentation (resize, crop, no flip, rotate 0)."""
    resized = image.resize((RESIZED_WIDTH, RESIZED_HEIGHT))
    cropped = resized.crop(CROP_BOX)
    return cropped.rotate(0)


def normalize(image: Image.Image) -> FloatArray:
    """mmlabNormalize: (H, W, 3) uint8 RGB -> (3, H, W) float32, channels in BGR order."""
    array = np.array(image).copy().astype(np.float32)
    mean = np.float64(np.array(NORM_MEAN, dtype=np.float32).reshape(1, -1))
    stdinv = 1 / np.float64(np.array(NORM_STD, dtype=np.float32).reshape(1, -1))
    cv2.cvtColor(array, cv2.COLOR_BGR2RGB, array)
    cv2.subtract(array, mean, array)
    cv2.multiply(array, stdinv, array)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def image_post_transform() -> tuple[FloatArray, FloatArray]:
    """post_rot (3,3) and post_tran (3,) of img_transform for the test-time augmentation."""
    post_rot = np.eye(2, dtype=np.float32) * np.float32(RESIZE)
    post_tran = np.zeros(2, dtype=np.float32) - np.asarray(CROP_BOX[:2], dtype=np.float32)
    angle = 0.0  # rotate = 0 at test time; kept to mirror the original sequence of operations
    rot = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]], dtype=np.float32
    )
    half = np.array(
        [CROP_BOX[2] - CROP_BOX[0], CROP_BOX[3] - CROP_BOX[1]], dtype=np.float32
    ) / np.float32(2)
    offset = rot @ -half + half
    post_rot = rot @ post_rot
    post_tran = rot @ post_tran + offset
    rot3 = np.eye(3, dtype=np.float32)
    rot3[:2, :2] = post_rot
    tran3 = np.zeros(3, dtype=np.float32)
    tran3[:2] = post_tran
    return rot3, tran3


def bda_matrix() -> FloatArray:
    """BEVAug test mode: no rotation, scale 1, no flips, zero translation -> identity."""
    return np.eye(4, dtype=np.float32)


def build_inputs(images: Mapping[str, Image.Image], rig: RigCalibration) -> ModelInputs:
    """Six RGB 1600x900 images (by camera name) + calibration -> model inputs."""
    if set(images) != set(CAMERA_ORDER):
        raise ValueError(f"images must be exactly {CAMERA_ORDER}")
    planes = []
    for name in CAMERA_ORDER:
        image = images[name]
        if image.size != (SRC_WIDTH, SRC_HEIGHT) or image.mode != "RGB":
            raise ValueError(
                f"{name}: expected RGB {SRC_WIDTH}x{SRC_HEIGHT}, got {image.mode} {image.size}"
            )
        planes.append(normalize(transform_image(image)))
    sensor2egos, ego2globals, intrins = rig_matrices(rig)
    post_rot, post_tran = image_post_transform()
    inputs = ModelInputs(
        imgs=np.stack(planes),
        sensor2egos=sensor2egos,
        ego2globals=ego2globals,
        intrins=intrins,
        post_rots=np.stack([post_rot] * N_CAMS),
        post_trans=np.stack([post_tran] * N_CAMS),
        bda=bda_matrix(),
    )
    inputs.check()
    return inputs
