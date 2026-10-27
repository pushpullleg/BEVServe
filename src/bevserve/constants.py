"""Input contract of BEVDet-R50 (configs/bevdet/bevdet-r50-cbgs.py, BEVDet dev3.0).

These values are fixed by the trained model. Changing any of them silently changes
model behaviour, so each one is covered by a test.
"""

from __future__ import annotations

from typing import Final

# Canonical camera order (data_config['cams']). Index 0 defines the reference ("key") ego frame
# in BEVDet.prepare_inputs, so the order is part of the output-frame contract.
CAMERA_ORDER: Final[tuple[str, ...]] = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
KEY_CAMERA: Final[str] = CAMERA_ORDER[0]

# Source image size the model was trained on (data_config['src_size'] = (900, 1600)).
SRC_WIDTH: Final[int] = 1600
SRC_HEIGHT: Final[int] = 900

# Network input size (data_config['input_size'] = (256, 704)).
INPUT_HEIGHT: Final[int] = 256
INPUT_WIDTH: Final[int] = 704

# Test-time geometry, from PrepareImageInputs.sample_augmentation (test branch, resize_test=0,
# crop_h=(0, 0)): resize = 704 / 1600, resize_dims = (704, 396), crop = (0, 140, 704, 396).
RESIZE: Final[float] = INPUT_WIDTH / SRC_WIDTH
RESIZED_WIDTH: Final[int] = int(SRC_WIDTH * RESIZE)
RESIZED_HEIGHT: Final[int] = int(SRC_HEIGHT * RESIZE)
CROP_TOP: Final[int] = RESIZED_HEIGHT - INPUT_HEIGHT
CROP_LEFT: Final[int] = int(max(0, RESIZED_WIDTH - INPUT_WIDTH) / 2)
CROP_BOX: Final[tuple[int, int, int, int]] = (
    CROP_LEFT,
    CROP_TOP,
    CROP_LEFT + INPUT_WIDTH,
    CROP_TOP + INPUT_HEIGHT,
)

# mmlabNormalize (loading.py): applied after a BGR<->RGB channel swap, see preprocess.py.
NORM_MEAN: Final[tuple[float, float, float]] = (123.675, 116.28, 103.53)
NORM_STD: Final[tuple[float, float, float]] = (58.395, 57.12, 57.375)

# CenterHead task class order (one task, 10 classes).
CLASS_NAMES: Final[tuple[str, ...]] = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

# Decoder limits from test_cfg: the head drops scores below 0.1 and keeps at most 500 boxes.
MODEL_SCORE_THRESHOLD: Final[float] = 0.1
MODEL_MAX_BOXES: Final[int] = 500

# Name of the reference frame of every returned box.
OUTPUT_FRAME: Final[str] = f"ego@{KEY_CAMERA}"
