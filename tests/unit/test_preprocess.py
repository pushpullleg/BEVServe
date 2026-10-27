from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from bevserve import constants as c
from bevserve.preprocess import (
    build_inputs,
    image_post_transform,
    normalize,
    transform_image,
)
from tests.synthetic import synthetic_image, synthetic_rig


def _six(seed0: int = 0) -> dict[str, Image.Image]:
    return {name: synthetic_image(seed0 + i) for i, name in enumerate(c.CAMERA_ORDER)}


def test_geometry_constants_match_bevdet_test_branch() -> None:
    assert c.RESIZE == 0.44
    assert (c.RESIZED_WIDTH, c.RESIZED_HEIGHT) == (704, 396)
    assert c.CROP_BOX == (0, 140, 704, 396)
    assert (c.INPUT_HEIGHT, c.INPUT_WIDTH) == (256, 704)


def test_build_inputs_shapes_dtypes_and_fixed_matrices() -> None:
    inputs = build_inputs(_six(), synthetic_rig())
    assert inputs.imgs.shape == (6, 3, 256, 704)
    assert all(a.dtype == np.float32 for a in inputs.as_tuple())
    expected_rot = np.diag(np.array([0.44, 0.44, 1.0], dtype=np.float32))
    for i in range(6):
        np.testing.assert_array_equal(inputs.post_rots[i], expected_rot)
        np.testing.assert_array_equal(inputs.post_trans[i], np.array([0, -140, 0], np.float32))
    np.testing.assert_array_equal(inputs.bda, np.eye(4, dtype=np.float32))


def test_post_transform_maps_source_pixel_to_network_pixel() -> None:
    rot, tran = image_post_transform()
    # Source pixel (800, 450) -> resized (352, 198) -> cropped (352, 58).
    np.testing.assert_allclose(
        rot @ np.array([800.0, 450.0, 1.0]) + tran, [352.0, 58.0, 1.0], atol=1e-4
    )


def test_channel_order_is_bgr_after_normalization() -> None:
    """Guard: the trained network receives BGR. Do not 'fix' this."""
    r, g, b = 200, 100, 50
    out = normalize(Image.new("RGB", (704, 256), (r, g, b)))
    assert out.shape == (3, 256, 704)
    np.testing.assert_allclose(out[0], (b - 123.675) / 58.395, rtol=1e-6)
    np.testing.assert_allclose(out[1], (g - 116.28) / 57.12, rtol=1e-6)
    np.testing.assert_allclose(out[2], (r - 103.53) / 57.375, rtol=1e-6)


def test_crop_removes_the_top_of_the_resized_image() -> None:
    src = Image.new("RGB", (1600, 900), (0, 0, 0))
    src.paste((255, 255, 255), (0, 450, 1600, 470))  # band at source rows 450-469
    rows = np.array(transform_image(src))[:, :, 0].mean(axis=1)
    bright = np.flatnonzero(rows > 128)
    # 450 * 0.44 = 198 and 470 * 0.44 = 206.8 in the resized image; minus the 140 cropped rows.
    assert bright.min() >= 57 and bright.max() <= 67
    assert rows[:50].max() < 10 and rows[80:].max() < 10


def test_resize_uses_pillow_default_bicubic() -> None:
    img = synthetic_image(7)
    expected = img.resize((704, 396), Image.Resampling.BICUBIC).crop((0, 140, 704, 396))
    np.testing.assert_array_equal(np.array(transform_image(img)), np.array(expected))


def test_preprocessing_is_deterministic_and_does_not_mutate_input() -> None:
    images = _six(3)
    before = np.array(images["CAM_BACK"]).copy()
    a = build_inputs(images, synthetic_rig())
    b = build_inputs(images, synthetic_rig())
    for x, y in zip(a.as_tuple(), b.as_tuple(), strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(np.array(images["CAM_BACK"]), before)


def test_camera_planes_follow_canonical_order() -> None:
    images = _six()
    images["CAM_FRONT_LEFT"] = Image.new("RGB", (1600, 900), (0, 0, 0))
    images["CAM_BACK_RIGHT"] = Image.new("RGB", (1600, 900), (255, 255, 255))
    shuffled = dict(reversed(list(images.items())))
    inputs = build_inputs(shuffled, synthetic_rig())
    assert inputs.imgs[0].max() < 0 < inputs.imgs[5].min()


@pytest.mark.parametrize(
    "bad",
    [Image.new("RGB", (1280, 720)), Image.new("L", (1600, 900))],
    ids=["wrong-size", "not-rgb"],
)
def test_build_inputs_rejects_unexpected_images(bad: Image.Image) -> None:
    images = _six()
    images["CAM_FRONT"] = bad
    with pytest.raises(ValueError, match="CAM_FRONT"):
        build_inputs(images, synthetic_rig())


def test_build_inputs_rejects_missing_camera() -> None:
    images = _six()
    del images["CAM_BACK"]
    with pytest.raises(ValueError, match="exactly"):
        build_inputs(images, synthetic_rig())
