"""Parallel preprocessing (camera_pool) is byte-identical to the serial path and fails the same way.

Reference = the serial path exactly as the API runs it: decode all six with decode_rgb, then
build_inputs(images, calibration.to_rig()).
"""

from __future__ import annotations

import functools
import io
import threading
import time
from collections.abc import Callable, Iterator

import cv2
import numpy as np
import pytest
from PIL import Image

from bevserve import camera_pool
from bevserve.calibration import RigCalibration
from bevserve.camera_pool import CameraDecodeError, CameraPool
from bevserve.constants import CAMERA_ORDER
from bevserve.imaging import ImageRejectedError, decode_rgb
from bevserve.preprocess import EXPECTED_SHAPES, ModelInputs, build_inputs
from bevserve.service import Service
from bevserve.settings import Settings
from tests.synthetic import encode, synthetic_image, synthetic_rig

POOL_SIZES = (1, 6, 12, 24, 32)


def frame(seed: int, fmt: str = "JPEG", mode: str = "RGB") -> dict[str, bytes]:
    """Six distinct images (one per camera) for one request (a fresh dict; bytes are shared)."""
    return dict(_frame(seed, fmt, mode))


@functools.cache
def _frame(seed: int, fmt: str, mode: str) -> dict[str, bytes]:
    out = {}
    for i, name in enumerate(CAMERA_ORDER):
        image = synthetic_image(100 * seed + i)
        out[name] = encode(image if mode == "RGB" else image.convert(mode), fmt)
    return out


def serial(raw: dict[str, bytes], to_rig: Callable[[], RigCalibration]) -> ModelInputs:
    images = {name: decode_rgb(data) for name, data in raw.items()}
    return build_inputs(images, to_rig())


def assert_identical(a: ModelInputs, b: ModelInputs) -> None:
    for name in EXPECTED_SHAPES:
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype == np.float32, name
        assert x.shape == y.shape == EXPECTED_SHAPES[name], name
        assert x.flags.c_contiguous and y.flags.c_contiguous, name
        assert x.strides == y.strides, name
        assert x.tobytes() == y.tobytes(), name  # bit-identical, including -0.0 and NaN payloads


@pytest.fixture(params=POOL_SIZES, ids=lambda n: f"threads{n}")
def pool(request: pytest.FixtureRequest) -> Iterator[CameraPool]:
    p = CameraPool(request.param)
    yield p
    p.shutdown()


@pytest.mark.parametrize(
    ("fmt", "mode"), [("JPEG", "RGB"), ("PNG", "RGB"), ("JPEG", "L"), ("PNG", "RGBA"), ("PNG", "P")]
)
def test_all_seven_arrays_byte_identical(pool: CameraPool, fmt: str, mode: str) -> None:
    for seed in range(2 if (fmt, mode) == ("JPEG", "RGB") else 1):
        raw = frame(seed, fmt, mode)
        images = {name: decode_rgb(data) for name, data in raw.items()}
        for poses in (False, True):
            rig = synthetic_rig(with_ego_poses=poses)
            inputs, decode_ms, pre_ms = pool.build(raw, lambda rig=rig: rig)
            assert_identical(inputs, build_inputs(images, rig))
            assert decode_ms >= 0 and pre_ms >= 0


def test_planes_follow_camera_order_not_completion_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Camera 0 finishes last, camera 5 first; planes must still be in CAMERA_ORDER."""
    delay = {encode(synthetic_image(i)): 0.05 * (5 - i) for i in range(6)}
    original = camera_pool.decode_rgb

    def slow_decode(data: bytes) -> Image.Image:
        time.sleep(delay.get(data, 0.0))
        return original(data)

    monkeypatch.setattr(camera_pool, "decode_rgb", slow_decode)
    raw = {name: encode(synthetic_image(i)) for i, name in enumerate(CAMERA_ORDER)}
    p = CameraPool(6)
    try:
        inputs, _, _ = p.build(raw, synthetic_rig)
    finally:
        p.shutdown()
    reference = serial(raw, synthetic_rig)
    assert_identical(inputs, reference)
    for i in range(6):  # each plane is its own camera's (the images all differ)
        assert not np.array_equal(inputs.imgs[i], reference.imgs[(i + 1) % 6])


def test_repeated_invocations_are_identical() -> None:
    raw = frame(3)
    p = CameraPool(12)
    try:
        first, _, _ = p.build(raw, synthetic_rig)
        for _ in range(10):
            assert_identical(p.build(raw, synthetic_rig)[0], first)
    finally:
        p.shutdown()
    assert_identical(first, serial(raw, synthetic_rig))


def test_concurrent_requests_each_get_their_own_inputs() -> None:
    frames = [frame(s) for s in range(8)]
    expected = [serial(f, synthetic_rig) for f in frames]
    results: dict[int, ModelInputs] = {}
    errors: list[BaseException] = []
    p = CameraPool(12)

    def request(k: int) -> None:
        try:
            for _ in range(3):
                results[k] = p.build(frames[k], synthetic_rig)[0]
                assert_identical(results[k], expected[k])
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=request, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    p.shutdown()
    assert not errors, errors
    assert len(results) == 8


def _truncated(data: bytes) -> bytes:
    return data[: len(data) // 3]


def _serial_error(raw: dict[str, bytes], to_rig: Callable[[], RigCalibration]) -> tuple[str, str]:
    """(camera, code) of the error the API's serial path reports."""
    for name, data in raw.items():
        try:
            decode_rgb(data)
        except ImageRejectedError as exc:
            return name, exc.code
    to_rig()
    raise AssertionError("no decode error")


@pytest.mark.parametrize("bad", [(2,), (4, 1), (5, 0, 3), tuple(range(6))])
def test_decode_errors_report_the_first_camera_in_order(
    pool: CameraPool, bad: tuple[int, ...]
) -> None:
    raw = frame(0)
    for i in bad:
        raw[CAMERA_ORDER[i]] = _truncated(raw[CAMERA_ORDER[i]])
    with pytest.raises(CameraDecodeError) as caught:
        pool.build(raw, synthetic_rig)
    assert isinstance(caught.value.error, ImageRejectedError)
    assert (caught.value.camera, caught.value.error.code) == _serial_error(raw, synthetic_rig)
    assert caught.value.camera == CAMERA_ORDER[min(bad)]


def test_error_precedence_matches_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """decode error > calibration error > per-camera preprocessing error, as in the serial path."""
    raw = frame(1)

    def bad_rig() -> RigCalibration:
        raise ValueError("calibration")

    original = camera_pool.camera_plane

    def failing_plane(name: str, image: Image.Image) -> np.ndarray:
        if name in ("CAM_BACK", "CAM_FRONT_RIGHT"):
            raise ValueError(f"plane {name}")
        return original(name, image)

    p = CameraPool(6)
    try:
        with pytest.raises(ValueError, match="calibration"):  # beats per-camera errors
            monkeypatch.setattr(camera_pool, "camera_plane", failing_plane)
            p.build(raw, bad_rig)
        with pytest.raises(ValueError, match="plane CAM_FRONT_RIGHT"):  # first in CAMERA_ORDER
            p.build(raw, synthetic_rig)
        broken = dict(raw, CAM_BACK_LEFT=_truncated(raw["CAM_BACK_LEFT"]))
        with pytest.raises(CameraDecodeError) as caught:  # beats calibration and plane errors
            p.build(broken, bad_rig)
        assert caught.value.camera == "CAM_BACK_LEFT"
    finally:
        p.shutdown()


def test_no_task_outlives_a_failed_request(monkeypatch: pytest.MonkeyPatch) -> None:
    finished: list[str] = []
    original = camera_pool._camera_task

    def tracked(name: str, data: bytes) -> object:
        time.sleep(0.02 if name != "CAM_FRONT_LEFT" else 0.0)
        result = original(name, data)
        finished.append(name)
        return result

    monkeypatch.setattr(camera_pool, "_camera_task", tracked)
    raw = frame(0)
    raw["CAM_FRONT_LEFT"] = _truncated(raw["CAM_FRONT_LEFT"])  # fails first and fastest
    p = CameraPool(6)
    try:
        with pytest.raises(CameraDecodeError):
            p.build(raw, synthetic_rig)
        assert sorted(finished) == sorted(CAMERA_ORDER)  # all six done before the raise
    finally:
        p.shutdown()


def test_requests_must_name_the_six_cameras_in_order() -> None:
    p = CameraPool(2)
    try:
        raw = frame(0)
        with pytest.raises(ValueError, match="exactly"):
            p.build(dict(reversed(raw.items())), synthetic_rig)
        with pytest.raises(ValueError, match="exactly"):
            p.build({k: raw[k] for k in CAMERA_ORDER[:5]}, synthetic_rig)
    finally:
        p.shutdown()


def _prep_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(camera_pool.THREAD_PREFIX)]


def test_lifecycle_threads_are_bounded_and_joined() -> None:
    before = cv2.getNumThreads()
    assert not _prep_threads()
    p = CameraPool(4)
    assert not _prep_threads()  # workers start lazily
    assert cv2.getNumThreads() == 1
    for s in range(3):
        p.build(frame(s), synthetic_rig)
    assert 1 <= len(_prep_threads()) <= 4
    p.shutdown()
    p.shutdown()  # idempotent
    assert not _prep_threads()
    assert cv2.getNumThreads() == before
    with pytest.raises(RuntimeError, match="shut down"):
        p.build(frame(0), synthetic_rig)


def test_opencv_threads_restored_only_after_the_last_pool() -> None:
    before = cv2.getNumThreads()
    a, b = CameraPool(2), CameraPool(2)
    a.shutdown()
    assert cv2.getNumThreads() == 1
    b.shutdown()
    assert cv2.getNumThreads() == before


@pytest.mark.parametrize("threads", [0, 33, -1])
def test_pool_size_bounds(threads: int) -> None:
    with pytest.raises(ValueError):
        CameraPool(threads)


def test_setting_bounds_and_service_ownership() -> None:
    assert Settings().preprocess_threads == 0
    with pytest.raises(ValueError):
        Settings(preprocess_threads=33)
    with pytest.raises(ValueError):
        Settings(preprocess_threads=-1)
    assert Service(Settings(), lambda: None).camera_pool is None  # type: ignore[arg-type,return-value]
    service = Service(Settings(preprocess_threads=12), lambda: None)  # type: ignore[arg-type,return-value]
    assert service.camera_pool is not None and service.camera_pool.threads == 12
    service.shutdown()
    assert not _prep_threads()


def test_serial_reference_is_unchanged_by_decode_order() -> None:
    """The serial API path decodes everything before building; check it equals build_inputs."""
    raw = frame(2)
    images = {n: Image.open(io.BytesIO(raw[n])).convert("RGB") for n in CAMERA_ORDER}
    assert_identical(serial(raw, synthetic_rig), build_inputs(images, synthetic_rig()))
