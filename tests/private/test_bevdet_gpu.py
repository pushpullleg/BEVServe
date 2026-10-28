"""Private GPU tests (T2). Never run in public CI.

Run only on an authorized GPU host, from a trusted checkout, in the model environment:

    BEVSERVE_PRIVATE_GPU=1 BEVSERVE_BEVDET_CONFIG=... BEVSERVE_CHECKPOINT=... \\
    BEVSERVE_CHECKPOINT_SHA256=... python -m pytest -q tests/private

Uses synthetic inputs only; the real-data comparison is tools/m2_compare_direct.py.
"""

from __future__ import annotations

import os

import pytest

if os.environ.get("BEVSERVE_PRIVATE_GPU") != "1":
    pytest.skip("private GPU tests: set BEVSERVE_PRIVATE_GPU=1", allow_module_level=True)
torch = pytest.importorskip("torch")
pytest.importorskip("mmdet3d")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from bevserve.cli import make_factory  # noqa: E402
from bevserve.detector import DetectorInitError  # noqa: E402
from bevserve.model.bevdet_detector import (  # noqa: E402
    BEVDetDetector,
    _downloads_disabled,
    load_bevdet,
    warmup_inputs,
)
from bevserve.schemas import DetectResponse  # noqa: E402
from bevserve.settings import Settings  # noqa: E402
from tests.api.helpers import client_for  # noqa: E402
from tests.synthetic import multipart_files  # noqa: E402

pytestmark = pytest.mark.private


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(detector="bevdet")


@pytest.fixture(scope="module")
def detector(settings: Settings) -> BEVDetDetector:
    return load_bevdet(settings)


def test_identity_is_the_trusted_checkpoint(detector: BEVDetDetector, settings: Settings) -> None:
    assert detector.info.name == "bevdet-r50-cbgs"
    assert detector.info.checkpoint_sha256 == settings.checkpoint_sha256
    assert detector.info.config_sha256 is not None


def test_detect_returns_valid_detections(detector: BEVDetDetector) -> None:
    dets = detector.detect(warmup_inputs())
    assert all(0 <= d.label_id < 10 and 0.1 <= d.score <= 1.0 for d in dets)


def test_memory_is_stable_across_calls(detector: BEVDetDetector) -> None:
    inputs = warmup_inputs()
    detector.detect(inputs)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(30):
        detector.detect(inputs)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before < 16 * 2**20


def test_wrong_hash_is_rejected(settings: Settings) -> None:
    bad = settings.model_copy(update={"checkpoint_sha256": "0" * 64})
    with pytest.raises(DetectorInitError) as err:
        load_bevdet(bad)
    assert err.value.reason == "checkpoint_hash_mismatch"


def test_download_helpers_are_disabled_during_build() -> None:
    import torch.hub

    with _downloads_disabled(), pytest.raises(DetectorInitError) as err:
        torch.hub.load_state_dict_from_url("https://example.invalid/w.pth")
    assert err.value.reason == "network_download_attempted"


@pytest.mark.anyio
async def test_api_contract_is_unchanged_with_the_real_model(settings: Settings) -> None:
    from bevserve.api import create_app

    factory = make_factory(settings)
    app = create_app(settings, factory, exit_fn=lambda code: None)
    files, data = multipart_files()
    async with client_for(app) as client:
        resp = await client.post("/v1/detect", files=files, data=data)
    assert resp.status_code == 200, resp.text
    body = DetectResponse.model_validate(resp.json())
    assert body.model.checkpoint_sha256 == settings.checkpoint_sha256
    assert resp.headers["x-model-version"] == f"bevdet-r50-cbgs@{settings.checkpoint_sha256[:8]}"


def test_timing_instrumentation_does_not_change_outputs(detector: BEVDetDetector) -> None:
    """M5: CUDA-event timing (BEVSERVE_SERVER_TIMING) must leave every output bit-identical."""
    import numpy as np

    inputs = warmup_inputs()
    inputs = type(inputs)(
        np.random.default_rng(5).standard_normal(inputs.imgs.shape).astype(np.float32),
        *inputs.as_tuple()[1:],
    )
    detector.timing_enabled = False
    plain = detector.detect_native(inputs)
    assert detector.last_timing is None
    detector.timing_enabled = True
    try:
        timed = detector.detect_native(inputs)
        assert set(detector.last_timing or {}) == {"h2d", "forward"}
        assert all(v > 0 for v in (detector.last_timing or {}).values())
    finally:
        detector.timing_enabled = False
    for a, b in zip(plain, timed, strict=True):
        np.testing.assert_array_equal(a, b)


def test_parallel_preprocessing_gives_identical_outputs(detector: BEVDetDetector) -> None:
    """M6: model outputs from serial and pooled preprocessing are bit-identical."""
    import numpy as np

    from bevserve.api import _decode_and_build, _decode_and_build_parallel
    from bevserve.camera_pool import CameraPool
    from bevserve.constants import CAMERA_ORDER
    from bevserve.schemas import CalibrationRequest
    from tests.synthetic import calibration_payload, encode, synthetic_image

    calibration = CalibrationRequest.model_validate(calibration_payload(with_ego_poses=True))
    for size in (6, 12, 24):
        pool = CameraPool(size)
        try:
            for seed in range(2):
                raw = {
                    n: encode(synthetic_image(10 * seed + i)) for i, n in enumerate(CAMERA_ORDER)
                }
                serial = detector.detect_native(_decode_and_build(raw, calibration)[0])
                pooled = detector.detect_native(
                    _decode_and_build_parallel(pool, raw, calibration)[0]
                )
                for a, b in zip(serial, pooled, strict=True):
                    np.testing.assert_array_equal(a, b)
        finally:
            pool.shutdown()
