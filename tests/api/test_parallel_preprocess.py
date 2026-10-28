"""BEVSERVE_PREPROCESS_THREADS: the API behaves identically in serial and parallel mode."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Sequence

import pytest

from bevserve.constants import CAMERA_ORDER
from bevserve.detector import DEFAULT_FAKE_BOXES, Detection, DetectorInfo
from bevserve.preprocess import EXPECTED_SHAPES, ModelInputs
from tests.api.helpers import build_app, client_for
from tests.synthetic import calibration_payload, encode, multipart_files, synthetic_image

pytestmark = pytest.mark.anyio


class RecordingDetector:
    """Keeps a copy of every ModelInputs it receives."""

    def __init__(self) -> None:
        self.seen: list[ModelInputs] = []
        self._lock = threading.Lock()

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(name="recording", checkpoint_sha256=None, config_sha256=None)

    def detect(self, inputs: ModelInputs) -> Sequence[Detection]:
        inputs.check()
        with self._lock:
            self.seen.append(inputs)
        return DEFAULT_FAKE_BOXES


def _images(seed: int) -> dict[str, bytes]:
    return {n: encode(synthetic_image(100 * seed + i)) for i, n in enumerate(CAMERA_ORDER)}


def _strip(body: dict) -> dict:
    body = dict(body)
    body.pop("timing_ms", None)
    body.pop("request_id", None)
    return body


async def _post(threads: int, requests: list[dict[str, bytes]], concurrent: bool = False):
    detector = RecordingDetector()
    app, _ = build_app(factory=lambda: detector, preprocess_threads=threads, max_in_flight=16)
    async with client_for(app) as client:

        async def one(images: dict[str, bytes]):
            files, data = multipart_files(images, calibration_payload(with_ego_poses=True))
            r = await client.post("/v1/detect", files=files, data=data)
            return r.status_code, r.json()

        if concurrent:
            responses = await asyncio.gather(*(one(i) for i in requests))
        else:
            responses = [await one(i) for i in requests]
    return responses, detector.seen


@pytest.mark.parametrize("threads", [6, 12, 24])
async def test_inputs_and_responses_identical_to_serial(threads: int) -> None:
    requests = [_images(s) for s in range(3)]
    serial, serial_inputs = await _post(0, requests)
    parallel, parallel_inputs = await _post(threads, requests)
    assert [s for s, _ in serial] == [s for s, _ in parallel] == [200, 200, 200]
    assert [_strip(b) for _, b in serial] == [_strip(b) for _, b in parallel]
    for a, b in zip(serial_inputs, parallel_inputs, strict=True):
        for name in EXPECTED_SHAPES:
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name


async def test_concurrent_requests_receive_their_own_inputs() -> None:
    requests = [_images(s) for s in range(8)]
    _, serial_inputs = await _post(0, requests)
    by_frame = {a.imgs.tobytes(): k for k, a in enumerate(serial_inputs)}
    responses, parallel_inputs = await _post(12, requests, concurrent=True)
    assert all(status == 200 for status, _ in responses)
    assert sorted(by_frame[a.imgs.tobytes()] for a in parallel_inputs) == list(range(8))


@pytest.mark.parametrize("bad", [("CAM_BACK",), ("CAM_BACK_RIGHT", "CAM_FRONT"), CAMERA_ORDER])
async def test_corrupt_images_give_the_same_error(bad: tuple[str, ...]) -> None:
    images = _images(0)
    for name in bad:
        images[name] = images[name][: len(images[name]) // 3]  # header intact, data truncated
    (serial,), _ = await _post(0, [images])
    (parallel,), seen = await _post(6, [images])
    assert serial[0] == parallel[0] == 422
    assert _strip(serial[1]) == _strip(parallel[1])
    first = next(n for n in CAMERA_ORDER if n in bad)
    assert parallel[1]["errors"][0]["field"] == first, json.dumps(parallel[1])
    assert not seen
