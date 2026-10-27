"""Upload limits are enforced before any image is decoded (review correction 5)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from bevserve import api
from bevserve.constants import CAMERA_ORDER
from tests.api.helpers import build_app, client_for
from tests.api.test_detect import assert_problem
from tests.synthetic import calibration_payload, encode, multipart_files, synthetic_image

pytestmark = pytest.mark.anyio


@pytest.fixture
def decode_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = api.decode_rgb

    def spy(data: bytes):  # type: ignore[no-untyped-def]
        calls.append(len(data))
        return real(data)

    monkeypatch.setattr(api, "decode_rgb", spy)
    return calls


async def test_content_length_over_limit_is_refused_without_reading(
    decode_calls: list[int],
) -> None:
    app, _ = build_app(max_request_bytes=1000)
    files, data = multipart_files()
    async with client_for(app) as client:
        body = assert_problem(
            await client.post("/v1/detect", files=files, data=data), 413, "request_too_large"
        )
    assert "1000" in body["detail"] and decode_calls == []


async def test_streamed_body_over_limit_is_cut_off(decode_calls: list[int]) -> None:
    """A valid multipart body sent chunked (no Content-Length) that grows past the limit."""
    files, data = multipart_files()
    template = httpx.Request("POST", "http://test/v1/detect", files=files, data=data)
    body = template.read()
    limit = len(body) // 3
    chunk = 4096
    total_chunks = -(-len(body) // chunk)
    sent: list[int] = []

    async def chunks() -> AsyncIterator[bytes]:
        for start in range(0, len(body), chunk):
            sent.append(start)
            yield body[start : start + chunk]

    app, _ = build_app(max_request_bytes=limit)
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/detect",
            content=chunks(),
            headers={"content-type": template.headers["content-type"]},
        )
    assert_problem(resp, 413, "request_too_large")
    assert decode_calls == []
    assert len(sent) < total_chunks  # the server stopped reading early


async def test_image_part_over_limit(decode_calls: list[int]) -> None:
    app, _ = build_app(max_image_bytes=20_000)
    images = {n: encode(synthetic_image(i)) for i, n in enumerate(CAMERA_ORDER)}
    assert max(len(b) for b in images.values()) > 20_000
    files, data = multipart_files(images=images)
    async with client_for(app) as client:
        assert_problem(
            await client.post("/v1/detect", files=files, data=data), 413, "image_too_large"
        )
    assert decode_calls == []


async def test_calibration_field_over_limit() -> None:
    app, _ = build_app(max_calibration_bytes=512)
    files, data = multipart_files()
    assert len(data["calibration"]) > 512
    async with client_for(app) as client:
        assert_problem(
            await client.post("/v1/detect", files=files, data=data), 413, "calibration_too_large"
        )


async def test_wrong_size_rejected_before_decode(decode_calls: list[int]) -> None:
    app, _ = build_app()
    images = {n: encode(synthetic_image(i)) for i, n in enumerate(CAMERA_ORDER)}
    images["CAM_BACK_RIGHT"] = encode(synthetic_image(9, 1600, 1200))
    files, data = multipart_files(images=images)
    async with client_for(app) as client:
        assert_problem(await client.post("/v1/detect", files=files, data=data), 422, "image_size")
    assert decode_calls == []


async def test_invalid_calibration_rejected_before_decode(decode_calls: list[int]) -> None:
    app, _ = build_app()
    files, _ = multipart_files()
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/detect", files=files, data={"calibration": json.dumps({"schema_version": "1"})}
        )
    assert resp.status_code == 422 and decode_calls == []


async def test_valid_request_decodes_each_image_once(decode_calls: list[int]) -> None:
    app, _ = build_app()
    files, data = multipart_files(calibration=calibration_payload())
    async with client_for(app) as client:
        assert (await client.post("/v1/detect", files=files, data=data)).status_code == 200
    assert len(decode_calls) == 6
