"""POST /v1/detect: contract, validation and error shapes."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from bevserve.constants import CAMERA_ORDER, CLASS_NAMES
from tests.api.helpers import build_app, client_for
from tests.synthetic import calibration_payload, encode, multipart_files, synthetic_image

pytestmark = pytest.mark.anyio


def assert_problem(resp: httpx.Response, status: int, code: str) -> dict[str, Any]:
    assert resp.status_code == status, resp.text
    assert resp.headers["content-type"].startswith("application/problem+json")
    body: dict[str, Any] = resp.json()
    assert body["status"] == status and body["type"] == f"/errors/{code}", body
    assert body["request_id"] == resp.headers["x-request-id"]
    assert "Traceback" not in resp.text
    return body


async def test_detect_returns_fake_boxes_with_identity_headers() -> None:
    app, _ = build_app()
    files, data = multipart_files()
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/detect", files=files, data=data, headers={"X-Request-ID": "abc-123"}
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert resp.headers["x-request-id"] == "abc-123" == body["request_id"]
    assert resp.headers["x-model-version"] == "fake-detector@none"
    assert body["frame"] == "ego@CAM_FRONT_LEFT"
    assert body["units"] == {"length": "m", "angle": "rad", "velocity": "m/s"}
    assert body["model"]["camera_order"] == list(CAMERA_ORDER)
    assert body["model"]["class_names"] == list(CLASS_NAMES)
    scores = [b["score"] for b in body["boxes"]]
    assert scores == sorted(scores, reverse=True) and len(scores) == 3
    assert [b["label"] for b in body["boxes"]] == ["car", "pedestrian", "traffic_cone"]
    assert set(body["timing_ms"]) == {"decode", "preprocess", "inference", "total"}
    assert body["timing_ms"]["total"] >= body["timing_ms"]["inference"]


async def test_options_filter_and_truncate() -> None:
    app, _ = build_app()
    payload = calibration_payload()
    payload["options"] = {"score_threshold": 0.3, "max_boxes": 1}
    files, data = multipart_files(calibration=payload)
    async with client_for(app) as client:
        resp = await client.post("/v1/detect", files=files, data=data)
    assert [b["label"] for b in resp.json()["boxes"]] == ["car"]


async def test_png_and_calibration_with_ego_poses_accepted() -> None:
    app, _ = build_app()
    images = {n: encode(synthetic_image(i), "PNG") for i, n in enumerate(CAMERA_ORDER)}
    files, data = multipart_files(
        images=images, calibration=calibration_payload(with_ego_poses=True)
    )
    files = [(n, (f"{n}.png", b, "image/png")) for n, (_, b, _) in files]
    async with client_for(app) as client:
        resp = await client.post("/v1/detect", files=files, data=data)
    assert resp.status_code == 200, resp.text


async def test_malformed_request_id_is_replaced() -> None:
    app, _ = build_app()
    async with client_for(app) as client:
        resp = await client.get("/healthz", headers={"X-Request-ID": "bad id with spaces"})
    rid = resp.headers["x-request-id"]
    assert rid != "bad id with spaces" and len(rid) == 32


async def test_missing_camera_part() -> None:
    app, _ = build_app()
    files, data = multipart_files()
    async with client_for(app) as client:
        body = assert_problem(
            await client.post("/v1/detect", files=files[1:], data=data), 422, "camera_set"
        )
    assert "CAM_FRONT_LEFT" in body["detail"]


async def test_duplicate_camera_part() -> None:
    app, _ = build_app()
    files, data = multipart_files()
    async with client_for(app) as client:
        assert_problem(
            await client.post("/v1/detect", files=[*files, files[0]], data=data), 422, "camera_set"
        )


async def test_unexpected_parts() -> None:
    app, _ = build_app()
    files, data = multipart_files()
    async with client_for(app) as client:
        extra_file = [*files, ("CAM_ROOF", ("r.jpg", files[0][1][1], "image/jpeg"))]
        assert_problem(
            await client.post("/v1/detect", files=extra_file, data=data), 422, "unexpected_part"
        )
        assert_problem(
            await client.post("/v1/detect", files=files, data={**data, "debug": "1"}),
            422,
            "unexpected_part",
        )


async def test_calibration_missing_bad_json_and_invalid() -> None:
    app, _ = build_app()
    files, _ = multipart_files()
    bad = calibration_payload()
    bad["cameras"]["CAM_BACK"]["intrinsic"][0][0] = 5.0
    async with client_for(app) as client:
        assert_problem(await client.post("/v1/detect", files=files), 422, "calibration_missing")
        assert_problem(
            await client.post("/v1/detect", files=files, data={"calibration": "{nope"}),
            400,
            "calibration_json",
        )
        body = assert_problem(
            await client.post("/v1/detect", files=files, data={"calibration": json.dumps(bad)}),
            422,
            "intrinsic",
        )
    assert body["errors"][0]["field"] == "calibration.cameras.CAM_BACK.intrinsic"


async def test_wrong_image_size() -> None:
    app, _ = build_app()
    images = {n: encode(synthetic_image(i)) for i, n in enumerate(CAMERA_ORDER)}
    images["CAM_BACK"] = encode(synthetic_image(9, 1280, 720))
    files, data = multipart_files(images=images)
    async with client_for(app) as client:
        body = assert_problem(
            await client.post("/v1/detect", files=files, data=data), 422, "image_size"
        )
    assert body["errors"][0]["field"] == "CAM_BACK" and "1280x720" in body["detail"]


async def test_not_an_image_and_wrong_part_type() -> None:
    app, _ = build_app()
    files, data = multipart_files()
    garbage = [
        (n, (f, b"definitely not a jpeg", t)) if n == "CAM_FRONT" else (n, (f, b, t))
        for n, (f, b, t) in files
    ]
    texty = [
        (n, (f, b, "text/plain")) if n == "CAM_FRONT" else (n, (f, b, t)) for n, (f, b, t) in files
    ]
    async with client_for(app) as client:
        assert_problem(
            await client.post("/v1/detect", files=garbage, data=data), 415, "image_format"
        )
        assert_problem(
            await client.post("/v1/detect", files=texty, data=data), 415, "part_content_type"
        )


async def test_non_multipart_body() -> None:
    app, _ = build_app()
    async with client_for(app) as client:
        assert_problem(await client.post("/v1/detect", json={"a": 1}), 415, "content_type")


async def test_unknown_route_is_a_problem() -> None:
    app, _ = build_app()
    async with client_for(app) as client:
        assert_problem(await client.get("/v2/detect"), 404, "not_found")
        assert_problem(await client.get("/v1/detect"), 405, "method_not_allowed")
