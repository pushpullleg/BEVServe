"""Readiness, init failure vs fatal runtime failure, stall, and slot safety over HTTP
(review corrections 1 and 2)."""

from __future__ import annotations

import asyncio
import threading

import pytest

from bevserve.detector import Detector, DetectorInitError, FakeDetector
from tests.api.helpers import build_app, client_for, wait_until
from tests.api.test_detect import assert_problem
from tests.synthetic import multipart_files

pytestmark = pytest.mark.anyio


async def test_health_and_ready() -> None:
    app, _ = build_app()
    async with client_for(app) as client:
        assert (await client.get("/healthz")).json() == {"status": "ok"}
        ready = await client.get("/readyz")
        assert ready.status_code == 200 and ready.json()["status"] == "ready"
        model = (await client.get("/v1/model")).json()
        assert model["name"] == "fake-detector" and model["checkpoint_sha256"] is None


async def test_init_failure_stays_up_and_reports_reason() -> None:
    def failing() -> Detector:
        raise DetectorInitError(
            "checkpoint_hash_mismatch", "sha256 does not match the configured value"
        )

    app, exits = build_app(factory=failing)
    files, data = multipart_files()
    async with client_for(app, wait_ready=False) as client:
        assert (await client.get("/healthz")).status_code == 200
        ready = await client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json() == {
            "status": "init_failed",
            "reason": "checkpoint_hash_mismatch",
            "detail": "sha256 does not match the configured value",
        }
        assert_problem(await client.post("/v1/detect", files=files, data=data), 503, "not_ready")
        assert (await client.get("/v1/model")).json()["name"] == "unavailable"
    assert exits.codes == []  # an init failure never exits: no crash loop


async def test_unexpected_init_exception_is_contained() -> None:
    def broken() -> Detector:
        raise RuntimeError("secret path " + "/" + "home/someone/ckpt.pth")

    app, exits = build_app(factory=broken)
    async with client_for(app, wait_ready=False) as client:
        body = (await client.get("/readyz")).json()
    assert body["status"] == "init_failed" and body["reason"] == "unexpected"
    assert "secret" not in str(body) and exits.codes == []


async def test_ordinary_inference_error_keeps_service_ready() -> None:
    fake = FakeDetector(fail="error")
    app, exits = build_app(fake)
    files, data = multipart_files()
    async with client_for(app) as client:
        assert_problem(
            await client.post("/v1/detect", files=files, data=data), 500, "inference_failed"
        )
        assert (await client.get("/readyz")).status_code == 200
    assert exits.codes == []


async def test_fatal_error_goes_not_ready_then_exits_nonzero_once() -> None:
    fake = FakeDetector(fail="fatal")
    app, exits = build_app(fake, fatal_exit_grace_s=0.05)
    files, data = multipart_files()
    async with client_for(app) as client:
        assert_problem(
            await client.post("/v1/detect", files=files, data=data), 500, "inference_fatal"
        )
        ready = await client.get("/readyz")
        assert ready.status_code == 503 and ready.json()["status"] == "fatal"
        assert_problem(await client.post("/v1/detect", files=files, data=data), 503, "not_ready")
        await wait_until(lambda: exits.codes)
        await asyncio.sleep(0.1)
    assert exits.codes == [70]
    assert fake.calls == 1


async def test_stalled_job_flips_readiness() -> None:
    gate = threading.Event()
    fake = FakeDetector(gate=gate)
    app, _ = build_app(fake, stall_after_s=0.1, deadline_s=0.05)
    files, data = multipart_files()
    async with client_for(app) as client:
        assert_problem(
            await client.post("/v1/detect", files=files, data=data), 504, "deadline_exceeded"
        )
        await asyncio.sleep(0.2)
        ready = await client.get("/readyz")
        assert ready.status_code == 503 and ready.json()["status"] == "inference_stalled"
        gate.set()
        await wait_until(lambda: app.state.service.executor.in_flight == 0)
        assert (await client.get("/readyz")).status_code == 200


async def test_timed_out_request_never_lets_a_second_job_start() -> None:
    gate = threading.Event()
    fake = FakeDetector(gate=gate)
    app, _ = build_app(fake, deadline_s=0.2, max_in_flight=2)
    files, data = multipart_files()
    async with client_for(app) as client:
        first = await client.post("/v1/detect", files=files, data=data)
        body = assert_problem(first, 504, "deadline_exceeded")
        assert "running" in body["detail"]
        second = await client.post("/v1/detect", files=files, data=data)
        body = assert_problem(second, 504, "deadline_exceeded")
        assert "queued" in body["detail"]
        assert fake.calls == 1  # the second job was cancelled before it could start
        gate.set()
        await wait_until(lambda: app.state.service.executor.in_flight == 0)
        third = await client.post("/v1/detect", files=files, data=data)
        assert third.status_code == 200
    assert fake.calls == 2 and fake.max_active == 1


async def test_admission_full_returns_503_with_retry_after() -> None:
    gate = threading.Event()
    fake = FakeDetector(gate=gate)
    app, _ = build_app(fake, deadline_s=0.1, max_in_flight=1)
    files, data = multipart_files()
    async with client_for(app) as client:
        assert (await client.post("/v1/detect", files=files, data=data)).status_code == 504
        busy = await client.post("/v1/detect", files=files, data=data)
        assert_problem(busy, 503, "queue_full")
        assert busy.headers["retry-after"] == "1"
        gate.set()
        await wait_until(lambda: app.state.service.executor.in_flight == 0)
        assert (await client.post("/v1/detect", files=files, data=data)).status_code == 200


async def test_concurrent_requests_run_one_at_a_time() -> None:
    fake = FakeDetector(delay_s=0.005)
    app, _ = build_app(fake, max_in_flight=16)
    files, data = multipart_files()
    async with client_for(app) as client:
        responses = await asyncio.gather(
            *(client.post("/v1/detect", files=files, data=data) for _ in range(16))
        )
    assert [r.status_code for r in responses] == [200] * 16
    assert fake.calls == 16 and fake.max_active == 1
