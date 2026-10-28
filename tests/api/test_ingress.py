"""Pre-admission (ingress) behaviour, CPU only: uploads, malformed bodies, disconnects (M7 R5/R6).

Parsing, part checks and header checks run before admission, so none of these may consume an
admission token, leave threads running or keep request memory alive. Concurrency here is bounded
(at most 32 in-process clients with ~1 MB bodies); nothing floods.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import tracemalloc
from typing import Any

import httpx
import pytest

from bevserve.detector import FakeDetector
from tests.api.helpers import build_app, client_for
from tests.synthetic import multipart_files

pytestmark = pytest.mark.anyio


def _body() -> tuple[bytes, str]:
    files, data = multipart_files()
    req = httpx.Request("POST", "http://x/v1/detect", files=files, data=data)
    return req.read(), req.headers["content-type"]


async def _asgi(
    app: Any, body: bytes, ctype: str, messages: list[dict[str, Any]]
) -> tuple[int | None, str | None]:
    """Drive the app directly with chosen receive messages; returns (status, escaped exception)."""
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if messages:
            msg = messages.pop(0)
            if msg.get("wait") is not None:
                await msg["wait"].wait()
                return {"type": "http.disconnect"}
            return msg
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    headers = [(b"content-type", ctype.encode()), (b"content-length", str(len(body)).encode())]
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": "/v1/detect", "raw_path": b"/v1/detect", "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8000), "state": {},
    }  # fmt: skip
    escaped = None
    try:
        await app(scope, receive, send)
    except Exception as exc:
        escaped = type(exc).__name__
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    return status, escaped


def _prep_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith("bevserve-prep"))


@pytest.mark.parametrize("threads", [0, 12])
async def test_bounded_concurrent_uploads_release_tokens_threads_and_memory(threads: int) -> None:
    app, _ = build_app(FakeDetector(delay_s=0.01), preprocess_threads=threads, max_in_flight=5)
    service = app.state.service
    files, data = multipart_files()
    async with client_for(app) as client:
        await client.post("/v1/detect", files=files, data=data)  # warm caches before measuring
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        statuses: dict[str, int] = {}

        async def one() -> None:
            r = await client.post("/v1/detect", files=files, data=data)
            key = f"{r.status_code} {r.json().get('type', 'ok')}"
            statuses[key] = statuses.get(key, 0) + 1

        for _ in range(2):
            await asyncio.gather(*(one() for _ in range(32)))
        assert (
            set(statuses) <= {"200 ok", "503 /errors/queue_full"} and statuses.get("200 ok", 0) >= 5
        )
        assert service.executor.in_flight == 0
        gc.collect()
        retained = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.stop()
        assert retained < 8 * 2**20, f"{retained / 2**20:.1f} MiB still allocated after 64 requests"
        assert _prep_threads() <= threads
        r = await client.post("/v1/detect", files=files, data=data)
        assert r.status_code == 200


@pytest.mark.parametrize(
    ("cut", "status", "problem"),
    [
        (lambda b: b[: len(b) // 2], 422, "/errors/camera_set"),  # upload stops mid-body
        (lambda b: b[:-40], 422, "/errors/camera_set"),  # final part incomplete
        (lambda b: b"x" * 4000, 400, "/errors/multipart"),  # not multipart at all
    ],
    ids=["truncated", "no-final-boundary", "garbage"],
)
async def test_incomplete_or_malformed_uploads_are_rejected_before_admission(
    cut: Any, status: int, problem: str
) -> None:
    app, _ = build_app(preprocess_threads=12)
    service = app.state.service
    body, ctype = _body()
    async with client_for(app) as client:
        r = await client.post("/v1/detect", content=cut(body), headers={"content-type": ctype})
        assert (r.status_code, r.json()["type"]) == (status, problem)
        assert service.executor.in_flight == 0
        files, data = multipart_files()
        assert (await client.post("/v1/detect", files=files, data=data)).status_code == 200


async def test_client_disconnect_during_upload_is_pre_admission_and_releases_everything() -> None:
    """Documents R5: Starlette raises ClientDisconnect from request.form(); BEVServe does not catch
    it, so the request is answered 500 'internal' (unsendable: the client is gone), logged as an
    unhandled error, and the exception reaches the ASGI server. Nothing leaks: it happens before
    admission and the form is closed."""
    app, _ = build_app(preprocess_threads=12)
    service = app.state.service
    await service.initialize()
    body, ctype = _body()
    try:
        for _ in range(3):
            messages = [{"type": "http.request", "body": body[: len(body) // 2], "more_body": True},
                        {"type": "http.disconnect"}]  # fmt: skip
            status, escaped = await _asgi(app, body, ctype, messages)
            assert (status, escaped) == (500, "ClientDisconnect")
            assert service.executor.in_flight == 0
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files, data = multipart_files()
            assert (await client.post("/v1/detect", files=files, data=data)).status_code == 200
    finally:
        service.shutdown()


async def test_client_disconnect_after_admission_completes_and_frees_the_token() -> None:
    """uvicorn does not cancel a handler when its client leaves; the job runs to the end, the token
    is released when the GPU job finishes, and the response is simply undeliverable."""
    gate = threading.Event()
    app, _ = build_app(FakeDetector(gate=gate), preprocess_threads=12, max_in_flight=1)
    service = app.state.service
    await service.initialize()
    body, ctype = _body()
    try:
        left = asyncio.Event()
        task = asyncio.create_task(_asgi(app, body, ctype, [
            {"type": "http.request", "body": body, "more_body": False}, {"wait": left},
        ]))  # fmt: skip
        for _ in range(200):  # until the job holds the only token
            if service.executor.in_flight == 1 and service.executor.running_for() is not None:
                break
            await asyncio.sleep(0.01)
        assert service.executor.in_flight == 1
        left.set()  # the client disconnects while the GPU job runs
        await asyncio.sleep(0.05)
        assert service.executor.in_flight == 1  # still held: the job keeps its token
        gate.set()
        status, escaped = await asyncio.wait_for(task, 5)
        assert (status, escaped) == (200, None) and service.executor.in_flight == 0
    finally:
        gate.set()
        service.shutdown()


async def test_many_invalid_requests_alongside_valid_ones_return_every_token() -> None:
    app, _ = build_app(FakeDetector(delay_s=0.01), preprocess_threads=12, max_in_flight=5)
    service = app.state.service
    files, data = multipart_files()
    name, (fname, jpeg, mime) = files[2]
    bad_files = [*files[:2], (name, (fname, jpeg[: len(jpeg) // 3], mime)), *files[3:]]
    async with client_for(app) as client:
        seen: dict[str, int] = {}

        async def one(k: int) -> None:
            r = await client.post("/v1/detect", files=bad_files if k % 2 else files, data=data)
            key = f"{r.status_code} {r.json().get('type', 'ok')}"
            seen[key] = seen.get(key, 0) + 1

        await asyncio.gather(*(one(k) for k in range(32)))
        assert set(seen) <= {"200 ok", "422 /errors/image_corrupt", "503 /errors/queue_full"}, seen
        assert service.executor.in_flight == 0
        assert (await client.post("/v1/detect", files=files, data=data)).status_code == 200
