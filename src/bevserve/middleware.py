"""Pure-ASGI middleware: request IDs and a request-body size limit.

The body limit runs before routing and multipart parsing (review correction 5): an oversized
Content-Length is refused without reading the body, and a body that grows past the limit while
streaming (e.g. chunked) is cut off as soon as the limit is crossed.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from bevserve.errors import BodyTooLargeError, problem_response

log = logging.getLogger("bevserve.access")
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
REQUEST_ID_HEADER = "x-request-id"


class RequestIdMiddleware:
    """Accept a well-formed X-Request-ID or mint one; echo it; log one line per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = Headers(scope=scope).get(REQUEST_ID_HEADER, "")
        request_id = supplied if _VALID_ID.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = 0

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            log.info(
                "request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
                request_id,
                scope["method"],
                scope["path"],
                status,
                (time.perf_counter() - started) * 1000,
            )


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = scope.get("state", {}).get("request_id")
        length = Headers(scope=scope).get("content-length")
        if length is not None and (not length.isdigit() or int(length) > self.max_bytes):
            response = problem_response(
                413,
                "request_too_large",
                "Request too large",
                f"request body exceeds {self.max_bytes} bytes",
                request_id,
            )
            await response(scope, receive, send)
            return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLargeError(self.max_bytes)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLargeError as exc:
            if started:
                raise
            await problem_response(exc.status, exc.code, exc.title, exc.detail, request_id)(
                scope, receive, send
            )
