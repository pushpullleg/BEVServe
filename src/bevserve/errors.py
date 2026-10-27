"""Structured errors: application/problem+json (RFC 9457), never a stack trace."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from bevserve.schemas import FieldError, Problem

PROBLEM_MEDIA_TYPE = "application/problem+json"
log = logging.getLogger("bevserve")


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        title: str,
        detail: str,
        errors: list[FieldError] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        self.errors = errors or []
        self.headers = headers or {}


class BodyTooLargeError(ApiError):
    def __init__(self, limit: int) -> None:
        super().__init__(
            413, "request_too_large", "Request too large", f"request body exceeds {limit} bytes"
        )


def request_id_of(request: Request) -> str | None:
    rid = request.scope.get("state", {}).get("request_id")
    return rid if isinstance(rid, str) else None


def problem_response(
    status: int,
    code: str,
    title: str,
    detail: str,
    request_id: str | None,
    errors: list[FieldError] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = Problem(
        type=f"/errors/{code}",
        title=title,
        status=status,
        detail=detail,
        request_id=request_id,
        errors=errors or [],
    )
    return JSONResponse(
        body.model_dump(), status_code=status, media_type=PROBLEM_MEDIA_TYPE, headers=headers
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return problem_response(
            exc.status,
            exc.code,
            exc.title,
            exc.detail,
            request_id_of(request),
            exc.errors,
            exc.headers,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return problem_response(
            exc.status_code,
            code,
            str(exc.detail),
            str(exc.detail),
            request_id_of(request),
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            FieldError(field=".".join(str(p) for p in e["loc"]), code=e["type"], message=e["msg"])
            for e in exc.errors()
        ]
        return problem_response(
            422, "invalid_request", "Invalid request", "see errors", request_id_of(request), errors
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        rid = request_id_of(request)
        log.exception("unhandled error request_id=%s", rid)
        return problem_response(500, "internal", "Internal error", "unexpected server error", rid)
