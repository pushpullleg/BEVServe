"""FastAPI application: /v1/detect, /v1/model, /healthz, /readyz."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from bevserve import __version__
from bevserve.constants import CAMERA_ORDER, OUTPUT_FRAME, SRC_HEIGHT, SRC_WIDTH
from bevserve.detector import Detection, DetectorError, DetectorFactory, FatalDetectorError
from bevserve.errors import ApiError, install_error_handlers, request_id_of
from bevserve.executor import DeadlineExceededError, QueueFullError
from bevserve.imaging import ImageRejectedError, check_size, decode_rgb, inspect_header
from bevserve.middleware import BodyLimitMiddleware, RequestIdMiddleware
from bevserve.preprocess import ModelInputs, build_inputs
from bevserve.schemas import (
    Box3D,
    CalibrationRequest,
    DetectOptions,
    DetectResponse,
    FieldError,
    ModelInfo,
    Readiness,
    Timing,
)
from bevserve.service import ExitFn, Service
from bevserve.settings import Settings

CALIBRATION_FIELD = "calibration"
IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png"}
# Parser caps above the legal counts so that extras are reported as 422, not as parser errors.
MAX_FILE_PARTS = 12
MAX_FIELD_PARTS = 4


def create_app(
    settings: Settings, factory: DetectorFactory, exit_fn: ExitFn | None = None
) -> FastAPI:
    service = Service(settings, factory, exit_fn) if exit_fn else Service(settings, factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        init = asyncio.create_task(service.initialize())
        try:
            yield
        finally:
            init.cancel()
            service.shutdown()

    app = FastAPI(title="BEVServe", version=__version__, lifespan=lifespan)
    app.state.service = service
    install_error_handlers(app)
    # add_middleware wraps from the inside out: RequestId runs first, then the body limit.
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(RequestIdMiddleware)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=Readiness)
    async def readyz() -> JSONResponse:
        state = service.readiness()
        headers = {"Retry-After": "1"} if state.status == "initializing" else None
        return JSONResponse(
            state.model_dump(), status_code=200 if state.status == "ready" else 503, headers=headers
        )

    @app.get("/v1/model", response_model=ModelInfo)
    async def model_info() -> ModelInfo:
        return _model_info(service)

    @app.post("/v1/detect", response_model=DetectResponse)
    async def detect(request: Request) -> JSONResponse:
        return await _detect(request, service)

    return app


# ------------------------------------------------------------------ helpers


def _model_info(service: Service) -> ModelInfo:
    info = service.detector.info if service.detector else None
    return ModelInfo(
        name=info.name if info else "unavailable",
        checkpoint_sha256=info.checkpoint_sha256 if info else None,
        config_sha256=info.config_sha256 if info else None,
        service_version=__version__,
        git_commit=service.settings.git_commit,
        class_names=list(info.class_names) if info else [],
        frame=OUTPUT_FRAME,
        camera_order=list(CAMERA_ORDER),
        source_size=(SRC_WIDTH, SRC_HEIGHT),
    )


def _model_version(info: ModelInfo) -> str:
    sha = info.checkpoint_sha256
    return f"{info.name}@{sha[:8] if sha else 'none'}"


def _require_ready(service: Service) -> None:
    state = service.readiness()
    if state.status == "ready":
        return
    headers = {"Retry-After": "1"} if state.status in ("initializing", "inference_stalled") else {}
    raise ApiError(503, "not_ready", "Service not ready", f"status={state.status}", headers=headers)


async def _read_form(request: Request, settings: Settings) -> FormData:
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        raise ApiError(
            415, "content_type", "Unsupported media type", "expected multipart/form-data"
        )
    try:
        return await request.form(
            max_files=MAX_FILE_PARTS,
            max_fields=MAX_FIELD_PARTS,
            max_part_size=settings.max_calibration_bytes,
        )
    except StarletteHTTPException as exc:
        if "maximum size" in str(exc.detail):
            raise ApiError(
                413,
                "calibration_too_large",
                "Calibration too large",
                f"calibration exceeds {settings.max_calibration_bytes} bytes",
            ) from exc
        raise ApiError(400, "multipart", "Malformed multipart body", str(exc.detail)) from exc


def _collect_parts(form: FormData, settings: Settings) -> tuple[dict[str, UploadFile], str]:
    images: dict[str, UploadFile] = {}
    calibration: str | None = None
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            if key not in CAMERA_ORDER:
                raise _field_error(422, "unexpected_part", key, f"unexpected file part {key!r}")
            if key in images:
                raise _field_error(422, "camera_set", key, f"duplicate part {key!r}")
            if value.content_type not in IMAGE_CONTENT_TYPES:
                raise _field_error(
                    415, "part_content_type", key, f"{key}: expected image/jpeg or image/png"
                )
            if value.size is not None and value.size > settings.max_image_bytes:
                raise _field_error(
                    413, "image_too_large", key, f"{key} exceeds {settings.max_image_bytes} bytes"
                )
            images[key] = value
        else:
            if key != CALIBRATION_FIELD:
                raise _field_error(422, "unexpected_part", key, f"unexpected field {key!r}")
            if calibration is not None:
                raise _field_error(422, "calibration_duplicate", key, "calibration given twice")
            calibration = value
    missing = [n for n in CAMERA_ORDER if n not in images]
    if missing:
        raise _field_error(422, "camera_set", ",".join(missing), f"missing image parts: {missing}")
    if calibration is None:
        raise _field_error(
            422, "calibration_missing", CALIBRATION_FIELD, "calibration field is required"
        )
    return images, calibration


def _parse_calibration(text: str) -> CalibrationRequest:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _field_error(
            400, "calibration_json", CALIBRATION_FIELD, f"invalid JSON: {exc.msg}"
        ) from exc
    try:
        return CalibrationRequest.model_validate(payload)
    except ValidationError as exc:
        errors = [
            FieldError(
                field=".".join([CALIBRATION_FIELD, *(str(p) for p in e["loc"])]),
                code=e["type"],
                message=e["msg"],
            )
            for e in exc.errors()
        ]
        raise ApiError(
            422, errors[0].code, "Invalid calibration", errors[0].message, errors
        ) from exc


async def _check_images(parts: dict[str, UploadFile]) -> dict[str, bytes]:
    """Read each part and run the header-only checks; no pixel data is decoded here."""
    raw: dict[str, bytes] = {}
    for name in CAMERA_ORDER:
        data = await parts[name].read()
        try:
            check_size(inspect_header(data))
        except ImageRejectedError as exc:
            status = 415 if exc.code == "image_format" else 422
            raise _field_error(status, exc.code, name, f"{name}: {exc.detail}") from exc
        raw[name] = data
    return raw


def _decode_and_build(
    raw: dict[str, bytes], calibration: CalibrationRequest
) -> tuple[ModelInputs, float, float]:
    t0 = time.perf_counter()
    images: dict[str, Image.Image] = {}
    for name, data in raw.items():
        try:
            images[name] = decode_rgb(data)
        except ImageRejectedError as exc:
            raise _field_error(422, exc.code, name, f"{name}: {exc.detail}") from exc
    t1 = time.perf_counter()
    inputs = build_inputs(images, calibration.to_rig())
    return inputs, (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000


def _boxes(
    detections: Sequence[Detection], options: DetectOptions, class_names: list[str]
) -> list[Box3D]:
    kept = sorted(
        (d for d in detections if d.score >= options.score_threshold),
        key=lambda d: d.score,
        reverse=True,
    )[: options.max_boxes]
    out = []
    for d in kept:
        if not 0 <= d.label_id < len(class_names):
            raise ApiError(500, "internal", "Internal error", "detector returned an unknown label")
        out.append(
            Box3D(
                label=class_names[d.label_id],
                label_id=d.label_id,
                score=d.score,
                center=d.center,
                size=d.size,
                yaw=d.yaw,
                velocity=d.velocity,
            )
        )
    return out


def _field_error(status: int, code: str, field: str, message: str) -> ApiError:
    title = {400: "Bad request", 413: "Payload too large", 415: "Unsupported media type"}.get(
        status, "Invalid request"
    )
    return ApiError(
        status, code, title, message, [FieldError(field=field, code=code, message=message)]
    )


async def _detect(request: Request, service: Service) -> JSONResponse:
    started = time.perf_counter()
    settings = service.settings
    _require_ready(service)
    form = await _read_form(request, settings)
    stages: dict[str, float] = {"parse": (time.perf_counter() - started) * 1000}
    try:
        t_validate = time.perf_counter()
        parts, calibration_text = _collect_parts(form, settings)
        calibration = _parse_calibration(calibration_text)
        raw = await _check_images(parts)
        stages["validate"] = (time.perf_counter() - t_validate) * 1000

        executor = service.executor
        if executor is None:
            raise ApiError(503, "not_ready", "Service not ready", "detector not initialized")
        try:
            admission = executor.try_admit()
        except QueueFullError as exc:
            raise ApiError(
                503, "queue_full", "Service busy", str(exc), headers={"Retry-After": "1"}
            ) from exc
        except FatalDetectorError as exc:
            raise ApiError(503, "not_ready", "Service not ready", "status=fatal") from exc

        with admission:
            inputs, decode_ms, preprocess_ms = await anyio.to_thread.run_sync(
                _decode_and_build, raw, calibration
            )
            t_inf = time.perf_counter()
            try:
                job_timing: dict[str, float] | None = {} if settings.server_timing else None
                detections = await executor.run(
                    admission, inputs, settings.deadline_s, timing=job_timing
                )
            except DeadlineExceededError as exc:
                raise ApiError(
                    504,
                    "deadline_exceeded",
                    "Inference deadline exceeded",
                    f"no result within {settings.deadline_s}s (job was {exc.stage})",
                ) from exc
            except FatalDetectorError as exc:
                raise ApiError(
                    500,
                    "inference_fatal",
                    "Inference failed",
                    "unrecoverable detector error; restarting",
                ) from exc
            except DetectorError as exc:
                raise ApiError(500, "inference_failed", "Inference failed", str(exc)) from exc
            inference_ms = (time.perf_counter() - t_inf) * 1000
    finally:
        await form.close()

    info = _model_info(service)
    body = DetectResponse(
        request_id=request_id_of(request) or "",
        model=info,
        frame=OUTPUT_FRAME,
        boxes=_boxes(detections, calibration.options, info.class_names),
        timing_ms=Timing(
            decode=round(decode_ms, 3),
            preprocess=round(preprocess_ms, 3),
            inference=round(inference_ms, 3),
            total=round((time.perf_counter() - started) * 1000, 3),
        ),
    )
    t_serialize = time.perf_counter()
    response = JSONResponse(body.model_dump(), headers={"X-Model-Version": _model_version(info)})
    if settings.server_timing:
        stages.update(decode=decode_ms, preprocess=preprocess_ms, **(job_timing or {}))
        stages["serialize"] = (time.perf_counter() - t_serialize) * 1000
        stages["total"] = (time.perf_counter() - started) * 1000
        response.headers["Server-Timing"] = server_timing_header(stages)
    return response


SERVER_TIMING_ORDER = (
    "parse",
    "validate",
    "decode",
    "preprocess",
    "queue",
    "gpu",
    "h2d",
    "forward",
    "serialize",
    "total",
)


def server_timing_header(stages: dict[str, float]) -> str:
    """RFC-style Server-Timing value, e.g. "decode;dur=15.20, gpu;dur=12.01" (milliseconds)."""
    return ", ".join(
        f"{name};dur={stages[name]:.3f}" for name in SERVER_TIMING_ORDER if name in stages
    )
