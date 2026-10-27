from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI

from bevserve.api import create_app
from bevserve.detector import Detector, DetectorFactory, FakeDetector
from bevserve.settings import Settings


class ExitRecorder:
    def __init__(self) -> None:
        self.codes: list[int] = []

    def __call__(self, code: int) -> None:
        self.codes.append(code)


def build_app(
    fake: FakeDetector | None = None,
    factory: DetectorFactory | None = None,
    **overrides: Any,
) -> tuple[FastAPI, ExitRecorder]:
    detector: Detector = fake or FakeDetector()
    exits = ExitRecorder()
    settings = Settings(**overrides)
    return create_app(settings, factory or (lambda: detector), exit_fn=exits), exits


@asynccontextmanager
async def client_for(app: FastAPI, wait_ready: bool = True) -> AsyncIterator[httpx.AsyncClient]:
    """ASGI client; runs service initialization like the lifespan does."""
    service = app.state.service
    await service.initialize()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        if wait_ready:
            assert service.status == "ready", service.readiness()
        try:
            yield client
        finally:
            service.shutdown()


async def wait_until(predicate: Any, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        if loop.time() > end:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)
