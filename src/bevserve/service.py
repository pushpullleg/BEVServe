"""Service lifecycle (review correction 2).

- Initialization failure: stay up, report init_failed with a reason; never exit (no crash loop).
- Fatal runtime failure: become not-ready immediately, then exit with a non-zero code after a
  short grace period, so that a process supervisor (restart policy) restarts the service.
- Ordinary per-request failures do not change readiness.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable
from typing import Literal

import anyio

from bevserve.detector import Detector, DetectorFactory, DetectorInitError, FatalDetectorError
from bevserve.executor import InferenceExecutor
from bevserve.schemas import Readiness
from bevserve.settings import Settings

log = logging.getLogger("bevserve")
Status = Literal["initializing", "ready", "init_failed", "fatal"]
ExitFn = Callable[[int], None]


class Service:
    def __init__(
        self, settings: Settings, factory: DetectorFactory, exit_fn: ExitFn = os._exit
    ) -> None:
        self.settings = settings
        self._factory = factory
        self._exit_fn = exit_fn
        self._lock = threading.Lock()
        self._status: Status = "initializing"
        self.reason: str | None = None
        self.detail: str | None = None
        self.detector: Detector | None = None
        self.executor: InferenceExecutor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._exit_scheduled = False

    @property
    def status(self) -> Status:
        with self._lock:
            return self._status

    async def initialize(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            detector = await anyio.to_thread.run_sync(self._factory)
        except DetectorInitError as exc:
            self._set_failed(exc.reason, exc.detail)
            return
        except Exception as exc:
            log.exception("detector initialization raised")
            self._set_failed("unexpected", type(exc).__name__)
            return
        self.detector = detector
        self.executor = InferenceExecutor(detector, self.settings.max_in_flight, self._on_fatal)
        with self._lock:
            self._status = "ready"
        log.info("detector ready: %s", detector.info.name)

    def _set_failed(self, reason: str, detail: str) -> None:
        with self._lock:
            self._status = "init_failed"
            self.reason, self.detail = reason, detail
        log.error("detector initialization failed: reason=%s detail=%s", reason, detail)

    def _on_fatal(self, exc: FatalDetectorError) -> None:
        """Called from the slot thread. Not-ready now; exit after the grace period."""
        with self._lock:
            self._status = "fatal"
            self.reason, self.detail = "fatal_runtime_error", str(exc)
            if self._exit_scheduled or self._loop is None:
                return
            self._exit_scheduled = True
        code, grace = self.settings.fatal_exit_code, self.settings.fatal_exit_grace_s
        log.critical("fatal detector error; exiting with code %d in %.1fs: %s", code, grace, exc)
        self._loop.call_soon_threadsafe(self._loop.call_later, grace, self._exit_fn, code)

    def readiness(self) -> Readiness:
        status = self.status
        if status == "ready" and self.executor is not None:
            running = self.executor.running_for()
            if running is not None and running > self.settings.stall_after_s:
                return Readiness(
                    status="inference_stalled",
                    reason="inference_stalled",
                    detail=f"current job running for {running:.1f}s",
                )
        return Readiness(status=status, reason=self.reason, detail=self.detail)

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown()
