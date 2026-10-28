"""Single GPU execution slot with bounded admission (review correction 1).

Invariants:
- One worker thread owns the only execution slot, so at most one detect() runs at a time.
- The slot is released when detect() returns or raises, never when a request gives up.
- Admission bounds jobs in flight (queued + running). A token is taken without blocking and is
  released when its job is done, cancelled, or was never submitted.
- A request deadline covers queue wait and run time. On expiry, a job that has not started is
  cancelled and never runs; a running job finishes, keeps its slot and token until then, and its
  result is discarded.
- After a FatalDetectorError the executor refuses all further jobs without calling the detector.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Literal

from bevserve.detector import Detection, Detector, FatalDetectorError
from bevserve.preprocess import ModelInputs

JobFuture = concurrent.futures.Future[Sequence[Detection]]


class QueueFullError(Exception):
    pass


class DeadlineExceededError(Exception):
    def __init__(self, stage: Literal["queued", "running"]) -> None:
        super().__init__(f"deadline exceeded while {stage}")
        self.stage = stage


class Admission:
    """One in-flight token. Use as a context manager; submit() hands it to the job."""

    def __init__(self, executor: InferenceExecutor) -> None:
        self._executor = executor
        self._released = False
        self._owned_by_job = False

    def submit(self, inputs: ModelInputs, timing: dict[str, float] | None = None) -> JobFuture:
        if self._owned_by_job or self._released:
            raise RuntimeError("admission already used")
        future = self._executor._submit(inputs, timing)
        self._owned_by_job = True
        future.add_done_callback(lambda _: self.release())
        return future

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._executor._release()

    def __enter__(self) -> Admission:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._owned_by_job:
            self.release()


class InferenceExecutor:
    def __init__(
        self,
        detector: Detector,
        max_in_flight: int,
        on_fatal: Callable[[FatalDetectorError], None],
    ) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        self._detector = detector
        self._max_in_flight = max_in_flight
        self._on_fatal = on_fatal
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bevserve-slot"
        )
        self._lock = threading.Lock()
        self._in_flight = 0
        self._running_since: float | None = None
        self._fatal = False

    # ------------------------------------------------------------ admission

    def try_admit(self) -> Admission:
        with self._lock:
            if self._fatal:
                raise FatalDetectorError("detector is in a fatal state")
            if self._in_flight >= self._max_in_flight:
                raise QueueFullError(
                    f"{self._in_flight} jobs in flight (limit {self._max_in_flight})"
                )
            self._in_flight += 1
        return Admission(self)

    def _release(self) -> None:
        with self._lock:
            self._in_flight -= 1

    # ------------------------------------------------------------ execution

    def _submit(self, inputs: ModelInputs, timing: dict[str, float] | None) -> JobFuture:
        return self._pool.submit(self._job, inputs, timing, time.perf_counter())

    def _job(
        self, inputs: ModelInputs, timing: dict[str, float] | None, submitted: float
    ) -> Sequence[Detection]:
        with self._lock:
            if self._fatal:
                raise FatalDetectorError("detector is in a fatal state")
            self._running_since = time.monotonic()
        started = time.perf_counter()
        try:
            detections = self._detector.detect(inputs)
            if timing is not None:  # optional instrumentation; read on the slot thread
                timing["queue"] = (started - submitted) * 1000
                timing["gpu"] = (time.perf_counter() - started) * 1000
                extra = getattr(self._detector, "last_timing", None)
                if isinstance(extra, dict):
                    timing.update(extra)
            return detections
        except FatalDetectorError as exc:
            with self._lock:
                self._fatal = True
            self._on_fatal(exc)
            raise
        finally:
            with self._lock:
                self._running_since = None

    async def run(
        self,
        admission: Admission,
        inputs: ModelInputs,
        deadline_s: float,
        timing: dict[str, float] | None = None,
    ) -> Sequence[Detection]:
        """Submit and wait up to deadline_s. Raises DeadlineExceededError, or the job's error.

        timing (optional): filled with "queue" (submit to start) and "gpu" (detect wall time; the
        detector synchronizes the GPU before returning) in ms, plus any detector-specific stages.
        """
        future = admission.submit(inputs, timing)
        waiter = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), timeout=deadline_s)
        except asyncio.TimeoutError:
            if future.cancel():
                raise DeadlineExceededError("queued") from None
            waiter.add_done_callback(_consume_result)
            raise DeadlineExceededError("running") from None
        except asyncio.CancelledError:
            if not future.cancel():
                waiter.add_done_callback(_consume_result)
            raise

    # ------------------------------------------------------------ status

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def fatal(self) -> bool:
        with self._lock:
            return self._fatal

    def running_for(self) -> float | None:
        """Seconds the current job has been running, or None if the slot is idle."""
        with self._lock:
            started = self._running_since
        return None if started is None else time.monotonic() - started

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _consume_result(fut: asyncio.Future[Sequence[Detection]]) -> None:
    """Retrieve an abandoned job's outcome so asyncio does not log it as unhandled."""
    if not fut.cancelled():
        fut.exception()
