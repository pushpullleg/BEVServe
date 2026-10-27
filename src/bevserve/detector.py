"""Detector protocol and the fake implementation used by M1 and the public tests.

The real BEVDet adapter (M2) implements the same protocol; nothing else in the service
imports mmdet3d.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from bevserve.constants import CLASS_NAMES
from bevserve.preprocess import ModelInputs


@dataclass(frozen=True)
class Detection:
    """One box in the output frame (ego@CAM_FRONT_LEFT), 9 values as BEVDet emits them."""

    label_id: int
    score: float
    center: tuple[float, float, float]
    size: tuple[float, float, float]  # x_size, y_size, z_size
    yaw: float
    velocity: tuple[float, float]


@dataclass(frozen=True)
class DetectorInfo:
    name: str
    checkpoint_sha256: str | None
    config_sha256: str | None
    class_names: tuple[str, ...] = CLASS_NAMES


class DetectorInitError(Exception):
    """Initialization failed. The service stays up and reports not-ready with `reason`."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class DetectorError(Exception):
    """A per-request failure; the detector remains usable."""


class FatalDetectorError(Exception):
    """The detector is unusable (e.g. a CUDA context error). The process must exit."""


class Detector(Protocol):
    @property
    def info(self) -> DetectorInfo: ...

    def detect(self, inputs: ModelInputs) -> Sequence[Detection]: ...


DetectorFactory = Callable[[], Detector]


DEFAULT_FAKE_BOXES: tuple[Detection, ...] = (
    Detection(0, 0.91, (12.0, 3.5, 0.8), (4.6, 1.9, 1.7), 0.05, (2.0, 0.1)),
    Detection(8, 0.42, (8.2, -4.0, 0.9), (0.7, 0.7, 1.8), 1.40, (0.0, 0.5)),
    Detection(9, 0.15, (6.0, 2.0, 0.3), (0.4, 0.4, 0.7), 0.00, (0.0, 0.0)),
)


@dataclass
class FakeDetector:
    """Deterministic stand-in for the model. Thread-safe counters let tests check concurrency.

    gate: if set, detect() blocks until the event is set (simulates a long GPU call).
    fail: "error" raises DetectorError, "fatal" raises FatalDetectorError.
    """

    boxes: Sequence[Detection] = DEFAULT_FAKE_BOXES
    delay_s: float = 0.0
    gate: threading.Event | None = None
    fail: Literal["error", "fatal"] | None = None
    calls: int = 0
    active: int = 0
    max_active: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(name="fake-detector", checkpoint_sha256=None, config_sha256=None)

    def detect(self, inputs: ModelInputs) -> Sequence[Detection]:
        inputs.check()
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.gate is not None and not self.gate.wait(timeout=30):
                raise DetectorError("fake gate never opened")
            if self.delay_s:
                time.sleep(self.delay_s)
            if self.fail == "error":
                raise DetectorError("fake per-request failure")
            if self.fail == "fatal":
                raise FatalDetectorError("fake unrecoverable failure")
            return list(self.boxes)
        finally:
            with self._lock:
                self.active -= 1
