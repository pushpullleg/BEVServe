"""Fake BEVServe process for orchestration tests: the real app with a FakeDetector, plus knobs.

Started by the fake `docker compose up` with the run's BEVSERVE_* environment. Without knobs it is
`bevserve.cli` with the fake detector. Knobs (environment) inject the failures the M7 harness must
catch:
  FAKE_FAIL_EVERY=N          every Nth detect raises DetectorError (HTTP 500 inference_failed)
  FAKE_WRONG_OUTPUT_AFTER=N  after N detects, boxes change (wrong model output)
  FAKE_LEAK_THREADS=N        each detect starts N threads that never exit (thread leak)
  FAKE_RSS_LEAK_MIB=N        each detect keeps N MiB allocated (memory growth)
  FAKE_NOT_READY_AFTER=N     after N detects, readiness reports "initializing"
"""

from __future__ import annotations

import dataclasses
import os
import sys
import threading
from collections.abc import Sequence

import uvicorn

from bevserve.api import create_app
from bevserve.detector import Detection, DetectorError, FakeDetector
from bevserve.preprocess import ModelInputs
from bevserve.schemas import Readiness
from bevserve.settings import Settings


def knob(name: str) -> int:
    return int(os.environ.get(name, "0") or 0)


class KnobDetector(FakeDetector):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.lock = threading.Lock()
        self.kept: list[bytearray] = []

    def detect(self, inputs: ModelInputs) -> Sequence[Detection]:
        with self.lock:
            self.n += 1
            n = self.n
        if knob("FAKE_FAIL_EVERY") and n % knob("FAKE_FAIL_EVERY") == 0:
            raise DetectorError("injected failure")
        for _ in range(knob("FAKE_LEAK_THREADS")):
            threading.Thread(target=threading.Event().wait, daemon=True).start()
        if knob("FAKE_RSS_LEAK_MIB"):
            self.kept.append(bytearray(os.urandom(knob("FAKE_RSS_LEAK_MIB") * 2**20)))
        boxes = super().detect(inputs)
        if knob("FAKE_WRONG_OUTPUT_AFTER") and n > knob("FAKE_WRONG_OUTPUT_AFTER"):
            boxes = [dataclasses.replace(b, score=b.score * 0.5) for b in boxes]
        return boxes


def main() -> int:
    settings = Settings()
    detector = KnobDetector()
    app = create_app(settings, lambda: detector)
    service = app.state.service
    if knob("FAKE_NOT_READY_AFTER"):
        original = service.readiness

        def readiness() -> Readiness:
            if detector.n > knob("FAKE_NOT_READY_AFTER"):
                return Readiness(status="initializing", reason="injected", detail="fake")
            return original()

        service.readiness = readiness
    uvicorn.run(app, host=settings.host, port=settings.port, server_header=False, access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
