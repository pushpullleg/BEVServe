"""TEST-ONLY server entrypoint that injects a fatal detector error, to verify supervised restart.

    BEVSERVE_FAULT_INJECTION_ACK=1 python -m bevserve.faultinject --fatal-on-call 2

Runs the normal server (same settings, same bind policy) with the real detector wrapped so that
its Nth detect() call raises FatalDetectorError. The service then follows the production fatal
path: 500 to that request, not-ready, process exit with the fatal exit code. Never used by the
`bevserve` command; refuses to start without BEVSERVE_FAULT_INJECTION_ACK=1.
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Sequence

import uvicorn

from bevserve.api import create_app
from bevserve.bind import BindNotAllowedError, check_bind_host
from bevserve.cli import configure_logging, make_factory
from bevserve.detector import Detection, Detector, DetectorInfo, FatalDetectorError
from bevserve.preprocess import ModelInputs
from bevserve.settings import Settings


class FatalOnCall:
    def __init__(self, inner: Detector, call: int) -> None:
        self._inner = inner
        self._call = call
        self._count = 0
        self._lock = threading.Lock()

    @property
    def info(self) -> DetectorInfo:
        return self._inner.info

    def detect(self, inputs: ModelInputs) -> Sequence[Detection]:
        with self._lock:
            self._count += 1
            count = self._count
        if count == self._call:
            raise FatalDetectorError(f"injected fatal error on call {count}")
        return self._inner.detect(inputs)


def main(argv: Sequence[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(prog="python -m bevserve.faultinject")
    parser.add_argument("--fatal-on-call", type=int, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("BEVSERVE_FAULT_INJECTION_ACK") != "1":
        print(
            "refusing: set BEVSERVE_FAULT_INJECTION_ACK=1 (test-only entrypoint)", file=sys.stderr
        )
        return 2
    configure_logging()
    settings = Settings()
    try:
        check_bind_host(settings.host, settings.allowed_bind_hosts)
    except BindNotAllowedError as exc:
        print(f"bevserve: {exc}", file=sys.stderr)
        return 2
    inner_factory = make_factory(settings)

    def factory() -> Detector:
        return FatalOnCall(inner_factory(), args.fatal_on_call)

    app = create_app(settings, factory)
    uvicorn.run(app, host=settings.host, port=settings.port, server_header=False, access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
