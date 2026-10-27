"""`bevserve` command: serve the API on an authorized interface."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

import uvicorn

from bevserve.api import create_app
from bevserve.bind import BindNotAllowedError, check_bind_host
from bevserve.detector import Detector, DetectorFactory, FakeDetector
from bevserve.settings import Settings


def fake_factory() -> Detector:
    return FakeDetector()


def make_factory(settings: Settings) -> DetectorFactory:
    if settings.detector == "fake":
        return fake_factory

    def bevdet_factory() -> Detector:
        # Imported here so that the fake backend and public CI never need torch or mmdet3d.
        from bevserve.model.bevdet_detector import load_bevdet

        return load_bevdet(settings)

    return bevdet_factory


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(stream: TextIO | None = None) -> None:
    """Give the bevserve loggers their own handler.

    mmcv's get_logger (called while BEVDet loads) raises every root StreamHandler to ERROR, which
    would silence request logs that rely on the root handler. A handler on the "bevserve" logger
    with propagate=False is not affected.
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger = logging.getLogger("bevserve")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def main() -> int:
    configure_logging()
    settings = Settings()
    try:
        check_bind_host(settings.host, settings.allowed_bind_hosts)
    except BindNotAllowedError as exc:
        print(f"bevserve: {exc}", file=sys.stderr)
        return 2
    if settings.detector == "fake":
        logging.getLogger("bevserve").warning(
            "detector=fake: responses contain fixed synthetic boxes, not model predictions"
        )
    app = create_app(settings, make_factory(settings))
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        server_header=False,
        access_log=False,  # bevserve logs one line per request itself
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
