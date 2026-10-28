"""Optional benchmark instrumentation (M5): Server-Timing header, off by default, body unchanged."""

from __future__ import annotations

import asyncio
import json

import pytest

from bevserve.api import server_timing_header
from bevserve.detector import FakeDetector
from tests.api.helpers import build_app, client_for
from tests.synthetic import multipart_files

pytestmark = pytest.mark.anyio
REQUIRED = {"parse", "validate", "decode", "preprocess", "queue", "gpu", "serialize", "total"}


def parse(header: str) -> dict[str, float]:
    out = {}
    for item in header.split(","):
        name, dur = item.strip().split(";dur=")
        out[name] = float(dur)
    return out


async def test_off_by_default() -> None:
    app, _ = build_app()
    files, data = multipart_files()
    async with client_for(app) as client:
        resp = await client.post("/v1/detect", files=files, data=data)
    assert resp.status_code == 200 and "server-timing" not in resp.headers


async def test_stages_reported_and_body_unchanged() -> None:
    files, data = multipart_files()
    app_off, _ = build_app()
    app_on, _ = build_app(server_timing=True)
    async with client_for(app_off) as client:
        off = await client.post("/v1/detect", files=files, data=data)
    async with client_for(app_on) as client:
        on = await client.post("/v1/detect", files=files, data=data)
    stages = parse(on.headers["server-timing"])
    assert set(stages) >= REQUIRED, stages
    assert all(v >= 0 for v in stages.values())
    assert stages["total"] >= stages["gpu"] and stages["total"] >= stages["decode"]
    assert json.dumps(on.json()["boxes"]) == json.dumps(off.json()["boxes"])
    assert set(on.json()["timing_ms"]) == set(off.json()["timing_ms"])  # body schema unchanged


async def test_queue_time_reflects_waiting_behind_a_running_job() -> None:
    fake = FakeDetector(delay_s=0.3)
    app, _ = build_app(fake, server_timing=True, max_in_flight=4)
    files, data = multipart_files()
    async with client_for(app) as client:
        first, second = await asyncio.gather(
            client.post("/v1/detect", files=files, data=data),
            client.post("/v1/detect", files=files, data=data),
        )
    queues = sorted(parse(r.headers["server-timing"])["queue"] for r in (first, second))
    gpus = [parse(r.headers["server-timing"])["gpu"] for r in (first, second)]
    assert all(g >= 290 for g in gpus)  # fake "GPU" work is 300 ms
    assert queues[1] >= 150  # one request waited for the other's slot


def test_header_format_and_order() -> None:
    header = server_timing_header({"total": 9.0, "decode": 1.23456, "unknown": 5.0})
    assert header == "decode;dur=1.235, total;dur=9.000"
