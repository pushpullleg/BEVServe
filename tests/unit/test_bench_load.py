"""Statistics of the M5 load client (pure functions; no server, no GPU)."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_load", ROOT / "tools/bench/load.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load = _load()


def test_nearest_rank_percentiles_and_intervals() -> None:
    values = [float(v) for v in range(1, 101)]  # 1..100
    s = load.stats(values, np.random.default_rng(0))
    assert (s["n"], s["p50"], s["p95"], s["p99"], s["max"]) == (100, 50.0, 95.0, 99.0, 100.0)
    for q in ("p50", "p95", "p99"):
        lo, hi = s[f"{q}_ci95"]
        assert lo <= s[q] <= hi


def test_empty_stats() -> None:
    assert load.stats([], np.random.default_rng(0)) == {"n": 0}


def test_parse_server_timing() -> None:
    assert load.parse_server_timing("decode;dur=1.5, gpu;dur=12.25") == {
        "decode": 1.5,
        "gpu": 12.25,
    }
    assert load.parse_server_timing(None) == {}


def _row(i: int, status: int, start: float, end: float, warmup: bool = False, **kw: object) -> dict:
    return {
        "i": i,
        "status": status,
        "t_start_s": start,
        "t_end_s": end,
        "warmup": warmup,
        "latency_ms": (end - start) * 1000,
        "server": {},
        **kw,
    }


def test_summary_excludes_warmup_and_counts_statuses() -> None:
    rows = [
        _row(0, 200, 0.0, 0.5, warmup=True, boxes=3),
        _row(1, 200, 1.0, 1.1, boxes=3, server={"gpu": 10.0}),
        _row(2, 503, 1.1, 1.12, type="/errors/queue_full", retry_after="1"),
        _row(3, 200, 1.2, 2.0, boxes=5, server={"gpu": 12.0}),
        _row(4, 0, 2.0, 2.5, error="ReadTimeout"),
    ]
    args = argparse.Namespace(label="t", url="u", concurrency=2, warmup=1, honor_retry_after=True)
    s = load.summarize(rows, args, {"kind": "synthetic"})
    assert s["requests_measured"] == 4 and s["success"] == 2
    assert s["statuses"] == {"200": 2, "503 /errors/queue_full": 1, "0 ReadTimeout": 1}
    assert s["retry_after_values"] == ["1"]
    assert abs(s["window_s"] - 1.5) < 1e-9 and abs(s["throughput_rps"] - 2 / 1.5) < 1e-9
    assert s["latency_ms"]["n"] == 2 and s["server_stages_ms"]["gpu"]["n"] == 2
    assert s["honor_retry_after"] is True
