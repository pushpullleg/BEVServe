"""M7 diagnostic (stage rel-diag): serial (A) vs parallel (B) under the C = 32 no-retry burst.

    python tools/bench/diag.py --dir RUN > RUN/diag.json

Descriptive only (one container per arm, no replicates): no pass/fail and no causal claim.

Per arm and phase (C = 5 baseline, C = 32 burst): attempts, outcomes, duration, stop reason,
goodput; latency of successful (200) and rejected (503 queue_full) requests kept separate; and,
for successful requests only (the server sends Server-Timing on 200 responses only), the
distribution of every Server-Timing stage, each reported on its own (percentiles of different
stages are never subtracted from each other). Stages:
  parse, validate      before admission (multipart parsing; part, calibration and header checks)
  decode, preprocess   decode-and-build in a worker thread (in parallel mode, decode = until the
                       last camera is decoded, preprocess = the rest; compare their sum)
  queue                admitted job waiting for the single GPU slot (submit -> start)
  gpu                  wall time of the slot's detect call (host thread, GPU-synchronized)
  h2d                  input upload, synchronized
  forward              CUDA-event span around BEVDet forward_test on the GPU stream; it includes
                       stream idle gaps while the host is late to launch work, so it is an upper
                       bound on kernel time, not kernel time itself
  serialize, total     response building; handler wall time from its start
Per-request derived values (computed within each request, then summarized):
  slot_host = gpu - h2d - forward   host work in the slot outside the CUDA-event span
  outside   = client latency - total   client, transport and pre-handler time
Resources per phase window (0.5 s samples): server process CPU, main-thread (event loop) CPU,
busiest thread CPU, threads, pids, RSS, GPU utilization, server GPU memory; the load generator's
own CPU (single asyncio thread: ~100% = saturated); readiness probes in the window.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

STAGES = (
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


def dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)

    def q(x: float) -> float:
        return round(float(np.percentile(a, x, method="inverted_cdf")), 2)

    return {"n": len(a), "p50": q(50), "p95": q(95), "p99": q(99), "max": round(float(a.max()), 2)}


def _rows(d: Path, label: str) -> list[dict[str, Any]]:
    f = d / f"{label}.requests.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def phase(
    d: Path,
    label: str,
    window: list[float] | None,
    samples: list[dict[str, str]],
    probes: list[dict[str, str]],
) -> dict[str, Any]:
    rows = _rows(d, label)
    summary = (
        json.loads((d / f"{label}.summary.json").read_text())
        if (d / f"{label}.summary.json").exists()
        else {}
    )
    ok = [r for r in rows if r["status"] == 200 and not r.get("error")]
    rejected = [r for r in rows if r["status"] == 503 and r.get("type") == "/errors/queue_full"]
    stages = {s: dist([r["server"][s] for r in ok if s in r.get("server", {})]) for s in STAGES}
    stages["decode+preprocess"] = dist(
        [
            r["server"]["decode"] + r["server"]["preprocess"]
            for r in ok
            if {"decode", "preprocess"} <= set(r.get("server", {}))
        ]
    )
    derived = {
        "slot_host": dist(
            [
                r["server"]["gpu"] - r["server"]["h2d"] - r["server"]["forward"]
                for r in ok
                if {"gpu", "h2d", "forward"} <= set(r.get("server", {}))
            ]
        ),
        "outside": dist(
            [r["latency_ms"] - r["server"]["total"] for r in ok if "total" in r.get("server", {})]
        ),
    }
    in_window = [s for s in samples if window and window[0] <= float(s["t_unix"]) <= window[1]]

    def col(name: str) -> list[float]:
        return [float(s[name]) for s in in_window if s.get(name) not in (None, "")]

    def mean_max(name: str) -> dict[str, Any]:
        v = col(name)
        return (
            {"mean": round(sum(v) / len(v), 1), "max": round(max(v), 1)}
            if v
            else {"mean": None, "max": None}
        )

    probes_in = [p for p in probes if window and window[0] <= float(p["t_unix"]) <= window[1]]
    duration = (
        (max(r["t_end_s"] for r in rows) - min(r["t_start_s"] for r in rows)) if rows else None
    )
    return {
        "attempts": len(rows),
        "outcomes": summary.get("outcomes"),
        "stop_reason": summary.get("stop_reason"),
        "duration_s": round(duration, 3) if duration is not None else None,
        "goodput_rps": round(len(ok) / duration, 2) if duration else None,
        "latency_success_ms": dist([r["latency_ms"] for r in ok]),
        "latency_rejected_ms": dist([r["latency_ms"] for r in rejected]),
        "server_stages_ms": stages,
        "derived_per_request_ms": derived,
        "resources": {
            "samples": len(in_window),
            "server_cpu_pct": mean_max("cpu_pct"),
            "main_thread_cpu_pct": mean_max("main_thread_cpu_pct"),
            "max_thread_cpu_pct": mean_max("max_thread_cpu_pct"),
            "tree_threads": mean_max("tree_threads"),
            "pids_current": mean_max("pids_current"),
            "rss_mib": mean_max("rss_mib"),
            "gpu_util_pct": mean_max("gpu_util_pct"),
            "server_gpu_mem_mib": mean_max("proc_gpu_mem_mib"),
            "client_cpu_pct": (summary.get("client_process") or {}).get("cpu_pct"),
        },
        "client": summary.get("multiprocess"),
        "readiness": {
            "probes": len(probes_in),
            "not_ready": sum(
                1 for p in probes_in if p["http_status"] != "200" or p["status"] != "ready"
            ),
            "max_latency_ms": max((float(p["latency_ms"]) for p in probes_in), default=None),
        },
    }


def ratio(
    base: dict[str, Any], burst: dict[str, Any], key: str, stat: str, src: str = "server_stages_ms"
) -> float | None:
    """burst / baseline of one summary statistic (same statistic, same arm)."""
    b = (base.get(src, {}).get(key) if src else base.get(key)) or {}
    u = (burst.get(src, {}).get(key) if src else burst.get(key)) or {}
    return round(u[stat] / b[stat], 2) if b.get(stat) and u.get(stat) is not None else None


def analyze(d: Path, prefix: str = "rel-diag") -> dict[str, Any]:
    windows: dict[str, list[float]] = {}
    runs = d / "runs.jsonl"
    for line in runs.read_text().splitlines() if runs.exists() else []:
        r = json.loads(line)
        if r.get("kind") == "phase":
            windows[r["label"]] = r["window"]
    out: dict[str, Any] = {"note": "descriptive diagnostic: one container per arm, no replicates"}
    for arm, threads in (("a", 0), ("b", None)):
        label = f"{prefix}-{arm}"
        try:
            samples = list(csv.DictReader((d / f"{label}.samples.csv").open()))
        except OSError:
            samples = []
        try:
            probes = list(csv.DictReader((d / f"{label}.readiness.csv").open()))
        except OSError:
            probes = []
        base = phase(d, f"{label}-base-c5", windows.get(f"{label}-base-c5"), samples, probes)
        burst = phase(
            d, f"{label}-c32-noretry", windows.get(f"{label}-c32-noretry"), samples, probes
        )

        out[arm] = {
            "preprocess_threads": threads if threads is not None else "parallel",
            "baseline_c5": base,
            "burst_c32_noretry": burst,
            "burst_over_own_baseline": {
                "success_latency_p50": ratio(base, burst, "latency_success_ms", "p50", ""),
                "success_latency_p99": ratio(base, burst, "latency_success_ms", "p99", ""),
                **{
                    f"{s}_p50": ratio(base, burst, s, "p50")
                    for s in ("queue", "gpu", "forward", "decode+preprocess", "parse")
                },
            },
            "criterion_reference": {  # the frozen C1b rule, evaluated per arm for context only
                "rule": "burst success p99 <= 2 x own baseline p99",
                "baseline_p99_ms": base["latency_success_ms"].get("p99"),
                "burst_p99_ms": burst["latency_success_ms"].get("p99"),
            },
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--prefix", default="rel-diag")
    a = p.parse_args()
    print(json.dumps(analyze(Path(a.dir), a.prefix), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
