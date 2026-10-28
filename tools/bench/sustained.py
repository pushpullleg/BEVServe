"""M7 sustained-run analysis: memory trend, GPU-memory plateau, threads, latency drift.

    python tools/bench/sustained.py --samples S.csv --requests a.jsonl [b.jsonl ...] \\
        --start T0 --warmup-s 300 [--bin-s 60] > sustained.json

T0 is the unix time the sustained load started. The first --warmup-s seconds are warm-up
(allocator and cache growth, lazily created threads) and are excluded from every trend and range
below; they are reported separately.

- RSS trend: the 1 Hz / 0.5 s samples are autocorrelated, so they are first averaged into
  --bin-s bins; the slope is a least-squares fit over the post-warm-up bins (MiB per hour) with a
  95% t interval. The interval is widened for residual lag-1 autocorrelation r1 > 0 by the usual
  effective-sample-size factor sqrt((1 + r1) / (1 - r1)) and uses df = n_eff - 2. Fewer than 6
  bins, or n_eff < 4, gives no interval (insufficient data).
- RSS peaks vs growth: raw maximum and 99th percentile are reported next to the bin means, so a
  temporary peak is not mistaken for a trend (and vice versa).
- GPU memory (process, nvidia-smi, includes the CUDA caching allocator): warm-up peak, maximum
  after warm-up, and their difference (allocator steps after warm-up).
- Threads (target process tree): minimum, maximum and range after warm-up.
- Latency drift: median latency of successful requests in the last window over the first window
  after warm-up; window = min(300 s, a quarter of the post-warm-up time).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overhead import T975


def t975(df: float) -> float:
    """Two-sided 95% Student t quantile; table for df <= 20, 1.96 + 2.5 / df beyond (< 0.2% off)."""
    k = max(1, math.floor(df))
    return T975[k - 1] if k <= len(T975) else 1.96 + 2.5 / k


def bins(
    points: list[tuple[float, float]], start: float, width: float
) -> list[tuple[float, float]]:
    """[(bin centre in hours from start, mean value)] for bins that contain samples."""
    groups: dict[int, list[float]] = {}
    for t, v in points:
        groups.setdefault(int((t - start) // width), []).append(v)
    return [((k + 0.5) * width / 3600, sum(v) / len(v)) for k, v in sorted(groups.items())]


def trend(series: list[tuple[float, float]]) -> dict[str, Any]:
    n = len(series)
    out: dict[str, Any] = {"bins": n, "slope_per_h": None, "ci95": None}
    if n < 3:
        return out
    xs, ys = [x for x, _ in series], [y for _, y in series]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return out
    slope = sum((x - mx) * (y - my) for x, y in series) / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in series]
    sse = sum(r * r for r in resid)
    denom = sse or 1e-12
    r1 = sum(a * b for a, b in itertools.pairwise(resid)) / denom
    out.update(slope_per_h=slope, lag1_autocorrelation=r1)
    r1 = max(0.0, min(r1, 0.95))
    n_eff = n * (1 - r1) / (1 + r1)
    out["n_effective"] = n_eff
    if n < 6 or n_eff < 4:
        return out
    se = math.sqrt(sse / (n - 2) / sxx) * math.sqrt((1 + r1) / (1 - r1))
    half = t975(n_eff - 2) * se
    out["ci95"] = [slope - half, slope + half]
    return out


def analyze(
    samples: Path, requests: list[Path], start: float, warmup_s: float, bin_s: float
) -> dict[str, Any]:
    rows = list(csv.DictReader(samples.open()))
    cut = start + warmup_s

    def col(name: str, after: bool) -> list[tuple[float, float]]:
        return [
            (float(r["t_unix"]), float(r[name]))
            for r in rows
            if r.get(name) not in (None, "") and (float(r["t_unix"]) >= cut) == after
        ]

    rss_post, rss_warm = col("rss_mib", True), col("rss_mib", False)
    gpu_post, gpu_warm = col("proc_gpu_mem_mib", True), col("proc_gpu_mem_mib", False)
    thr_post = [v for _, v in col("tree_threads", True)] or [v for _, v in col("threads", True)]
    rss_values = sorted(v for _, v in rss_post)
    out: dict[str, Any] = {
        "warmup_s": warmup_s,
        "bin_s": bin_s,
        "samples_post_warmup": len(rss_post),
        "rss": {
            "warmup_max_mib": max((v for _, v in rss_warm), default=None),
            "post_max_mib": rss_values[-1] if rss_values else None,
            "post_p99_mib": rss_values[int(0.99 * (len(rss_values) - 1))] if rss_values else None,
            "trend_mib": trend(bins(rss_post, cut, bin_s)),
            "bin_means_mib": [round(v, 1) for _, v in bins(rss_post, cut, bin_s)],
        },
        "gpu": {
            "warmup_peak_mib": max((v for _, v in gpu_warm), default=None),
            "post_max_mib": max((v for _, v in gpu_post), default=None),
        },
        "threads": {
            "post_min": min(thr_post) if thr_post else None,
            "post_max": max(thr_post) if thr_post else None,
        },
    }
    g = out["gpu"]
    g["growth_after_warmup_mib"] = (
        g["post_max_mib"] - g["warmup_peak_mib"]
        if g["post_max_mib"] is not None and g["warmup_peak_mib"] is not None
        else None
    )
    t = out["threads"]
    t["post_range"] = t["post_max"] - t["post_min"] if thr_post else None
    lat: list[tuple[float, float]] = []
    for f in requests:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == 200 and not r.get("error") and r["t_unix"] >= cut:
                lat.append((r["t_unix"], r["latency_ms"]))
    if lat:
        end = max(t for t, _ in lat)
        window = min(300.0, (end - cut) / 4)
        first = [v for t, v in lat if t < cut + window]
        last = [v for t, v in lat if t >= end - window]
        out["latency"] = {
            "window_s": window,
            "first_median_ms": statistics.median(first) if first else None,
            "last_median_ms": statistics.median(last) if last else None,
        }
        lt = out["latency"]
        lt["drift_ratio"] = (
            lt["last_median_ms"] / lt["first_median_ms"]
            if lt["first_median_ms"] and lt["last_median_ms"]
            else None
        )
    else:
        out["latency"] = {"drift_ratio": None}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--requests", nargs="+", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--warmup-s", type=float, required=True)
    p.add_argument("--bin-s", type=float, default=60.0)
    a = p.parse_args()
    result = analyze(Path(a.samples), [Path(f) for f in a.requests], a.start, a.warmup_s, a.bin_s)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
