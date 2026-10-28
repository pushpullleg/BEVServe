"""Stop conditions between phases of an M7 run (exit 1 = stop the run, with the reasons as JSON).

    python tools/bench/budget.py --dir RUN --sampler LABEL --readiness LABEL --since T0 \\
        --rss-max-mib 8192 --gpu-max-mib 4096 --threads-max 400 [--summaries L1 L2 ...]

Checks the samples and readiness probes recorded since T0 (the stage start): server RSS, server
GPU memory and server process-tree threads against their stop budgets; the longest run of
not-ready probes (> 5 s is a stop); and, for each listed client summary, any transport error,
non-admission 5xx or malformed response. Missing files are a stop (fail closed).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

MAX_NOT_READY_S = 5.0


def check(a: argparse.Namespace) -> list[str]:
    d = Path(a.dir)
    reasons: list[str] = []
    try:
        samples = [
            r
            for r in csv.DictReader((d / f"{a.sampler}.samples.csv").open())
            if float(r["t_unix"]) >= a.since
        ]
        probes = [
            r
            for r in csv.DictReader((d / f"{a.readiness}.readiness.csv").open())
            if float(r["t_unix"]) >= a.since
        ]
    except OSError as exc:
        return [f"monitoring record missing: {exc}"]

    def peak(col: str) -> float | None:
        vals = [float(r[col]) for r in samples if r.get(col) not in (None, "")]
        return max(vals) if vals else None

    for col, limit, name in (
        ("rss_mib", a.rss_max_mib, "server RSS MiB"),
        ("proc_gpu_mem_mib", a.gpu_max_mib, "server GPU memory MiB"),
        ("tree_threads", a.threads_max, "server threads"),
    ):
        value = peak(col)
        if value is not None and value > limit:
            reasons.append(f"{name} {value:.0f} > stop budget {limit}")
    if not samples:
        reasons.append("no resource samples since the stage started")
    streak, start = 0.0, None
    for r in probes:
        bad = r["http_status"] != "200" or r["status"] != "ready"
        t = float(r["t_unix"])
        if bad:
            start = t if start is None else start
            streak = max(streak, t - start)
        else:
            start = None
    if streak > MAX_NOT_READY_S:
        reasons.append(f"not ready for {streak:.1f} s (> {MAX_NOT_READY_S} s)")
    for label in a.summaries:
        try:
            s = json.loads((d / f"{label}.summary.json").read_text())
        except (OSError, ValueError):
            reasons.append(f"{label}: summary missing or malformed")
            continue
        bad = {k: v for k, v in (s.get("outcomes") or {}).items() if k not in ("ok", "rejected")}
        if bad:
            reasons.append(f"{label}: unexpected outcomes {bad}")
    return reasons


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--sampler", required=True)
    p.add_argument("--readiness", required=True)
    p.add_argument("--since", type=float, required=True)
    p.add_argument("--rss-max-mib", type=float, required=True)
    p.add_argument("--gpu-max-mib", type=float, required=True)
    p.add_argument("--threads-max", type=float, required=True)
    p.add_argument("--summaries", nargs="*", default=[])
    a = p.parse_args()
    reasons = check(a)
    print(json.dumps({"stop": bool(reasons), "reasons": reasons}))
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
