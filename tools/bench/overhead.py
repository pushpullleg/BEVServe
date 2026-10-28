"""Server-Timing instrumentation overhead from repeated, balanced warm trials (M5).

    python tools/bench/overhead.py order 8          # trial arms, one per line (ABBA blocks)
    python tools/bench/overhead.py analyze --dir RUN_DIR --trials 8 > RUN_DIR/overhead.json

Each trial is a fresh service with timing off (A) or on (B), warm-up excluded, at concurrency 1.
Trials run in ABBA blocks, so a linear drift over a block affects both arms equally. Within block
b the pairs are (trial 4b+1, 4b+2) and (4b+4, 4b+3), i.e. each B trial paired with its adjacent A
trial. The estimate is the mean paired difference of per-trial client p50 latency (on - off), with
a two-sided 95% Student t interval over the pairs. Per-trial values are reported in full.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ARMS = ("false", "true")  # server timing off, on
# Two-sided 95% Student t quantiles, df = 1..20.
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228)  # fmt: skip
T975 += (2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086)  # fmt: skip


def order(trials: int) -> list[str]:
    if trials < 4 or trials % 4:
        raise ValueError("trials must be a positive multiple of 4 (whole ABBA blocks)")
    return [ARMS[i] for _ in range(trials // 4) for i in (0, 1, 1, 0)]


def pairs(trials: int) -> list[tuple[int, int]]:
    """(off trial, on trial), 1-based."""
    return [p for b in range(trials // 4) for p in ((4 * b + 1, 4 * b + 2), (4 * b + 4, 4 * b + 3))]


def t_interval(diffs: list[float]) -> tuple[float, float, float] | None:
    n = len(diffs)
    if n < 2 or n - 1 > len(T975):
        return None
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    half = T975[n - 2] * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def analyze(d: Path, trials: int) -> dict[str, Any]:
    arms = order(trials)
    per_trial: list[dict[str, Any]] = []
    for k, arm in enumerate(arms, start=1):
        f = d / f"overhead-t{k}-timing-{arm}.summary.json"
        s = json.loads(f.read_text()) if f.exists() else None
        lat = (s or {}).get("latency_ms", {})
        per_trial.append(
            {
                "trial": k,
                "server_timing": arm == "true",
                "measured": (s or {}).get("requests_measured"),
                "ok": (s or {}).get("success"),
                "p50_ms": lat.get("p50"),
                "mean_ms": lat.get("mean"),
                "p95_ms": lat.get("p95"),
            }
        )
    diffs, complete = [], True
    for off, on in pairs(trials):
        a, b = per_trial[off - 1]["p50_ms"], per_trial[on - 1]["p50_ms"]
        if a is None or b is None:
            complete = False
            continue
        diffs.append(b - a)
    offs = [t["p50_ms"] for t in per_trial if not t["server_timing"] and t["p50_ms"] is not None]
    ci = t_interval(diffs) if complete else None
    return {
        "trials": per_trial,
        "pairs": pairs(trials),
        "paired_p50_diff_ms": diffs,
        "complete": complete,
        "mean_diff_ms": ci[0] if ci else None,
        "ci95_ms": [ci[1], ci[2]] if ci else None,
        "off_p50_mean_ms": sum(offs) / len(offs) if offs else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("order")
    o.add_argument("trials", type=int)
    a = sub.add_parser("analyze")
    a.add_argument("--dir", required=True)
    a.add_argument("--trials", type=int, required=True)
    args = p.parse_args()
    if args.cmd == "order":
        print("\n".join(order(args.trials)))
    else:
        print(json.dumps(analyze(Path(args.dir), args.trials), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
