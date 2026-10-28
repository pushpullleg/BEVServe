"""M6 A/B analysis: serial (A, PREPROCESS_THREADS=0) vs parallel (B) preprocessing.

    python tools/bench/ab.py analyze --dir RUN_DIR --trials 12 > RUN_DIR/ab.json
    python tools/bench/ab.py pilot-gate --pilot PILOT_DIR --image-id ID --threads 12

Design (tools/m5_bench.sh, stage `ab`): one verified image; each trial is a fresh service with
arm A or B in ABBA blocks (tools/bench/overhead.py order: "false" = A, "true" = B); per trial,
concurrency 1 then concurrency 8 (warm-up excluded), then the 10-frame golden comparison; one
resource sampler per trial, split into phases by the recorded phase times.

Replicates are trial pairs, not requests: within ABBA block b the pairs are (4b+1, 4b+2) and
(4b+4, 4b+3). For each metric the estimate is the geometric mean of the paired ratios B/A with a
two-sided 95% Student t interval on the log ratios (df = pairs - 1).

Per-trial metrics: client latency p50/p95/p99 and successful throughput at C = 1 and C = 8;
decode + preprocessing time from the response body (`timing_ms.decode + timing_ms.preprocess`,
the wall time of the whole decode-and-build step in both modes); Server-Timing gpu and queue
medians; serving-process peak RSS and peak GPU memory (max of the samples over the whole trial,
sampled every 0.5 s; GPU memory is the process's nvidia-smi figure, which includes the CUDA
caching allocator); mean process CPU % and mean GPU utilization per phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overhead import T975, order, pairs

ARM = {"false": "a", "true": "b"}
RATIO_METRICS = (
    "c1_p50_ms", "c1_p95_ms", "c1_p99_ms", "c1_rps", "c1_prep_p50_ms",
    "c8_p50_ms", "c8_p95_ms", "c8_p99_ms", "c8_rps", "c8_prep_p50_ms",
    "c8_gpu_p50_ms", "c8_queue_p50_ms", "peak_rss_mib", "peak_gpu_mem_mib",
    "c1_cpu_pct_mean", "c8_cpu_pct_mean", "c8_gpu_util_mean",
)  # fmt: skip


def labels(trials: int) -> list[tuple[int, str]]:
    """[(trial, arm "a"|"b")] in execution order."""
    return [(k, ARM[x]) for k, x in enumerate(order(trials), start=1)]


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _phase_mean(rows: list[dict[str, str]], window: list[float] | None, col: str) -> float | None:
    if not window:
        return None
    vals = [
        float(r[col]) for r in rows if r.get(col) and window[0] <= float(r["t_unix"]) <= window[1]
    ]
    return sum(vals) / len(vals) if vals else None


def trial_metrics(d: Path, k: int, arm: str, phases: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"trial": k, "arm": arm}
    for c in (1, 8):
        label = f"ab-t{k}-{arm}-c{c}"
        s = _load(d / f"{label}.summary.json") or {}
        lat = s.get("latency_ms", {})
        stages = s.get("server_stages_ms", {})
        out[f"c{c}_measured"] = s.get("requests_measured")
        out[f"c{c}_ok"] = s.get("success")
        out[f"c{c}_outcomes"] = s.get("outcomes")
        for q in ("p50", "p95", "p99"):
            out[f"c{c}_{q}_ms"] = lat.get(q)
        out[f"c{c}_rps"] = s.get("throughput_rps")
        out[f"c{c}_gpu_p50_ms"] = stages.get("gpu", {}).get("p50")
        out[f"c{c}_queue_p50_ms"] = stages.get("queue", {}).get("p50")
        prep = []
        try:
            for line in (d / f"{label}.requests.jsonl").read_text().splitlines():
                r = json.loads(line)
                if not r["warmup"] and r["status"] == 200 and not r.get("error"):
                    prep.append(r["timing_ms"]["decode"] + r["timing_ms"]["preprocess"])
        except (OSError, ValueError, KeyError):
            prep = []
        out[f"c{c}_prep_p50_ms"] = statistics.median(prep) if prep else None
    try:
        rows = list(csv.DictReader((d / f"ab-t{k}.samples.csv").open()))
    except OSError:
        rows = []
    rss = [float(r["rss_mib"]) for r in rows if r.get("rss_mib")]
    gpu = [float(r["proc_gpu_mem_mib"]) for r in rows if r.get("proc_gpu_mem_mib")]
    out["samples"] = len(rows)
    out["peak_rss_mib"] = max(rss) if rss else None
    out["peak_gpu_mem_mib"] = max(gpu) if gpu else None
    window = phases.get(f"ab-t{k}", {})
    for c in (1, 8):
        out[f"c{c}_cpu_pct_mean"] = _phase_mean(rows, window.get(f"c{c}"), "cpu_pct")
        out[f"c{c}_gpu_util_mean"] = _phase_mean(rows, window.get(f"c{c}"), "gpu_util_pct")
    return out


def ratio_interval(ratios: list[float]) -> dict[str, Any] | None:
    n = len(ratios)
    if n < 2 or n - 1 > len(T975) or any(r <= 0 for r in ratios):
        return None
    logs = [math.log(r) for r in ratios]
    mean = sum(logs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / (n - 1))
    half = T975[n - 2] * sd / math.sqrt(n)
    return {
        "ratio": math.exp(mean),
        "ci95": [math.exp(mean - half), math.exp(mean + half)],
        "pairs": n,
    }


def analyze(d: Path, trials: int) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    runs = d / "runs.jsonl"
    for line in runs.read_text().splitlines() if runs.exists() else []:
        r = json.loads(line)
        if r.get("kind") == "phase":
            phases[r["label"]] = r
    per_trial = [trial_metrics(d, k, arm, phases) for k, arm in labels(trials)]
    estimates: dict[str, Any] = {}
    for m in RATIO_METRICS:
        ratios, complete = [], True
        for a_k, b_k in pairs(trials):
            a, b = per_trial[a_k - 1][m], per_trial[b_k - 1][m]
            if a is None or b is None or a <= 0 or b <= 0:
                complete = False
                continue
            ratios.append(b / a)
        est = ratio_interval(ratios) if complete else None
        estimates[m] = {"paired_ratios_b_over_a": ratios, "complete": complete, **(est or {})}
    arms = {}
    for arm in ("a", "b"):
        mine = [t for t in per_trial if t["arm"] == arm]
        arms[arm] = {
            m: statistics.median(v) if (v := [t[m] for t in mine if t[m] is not None]) else None
            for m in RATIO_METRICS
        }
    return {
        "trials": per_trial,
        "pairs": pairs(trials),
        "estimates": estimates,
        "arm_medians": arms,
    }


def pilot_gate(pilot: Path, image_id: str, threads: int) -> list[str]:
    """Reasons a full A/B run may not start (empty = allowed)."""
    meta, crit = _load(pilot / "meta.json"), _load(pilot / "criteria.json")
    if not isinstance(meta, dict) or not isinstance(crit, dict):
        return [f"pilot {pilot.name}: meta.json or criteria.json missing or unreadable"]
    reasons = []
    params = meta.get("params", {})
    if meta.get("test_mode") is not False:
        reasons.append("pilot was a test-mode run (or predates test-mode recording)")
    if params.get("mode") != "pilot" or "ab" not in params.get("stages", []):
        reasons.append("pilot run is not MODE=pilot with STAGES containing ab")
    if (meta.get("image") or {}).get("Id") != image_id:
        reasons.append("pilot ran on a different image")
    if params.get("ab_threads") != threads:
        reasons.append("pilot used a different AB_THREADS")
    failed = [
        k
        for k, v in crit.items()
        if isinstance(v, dict) and v.get("counted") and v.get("status") != "pass"
    ]
    if failed:
        reasons.append(f"pilot has failed counted criteria: {failed}")
    if not any(k.startswith("ab:") for k in crit):
        reasons.append("pilot has no ab criteria")
    return reasons


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--dir", required=True)
    a.add_argument("--trials", type=int, required=True)
    g = sub.add_parser("pilot-gate")
    g.add_argument("--pilot", required=True)
    g.add_argument("--image-id", required=True)
    g.add_argument("--threads", type=int, required=True)
    args = p.parse_args()
    if args.cmd == "analyze":
        print(json.dumps(analyze(Path(args.dir), args.trials), indent=2))
        return 0
    reasons = pilot_gate(Path(args.pilot), args.image_id, args.threads)
    print(json.dumps({"pass": not reasons, "reasons": reasons, "pilot": Path(args.pilot).name}))
    return 0 if not reasons else 1


if __name__ == "__main__":
    sys.exit(main())
