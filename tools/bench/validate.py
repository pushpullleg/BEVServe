r"""Execution records for M5 benchmark steps (one JSON line each, appended to RUN_DIR/runs.jsonl).

    python tools/bench/validate.py run --dir D --label L --requests N --warmup W \
        --concurrency C --exit RC
    python tools/bench/validate.py sampler --dir D --label L --exit RC --t0 T0 --t1 T1
    python tools/bench/validate.py golden --file F --golden G --exit RC --label L
    python tools/bench/validate.py readiness --dir D --label L --exit RC --t0 T0 --t1 T1

Time-bounded runs (summary "mode": "duration"): --requests is the attempt cap; the run must have
stopped for "duration" or "max_attempts", with one well-formed record per attempt, indices
0..n-1 and n <= the cap.

A record says whether the step *executed* correctly (the client exited 0 and wrote complete,
well-formed results; the sampler ran throughout and attributed GPU memory to the serving process;
the golden comparison ran over every golden frame). HTTP outcomes such as 503 admission rejections
are results, judged by the acceptance criteria, not execution failures. Exit status 1 if invalid.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

MAX_GAP_FACTOR = 3.0  # a gap between samples longer than 3 intervals (+1 s slack) is a gap
MIN_GPU_ATTRIBUTED = 0.9  # fraction of samples with the serving process's GPU memory


def check_run(
    d: Path, label: str, requests: int, warmup: int, concurrency: int, rc: int
) -> list[str]:
    problems = [] if rc == 0 else [f"client exited {rc}"]
    summary_f, rows_f = d / f"{label}.summary.json", d / f"{label}.requests.jsonl"
    try:
        summary = json.loads(summary_f.read_text())
        rows = [json.loads(line) for line in rows_f.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        return [*problems, f"results missing or malformed ({type(exc).__name__})"]
    if not isinstance(summary, dict) or summary.get("label") != label:
        return [*problems, "summary is not for this label"]
    if summary.get("mode") == "duration":
        return problems + check_timed_run(summary, rows, requests, warmup, concurrency)
    if len(rows) != requests + warmup:
        problems.append(f"{len(rows)} request records, expected {requests + warmup}")
    if sorted(r.get("i") for r in rows) != list(range(requests + warmup)):
        problems.append("request indices incomplete")
    if any(not isinstance(r.get("status"), int) or "latency_ms" not in r for r in rows):
        problems.append("request record without status or latency")
    if summary.get("requests_measured") != requests or summary.get("warmup") != warmup:
        problems.append("summary counts differ from the plan")
    if summary.get("concurrency") != concurrency:
        problems.append("summary concurrency differs from the plan")
    outcomes = summary.get("outcomes")
    if not isinstance(outcomes, dict) or sum(outcomes.values()) != requests:
        problems.append("outcome counts do not cover the measured requests")
    return problems


def check_timed_run(
    summary: dict[str, Any], rows: list[dict[str, Any]], cap: int, warmup: int, concurrency: int
) -> list[str]:
    problems = []
    n = len(rows)
    if summary.get("stop_reason") not in ("duration", "max_attempts"):
        problems.append(f"run stopped for {summary.get('stop_reason')!r}")
    if n == 0 or n > cap or summary.get("attempts_total") != n:
        problems.append(f"{n} request records for an attempt cap of {cap}")
    if sorted(r.get("i") for r in rows) != list(range(n)):
        problems.append("request indices incomplete")
    if any(not isinstance(r.get("status"), int) or "latency_ms" not in r for r in rows):
        problems.append("request record without status or latency")
    acct = summary.get("accounting") or {}
    if summary.get("warmup") != warmup or acct.get("attempts") != max(0, n - warmup):
        problems.append("summary counts differ from the records")
    if summary.get("concurrency") != concurrency:
        problems.append("summary concurrency differs from the plan")
    outcomes = summary.get("outcomes")
    if not isinstance(outcomes, dict) or sum(outcomes.values()) != acct.get("attempts"):
        problems.append("outcome counts do not cover the measured attempts")
    if (summary.get("outputs") or {}).get("reference_error"):
        problems.append(summary["outputs"]["reference_error"])
    return problems


def check_readiness(
    d: Path, label: str, rc: int, t0: float, t1: float
) -> tuple[list[str], dict[str, Any]]:
    problems = [] if rc == 0 else [f"readiness poller exited {rc}"]
    try:
        rows = list(csv.DictReader((d / f"{label}.readiness.csv").open()))
    except OSError as exc:
        return [*problems, f"readiness record missing ({type(exc).__name__})"], {}
    times = [t0, *(float(r["t_unix"]) for r in rows), t1]
    max_gap = max(b - a for a, b in itertools.pairwise(times))
    if not rows:
        problems.append("no readiness probes")
    elif max_gap > 5.0:  # 1 s interval + 2 s probe timeout + slack
        problems.append(f"readiness probing gap of {max_gap:.1f} s")
    not_ready = sum(1 for r in rows if r["http_status"] != "200" or r["status"] != "ready")
    return problems, {"probes": len(rows), "not_ready": not_ready, "max_gap_s": round(max_gap, 2)}


def check_sampler(
    d: Path, label: str, rc: int, t0: float, t1: float
) -> tuple[list[str], dict[str, Any]]:
    problems = [] if rc == 0 else [f"sampler exited {rc}"]
    try:
        meta = json.loads((d / f"{label}.sampler.json").read_text())
        rows = list(csv.DictReader((d / f"{label}.samples.csv").open()))
    except (OSError, ValueError) as exc:
        return [*problems, f"sampler output missing or malformed ({type(exc).__name__})"], {}
    if meta.get("exit_reason") != "stopped":
        problems.append(f"sampler ended early: {meta.get('exit_reason')}")
    interval = float(meta.get("interval_s", 1.0))
    times = [t0, *(float(r["t_unix"]) for r in rows), t1]
    max_gap = max(b - a for a, b in itertools.pairwise(times))
    if max_gap > MAX_GAP_FACTOR * interval + 1.0:
        problems.append(f"sampling gap of {max_gap:.1f} s")
    attributed = sum(1 for r in rows if r.get("gpu_status") == "ok")
    fraction = attributed / len(rows) if rows else 0.0
    if rows and fraction < MIN_GPU_ATTRIBUTED:
        problems.append(f"serving-process GPU memory present in {fraction:.0%} of samples")
    pids = meta.get("gpu_pids", [])
    if len(pids) != 1:
        problems.append(
            f"GPU memory attributed to {len(pids)} process(es), expected the server only"
        )
    evidence = {
        "samples": len(rows),
        "max_gap_s": round(max_gap, 2),
        "gpu_attributed_fraction": round(fraction, 3),
        "gpu_status_counts": meta.get("gpu_status_counts"),
        "gpu_pids": pids,
        "target_pid": meta.get("target_pid"),
    }
    return problems, evidence


def check_golden(f: Path, golden: Path, rc: int) -> tuple[list[str], dict[str, Any]]:
    problems = [] if rc == 0 else [f"comparison exited {rc}"]
    try:
        result = json.loads(f.read_text())
        expected = len(json.loads(golden.read_text())["frames"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [*problems, f"comparison output missing or malformed ({type(exc).__name__})"], {}
    frames = result.get("frames") if isinstance(result, dict) else None
    if not isinstance(frames, list) or len(frames) != expected:
        problems.append(
            f"compared {len(frames) if isinstance(frames, list) else 0} of {expected} frames"
        )
    if result.get("pass") is not True or result.get("model_identity_ok") is not True:
        problems.append("golden parity not met")
    return problems, {"frames_compared": len(frames or []), "frames_expected": expected}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["run", "sampler", "golden", "readiness"])
    p.add_argument("--dir")
    p.add_argument("--label", required=True)
    p.add_argument("--exit", type=int, required=True)
    p.add_argument("--requests", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--t0", type=float)
    p.add_argument("--t1", type=float)
    p.add_argument("--file")
    p.add_argument("--golden")
    a = p.parse_args()
    evidence: dict[str, Any] = {}
    if a.kind == "run":
        problems = check_run(Path(a.dir), a.label, a.requests, a.warmup, a.concurrency, a.exit)
    elif a.kind == "sampler":
        problems, evidence = check_sampler(Path(a.dir), a.label, a.exit, a.t0, a.t1)
    elif a.kind == "readiness":
        problems, evidence = check_readiness(Path(a.dir), a.label, a.exit, a.t0, a.t1)
    else:
        problems, evidence = check_golden(Path(a.file), Path(a.golden), a.exit)
    record = {
        "kind": a.kind,
        "label": a.label,
        "valid": not problems,
        "problems": problems,
        **evidence,
    }
    print(json.dumps(record))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
