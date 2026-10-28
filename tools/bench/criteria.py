"""Evaluate M5 acceptance criteria from a benchmark run directory.

    python tools/bench/criteria.py --dir RUN_DIR > RUN_DIR/criteria.json

The plan (stages and request counts) comes from RUN_DIR/meta.json; execution records from
RUN_DIR/runs.jsonl (tools/bench/validate.py and the orchestrator). A planned step without a valid
execution record fails: a crashed client, missing or malformed results, a sampler that died or
could not attribute GPU memory, or a golden comparison that did not run never count as success.
HTTP 503 queue_full responses under saturation are expected admission results, judged separately
from execution failures.

Statuses: "pass", "fail", "insufficient_data" (a needed measure is missing or the interval is too
wide; never a pass), "not_run" (stage not requested). Kinds: "harness" and "correctness" count in
every mode; "statistical" criteria count in full mode and are reported uncounted in pilot mode
(pilot runs are too small for them). Exit status 1 if any counted criterion is not "pass".
Latency and throughput values are baselines, not pass/fail; the provisional design target (p95
server total <= 250 ms at concurrency 1) is information only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

OVERHEAD_LIMIT_MS = 3.0
OVERHEAD_LIMIT_FRACTION = 0.03
COLD_READY_LIMIT_S = 30.0
SOAK_RSS_GROWTH_MIB = 200.0
SOAK_GPU_GROWTH_MIB = 100.0
SOAK_P50_DRIFT_RATIO = 1.25
LEVELS = (1, 2, 4, 8)
SAMPLED_STAGES = ("levels", "admission", "soak")


def plan(p: dict[str, Any]) -> dict[str, list[tuple[str, int, int, int]]]:
    """stage -> [(label, requests, warmup, concurrency)], mirroring tools/m5_bench.sh."""
    out: dict[str, list[tuple[str, int, int, int]]] = {
        "cold": [(f"cold{k}-first", 1, 0, 1) for k in range(1, p["cold_runs"] + 1)],
        "levels": [(f"levels-c{c}", p["level_requests"], p["level_warmup"], c) for c in LEVELS],
        "overhead": [
            (f"overhead-t{k}-timing-{arm}", p["overhead_requests"], p["overhead_warmup"], 1)
            for k, arm in enumerate(overhead_order(p["overhead_trials"]), start=1)
        ],
        "admission": [
            (f"admission-c{c}-{m}", p["admission_requests"], 0, c)
            for c in p["admission_levels"]
            for m in ("noretry", "honor")
        ]
        + [("admission-recovery", 5, 0, 1)],
        "soak": [(f"soak-{p['soak_requests']}", p["soak_requests"], p["soak_warmup"], 2)],
    }
    return {s: out[s] for s in p["stages"]}


def overhead_order(trials: int) -> list[str]:
    return [("false", "true")[i] for _ in range(trials // 4) for i in (0, 1, 1, 0)]


def load_json(f: Path) -> Any:
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def ready(f: Path) -> bool | None:
    doc = load_json(f)
    return None if not isinstance(doc, dict) else doc.get("status") == "ready"


def growth(path: Path, column: str) -> float | None:
    try:
        rows = [r for r in csv.DictReader(path.open()) if r.get(column) not in (None, "")]
    except OSError:
        return None
    if len(rows) < 10:
        return None
    k = max(1, len(rows) // 10)
    first = sum(float(r[column]) for r in rows[:k]) / k
    last = sum(float(r[column]) for r in rows[-k:]) / k
    return last - first


def median(values: list[float]) -> float:
    s = sorted(values)
    return s[(len(s) - 1) // 2]


def evaluate(d: Path) -> dict[str, Any]:
    meta = load_json(d / "meta.json") or {}
    params = meta.get("params")
    if not isinstance(params, dict):
        return {
            "plan": {
                "status": "fail",
                "kind": "harness",
                "counted": True,
                "reason": "meta.json has no plan",
            }
        }
    pilot = params.get("mode") == "pilot"
    stages = plan(params)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    cleanups: list[dict[str, Any]] = []
    services = 0
    runs_f = d / "runs.jsonl"
    for line in runs_f.read_text().splitlines() if runs_f.exists() else []:
        r = json.loads(line)
        if r.get("kind") == "cleanup":
            cleanups.append(r)
        elif r.get("kind") == "service":
            services += 1
        else:
            records[(r["kind"], r["label"])] = r
    out: dict[str, Any] = {"mode": "pilot" if pilot else "full"}

    def put(name: str, kind: str, stage: str | None, ok: bool | None, **evidence: Any) -> None:
        if stage is not None and stage not in stages:
            status = "not_run"
        elif ok is None or any(v is None for v in evidence.values()):
            status = "insufficient_data"
        else:
            status = "pass" if ok else "fail"
        counted = status != "not_run" and not (pilot and kind == "statistical")
        out[name] = {"status": status, "kind": kind, "counted": counted, **evidence}

    def run_ok(label: str) -> bool:
        return records.get(("run", label), {}).get("valid") is True

    def summ(label: str) -> dict[str, Any] | None:
        s = load_json(d / f"{label}.summary.json")
        return s if run_ok(label) and isinstance(s, dict) else None

    # ---- harness
    planned = [label for runs in stages.values() for (label, *_rest) in runs]
    bad = [label for label in planned if not run_ok(label)]
    put(
        "execution: every planned client run completed with complete, well-formed results",
        "harness",
        None,
        not bad,
        planned=len(planned),
        invalid={
            label: records.get(("run", label), {}).get("problems", ["no record"]) for label in bad
        },
    )
    sampled = [s for s in SAMPLED_STAGES if s in stages]
    bad_s = [s for s in sampled if records.get(("sampler", s), {}).get("valid") is not True]
    put(
        "execution: resource samplers ran throughout and attributed GPU memory to the server",
        "harness",
        None if sampled else "levels",
        not bad_s,
        samplers={
            s: {
                k: v
                for k, v in records.get(("sampler", s), {"problems": ["no record"]}).items()
                if k not in ("kind", "label")
            }
            for s in sampled
        },
    )
    failed_cleanups = [c for c in cleanups if not c.get("ok")]
    put(
        "cleanup: every benchmark service was removed (containers and network gone)",
        "harness",
        None,
        services > 0 and len(cleanups) == services and not failed_cleanups,
        services_started=services,
        cleanups_ok=len(cleanups) - len(failed_cleanups),
        failed=[c.get("detail") for c in failed_cleanups],
    )

    # ---- correctness
    levels = {c: summ(f"levels-c{c}") for c in LEVELS}
    put(
        "levels: every request at C = 1, 2, 4, 8 succeeded (no rejections, errors or timeouts)",
        "correctness",
        "levels",
        all(
            s is not None and s["outcomes"] == {"ok": s["requests_measured"]}
            for s in levels.values()
        ),
        outcomes={f"c{c}": (s or {}).get("outcomes", "invalid run") for c, s in levels.items()},
    )
    if (c1 := levels.get(1)) and "total" in c1.get("server_stages_ms", {}):
        p95 = c1["server_stages_ms"]["total"].get("p95")
        out["info: provisional target p95 server total <= 250 ms at C=1"] = {
            "status": "met" if p95 is not None and p95 <= 250 else "not met",
            "kind": "info",
            "counted": False,
            "p95_ms": p95,
        }
    adm_labels = [
        label for label, *_ in stages.get("admission", []) if label != "admission-recovery"
    ]
    adm = {label: summ(label) for label in adm_labels}
    put(
        "admission: under saturation only 200 or 503 queue_full; every 503 has Retry-After: 1",
        "correctness",
        "admission",
        all(
            s is not None
            and set(s["outcomes"]) <= {"ok", "rejected"}
            and s["outcomes"].get("ok", 0) > 0
            and set(s["queue_full_retry_after"]) <= {"1"}
            for s in adm.values()
        )
        if adm
        else None,
        runs={
            label: {
                "outcomes": (s or {}).get("outcomes", "invalid run"),
                "retry_after": (s or {}).get("queue_full_retry_after"),
            }
            for label, s in adm.items()
        },
    )
    rec = summ("admission-recovery")
    put(
        "admission: ready and serving again after saturation",
        "correctness",
        "admission",
        rec is not None
        and rec["outcomes"] == {"ok": 5}
        and ready(d / "admission-recovery-readyz.json") is True,
    )
    cold = (
        [json.loads(x) for x in (d / "cold.jsonl").read_text().splitlines()]
        if (d / "cold.jsonl").exists()
        else []
    )
    firsts = [summ(label) for label, *_ in stages.get("cold", [])]
    put(
        f"cold start: ready within {COLD_READY_LIMIT_S:.0f} s and first request succeeds",
        "correctness",
        "cold",
        len(cold) == len(firsts)
        and all(c["ready_s"] <= COLD_READY_LIMIT_S for c in cold)
        and all(s is not None and s["outcomes"] == {"ok": 1} for s in firsts),
        ready_s=[c["ready_s"] for c in cold],
        first_request_ms=[(s or {}).get("latency_ms", {}).get("p50") for s in firsts],
    )

    # ---- overhead (statistical)
    ov = load_json(d / "overhead.json") if "overhead" in stages else None
    if (
        ov
        and ov.get("complete")
        and ov.get("ci95_ms")
        and all(run_ok(lb) for lb, *_ in stages["overhead"])
    ):
        limit = max(OVERHEAD_LIMIT_MS, OVERHEAD_LIMIT_FRACTION * ov["off_p50_mean_ms"])
        lo, hi = ov["ci95_ms"]
        # pass: the whole interval is within the limit; fail: the estimate itself exceeds it;
        # otherwise the trials cannot decide (insufficient_data, never a pass).
        verdict: bool | None = (
            True if hi <= limit else (False if ov["mean_diff_ms"] > limit else None)
        )
        put(
            "overhead: Server-Timing adds <= max(3 ms, 3%) to client p50 (95% CI upper bound)",
            "statistical",
            "overhead",
            verdict,
            mean_diff_ms=ov["mean_diff_ms"],
            ci95_ms=[lo, hi],
            limit_ms=limit,
            paired_p50_diff_ms=ov["paired_p50_diff_ms"],
        )
    else:
        put(
            "overhead: Server-Timing adds <= max(3 ms, 3%) to client p50 (95% CI upper bound)",
            "statistical",
            "overhead",
            None,
            overhead_json=None,
        )

    # ---- soak
    label = f"soak-{params['soak_requests']}"
    soak = summ(label) if "soak" in stages else None
    put(
        "soak: no failed requests",
        "correctness",
        "soak",
        soak is not None and soak["outcomes"] == {"ok": soak["requests_measured"]},
        outcomes=(soak or {}).get("outcomes", "invalid run"),
    )
    golden = records.get(("golden", "soak-golden"), {})
    put(
        "soak: ready afterwards and golden parity comparison ran over all frames and passed",
        "correctness",
        "soak",
        golden.get("valid") is True and ready(d / "soak-readyz.json") is True,
        golden={k: v for k, v in golden.items() if k not in ("kind", "label")} or "no record",
        readyz_ready=ready(d / "soak-readyz.json"),
    )
    samples = d / "soak.samples.csv"
    sampler_ok = records.get(("sampler", "soak"), {}).get("valid") is True
    rss = growth(samples, "rss_mib") if sampler_ok else None
    gpu = growth(samples, "proc_gpu_mem_mib") if sampler_ok else None
    put(
        f"soak: RSS growth <= {SOAK_RSS_GROWTH_MIB:.0f} MiB",
        "statistical",
        "soak",
        rss is not None and rss <= SOAK_RSS_GROWTH_MIB,
        rss_growth_mib=rss,
    )
    put(
        f"soak: serving-process GPU memory growth <= {SOAK_GPU_GROWTH_MIB:.0f} MiB",
        "statistical",
        "soak",
        gpu is not None and gpu <= SOAK_GPU_GROWTH_MIB,
        gpu_growth_mib=gpu,
    )
    drift = None
    if soak is not None:
        rows = [
            json.loads(x)
            for x in (d / f"{label}.requests.jsonl").read_text().splitlines()
            if x.strip()
        ]
        ok_rows = sorted(
            (r for r in rows if not r["warmup"] and r["status"] == 200 and not r.get("error")),
            key=lambda r: r["i"],
        )
        k = len(ok_rows) // 10
        if k >= 5:
            drift = median([r["latency_ms"] for r in ok_rows[-k:]]) / median(
                [r["latency_ms"] for r in ok_rows[:k]]
            )
    put(
        f"soak: latency drift (median last 10% / first 10%) <= {SOAK_P50_DRIFT_RATIO}",
        "statistical",
        "soak",
        drift is not None and drift <= SOAK_P50_DRIFT_RATIO,
        drift_ratio=drift,
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    out = evaluate(Path(p.parse_args().dir))
    print(json.dumps(out, indent=2))
    counted = [v for v in out.values() if isinstance(v, dict) and v.get("counted", True)]
    return 1 if any(v.get("status") != "pass" for v in counted) else 0


if __name__ == "__main__":
    sys.exit(main())
