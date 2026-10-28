"""Multi-process load generator (M7 diagnostic): removes the single-client CPU bottleneck.

    python tools/bench/multiload.py --url URL --workers 4 --concurrency 32 --duration 10 \\
        --max-attempts 1000 [--rate 250] [--honor-retry-after] --inputs nuscenes:even:20 \\
        --bevdet-root ROOT --out DIR --label L [--reference REF]

Request bodies are encoded once and inherited by --workers forked processes (no per-process
dataset loading). Aggregate concurrency is split as evenly as possible (32 over 4 = 8 each; 5 over
4 = 2,1,1,1). All workers start at the same instant and stop at the same deadline (--duration);
a cross-process counter enforces --max-attempts across all of them, exactly.

--rate R paces attempts: worker k starts attempt j no earlier than start + j / r_k, with
r_k = R x share_k / concurrency, so both arms of a comparison receive the same offered arrival
schedule. An attempt that is due but finds all of its worker's concurrency slots busy waits for one
(recorded as waited_for_slot); attempts then start late and the achieved rate falls below R, which
the summary reports. Without --rate each worker is a closed loop over its share.

Per attempt (in addition to tools/bench/load.py's fields): worker, pid, scheduled_unix, lag_ms
(actual start - scheduled start) and waited_for_slot. The merged run is written in load.py's
format (indices renumbered 0..n-1 in global start order), so the existing validation, criteria and
analysis apply, plus a "multiprocess" summary block: per-worker CPU % of one core (each worker is a
single asyncio thread), attempts, outcomes, lag; target and achieved attempt rates; time-weighted
mean and maximum in-flight requests.

Stop reasons: duration | max_attempts | interrupted (SIGINT/SIGTERM; exit 3) | hard_stop (a worker
did not finish within duration + timeout + 10 s; exit 3) | worker_failed (exit 4). Only the first
two are valid runs. BENCH_TEST_FAIL_WORKER=k makes worker k fail after its first attempt (tests).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import multiprocessing as mp
import os
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load  # same directory

SATURATION_CPU_PCT = 80.0  # a worker at >= 80% of one core is treated as client-saturated


def shares(total: int, workers: int) -> list[int]:
    base, extra = divmod(total, workers)
    return [base + (1 if k < extra else 0) for k in range(workers)]


async def _worker(
    k: int,
    share: int,
    a: argparse.Namespace,
    bodies: list[tuple[bytes, str]],
    counter: Any,
    stop: Any,
    t_start: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rate = a.rate * share / a.concurrency if a.rate else None
    deadline = t_start + a.duration
    sem = asyncio.Semaphore(share)
    fail_after = (
        int(os.environ["BENCH_TEST_FAIL_WORKER"]) == k
        if os.environ.get("BENCH_TEST_FAIL_WORKER")
        else False
    )
    limits = httpx.Limits(max_connections=share, max_keepalive_connections=share)
    tasks: set[asyncio.Task[None]] = set()

    async def one(idx: int, scheduled: float, waited: bool) -> None:
        body, ctype = bodies[idx % len(bodies)]
        t0u = time.time()
        t0 = time.perf_counter()
        row: dict[str, Any] = {
            "wi": len(rows),
            "worker": k,
            "pid": os.getpid(),
            "input": idx % len(bodies),
            "warmup": False,
            "bytes_sent": len(body),
            "t_unix": t0u,
            "scheduled_unix": scheduled,
            "lag_ms": (t0u - scheduled) * 1000,
            "waited_for_slot": waited,
            "attempt": idx,
        }
        try:
            resp = await client.post(
                "/v1/detect",
                content=body,
                headers={"content-type": ctype, "x-request-id": f"bench-{a.label}-{idx}"},
            )
            payload = resp.content
            row.update(
                latency_ms=(time.perf_counter() - t0) * 1000,
                status=resp.status_code,
                bytes_received=len(payload),
                retry_after=resp.headers.get("retry-after"),
                server=load.parse_server_timing(resp.headers.get("server-timing")),
            )
            if resp.status_code == 200:
                try:
                    doc = json.loads(payload)
                    row.update(
                        boxes=len(doc["boxes"]),
                        timing_ms=doc["timing_ms"],
                        output_sha256=load.canonical_output(doc),
                    )
                except (ValueError, KeyError, TypeError):
                    row["error"] = "MalformedResponse"
            else:
                try:
                    row["type"] = json.loads(payload).get("type")
                except ValueError:
                    row["type"] = None
        except httpx.HTTPError as exc:
            row.update(
                latency_ms=(time.perf_counter() - t0) * 1000, status=0, error=type(exc).__name__
            )
        row["t_end_unix"] = time.time()
        rows.append(row)
        if a.honor_retry_after and row.get("retry_after"):
            with contextlib.suppress(ValueError):
                await asyncio.sleep(
                    max(0.0, min(float(row["retry_after"]), 10.0, deadline - time.time()))
                )
        sem.release()

    async with httpx.AsyncClient(
        base_url=a.url, limits=limits, timeout=httpx.Timeout(a.timeout)
    ) as client:
        await asyncio.sleep(max(0.0, t_start - time.time()))
        j = 0
        while not stop.value and time.time() < deadline:
            scheduled = t_start + j / rate if rate else time.time()
            if rate and scheduled > time.time():
                await asyncio.sleep(scheduled - time.time())
            if stop.value or time.time() >= deadline:
                break
            waited = sem.locked()
            await sem.acquire()
            if stop.value or time.time() >= deadline:
                sem.release()
                break
            with counter.get_lock():
                if counter.value >= a.max_attempts:
                    sem.release()
                    break
                idx = counter.value
                counter.value += 1
            task = asyncio.ensure_future(one(idx, scheduled, waited))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            j += 1
            if fail_after and rows:
                raise RuntimeError(f"injected failure in worker {k}")
        if tasks:
            await asyncio.wait(set(tasks), timeout=a.timeout + 5)
    return rows


def _run_worker(
    k: int,
    share: int,
    a: argparse.Namespace,
    bodies: list[tuple[bytes, str]],
    counter: Any,
    stop: Any,
    t_start: float,
    out_dir: str,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the coordinator handles interruption
    signal.signal(signal.SIGTERM, lambda *_: setattr(stop, "value", 1))
    cpu0, w0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    rows = asyncio.run(_worker(k, share, a, bodies, counter, stop, t_start))
    cpu1, wall = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter() - w0
    cpu = (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)
    active = max((r["t_end_unix"] for r in rows), default=t_start) - t_start
    Path(out_dir, f"worker-{k}.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "cpu_s": cpu,
                "wall_s": wall,
                "active_s": active,
                "pid": os.getpid(),
                "share": share,
            }
        )
    )


def in_flight(rows: list[dict[str, Any]]) -> dict[str, float]:
    events = sorted([(r["t_unix"], 1) for r in rows] + [(r["t_end_unix"], -1) for r in rows])
    if not events:
        return {"mean": 0.0, "max": 0}
    level, peak, area, last = 0, 0, 0.0, events[0][0]
    for t, step in events:
        area += level * (t - last)
        level += step
        peak = max(peak, level)
        last = t
    span = events[-1][0] - events[0][0]
    return {"mean": round(area / span, 2) if span > 0 else 0.0, "max": peak}


def merge(
    results: list[dict[str, Any]], a: argparse.Namespace, t_start: float
) -> list[dict[str, Any]]:
    rows = sorted(
        (r for res in results for r in res["rows"]), key=lambda r: (r["t_unix"], r["attempt"])
    )
    for i, r in enumerate(rows):
        r["i"] = i
        r["t_start_s"] = r["t_unix"] - t_start
        r["t_end_s"] = r["t_end_unix"] - t_start
    return rows


def worker_report(res: dict[str, Any]) -> dict[str, Any]:
    rows = res["rows"]
    busy = res["active_s"] if res["active_s"] > 0 else res["wall_s"]
    lags = [r["lag_ms"] for r in rows if not r["waited_for_slot"]]
    cpu_pct = 100 * res["cpu_s"] / busy if busy > 0 else None
    return {
        "worker": res["k"],
        "share": res["share"],
        "attempts": len(rows),
        "ok": sum(1 for r in rows if load.outcome(r) == "ok"),
        "rejected": sum(1 for r in rows if load.outcome(r) == "rejected"),
        "cpu_pct_of_active_time": round(cpu_pct, 1) if cpu_pct is not None else None,
        "saturated": cpu_pct is not None and cpu_pct >= SATURATION_CPU_PCT,
        "waited_for_slot_fraction": round(
            sum(1 for r in rows if r["waited_for_slot"]) / len(rows), 3
        )
        if rows
        else None,
        "lag_ms_without_slot_wait": {
            "n": len(lags),
            "p50": round(float(np.percentile(lags, 50)), 2) if lags else None,
            "p95": round(float(np.percentile(lags, 95)), 2) if lags else None,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--max-attempts", type=int, required=True)
    p.add_argument("--rate", type=float)
    p.add_argument("--honor-retry-after", action="store_true")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--inputs", default="synthetic:6")
    p.add_argument("--bevdet-root")
    p.add_argument("--out", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--reference")
    a = p.parse_args()
    if not 1 <= a.workers <= 16 or a.workers > a.concurrency:
        raise SystemExit("--workers must be 1..16 and at most --concurrency")
    if a.duration <= 0 or a.max_attempts < 1 or (a.rate is not None and a.rate <= 0):
        raise SystemExit("--duration, --max-attempts and --rate must be positive")
    bodies, inputs = load.load_inputs(a.inputs, a.bevdet_root)
    inputs["mean_request_bytes"] = float(np.mean([len(b) for b, _ in bodies]))
    hashes = load.body_hashes(bodies)
    reference = json.loads(Path(a.reference).read_text()) if a.reference else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / f".{a.label}.workers"
    scratch.mkdir(exist_ok=True)
    ctx = mp.get_context("fork")
    counter, stop = ctx.Value("i", 0), ctx.Value("i", 0)
    t_start = time.time() + 1.0  # every worker forked and ready before the common start
    procs = [
        ctx.Process(
            target=_run_worker, args=(k, s, a, bodies, counter, stop, t_start, str(scratch))
        )
        for k, s in enumerate(shares(a.concurrency, a.workers))
    ]
    state: dict[str, str | None] = {"reason": None}

    def interrupt(*_: object) -> None:
        state["reason"] = "interrupted"
        stop.value = 1

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    for proc in procs:
        proc.start()
    hard = t_start + a.duration + a.timeout + 10
    for proc in procs:
        proc.join(timeout=max(0.0, hard - time.time()))
    if any(proc.is_alive() for proc in procs):
        stop.value = 1
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
        state["reason"] = state["reason"] or "hard_stop"
    results, failed = [], []
    for k, proc in enumerate(procs):
        f = scratch / f"worker-{k}.json"
        if proc.exitcode != 0 or not f.exists():
            failed.append(k)
            continue
        res = json.loads(f.read_text())
        res["k"] = k
        results.append(res)
        f.unlink()
    with contextlib.suppress(OSError):
        scratch.rmdir()
    reason = state["reason"] or (
        "worker_failed"
        if failed
        else ("max_attempts" if counter.value >= a.max_attempts else "duration")
    )
    rows = merge(results, a, t_start)
    (out / f"{a.label}.requests.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")
    )
    ns = argparse.Namespace(
        label=a.label,
        url=a.url,
        concurrency=a.concurrency,
        warmup=0,
        honor_retry_after=a.honor_retry_after,
        duration=a.duration,
        stop_reason=reason,
    )
    summary = load.summarize(rows, ns, inputs)
    summary["outputs"] = load.check_outputs(rows, reference, hashes)
    reports = [worker_report(r) for r in results]
    window = summary["window_s"]
    summary["multiprocess"] = {
        "workers": a.workers,
        "shares": shares(a.concurrency, a.workers),
        "failed_workers": failed,
        "target_rate_rps": a.rate,
        "achieved_attempt_rate_rps": round(len(rows) / window, 2) if window else None,
        "in_flight": in_flight(rows),
        "per_worker": reports,
        "any_worker_saturated": any(r["saturated"] for r in reports),
        "saturation_rule": f"worker CPU >= {SATURATION_CPU_PCT:.0f}% of one core while active",
    }
    summary["client_process"] = {
        "cpu_pct": max((r["cpu_pct_of_active_time"] or 0 for r in reports), default=None),
        "note": "maximum over worker processes",
    }
    (out / f"{a.label}.summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "label": a.label,
                "stop_reason": reason,
                "attempts": len(rows),
                "statuses": summary["statuses"],
                "achieved_rate": summary["multiprocess"]["achieved_attempt_rate_rps"],
                "worker_cpu_pct": [r["cpu_pct_of_active_time"] for r in reports],
                "failed_workers": failed,
            }
        )
    )
    if failed:
        return 4
    return 3 if reason in ("interrupted", "hard_stop") else 0


if __name__ == "__main__":
    sys.exit(main())
