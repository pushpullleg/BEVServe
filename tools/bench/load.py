"""Closed-loop load generator for BEVServe (M5).

    python tools/bench/load.py --url http://127.0.0.1:8000 --concurrency 4 --requests 300 \\
        --warmup 20 --inputs nuscenes:even:20 --bevdet-root /path/to/BEVDet --out DIR --label c4
    python tools/bench/load.py ... --inputs synthetic:6      # no dataset needed

Each of `concurrency` workers sends the next request as soon as its previous one completes. Request
bodies are multipart-encoded once up front, so per-request client work is only the HTTP exchange.
Warm-up requests (same concurrency) are sent first and excluded from the statistics.

Writes DIR/<label>.requests.jsonl (one line per measured request) and DIR/<label>.summary.json.
Latency percentiles use the nearest-rank method; 95% intervals come from a 2,000-resample
bootstrap (seed 0). Throughput = successful responses / measurement wall time.
nuScenes inputs are read locally and never written anywhere; outputs contain no image data.

Time-bounded mode (M7): `--duration S --max-attempts N` sends requests until S seconds have passed
or N attempts (warm-up included) were made, whichever comes first; requests in flight at the end
finish within --timeout, and a hard stop at S + timeout + 5 s cancels anything left. SIGINT and
SIGTERM stop new requests the same way; the run is then marked interrupted and exits 3.

Output checks (M7): every 200 response gets `output_sha256`, a hash of its canonical semantic
content (model identity, frame, units and boxes; JSON re-serialized with sorted keys and fixed
separators, so formatting cannot cause a mismatch; request id and timings excluded).
`--write-reference F` stores the hash per input (each input must give one hash); `--reference F`
compares every 200 response with it (same input set required, checked by request-body hashes).

Accounting: attempts (requests sent), completed (an HTTP response arrived), rejected (503
queue_full, i.e. refused by admission), accepted (completed and not rejected), success (200 with a
valid body). Latency is reported separately for successful and for rejected requests.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import platform
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def encode(files: list[Any], data: dict[str, str]) -> tuple[bytes, str]:
    req = httpx.Request("POST", "http://x/v1/detect", files=files, data=data)
    return req.read(), req.headers["content-type"]


def load_inputs(
    spec: str, bevdet_root: str | None
) -> tuple[list[tuple[bytes, str]], dict[str, Any]]:
    kind, _, rest = spec.partition(":")
    if kind == "synthetic":
        from tests.synthetic import calibration_payload, synthetic_image
        from tests.synthetic import encode as enc_img

        from bevserve.constants import CAMERA_ORDER

        bodies = []
        for k in range(int(rest or 1)):
            images = {n: enc_img(synthetic_image(100 * k + i)) for i, n in enumerate(CAMERA_ORDER)}
            files = [(n, (f"{n}.jpg", b, "image/jpeg")) for n, b in images.items()]
            bodies.append(encode(files, {"calibration": json.dumps(calibration_payload())}))
        return bodies, {"kind": "synthetic", "count": len(bodies)}
    if kind == "nuscenes":
        if not bevdet_root:
            raise SystemExit("--bevdet-root is required for nuscenes inputs")
        import os

        import mmcv

        sys.path.insert(0, str(ROOT / "tools"))
        from compare_direct import request_parts, select

        from bevserve.constants import CAMERA_ORDER

        cwd = os.getcwd()
        os.chdir(bevdet_root)  # info files use paths relative to the BEVDet root
        try:
            infos = sorted(
                mmcv.load("data/nuscenes/bevdetv3-trainval_infos_val.pkl")["infos"],
                key=lambda e: e["timestamp"],
            )
            indices = select(rest or "even:20", infos)
            bodies = []
            for idx in indices:
                images, calib = request_parts(infos[idx])
                files = [(n, (f"{n}.jpg", images[n], "image/jpeg")) for n in CAMERA_ORDER]
                bodies.append(encode(files, {"calibration": json.dumps(calib)}))
        finally:
            os.chdir(cwd)
        return bodies, {
            "kind": "nuscenes-val",
            "selection": rest or "even:20",
            "count": len(bodies),
        }
    raise SystemExit(f"unknown input spec {spec!r}")


def canonical_output(doc: dict[str, Any]) -> str:
    """sha256 of the semantically meaningful part of a 200 response (see module docstring)."""
    model = doc.get("model") or {}
    content = {
        "model": {k: model.get(k) for k in ("name", "checkpoint_sha256", "config_sha256")},
        "frame": doc["frame"],
        "units": doc.get("units"),
        "boxes": doc["boxes"],
    }
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def body_hashes(bodies: list[tuple[bytes, str]]) -> list[str]:
    """Identity of each input, independent of the multipart boundary chosen at encoding."""
    return [
        hashlib.sha256(b.replace(c.split("boundary=")[1].encode(), b"")).hexdigest()
        for b, c in bodies
    ]


def parse_server_timing(header: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in (header or "").split(","):
        if ";dur=" in item:
            name, dur = item.strip().split(";dur=")
            out[name] = float(dur)
    return out


async def run(
    args: argparse.Namespace, bodies: list[tuple[bytes, str]]
) -> tuple[list[dict[str, Any]], float, str]:
    """(rows, seconds, stop reason: requests, duration, max_attempts, interrupted, hard_stop)."""
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    timeout = httpx.Timeout(args.timeout)
    rows: list[dict[str, Any]] = []
    counter = {"next": 0}
    timed = args.duration is not None
    total = args.max_attempts if timed else args.warmup + args.requests
    stop = asyncio.Event()
    state = {"reason": "max_attempts" if timed else "requests"}
    loop = asyncio.get_running_loop()

    def interrupt() -> None:
        state["reason"] = "interrupted"
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, interrupt)

    async with httpx.AsyncClient(base_url=args.url, limits=limits, timeout=timeout) as client:
        t_start = time.perf_counter()
        t_unix0 = time.time()
        end_at = t_start + args.duration if timed else None

        def more() -> bool:
            if stop.is_set():
                return False
            if end_at is not None and time.perf_counter() >= end_at:
                if state["reason"] != "interrupted":
                    state["reason"] = "duration"
                return False
            return counter["next"] < total

        async def worker(wid: int) -> None:
            while more():
                i = counter["next"]
                counter["next"] = i + 1
                body, ctype = bodies[i % len(bodies)]
                row: dict[str, Any] = {
                    "i": i,
                    "worker": wid,
                    "input": i % len(bodies),
                    "warmup": i < args.warmup,
                    "bytes_sent": len(body),
                }
                t0 = time.perf_counter()
                row["t_start_s"] = t0 - t_start
                row["t_unix"] = t_unix0 + (t0 - t_start)
                try:
                    resp = await client.post(
                        "/v1/detect",
                        content=body,
                        headers={"content-type": ctype, "x-request-id": f"bench-{args.label}-{i}"},
                    )
                    payload = resp.content
                    row["latency_ms"] = (time.perf_counter() - t0) * 1000
                    row["status"] = resp.status_code
                    row["bytes_received"] = len(payload)
                    row["retry_after"] = resp.headers.get("retry-after")
                    row["server"] = parse_server_timing(resp.headers.get("server-timing"))
                    if resp.status_code == 200:
                        try:
                            doc = json.loads(payload)
                            row["boxes"] = len(doc["boxes"])
                            row["timing_ms"] = doc["timing_ms"]
                            row["output_sha256"] = canonical_output(doc)
                        except (ValueError, KeyError, TypeError):
                            row["error"] = "MalformedResponse"  # a 200 without a valid body fails
                    else:
                        try:
                            row["type"] = json.loads(payload).get("type")
                        except ValueError:
                            row["type"] = None
                except httpx.HTTPError as exc:
                    row["latency_ms"] = (time.perf_counter() - t0) * 1000
                    row["status"] = 0
                    row["error"] = type(exc).__name__
                row["t_end_s"] = time.perf_counter() - t_start
                rows.append(row)
                if args.honor_retry_after and row.get("retry_after"):
                    with contextlib.suppress(ValueError):
                        wait = min(float(row["retry_after"]), 10.0)
                        if end_at is not None:
                            wait = min(wait, max(0.0, end_at - time.perf_counter()))
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(stop.wait(), timeout=wait)

        tasks = [asyncio.ensure_future(worker(w)) for w in range(args.concurrency)]
        hard = (args.duration + args.timeout + 5) if timed else None
        done, pending = await asyncio.wait(tasks, timeout=hard)
        for t in pending:  # only if the server hangs past the client timeout
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            t.result()  # surface client bugs instead of hiding them
        if pending:
            state["reason"] = "hard_stop"
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.remove_signal_handler(sig)
    return rows, time.perf_counter() - t_start, state["reason"]


def check_outputs(
    rows: list[dict[str, Any]], reference: dict[str, Any] | None, hashes: list[str]
) -> dict[str, Any]:
    """Consistency of outputs per input within the run, and against the reference if given."""
    per_input: dict[int, set[str]] = {}
    for r in rows:
        if "output_sha256" in r:
            per_input.setdefault(r["input"], set()).add(r["output_sha256"])
    out: dict[str, Any] = {
        "checked": sum(1 for r in rows if "output_sha256" in r),
        "inconsistent_inputs": sorted(k for k, v in per_input.items() if len(v) > 1),
    }
    if reference is not None:
        if reference.get("body_sha256") != hashes:
            out["reference_error"] = "reference was recorded for different inputs"
            out["mismatches"] = None
        else:
            expected = reference["outputs"]
            out["mismatches"] = sum(
                1
                for r in rows
                if "output_sha256" in r and r["output_sha256"] != expected[r["input"]]
            )
    return out


def stats(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    a = np.asarray(values)
    out: dict[str, Any] = {
        "n": len(a),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
    }
    boot = rng.choice(a, size=(2000, len(a)), replace=True)
    for q in (50, 95, 99):
        out[f"p{q}"] = float(np.percentile(a, q, method="inverted_cdf"))
        lo, hi = np.percentile(np.percentile(boot, q, axis=1, method="inverted_cdf"), [2.5, 97.5])
        out[f"p{q}_ci95"] = [float(lo), float(hi)]
    return out


def outcome(r: dict[str, Any]) -> str:
    """ok | rejected (503 queue_full: expected admission control) | http_error | transport_error |
    malformed. Only "ok" counts as a success."""
    if r["status"] == 0:
        return "transport_error"
    if r["status"] == 200:
        return "malformed" if r.get("error") else "ok"
    if r["status"] == 503 and r.get("type") == "/errors/queue_full":
        return "rejected"
    return "http_error"


def summarize(
    rows: list[dict[str, Any]], args: argparse.Namespace, inputs: dict[str, Any]
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    measured = [r for r in rows if not r["warmup"]]
    ok = [r for r in measured if outcome(r) == "ok"]
    outcomes: dict[str, int] = {}
    for r in measured:
        outcomes[outcome(r)] = outcomes.get(outcome(r), 0) + 1
    window = (
        (max(r["t_end_s"] for r in measured) - min(r["t_start_s"] for r in measured))
        if measured
        else 0.0
    )
    statuses: dict[str, int] = {}
    for r in measured:
        key = (
            str(r["status"])
            + (f" {r.get('type')}" if r.get("type") else "")
            + (f" {r.get('error')}" if r.get("error") else "")
        )
        statuses[key] = statuses.get(key, 0) + 1
    stages = sorted({k for r in ok for k in r.get("server", {})})
    rejected = [r for r in measured if outcome(r) == "rejected"]
    completed = [r for r in measured if r["status"] != 0]
    return {
        "mode": "duration" if getattr(args, "duration", None) is not None else "requests",
        "stop_reason": getattr(args, "stop_reason", "requests"),
        "attempts_total": len(rows),
        "accounting": {
            "attempts": len(measured),
            "completed": len(completed),
            "rejected": len(rejected),
            "accepted": len(completed) - len(rejected),
            "success": len(ok),
        },
        "attempt_rate_rps": len(measured) / window if window > 0 else 0.0,
        "latency_rejected_ms": stats([r["latency_ms"] for r in rejected], rng),
        "label": args.label,
        "url": args.url,
        "concurrency": args.concurrency,
        "honor_retry_after": args.honor_retry_after,
        "requests_measured": len(measured),
        "warmup": args.warmup,
        "inputs": inputs,
        "statuses": statuses,
        "outcomes": outcomes,
        "queue_full_retry_after": sorted(
            {str(r.get("retry_after")) for r in measured if outcome(r) == "rejected"}
        ),
        "success": len(ok),
        "success_rate": len(ok) / len(measured) if measured else 0.0,
        "retry_after_values": sorted({r["retry_after"] for r in measured if r.get("retry_after")}),
        "window_s": window,
        "throughput_rps": len(ok) / window if window > 0 else 0.0,
        "latency_ms": stats([r["latency_ms"] for r in ok], rng),
        "server_stages_ms": {
            s: stats([r["server"][s] for r in ok if s in r["server"]], rng) for s in stages
        },
        "boxes_per_response": stats([float(r["boxes"]) for r in ok], rng),
        "client": {
            "python": platform.python_version(),
            "httpx": httpx.__version__,
            "host_cpus": __import__("os").cpu_count(),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument(
        "--honor-retry-after",
        action="store_true",
        help="after a response with Retry-After, wait that long (max 10 s) before the next request",
    )
    p.add_argument("--inputs", default="synthetic:6")
    p.add_argument("--bevdet-root")
    p.add_argument("--out", required=True)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--duration", type=float, help="time-bounded mode: seconds (needs --max-attempts)"
    )
    p.add_argument(
        "--max-attempts", type=int, help="time-bounded mode: cap on attempts incl. warm-up"
    )
    p.add_argument("--write-reference", help="store the output hash of every input (one each)")
    p.add_argument("--reference", help="compare every 200 response with this reference")
    args = p.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        raise SystemExit("concurrency and requests must be >= 1")
    if (args.duration is None) != (args.max_attempts is None):
        raise SystemExit("--duration and --max-attempts go together (both limits are required)")
    if args.duration is not None and (args.duration <= 0 or args.max_attempts < 1):
        raise SystemExit("--duration must be > 0 and --max-attempts >= 1")
    bodies, inputs = load_inputs(args.inputs, args.bevdet_root)
    inputs["mean_request_bytes"] = float(np.mean([len(b) for b, _ in bodies]))
    hashes = body_hashes(bodies)
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    cpu0, wall0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    rows, _, args.stop_reason = asyncio.run(run(args, bodies))
    cpu1, wall = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter() - wall0
    client_cpu = (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.requests.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sorted(rows, key=lambda r: r["i"])) + "\n"
    )
    summary = summarize(rows, args, inputs)
    summary["outputs"] = check_outputs(rows, reference, hashes)
    # the client is one asyncio thread: ~100% here means the load generator itself was saturated
    summary["client_process"] = {
        "cpu_s": round(client_cpu, 3),
        "wall_s": round(wall, 3),
        "cpu_pct": round(100 * client_cpu / wall, 1) if wall > 0 else None,
    }
    if args.write_reference:
        per: dict[int, set[str]] = {}
        for r in rows:
            if "output_sha256" in r:
                per.setdefault(r["input"], set()).add(r["output_sha256"])
        if sorted(per) != list(range(len(bodies))) or any(len(v) != 1 for v in per.values()):
            summary["outputs"]["reference_error"] = "every input needs exactly one output hash"
        else:
            Path(args.write_reference).write_text(json.dumps({
                "inputs": args.inputs, "body_sha256": hashes,
                "outputs": [next(iter(per[k])) for k in range(len(bodies))],
            }, indent=2))  # fmt: skip
    (out / f"{args.label}.summary.json").write_text(json.dumps(summary, indent=2))
    lat = summary["latency_ms"]
    print(
        json.dumps(
            {
                "label": args.label,
                "c": args.concurrency,
                "ok": summary["success"],
                "measured": summary["requests_measured"],
                "statuses": summary["statuses"],
                "rps": round(summary["throughput_rps"], 2),
                "p50": round(lat.get("p50", 0), 1),
                "p95": round(lat.get("p95", 0), 1),
                "p99": round(lat.get("p99", 0), 1),
            }
        )
    )
    return 3 if args.stop_reason in ("interrupted", "hard_stop") else 0


if __name__ == "__main__":
    sys.exit(main())
