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
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import platform
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


def parse_server_timing(header: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in (header or "").split(","):
        if ";dur=" in item:
            name, dur = item.strip().split(";dur=")
            out[name] = float(dur)
    return out


async def run(
    args: argparse.Namespace, bodies: list[tuple[bytes, str]]
) -> tuple[list[dict[str, Any]], float]:
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    timeout = httpx.Timeout(args.timeout)
    rows: list[dict[str, Any]] = []
    counter = {"next": 0}
    total = args.warmup + args.requests

    async with httpx.AsyncClient(base_url=args.url, limits=limits, timeout=timeout) as client:
        t_start = time.perf_counter()

        async def worker(wid: int) -> None:
            while True:
                i = counter["next"]
                if i >= total:
                    return
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
                        await asyncio.sleep(min(float(row["retry_after"]), 10.0))

        await asyncio.gather(*(worker(w) for w in range(args.concurrency)))
    return rows, time.perf_counter() - t_start


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
    return {
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
    args = p.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        raise SystemExit("concurrency and requests must be >= 1")
    bodies, inputs = load_inputs(args.inputs, args.bevdet_root)
    inputs["mean_request_bytes"] = float(np.mean([len(b) for b, _ in bodies]))
    rows, _ = asyncio.run(run(args, bodies))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.requests.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sorted(rows, key=lambda r: r["i"])) + "\n"
    )
    summary = summarize(rows, args, inputs)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
