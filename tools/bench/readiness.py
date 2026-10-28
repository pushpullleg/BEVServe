"""Readiness poller (M7): GET /readyz at a fixed interval until terminated.

    python tools/bench/readiness.py --url http://127.0.0.1:8000 --out ready.csv [--interval 1]

Columns: t_unix, http_status (0 = no response within --timeout or connection error), status (the
body's readiness status, empty if none), latency_ms. A probe that times out counts as not ready:
an orchestrator would treat it the same way. Exits 0 on SIGTERM/SIGINT.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
import urllib.error
import urllib.request


def probe(url: str, timeout: float) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{url}/readyz", timeout=timeout) as resp:
            code, body = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        code, body = exc.code, exc.read()
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, "", (time.perf_counter() - t0) * 1000
    try:
        status = str(json.loads(body).get("status", ""))
    except ValueError:
        status = ""
    return code, status, (time.perf_counter() - t0) * 1000


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=2.0)
    a = p.parse_args()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    with open(a.out, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["t_unix", "http_status", "status", "latency_ms"])
        while not stop["flag"]:
            t = time.time()
            code, status, ms = probe(a.url, a.timeout)
            out.writerow([f"{t:.3f}", code, status, f"{ms:.1f}"])
            f.flush()
            time.sleep(max(0.0, a.interval - (time.time() - t)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
