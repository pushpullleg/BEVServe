"""M6 CPU-only benchmark of decode + preprocessing: serial (M5) vs parallel camera pool.

Private tool (local nuScenes, no GPU). Preliminary: the GPU-container benchmark decides.

    nice -n 10 python tools/m6_cpu_bench.py --bevdet-root /path/to/BEVDet --out outputs/m6-cpu-<id>

Each (arm, repeat) runs in a fresh subprocess (`--arm serial|pool6|pool12|pool24`), so peak RSS
and thread counts are per arm; arms are interleaved across repeats so slow drift affects all of
them. Within a subprocess: concurrency 1, then concurrency 8 (closed loop, C client threads each
calling the API's decode + build function for successive frames). Reported per run: p50/p95
latency, throughput, peak RSS, peak thread count and process CPU time. Inputs are the 20 val frames
M5 used, read locally; only timings are written.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

ARMS = ("serial", "pool6", "pool12", "pool24")


def threads_now() -> int:
    with open("/proc/self/status") as f:
        return int(next(line.split()[1] for line in f if line.startswith("Threads:")))


def run_arm(arm: str, bevdet_root: str, per_client: int) -> dict[str, Any]:
    from profile_preprocess import jpeg_sets

    from bevserve.api import _decode_and_build, _decode_and_build_parallel
    from bevserve.camera_pool import CameraPool
    from bevserve.schemas import CalibrationRequest

    sets = [
        (j, CalibrationRequest.model_validate(c))
        for j, c in jpeg_sets("nuscenes:even:20", bevdet_root)
    ]
    pool = CameraPool(int(arm[4:])) if arm.startswith("pool") else None

    def call(k: int) -> None:
        raw, calibration = sets[k % len(sets)]
        if pool is None:
            _decode_and_build(raw, calibration)
        else:
            _decode_and_build_parallel(pool, raw, calibration)

    peak = {"threads": threads_now()}
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            peak["threads"] = max(peak["threads"], threads_now())
            time.sleep(0.01)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    for k in range(5):  # warm-up
        call(k)
    out: dict[str, Any] = {"arm": arm}
    for c in (1, 8):
        latencies: list[float] = []
        lock = threading.Lock()

        def client(w: int, latencies: list[float] = latencies, lock: threading.Lock = lock) -> None:
            for j in range(per_client):
                t0 = time.perf_counter()
                call(w * per_client + j)
                with lock:
                    latencies.append((time.perf_counter() - t0) * 1000)

        cpu0, t0 = time.process_time(), time.perf_counter()
        with cf.ThreadPoolExecutor(c) as clients:
            list(clients.map(client, range(c)))
        wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
        latencies.sort()
        out[f"c{c}"] = {
            "requests": len(latencies),
            "p50_ms": round(statistics.median(latencies), 2),
            "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 2),
            "throughput_rps": round(len(latencies) / wall, 2),
            "cpu_cores_used": round(cpu / wall, 2),
        }
    stop.set()
    if pool is not None:
        pool.shutdown()
    out["peak_threads"] = peak["threads"]
    out["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bevdet-root", required=True)
    p.add_argument("--out")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--per-client", type=int, default=30)
    p.add_argument("--arm", choices=ARMS)  # internal: one arm in this process
    a = p.parse_args()
    if a.arm:
        print(json.dumps(run_arm(a.arm, a.bevdet_root, a.per_client)))
        return 0
    os.umask(0o077)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    runs = []
    for r in range(a.repeats):
        order = ARMS if r % 2 == 0 else tuple(reversed(ARMS))  # balanced order across repeats
        for arm in order:
            res = subprocess.run(
                [sys.executable, __file__, "--arm", arm, "--bevdet-root", a.bevdet_root,
                 "--per-client", str(a.per_client)],
                capture_output=True, text=True, check=True,
            )  # fmt: skip
            run = json.loads(res.stdout.strip().splitlines()[-1]) | {"repeat": r}
            runs.append(run)
            print(json.dumps(run), flush=True)
    summary: dict[str, Any] = {}
    for arm in ARMS:
        mine = [x for x in runs if x["arm"] == arm]
        summary[arm] = {
            f"c{c}_{k}": [x[f"c{c}"][k] for x in mine]
            for c in (1, 8)
            for k in ("p50_ms", "p95_ms", "throughput_rps", "cpu_cores_used")
        } | {
            "peak_threads": [x["peak_threads"] for x in mine],
            "max_rss_mib": [x["max_rss_mib"] for x in mine],
        }
    (out / "runs.json").write_text(json.dumps({"runs": runs, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
