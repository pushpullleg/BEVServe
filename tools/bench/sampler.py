"""Resource sampler for one server process (M5), unprivileged: /proc and nvidia-smi only.

    python tools/bench/sampler.py --pid PID --out samples.csv --meta sampler.json [--interval 1.0]

Runs until terminated (SIGTERM/SIGINT: exit 0) or until the process disappears (exit 3: the
serving process died during sampling). Columns: wall time, process RSS (MiB), process CPU % (all
threads; 100 = one core), threads, GPU utilization % and memory used (whole GPU), the GPU memory of
the serving process (nvidia-smi compute apps whose PID is the target or one of its descendants,
host PID namespace), and gpu_status: ok | query_failed | not_listed | unavailable. A value that
nvidia-smi cannot provide is left empty (missing), never written as zero.

For a container, pass the host PID of its main process (`docker inspect -f '{{.State.Pid}}'`).
The --meta sidecar records the target, the GPU PIDs attributed to it, the sample count and the exit
reason, so a crashed sampler or unattributed GPU memory is visible to the acceptance criteria.
NVIDIA_SMI overrides the nvidia-smi command (tests).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CLK = os.sysconf("SC_CLK_TCK")
NVIDIA_SMI = os.environ.get("NVIDIA_SMI", "nvidia-smi")
EXIT_TARGET_GONE = 3


def proc_sample(pid: int) -> tuple[float, int, float] | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            status = dict(line.split(":", 1) for line in f if ":" in line)
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    rss_mib = int(status["VmRSS"].split()[0]) / 1024
    threads = int(status["Threads"])
    cpu_ticks = int(fields[11]) + int(fields[12])  # utime + stime
    return rss_mib, threads, cpu_ticks / CLK


def process_tree(pid: int) -> set[int]:
    """The target and all its descendants (host PIDs), from /proc parent links."""
    parents: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as f:
                    parents[int(entry)] = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
    tree, frontier = {pid}, [pid]
    while frontier:
        parent = frontier.pop()
        children = [c for c, p in parents.items() if p == parent and c not in tree]
        tree.update(children)
        frontier.extend(children)
    return tree


def number(text: str) -> str:
    """nvidia-smi value or "" when not a number ("[N/A]", "[Not Supported]", empty)."""
    try:
        float(text)
        return text.strip()
    except ValueError:
        return ""


def smi(*query: str) -> list[list[str]]:
    out = subprocess.run(
        [NVIDIA_SMI, *query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def gpu_sample(tree: set[int]) -> tuple[str, str, str, str, list[int]]:
    """(util %, GPU memory used MiB, serving-process GPU MiB, status, attributed GPU pids)."""
    try:
        util, used = smi("--query-gpu=utilization.gpu,memory.used")[0][:2]
        apps = smi("--query-compute-apps=pid,used_memory")
    except (subprocess.SubprocessError, OSError, IndexError, ValueError):
        return "", "", "", "query_failed", []
    own = [(int(a[0]), number(a[1])) for a in apps if len(a) >= 2 and a[0].isdigit()]
    own = [(p, m) for p, m in own if p in tree]
    if not own:
        return number(util), number(used), "", "not_listed", []
    if any(m == "" for _, m in own):
        return number(util), number(used), "", "unavailable", [p for p, _ in own]
    total = sum(float(m) for _, m in own)
    return number(util), number(used), f"{total:.0f}", "ok", [p for p, _ in own]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    meta: dict[str, object] = {
        "target_pid": args.pid,
        "interval_s": args.interval,
        "t_start_unix": time.time(),
    }
    statuses: dict[str, int] = {}
    gpu_pids: set[int] = set()
    n = 0
    reason = "stopped"
    with open(args.out, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(
            [
                "t_unix",
                "rss_mib",
                "cpu_pct",
                "threads",
                "gpu_util_pct",
                "gpu_mem_used_mib",
                "proc_gpu_mem_mib",
                "gpu_status",
            ]
        )
        prev = proc_sample(args.pid)
        prev_t = time.monotonic()
        if prev is None:
            reason = "target_not_found"
        while not stop["flag"] and prev is not None:
            time.sleep(args.interval)
            if stop["flag"]:
                break
            cur = proc_sample(args.pid)
            if cur is None:
                reason = "target_exited"
                break
            now = time.monotonic()
            cpu_pct = 100 * (cur[2] - prev[2]) / (now - prev_t)
            util, used, own, status, pids = gpu_sample(process_tree(args.pid))
            statuses[status] = statuses.get(status, 0) + 1
            gpu_pids.update(pids)
            out.writerow(
                [
                    f"{time.time():.3f}",
                    f"{cur[0]:.1f}",
                    f"{cpu_pct:.1f}",
                    cur[1],
                    util,
                    used,
                    own,
                    status,
                ]
            )
            f.flush()
            n += 1
            prev, prev_t = cur, now
    meta.update(
        t_stop_unix=time.time(),
        samples=n,
        gpu_status_counts=statuses,
        gpu_pids=sorted(gpu_pids),
        exit_reason=reason,
    )
    Path(args.meta).write_text(json.dumps(meta, indent=2))
    return 0 if reason == "stopped" else EXIT_TARGET_GONE


if __name__ == "__main__":
    sys.exit(main())
