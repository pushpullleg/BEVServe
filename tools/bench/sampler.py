"""Resource sampler for one server process (M5), unprivileged: /proc and nvidia-smi only.

    python tools/bench/sampler.py --pid PID --out samples.csv --meta sampler.json [--interval 1.0]

Runs until terminated (SIGTERM/SIGINT: exit 0) or until the process disappears (exit 3: the
serving process died during sampling). Columns: wall time, process RSS (MiB), process CPU % (all
threads; 100 = one core), threads, GPU utilization % and memory used (whole GPU), the GPU memory of
the serving process (nvidia-smi compute apps whose PID is the target or one of its descendants,
host PID namespace), and gpu_status: ok | query_failed | not_listed | unavailable. A value that
nvidia-smi cannot provide is left empty (missing), never written as zero. M7 adds tree_threads
(threads of the target and its descendants) and the container's cgroup pids.current / pids.max
(tasks = threads, the quantity Compose's `pids` limit bounds; empty if not readable).
M7 diagnostic adds per-thread CPU of the target process: main_thread_cpu_pct (the thread whose id
equals the process id; uvicorn's event loop runs there) and max_thread_cpu_pct (the busiest single
thread in the interval). 100 = one core. They show load per thread, not lock contention.

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


def thread_ticks(pid: int) -> dict[int, int]:
    """utime + stime (clock ticks) of every thread of the process."""
    out: dict[int, int] = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            out[int(tid)] = int(fields[11]) + int(fields[12])
        except (OSError, IndexError, ValueError):
            continue
    return out


def thread_cpu(
    pid: int, before: dict[int, int], after: dict[int, int], seconds: float
) -> tuple[str, str]:
    """(main-thread CPU %, busiest-thread CPU %) over an interval, or ("", "") if unknown."""
    if not after or seconds <= 0:
        return "", ""
    deltas = {t: after[t] - before.get(t, after[t]) for t in after}
    main = deltas.get(pid)
    busiest = max(deltas.values(), default=None)
    pct = lambda ticks: f"{100 * ticks / CLK / seconds:.1f}"  # noqa: E731
    return (pct(main) if main is not None else ""), (pct(busiest) if busiest is not None else "")


def tree_threads(tree: set[int]) -> int | None:
    total = 0
    for pid in tree:
        try:
            with open(f"/proc/{pid}/status") as f:
                total += int(next(line.split()[1] for line in f if line.startswith("Threads:")))
        except (OSError, StopIteration, ValueError):
            continue
    return total or None


def cgroup_pids(pid: int) -> tuple[str, str]:
    """(pids.current, pids.max) of the target's cgroup v2, or ("", "") if unavailable."""
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            path = next(line.strip().split("::", 1)[1] for line in f if line.startswith("0::"))
        base = f"/sys/fs/cgroup{path}"
        with open(f"{base}/pids.current") as f:
            current = f.read().strip()
        with open(f"{base}/pids.max") as f:
            limit = f.read().strip()
        return current, limit
    except (OSError, StopIteration, IndexError):
        return "", ""


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
                "tree_threads",
                "pids_current",
                "pids_max",
                "main_thread_cpu_pct",
                "max_thread_cpu_pct",
            ]
        )
        prev = proc_sample(args.pid)
        prev_threads = thread_ticks(args.pid)
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
            ticks = thread_ticks(args.pid)
            main_cpu, max_cpu = thread_cpu(args.pid, prev_threads, ticks, now - prev_t)
            tree = process_tree(args.pid)
            util, used, own, status, pids = gpu_sample(tree)
            threads_all = tree_threads(tree)
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
                    threads_all or "",
                    *cgroup_pids(args.pid),
                    main_cpu,
                    max_cpu,
                ]
            )
            f.flush()
            n += 1
            prev, prev_t, prev_threads = cur, now, ticks
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
