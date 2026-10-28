"""tools/bench/multiload.py against a local fake server (CPU only): caps, pacing, failures."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from harness import FAKES, ROOT, SUPPORTED

pytestmark = pytest.mark.skipif(not SUPPORTED, reason="needs Linux with curl, ss and setsid")


def _module() -> ModuleType:
    sys.path.insert(0, str(ROOT / "tools/bench"))
    spec = importlib.util.spec_from_file_location("multiload", ROOT / "tools/bench/multiload.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ml = _module()


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {
        **os.environ,
        "BEVSERVE_PORT": str(port),
        "BEVSERVE_MAX_IN_FLIGHT": "5",
        "BEVSERVE_SERVER_TIMING": "true",
        "BEVSERVE_PREPROCESS_THREADS": "12",
    }
    proc = subprocess.Popen(
        [sys.executable, str(FAKES / "fake_server.py")],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            import urllib.request

            if urllib.request.urlopen(f"{url}/readyz", timeout=1).status == 200:
                break
        except OSError:
            time.sleep(0.1)
    yield url
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=10)


def run(
    url: str, out: Path, *extra: str, env: dict[str, str] | None = None, label: str = "m"
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(ROOT / "tools/bench/multiload.py"),
        "--url",
        url,
        "--out",
        str(out),
        "--label",
        label,
        "--inputs",
        "synthetic:2",
        *extra,
    ]
    return subprocess.run(
        cmd, env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=120
    )


def summary(out: Path, label: str = "m") -> dict:
    return json.loads((out / f"{label}.summary.json").read_text())


def test_shares_and_in_flight() -> None:
    assert ml.shares(32, 4) == [8, 8, 8, 8] and ml.shares(5, 4) == [2, 1, 1, 1]
    rows = [{"t_unix": 0.0, "t_end_unix": 2.0}, {"t_unix": 1.0, "t_end_unix": 2.0}]
    assert ml.in_flight(rows) == {"mean": 1.5, "max": 2}


def test_aggregate_attempt_cap_is_exact_and_metrics_aggregate(server: str, tmp_path: Path) -> None:
    p = run(
        server,
        tmp_path,
        "--workers",
        "4",
        "--concurrency",
        "32",
        "--duration",
        "8",
        "--max-attempts",
        "120",
        "--rate",
        "200",
    )
    assert p.returncode == 0, p.stderr[-2000:]
    s = summary(tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "m.requests.jsonl").read_text().splitlines()]
    assert s["stop_reason"] == "max_attempts" and len(rows) == s["attempts_total"] == 120
    assert sorted(r["i"] for r in rows) == list(range(120)) and sorted(
        r["attempt"] for r in rows
    ) == list(range(120))
    m = s["multiprocess"]
    assert sum(w["attempts"] for w in m["per_worker"]) == 120 and len({r["pid"] for r in rows}) == 4
    assert (
        sum(w["ok"] + w["rejected"] for w in m["per_worker"])
        == s["accounting"]["success"] + s["accounting"]["rejected"]
    )
    assert m["in_flight"]["max"] <= 32 and all(
        w["cpu_pct_of_active_time"] is not None for w in m["per_worker"]
    )
    assert 0 < m["achieved_attempt_rate_rps"] <= 200 * 1.1
    assert set(s["outcomes"]) <= {"ok", "rejected"}


def test_duration_bound_and_closed_loop_baseline(server: str, tmp_path: Path) -> None:
    t0 = time.time()
    p = run(
        server,
        tmp_path,
        "--workers",
        "4",
        "--concurrency",
        "5",
        "--duration",
        "2",
        "--max-attempts",
        "100000",
        "--honor-retry-after",
    )
    assert p.returncode == 0 and time.time() - t0 < 20, p.stderr[-2000:]
    s = summary(tmp_path)
    assert s["stop_reason"] == "duration" and s["multiprocess"]["shares"] == [2, 1, 1, 1]
    assert s["outcomes"] == {
        "ok": s["accounting"]["attempts"]
    }  # 5 in flight never exceed the limit
    assert s["multiprocess"]["in_flight"]["max"] <= 5


def test_worker_failure_invalidates_the_run(server: str, tmp_path: Path) -> None:
    p = run(
        server,
        tmp_path,
        "--workers",
        "4",
        "--concurrency",
        "8",
        "--duration",
        "3",
        "--max-attempts",
        "200",
        env={"BENCH_TEST_FAIL_WORKER": "2"},
    )
    assert p.returncode == 4
    s = summary(tmp_path)
    assert s["stop_reason"] == "worker_failed" and s["multiprocess"]["failed_workers"] == [2]
    assert not any(
        r["worker"] == 2
        for r in map(json.loads, (tmp_path / "m.requests.jsonl").read_text().splitlines())
    )


def test_interrupt_stops_all_workers(server: str, tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "tools/bench/multiload.py"),
        "--url",
        server,
        "--out",
        str(tmp_path),
        "--label",
        "m",
        "--inputs",
        "synthetic:2",
        "--workers",
        "4",
        "--concurrency",
        "8",
        "--duration",
        "30",
        "--max-attempts",
        "100000",
        "--rate",
        "50",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    time.sleep(4)
    os.killpg(proc.pid, signal.SIGINT)  # Ctrl-C reaches the whole process group
    proc.communicate(timeout=60)
    assert proc.returncode == 3
    s = summary(tmp_path)
    assert s["stop_reason"] == "interrupted" and 0 < s["attempts_total"] < 1500
    left = subprocess.run(
        ["pgrep", "-s", str(proc.pid)], capture_output=True, text=True
    ).stdout.split()
    assert left == []
