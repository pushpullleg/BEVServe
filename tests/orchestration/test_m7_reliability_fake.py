"""M7 reliability stages (tools/m5_bench.sh rel-*) and fault retest, against fake Docker / GPU.

Positive runs must pass end to end; every injected failure class must fail closed: wrong model
outputs, unexpected HTTP statuses, readiness loss, thread and memory budgets, failed golden,
interruption, and gates that would let a failed or test-mode run authorize the next experiment.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from harness import FAKES, IMAGE_ID, ROOT, SUPPORTED, m6, record_fake_golden, records, run_bench

pytestmark = pytest.mark.skipif(not SUPPORTED, reason="needs Linux with curl, ss and setsid")

FAST = {"REL_BASE_S": "5", "REL_PHASE_S": "5", "REL_MAX_ATTEMPTS": "400"}


def rel(env: dict[str, str], stage: str = "rel-admission", **extra: str) -> dict[str, str]:
    return {**m6(env), "STAGES": stage, "MODE": "pilot", **FAST, **extra}


def passing(env: dict[str, str]) -> dict[str, str]:
    record_fake_golden(Path(env["GOLDEN_DIR"]) / "http_golden.json")
    return {"BENCH_TEST_MODE": "1", "GOLDEN_TOOL": str(FAKES / "golden_http.py")}


def crit(out: Path) -> dict:
    return json.loads((out / "criteria.json").read_text())


def counted_failures(out: Path) -> dict[str, str]:
    return {
        k: v["status"]
        for k, v in crit(out).items()
        if isinstance(v, dict) and v["counted"] and v["status"] != "pass"
    }


def test_rel_admission_pilot_passes_end_to_end(env: dict[str, str]) -> None:
    code, log, out = run_bench(rel(env, **passing(env), REL_OVERLOAD="8:honor 8:noretry"))
    assert out is not None and (out / "criteria.json").exists(), log
    assert code == 0 and "RESULT: all counted criteria passed (pilot)" in log, (
        counted_failures(out),
        log[-3000:],
    )
    recs = records(out)
    services = [
        (r["stage"], r["preprocess_threads_env"], r["max_in_flight"])
        for r in recs
        if r["kind"] == "service"
    ]
    assert services == [("rel-ref", 0, 5), ("rel-admission", 12, 5)]
    assert all(r["ok"] for r in recs if r["kind"] == "cleanup")
    assert [r["label"] for r in recs if r["kind"] == "recovery"] == [
        "rel-c8-honor",
        "rel-c8-noretry",
    ]
    s = json.loads((out / "rel-c8-noretry.summary.json").read_text())
    acct = s["accounting"]
    assert acct["attempts"] == acct["completed"] == acct["accepted"] + acct["rejected"]
    assert acct["success"] == acct["accepted"] and s["stop_reason"] in ("duration", "max_attempts")
    assert s["outputs"]["mismatches"] == 0 and s["outputs"]["checked"] == acct["success"]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["test_mode"] is True and meta["params"]["rel_threads"] == 12
    assert (out / "rel-admission.readiness.csv").exists()


def test_failed_golden_fails_the_run(env: dict[str, str]) -> None:
    code, _, out = run_bench(rel(env, REL_OVERLOAD="8:honor"))
    assert code == 1 and out is not None
    assert counted_failures(out) == {
        "rel-admission: golden parity (all frames byte-identical) after the stage": "fail"
    }


def test_wrong_model_outputs_are_detected(env: dict[str, str]) -> None:
    code, _, out = run_bench(
        rel(env, **passing(env), REL_OVERLOAD="8:honor", FAKE_WRONG_OUTPUT_AFTER="10")
    )
    assert code == 1 and out is not None
    failures = counted_failures(out)
    assert any("equals the serial reference" in name for name in failures), failures


@pytest.mark.parametrize(
    ("knobs", "reason"),
    [
        ({"FAKE_FAIL_EVERY": "7"}, "unexpected outcomes"),
        ({"FAKE_LEAK_THREADS": "3", "REL_THREADS_MAX": "60"}, "server threads"),
        ({"FAKE_RSS_LEAK_MIB": "4", "REL_RSS_MAX_MIB": "320"}, "server RSS"),
        ({"FAKE_NOT_READY_AFTER": "20"}, "stop condition reached"),
    ],
    ids=["unexpected-5xx", "thread-budget", "rss-budget", "readiness-lost"],
)
def test_stop_conditions_abort_and_clean_up(
    env: dict[str, str], knobs: dict[str, str], reason: str
) -> None:
    code, log, out = run_bench(rel(env, REL_OVERLOAD="8:honor", **knobs))
    assert code == 1 and "stop condition reached" in log and reason in log, log[-2000:]
    assert "RESULT: run aborted" in log
    assert out is not None and all(r["ok"] for r in records(out) if r["kind"] == "cleanup")


def test_interrupt_stops_clients_and_cleans_up(env: dict[str, str]) -> None:
    proc = subprocess.Popen(
        ["bash", str(ROOT / "tools/m5_bench.sh")],
        env=rel(env, REL_PHASE_S="30", REL_OVERLOAD="8:noretry"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.time() + 120
    while time.time() < deadline:  # wait until the overload phase is running
        outs = sorted((ROOT / "outputs").glob("m5-2*"))
        if outs and (outs[-1] / "rel-c8-noretry.client.txt").exists():
            break
        time.sleep(0.2)
    time.sleep(2)
    os.killpg(proc.pid, signal.SIGINT)  # what Ctrl-C does: the whole process group
    log, _ = proc.communicate(timeout=120)
    assert proc.returncode != 0 and "interrupted" in log, log[-2000:]
    out = sorted((ROOT / "outputs").glob("m5-2*"))[-1]
    assert all(r["ok"] for r in records(out) if r["kind"] == "cleanup")
    ps = subprocess.run(
        ["pgrep", "-f", f"--label rel-c8-noretry .*{out.name}"], capture_output=True, text=True
    )
    left = subprocess.run(
        ["pgrep", "-s", str(proc.pid)], capture_output=True, text=True
    ).stdout.split()
    assert ps.stdout.strip() == "" and left == [], (ps.stdout, left)


def _record_run(
    d: Path, mode: str, stage: str, status: str = "pass", test_mode: bool = False
) -> Path:
    d.mkdir()
    meta = {"test_mode": test_mode, "image": {"Id": IMAGE_ID},
            "params": {"mode": mode, "stages": [stage], "rel_threads": 12}}  # fmt: skip
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "criteria.json").write_text(json.dumps({"x": {"status": status, "counted": True}}))
    return d


def _record_fault(d: Path, test_mode: bool = False) -> Path:
    d.mkdir()
    (d / "run.json").write_text(
        json.dumps({"image_id": IMAGE_ID, "preprocess_threads": 12, "test_mode": test_mode})
    )
    lines = [
        f"PASS image id is the verified {IMAGE_ID}",
        "PASS container runs with BEVSERVE_PREPROCESS_THREADS=12",
        "PASS process exited with code 70 (docker die event, then restart; restart count 0 -> 1)",
        "PASS ready again after restart (1 s)",
        "PASS request after recovery -> 200",
        "PASS no further restarts (restart count 1 after 10 s)",
        "PASS clean shutdown with BEVSERVE_PREPROCESS_THREADS=12 (exit 0 in 1 s)",
    ]
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    return d


def test_gates_refuse_failed_or_test_mode_runs(env: dict[str, str], tmp_path: Path) -> None:
    calls = Path(env["FAKE_STATE"], "calls.log")

    def refused(extra: dict[str, str], message: str) -> None:
        code, log, _ = run_bench(rel(env, **extra))
        assert code == 1 and message in log, log[-1500:]
        assert "compose" not in (calls.read_text() if calls.exists() else "")

    good_c1 = _record_run(tmp_path / "c1", "pilot", "rel-admission")
    good_fault = _record_fault(tmp_path / "fault")
    refused({"MODE": "full"}, "C2 needs PILOT_DIR")
    refused({"MODE": "full", "PILOT_DIR": str(good_c1)}, "C2 needs FAULT_DIR")
    failed_c1 = _record_run(tmp_path / "c1-failed", "pilot", "rel-admission", status="fail")
    refused(
        {"MODE": "full", "PILOT_DIR": str(failed_c1), "FAULT_DIR": str(good_fault)},
        "earlier experiment did not pass",
    )
    fake_c1 = _record_run(tmp_path / "c1-test", "pilot", "rel-admission", test_mode=True)
    refused(
        {"MODE": "full", "PILOT_DIR": str(fake_c1), "FAULT_DIR": str(good_fault)},
        "earlier experiment did not pass",
    )
    fake_fault = _record_fault(tmp_path / "fault-test", test_mode=True)
    refused(
        {"MODE": "full", "PILOT_DIR": str(good_c1), "FAULT_DIR": str(fake_fault)},
        "fault retest did not pass",
    )
    refused({"STAGES": "rel-sustained"}, "C3 needs PRIOR_DIR")
    wrong_prior = _record_run(tmp_path / "c2-as-pilot", "pilot", "rel-admission")
    refused(
        {"STAGES": "rel-sustained", "PRIOR_DIR": str(wrong_prior)},
        "earlier experiment did not pass",
    )
    refused({"STAGES": "rel-sustained", "MODE": "full"}, "needs SUSTAINED_AUTH=1")
    refused({"STAGES": "rel-admission levels"}, "a rel stage must be the only stage")
    refused({"REL_RSS_MAX_MIB": "9000"}, "REL_RSS_MAX_MIB=9000 must be an integer in [256, 8192]")
    refused({"REL_OVERLOAD": "64:noretry"}, "REL_OVERLOAD entries")


def test_rel_sustained_pilot_runs_blocks_and_analysis(env: dict[str, str], tmp_path: Path) -> None:
    prior = _record_run(tmp_path / "c2", "full", "rel-admission")  # stands in for a passing C2
    timing = {"REL_SUSTAINED_S": "60", "REL_BLOCK_S": "30", "REL_WARMUP_S": "30"}
    code, log, out = run_bench(
        rel(env, "rel-sustained", **passing(env), PRIOR_DIR=str(prior), **timing)
    )
    assert out is not None and (out / "sustained.json").exists(), log[-3000:]
    assert code == 0, (counted_failures(out), log[-3000:])
    golden = [r for r in records(out) if r["kind"] == "golden"]
    assert [g["label"] for g in golden] == ["rel-sust-b1-golden", "rel-sust-b2-golden"]
    an = json.loads((out / "sustained.json").read_text())
    assert an["warmup_s"] == 30 and an["threads"]["post_range"] is not None
    stat = {
        k: v for k, v in crit(out).items() if isinstance(v, dict) and v["kind"] == "statistical"
    }
    assert stat and not any(v["counted"] for v in stat.values())  # C3 pilot: reported only
    assert json.loads((out / "gate-c2.json").read_text())["pass"]


def _fault_env(env: dict[str, str], **extra: str) -> dict[str, str]:
    return {**env, "DOCKER_CMD": str(FAKES / "docker_fault"), "PREPROCESS_THREADS": "12", **extra}


def _run_fault(env: dict[str, str]) -> tuple[int, str, Path]:
    p = subprocess.run(
        ["bash", str(ROOT / "tools/m5_fault_retest.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return p.returncode, p.stdout + p.stderr, sorted((ROOT / "outputs").glob("m5-fault-2*"))[-1]


def test_fault_retest_with_parallel_preprocessing(env: dict[str, str]) -> None:
    code, log, out = _run_fault(_fault_env(env))
    assert code == 0 and "RESULT: all checks passed" in log, log[-3000:]
    summary = (out / "summary.txt").read_text()
    for text in (
        "container runs with BEVSERVE_PREPROCESS_THREADS=12",
        "process exited with code 70",
        "ready again after restart",
        "request after recovery -> 200",
        "clean shutdown with BEVSERVE_PREPROCESS_THREADS=12 (exit 0",
    ):
        assert f"PASS {text}" in summary, summary
    run = json.loads((out / "run.json").read_text())
    assert run == {
        "run_id": run["run_id"],
        "image_id": IMAGE_ID,
        "preprocess_threads": 12,
        "test_mode": True,
    }


def test_fault_retest_fails_closed_on_wrong_runtime_threads(env: dict[str, str]) -> None:
    code, log, _ = _run_fault(_fault_env(env, FAKE_ENV_THREADS="0"))
    assert code == 1 and "BEVSERVE_PREPROCESS_THREADS=0, expected 12" in log, log[-2000:]


def test_rel_diag_runs_both_arms_with_server_timing(env: dict[str, str]) -> None:
    code, log, out = run_bench(rel(env, "rel-diag", **passing(env)))
    assert out is not None and (out / "diag.json").exists(), log[-3000:]
    assert code == 0 and "RESULT: all counted criteria passed (pilot)" in log, (
        counted_failures(out),
        log[-3000:],
    )
    services = [
        (r["stage"], r["preprocess_threads_env"], r["server_timing"])
        for r in records(out)
        if r["kind"] == "service"
    ]
    assert services == [("rel-ref", 0, False), ("rel-diag-a", 0, True), ("rel-diag-b", 12, True)]
    assert all(r["ok"] for r in records(out) if r["kind"] == "cleanup")
    diag = json.loads((out / "diag.json").read_text())
    for arm in ("a", "b"):
        burst = diag[arm]["burst_c32_noretry"]
        assert burst["attempts"] <= 400 and burst["stop_reason"] in ("duration", "max_attempts")
        assert burst["server_stages_ms"]["queue"]["n"] == burst["latency_success_ms"]["n"] > 0
        assert burst["latency_rejected_ms"]["n"] == burst["outcomes"].get("rejected", 0)
        assert burst["resources"]["client_cpu_pct"] is not None
        assert burst["resources"]["main_thread_cpu_pct"]["max"] is not None
        assert diag[arm]["criterion_reference"]["baseline_p99_ms"] is not None
    assert diag["b"]["preprocess_threads"] == "parallel" and diag["a"]["preprocess_threads"] == 0


def test_rel_diag_is_pilot_only(env: dict[str, str]) -> None:
    code, log, _ = run_bench(rel(env, "rel-diag", MODE="full"))
    assert code == 1 and "rel-diag runs only with MODE=pilot" in log


def test_rel_diagmp_multiprocess_client_both_arms(env: dict[str, str]) -> None:
    code, log, out = run_bench(rel(env, "rel-diagmp", **passing(env), REL_DIAG_RATE="150"))
    assert out is not None and (out / "diag.json").exists(), log[-3000:]
    assert code == 0 and "RESULT: all counted criteria passed (pilot)" in log, (
        counted_failures(out),
        log[-3000:],
    )
    services = [
        (r["stage"], r["preprocess_threads_env"], r["server_timing"])
        for r in records(out)
        if r["kind"] == "service"
    ]
    assert services == [
        ("rel-ref", 0, False),
        ("rel-diagmp-a", 0, True),
        ("rel-diagmp-b", 12, True),
    ]
    diag = json.loads((out / "diag.json").read_text())
    for arm in ("a", "b"):
        base, burst = diag[arm]["baseline_c5"]["client"], diag[arm]["burst_c32_noretry"]["client"]
        assert base["shares"] == [2, 1, 1, 1] and base["target_rate_rps"] is None
        assert burst["shares"] == [8, 8, 8, 8] and burst["target_rate_rps"] == 150
        assert len(burst["per_worker"]) == 4 and not burst["failed_workers"]
        assert diag[arm]["burst_c32_noretry"]["attempts"] <= 400
    assert all(r["ok"] for r in records(out) if r["kind"] == "cleanup")


def test_rel_diagmp_worker_failure_fails_closed(env: dict[str, str]) -> None:
    code, log, out = run_bench(rel(env, "rel-diagmp", **passing(env), BENCH_TEST_FAIL_WORKER="1"))
    assert code == 1 and "INVALID run rel-diagmp-a-base-c5" in log, log[-2000:]
    assert out is not None and all(r["ok"] for r in records(out) if r["kind"] == "cleanup")
    assert any(k.startswith("execution: every planned client run") for k in counted_failures(out))


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"MODE": "full"}, "rel-diagmp runs only with MODE=pilot"),
        ({"REL_PHASE_S": "11"}, "burst limited to 10 s and 1,000 attempts"),
        ({"REL_MAX_ATTEMPTS": "1001"}, "burst limited to 10 s and 1,000 attempts"),
        ({"LOAD_WORKERS": "6"}, "must not exceed the baseline concurrency 5"),
        ({"REL_DIAG_RATE": "900"}, "REL_DIAG_RATE=900 must be an integer in [50, 500]"),
    ],
)
def test_rel_diagmp_bounds(env: dict[str, str], extra: dict[str, str], message: str) -> None:
    code, log, _ = run_bench(rel(env, "rel-diagmp", **extra))
    assert code == 1 and message in log, log[-1500:]
