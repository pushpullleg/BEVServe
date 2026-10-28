"""tools/m5_bench.sh end to end against a fake docker and nvidia-smi (FakeDetector server, no GPU).

Checks the orchestration contract, not performance: the planned runs execute and are validated,
samplers attribute GPU memory to the serving process, every service is removed and verified, a
golden comparison that does not pass fails the run, cleanup failures are reported with a non-zero
status, and the safeguards refuse unverified images and unauthorized request counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness import (
    FAKES,
    IMAGE_ID,
    SUPPORTED,
)
from harness import m6 as _m6
from harness import record_fake_golden as _record_fake_golden
from harness import records as _records
from harness import run_bench as _run

pytestmark = pytest.mark.skipif(not SUPPORTED, reason="needs Linux with curl, ss and setsid")


def test_pilot_runs_every_stage_validates_and_cleans_up(env: dict[str, str]) -> None:
    code, log, out = _run(env, MODE="pilot")
    assert out is not None and (out / "criteria.json").exists(), log
    criteria = json.loads((out / "criteria.json").read_text())
    recs = _records(out)
    # every planned client run executed and validated; all four concurrency levels
    runs = {r["label"]: r for r in recs if r["kind"] == "run"}
    assert all(r["valid"] for r in runs.values()), runs
    assert {f"levels-c{c}" for c in (1, 2, 4, 8)} <= set(runs)
    assert sum(label.startswith("overhead-t") for label in runs) == 4
    # samplers attributed GPU memory to the server only; every service removed and verified
    samplers = [r for r in recs if r["kind"] == "sampler"]
    assert len(samplers) == 3 and all(s["valid"] for s in samplers), samplers
    services = [r for r in recs if r["kind"] == "service"]
    cleanups = [r for r in recs if r["kind"] == "cleanup"]
    assert len(services) == 8 and len(cleanups) == 8 and all(c["ok"] for c in cleanups)
    harness = [v for v in criteria.values() if isinstance(v, dict) and v["kind"] == "harness"]
    assert harness and all(v["status"] == "pass" for v in harness), criteria
    assert (
        criteria[
            "levels: every request at C = 1, 2, 4, 8 succeeded (no rejections, errors or timeouts)"
        ]["status"]
        == "pass"
    )
    # the golden comparison cannot pass against fake boxes / no dataset: the run must fail on it
    golden = criteria[
        "soak: ready afterwards and golden parity comparison ran over all frames and passed"
    ]
    assert golden["status"] == "fail" and golden["counted"], golden
    assert code == 1 and "acceptance criteria not met" in log
    # statistical criteria are reported but not counted in pilot mode
    stat = [v for v in criteria.values() if isinstance(v, dict) and v["kind"] == "statistical"]
    assert stat and not any(v["counted"] for v in stat)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["verified_image"]["pass"] and meta["image_package_matches_source"]
    assert meta["checkpoint"]["sha256"] == env["CKPT_SHA256"] and "harness_sha256" in meta
    assert (out / "overhead.json").exists() and "Paired p50 differences" in (
        out / "report.md"
    ).read_text()


def test_unattributed_gpu_memory_invalidates_the_sampler(env: dict[str, str]) -> None:
    code, log, out = _run(
        env,
        MODE="pilot",
        STAGES="levels",
        LEVEL_REQUESTS="3",
        LEVEL_WARMUP="0",
        FAKE_SMI_NOAPPS="1",
    )
    assert out is not None, log
    sampler = next(r for r in _records(out) if r["kind"] == "sampler")
    assert not sampler["valid"] and sampler["gpu_attributed_fraction"] == 0.0
    assert code == 1


def test_failed_cleanup_is_reported_and_nonzero(env: dict[str, str]) -> None:
    code, log, out = _run(env, MODE="pilot", STAGES="cold", FAKE_DOWN_FAIL="1")
    assert code == 2, log
    assert "CLEANUP FAILED" in log and "down --remove-orphans" in log
    assert out is not None
    assert [r["ok"] for r in _records(out) if r["kind"] == "cleanup"] == [False, False]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"MODE": ""}, "MODE must be pilot or full"),
        ({"MODE": "full", "SOAK_REQUESTS": "101"}, "needs SOAK_FULL=1"),
        (
            {"MODE": "full", "SOAK_REQUESTS": "1001", "SOAK_FULL": "1"},
            "must be an integer in [1, 1000]",
        ),
        ({"MODE": "pilot", "LEVEL_REQUESTS": "51"}, "must be an integer in [1, 50]"),
        ({"MODE": "full", "OVERHEAD_TRIALS": "6"}, "multiple of 4"),
        ({"MODE": "full", "ADMISSION_LEVELS": "32"}, "only 8 and 16"),
        (
            {"MODE": "full", "EXPECT_IMAGE_ID": "sha256:" + "0" * 64},
            "not the verified image",
        ),
    ],
)
def test_safeguards_refuse_before_starting_anything(
    env: dict[str, str], extra: dict[str, str], message: str
) -> None:
    code, log, _ = _run(env, **extra)
    assert code == 1 and message in log, log
    calls = Path(env["FAKE_STATE"], "calls.log")
    assert "compose" not in (calls.read_text() if calls.exists() else "")


def test_ab_pilot_alternates_verified_arms_and_validates(env: dict[str, str]) -> None:
    code, log, out = _run(_m6(env), MODE="pilot")
    assert out is not None and (out / "criteria.json").exists(), log
    recs = _records(out)
    services = [r for r in recs if r["kind"] == "service"]
    assert [r["preprocess_threads_env"] for r in services] == [0, 12, 12, 0]  # ABBA, as running
    assert len({r["container"] for r in services}) == 4  # a fresh container per trial
    assert all(r["ok"] for r in recs if r["kind"] == "cleanup") and len(services) == 4
    runs = {r["label"]: r for r in recs if r["kind"] == "run"}
    assert len(runs) == 8 and all(r["valid"] for r in runs.values()), runs
    samplers = [r for r in recs if r["kind"] == "sampler"]
    assert len(samplers) == 4 and all(s["valid"] for s in samplers), samplers
    assert len([r for r in recs if r["kind"] == "phase"]) == 4
    ab = json.loads((out / "ab.json").read_text())
    assert ab["pairs"] == [[1, 2], [4, 3]] and len(ab["trials"]) == 4
    assert ab["estimates"]["c1_p50_ms"]["complete"]
    crit = json.loads((out / "criteria.json").read_text())
    fresh = crit[
        "ab: every trial ran in a fresh container with the intended BEVSERVE_PREPROCESS_THREADS"
    ]
    assert fresh["status"] == "pass"
    assert crit["ab: every request at C = 1 and C = 8 succeeded in every trial"]["status"] == "pass"
    golden = crit[
        "ab: golden parity (all frames byte-identical) in every trial container, both arms"
    ]
    assert golden["status"] == "fail" and golden["counted"]  # fake boxes, no dataset
    stat = [v for k, v in crit.items() if k.startswith("ab:") and v["kind"] == "statistical"]
    assert len(stat) == 5 and not any(v["counted"] for v in stat)  # pilot: reported, not counted
    meta = json.loads((out / "meta.json").read_text())
    assert meta["verified_image"]["pass"] and "retest_dir" not in meta["verified_image"]
    assert "M6 A/B" in (out / "report.md").read_text()
    assert code == 1


def test_ab_fails_closed_when_the_container_runs_other_settings(env: dict[str, str]) -> None:
    code, log, out = _run(_m6(env), MODE="pilot", FAKE_ENV_THREADS="0")
    assert code == 1 and "runs with BEVSERVE_PREPROCESS_THREADS=0, expected 12" in log, log
    assert out is not None
    assert all(r["ok"] for r in _records(out) if r["kind"] == "cleanup")


def test_ab_full_needs_a_passing_pilot(env: dict[str, str], tmp_path: Path) -> None:
    code, log, _ = _run(_m6(env), MODE="full")
    assert code == 1 and "needs PILOT_DIR" in log
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "meta.json").write_text(json.dumps({
        "image": {"Id": IMAGE_ID}, "params": {"mode": "pilot", "stages": ["ab"], "ab_threads": 12},
    }))  # fmt: skip
    (pilot / "criteria.json").write_text(json.dumps({
        "ab: golden": {"status": "fail", "counted": True, "kind": "correctness"},
    }))  # fmt: skip
    code, log, _ = _run(_m6(env), MODE="full", PILOT_DIR=str(pilot))
    assert code == 1 and "pilot did not pass" in log, log
    calls = Path(env["FAKE_STATE"], "calls.log")
    assert "compose" not in (calls.read_text() if calls.exists() else "")


def test_ab_needs_the_parallel_verification_checks(env: dict[str, str]) -> None:
    code, log, _ = _run(_m6(env), MODE="pilot", AB_THREADS="24")  # stage C verified 12 only
    assert code == 1 and "not the verified image" in log, log


def test_ab_pilot_positive_end_to_end(env: dict[str, str], tmp_path: Path) -> None:
    """Valid simulated golden data: the whole pilot passes, with 4 fresh 0/12/12/0 containers."""
    _record_fake_golden(Path(env["GOLDEN_DIR"]) / "http_golden.json")
    run_env = _m6(env, BENCH_TEST_MODE="1", GOLDEN_TOOL=str(FAKES / "golden_http.py"))
    code, log, out = _run(run_env, MODE="pilot")
    assert out is not None, log
    assert code == 0 and "RESULT: all counted criteria passed (pilot)" in log, log
    recs = _records(out)
    services = [r for r in recs if r["kind"] == "service"]
    assert [r["preprocess_threads_env"] for r in services] == [0, 12, 12, 0]
    assert len({r["container"] for r in services}) == 4
    cleanups = [r for r in recs if r["kind"] == "cleanup"]
    assert len(cleanups) == 4 and all(c["ok"] for c in cleanups)
    golden = [r for r in recs if r["kind"] == "golden"]
    assert len(golden) == 4 and all(g["valid"] and g["frames_compared"] == 10 for g in golden)
    crit = json.loads((out / "criteria.json").read_text())
    counted = {k: v["status"] for k, v in crit.items() if isinstance(v, dict) and v["counted"]}
    assert counted and set(counted.values()) == {"pass"}, counted
    meta = json.loads((out / "meta.json").read_text())
    assert meta["test_mode"] is True and meta["golden_tool"]["path"].startswith("tests/")
    assert "TEST MODE" in log
    # a test-mode pilot can never authorize a full run
    code, log, _ = _run(run_env, MODE="full", PILOT_DIR=str(out))
    assert code == 1 and "pilot did not pass" in log, log


def test_golden_tool_override_needs_test_mode(env: dict[str, str]) -> None:
    code, log, _ = _run(_m6(env), MODE="pilot", GOLDEN_TOOL=str(FAKES / "golden_http.py"))
    assert code == 1 and "only be overridden with BENCH_TEST_MODE=1" in log, log
