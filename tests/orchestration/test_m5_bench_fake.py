"""tools/m5_bench.sh end to end against a fake docker and nvidia-smi (FakeDetector server, no GPU).

Checks the orchestration contract, not performance: the planned runs execute and are validated,
samplers attribute GPU memory to the serving process, every service is removed and verified, a
golden comparison that does not pass fails the run, cleanup failures are reported with a non-zero
status, and the safeguards refuse unverified images and unauthorized request counts.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FAKES = Path(__file__).parent / "fakes"
IMAGE_ID = "sha256:" + "f" * 64

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or not all(shutil.which(t) for t in ("curl", "ss", "setsid")),
    reason="needs Linux with curl, ss and setsid",
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def env(tmp_path: Path) -> Iterator[dict[str, str]]:
    state = tmp_path / "state"
    state.mkdir()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "bevdet-r50-cbgs.pth").write_bytes(b"not a real checkpoint")
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "http_golden.json").write_text(json.dumps({"model": {}, "frames": [{}] * 10}))
    stage_a = tmp_path / "stage-a"
    stage_a.mkdir()
    (stage_a / "image.txt").write_text(f"id={IMAGE_ID} size_bytes=1 created=now\n")
    (stage_a / "summary.txt").write_text(
        "PASS build\nFAIL fault: process exited with code 70 (saw: )\nPASS clean-shutdown\n"
    )
    retest = tmp_path / "retest"
    retest.mkdir()
    (retest / "summary.txt").write_text(
        f"PASS image id is the verified {IMAGE_ID}\n"
        "PASS process exited with code 70 (docker die event, then restart; restart count 0 -> 1)\n"
        "PASS request after recovery -> 200\n"
    )
    stage_c = tmp_path / "stage-c"  # an all-PASS M6 verification run (no fault retest)
    stage_c.mkdir()
    (stage_c / "image.txt").write_text(f"id={IMAGE_ID} size_bytes=1 created=now\n")
    (stage_c / "summary.txt").write_text(
        "PASS build\nPASS http-golden-parity (10 real frames, byte-identical boxes)\n"
        "PASS parallel: recreated container (new id) runs with BEVSERVE_PREPROCESS_THREADS=12\n"
        "PASS http-golden-parity with preprocess threads 12 (10 real frames, byte-identical)\n"
        "PASS clean-shutdown\n"
    )
    bevdet = tmp_path / "bevdet"
    bevdet.mkdir()
    yield {
        **os.environ,
        "FAKE_STATE": str(state),
        "FAKE_PY": sys.executable,
        "DOCKER_CMD": str(FAKES / "docker"),
        "NVIDIA_SMI": str(FAKES / "nvidia-smi"),
        "PORT": str(_free_port()),
        "IMAGE": "fake:tag",
        "EXPECT_IMAGE_ID": IMAGE_ID,
        "STAGE_A_DIR": str(stage_a),
        "FAULT_RETEST_DIR": str(retest),
        "BEVDET_ROOT": str(bevdet),
        "CKPT_DIR": str(ckpt),
        "CKPT_SHA256": hashlib.sha256(b"not a real checkpoint").hexdigest(),
        "GOLDEN_DIR": str(golden),
        "HOST_PY": sys.executable,
        "INPUTS": "synthetic:2",
    }
    pid = state / "pid"  # never leave a fake server behind
    if pid.exists():
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.killpg(int(pid.read_text()), signal.SIGTERM)


def _run(env: dict[str, str], **extra: str) -> tuple[int, str, Path | None]:
    before = set((ROOT / "outputs").glob("m5-2*")) if (ROOT / "outputs").exists() else set()
    p = subprocess.run(
        ["bash", str(ROOT / "tools/m5_bench.sh")],
        env={**env, **extra},
        capture_output=True,
        text=True,
        timeout=600,
    )
    new = sorted(set((ROOT / "outputs").glob("m5-2*")) - before)
    return p.returncode, p.stdout + p.stderr, new[-1] if new else None


@pytest.fixture(autouse=True)
def _remove_outputs() -> Iterator[None]:
    before = set((ROOT / "outputs").glob("m5-2*")) if (ROOT / "outputs").exists() else set()
    yield
    for d in set((ROOT / "outputs").glob("m5-2*")) - before:
        shutil.rmtree(d)


def _records(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines() if x.strip()]


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


def _m6(env: dict[str, str], **extra: str) -> dict[str, str]:
    out = {k: v for k, v in env.items() if k != "FAULT_RETEST_DIR"}
    out["STAGE_A_DIR"] = str(Path(env["STAGE_A_DIR"]).parent / "stage-c")
    return {**out, "STAGES": "ab", **extra}


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


def _record_fake_golden(path: Path) -> None:
    """Golden responses for 10 synthetic frames, recorded from an in-process FakeDetector app."""
    sys.path.insert(0, str(FAKES))
    from golden_http import frame_parts  # the same frames the fake comparison posts
    from tests.api.helpers import build_app, client_for

    async def record() -> dict:
        app, _ = build_app()
        async with client_for(app) as client:
            model = (await client.get("/v1/model")).json()
            frames = []
            for idx in range(10):
                files, data = frame_parts(idx)
                body = (await client.post("/v1/detect", files=files, data=data)).json()
                frames.append({"index": idx, "boxes": body["boxes"], "frame": body["frame"],
                               "units": body["units"]})  # fmt: skip
        return {"model": {k: model[k] for k in ("name", "checkpoint_sha256")}, "frames": frames}

    import asyncio

    path.write_text(json.dumps(asyncio.run(record())))


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
