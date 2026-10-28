"""M5 harness logic: execution validation, overhead statistics, sampler parsing, outcomes, gate."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"bench_{name}", ROOT / f"tools/bench/{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate, overhead, load, gate = (
    _load(n) for n in ("validate", "overhead", "load", "verified_image")
)
ID = "sha256:" + "a" * 64


def _write_run(d: Path, label: str, n: int, w: int, c: int = 1, **summary_over: object) -> None:
    rows = [{"i": i, "status": 200, "latency_ms": 1.0, "warmup": i < w} for i in range(n + w)]
    (d / f"{label}.requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = {"label": label, "requests_measured": n, "warmup": w, "concurrency": c,
               "outcomes": {"ok": n}, **summary_over}  # fmt: skip
    (d / f"{label}.summary.json").write_text(json.dumps(summary))


def test_valid_run(tmp_path: Path) -> None:
    _write_run(tmp_path, "r", 5, 2)
    assert validate.check_run(tmp_path, "r", 5, 2, 1, 0) == []


@pytest.mark.parametrize("damage", ["exit", "missing", "malformed", "short", "counts"])
def test_failed_client_or_bad_results_are_invalid(tmp_path: Path, damage: str) -> None:
    _write_run(tmp_path, "r", 5, 2)
    rc = 0
    if damage == "exit":
        rc = 1
    elif damage == "missing":
        (tmp_path / "r.summary.json").unlink()
    elif damage == "malformed":
        (tmp_path / "r.requests.jsonl").write_text("{not json\n")
    elif damage == "short":
        lines = (tmp_path / "r.requests.jsonl").read_text().splitlines()
        (tmp_path / "r.requests.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    else:
        _write_run(tmp_path, "r", 5, 2, outcomes={"ok": 3})
    assert validate.check_run(tmp_path, "r", 5, 2, 1, rc)


def _write_sampler(
    d: Path, times: list[float], gpu: list[str], pids: list[int], reason: str = "stopped"
) -> None:
    rows = [
        "t_unix,rss_mib,cpu_pct,threads,gpu_util_pct,gpu_mem_used_mib,proc_gpu_mem_mib,gpu_status"
    ]
    rows += [f"{t},1,1,1,1,1,{'' if g != 'ok' else 5},{g}" for t, g in zip(times, gpu, strict=True)]
    (d / "s.samples.csv").write_text("\n".join(rows) + "\n")
    meta = {"interval_s": 1.0, "exit_reason": reason, "gpu_pids": pids, "target_pid": 1}
    (d / "s.sampler.json").write_text(json.dumps(meta))


def test_sampler_record(tmp_path: Path) -> None:
    _write_sampler(tmp_path, [1.0, 2.0, 3.0], ["ok"] * 3, [42])
    problems, ev = validate.check_sampler(tmp_path, "s", 0, 0.0, 4.0)
    assert problems == [] and ev["gpu_attributed_fraction"] == 1.0
    _write_sampler(tmp_path, [1.0, 2.0, 9.0], ["ok"] * 3, [42])  # 7 s gap
    assert any("gap" in p for p in validate.check_sampler(tmp_path, "s", 0, 0.0, 10.0)[0])
    _write_sampler(tmp_path, [1.0, 2.0], ["ok", "not_listed"], [42])
    assert any(
        "GPU memory present" in p for p in validate.check_sampler(tmp_path, "s", 0, 0.0, 3.0)[0]
    )
    _write_sampler(tmp_path, [1.0], ["ok"], [42, 43])  # memory attributed to two processes
    assert validate.check_sampler(tmp_path, "s", 0, 0.0, 2.0)[0]
    _write_sampler(tmp_path, [1.0], ["ok"], [42], reason="target_exited")
    assert validate.check_sampler(tmp_path, "s", 3, 0.0, 2.0)[0]


def test_golden_record(tmp_path: Path) -> None:
    golden = tmp_path / "g.json"
    golden.write_text(json.dumps({"frames": [{}] * 10}))
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"pass": True, "model_identity_ok": True, "frames": [{}] * 10}))
    assert validate.check_golden(f, golden, 0)[0] == []
    f.write_text(json.dumps({"pass": True, "model_identity_ok": True, "frames": [{}] * 9}))
    assert validate.check_golden(f, golden, 0)[0]  # not every frame compared
    f.write_text("")
    assert validate.check_golden(f, golden, 1)[0]  # did not run


def test_abba_order_pairs_and_interval() -> None:
    assert overhead.order(8) == ["false", "true", "true", "false"] * 2
    assert overhead.pairs(8) == [(1, 2), (4, 3), (5, 6), (8, 7)]
    for off, on in overhead.pairs(12):
        arms = overhead.order(12)
        assert (arms[off - 1], arms[on - 1]) == ("false", "true")
    mean, lo, hi = overhead.t_interval([1.0, 2.0, 3.0])
    assert (
        mean == 2.0
        and lo == pytest.approx(2 - 4.303 / 3**0.5)
        and hi == pytest.approx(2 + 4.303 / 3**0.5)
    )
    assert overhead.t_interval([1.0]) is None
    with pytest.raises(ValueError):
        overhead.order(6)


def test_overhead_analysis_reports_trials(tmp_path: Path) -> None:
    for k, arm in enumerate(overhead.order(4), start=1):
        p50 = 100.0 + (1.0 if arm == "true" else 0.0) + k * 0.1
        _write_run(
            tmp_path, f"overhead-t{k}-timing-{arm}", 3, 0, latency_ms={"p50": p50, "mean": p50}
        )
    r = overhead.analyze(tmp_path, 4)
    assert r["complete"] and len(r["trials"]) == 4
    assert r["paired_p50_diff_ms"] == pytest.approx([1.1, 0.9])  # drift cancels within a block
    (tmp_path / "overhead-t2-timing-true.summary.json").unlink()
    assert overhead.analyze(tmp_path, 4)["ci95_ms"] is None


def test_outcomes_separate_rejections_from_failures() -> None:
    row = {"status": 503, "type": "/errors/queue_full"}
    assert load.outcome(row) == "rejected"
    assert load.outcome({"status": 503, "type": "/errors/not_ready"}) == "http_error"
    assert load.outcome({"status": 200, "error": "MalformedResponse"}) == "malformed"
    assert load.outcome({"status": 0, "error": "ReadTimeout"}) == "transport_error"
    assert load.outcome({"status": 200}) == "ok"


def test_sampler_attributes_only_the_serving_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smi = tmp_path / "nvidia-smi"
    smi.write_text(
        "#!/bin/sh\ncase \"$*\" in *query-gpu*) echo '[N/A], 2048';; "
        "*compute-apps*) echo '111, 900'; echo '222, [N/A]';; esac\n"
    )
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("NVIDIA_SMI", str(smi))
    sampler = _load("sampler")
    assert sampler.gpu_sample({111}) == ("", "2048", "900", "ok", [111])  # N/A util -> missing
    assert sampler.gpu_sample({333}) == ("", "2048", "", "not_listed", [])
    assert sampler.gpu_sample({222})[2:4] == ("", "unavailable")  # never zero
    monkeypatch.setenv("NVIDIA_SMI", str(tmp_path / "absent"))
    assert _load("sampler").gpu_sample({111})[3] == "query_failed"
    assert os.getpid() in sampler.process_tree(os.getppid())


def _gate_dirs(tmp_path: Path, stage_a: str, retest: str) -> tuple[Path, Path]:
    a, r = tmp_path / "a", tmp_path / "r"
    a.mkdir(exist_ok=True)
    r.mkdir(exist_ok=True)
    (a / "image.txt").write_text(f"id={ID} size_bytes=1\n")
    (a / "summary.txt").write_text(stage_a)
    (r / "summary.txt").write_text(retest)
    return a, r


RETEST_OK = (
    f"PASS image id is the verified {ID}\nPASS process exited with code 70 (docker die event, x)\n"
    "PASS request after recovery -> 200\n"
)


def test_verified_image_gate(tmp_path: Path) -> None:
    a, r = _gate_dirs(tmp_path, f"PASS x\n{gate.SUPERSEDED}\n", RETEST_OK)
    assert gate.gate(ID, ID, a, r)["pass"]
    assert not gate.gate(ID, "sha256:" + "b" * 64, a, r)["pass"]  # different image
    a, r = _gate_dirs(tmp_path, "PASS x\nFAIL something else\n", RETEST_OK)
    assert not gate.gate(ID, ID, a, r)["pass"]  # unaccepted stage A failure
    a, r = _gate_dirs(tmp_path, "PASS x\n", RETEST_OK.replace("PASS request", "FAIL request"))
    assert not gate.gate(ID, ID, a, r)["pass"]  # retest not fully passed
