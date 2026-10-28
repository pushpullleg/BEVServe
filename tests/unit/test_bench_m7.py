"""M7 harness logic: canonical output hashing, accounting, sustained trends, gates, budgets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
ID = "sha256:" + "e" * 64


def _load(name: str) -> ModuleType:
    sys.path.insert(0, str(ROOT / "tools/bench"))
    spec = importlib.util.spec_from_file_location(f"m7_{name}", ROOT / f"tools/bench/{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load, sustained, gate, budget, validate = (
    _load(n) for n in ("load", "sustained", "gate", "budget", "validate")
)

DOC = {
    "request_id": "r1",
    "model": {"name": "m", "checkpoint_sha256": "c", "config_sha256": "k", "git_commit": "x"},
    "frame": "ego@CAM_FRONT_LEFT",
    "units": {"distance": "m"},
    "boxes": [{"label": "car", "score": 0.5, "center": [1.0, 2.0, 0.5]}],
    "timing_ms": {"total": 10.0},
}


def test_canonical_hash_ignores_formatting_and_volatile_fields() -> None:
    reordered = json.loads(json.dumps(DOC, sort_keys=True, indent=4))
    reordered["request_id"], reordered["timing_ms"] = "r2", {"total": 99.0}
    reordered["model"]["git_commit"] = "y"  # deployment metadata, not model output
    assert load.canonical_output(reordered) == load.canonical_output(DOC)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["boxes"][0].update(score=0.5000001),
        lambda d: d["boxes"].append(dict(d["boxes"][0])),
        lambda d: d.update(frame="ego@CAM_FRONT"),
        lambda d: d["model"].update(checkpoint_sha256="other"),
    ],
    ids=["score", "extra-box", "frame", "checkpoint"],
)
def test_canonical_hash_detects_semantic_changes(change: object) -> None:
    other = json.loads(json.dumps(DOC))
    change(other)  # type: ignore[operator]
    assert load.canonical_output(other) != load.canonical_output(DOC)


def test_input_identity_is_independent_of_the_multipart_boundary() -> None:
    files = [("CAM_FRONT", ("f.jpg", b"\xff\xd8\xffdata", "image/jpeg"))]
    a = httpx.Request("POST", "http://x", files=files, data={"calibration": "{}"})
    b = httpx.Request("POST", "http://x", files=files, data={"calibration": "{}"})
    bodies = [(a.read(), a.headers["content-type"]), (b.read(), b.headers["content-type"])]
    assert bodies[0][0] != bodies[1][0]  # different random boundaries
    assert load.body_hashes(bodies)[0] == load.body_hashes(bodies)[1]


def test_check_outputs_against_reference_and_within_run() -> None:
    rows = [
        {"input": 0, "output_sha256": "a"},
        {"input": 1, "output_sha256": "b"},
        {"input": 0, "output_sha256": "a"},
        {"input": 1, "status": 503},
    ]
    ref = {"body_sha256": ["h0", "h1"], "outputs": ["a", "b"]}
    assert load.check_outputs(rows, ref, ["h0", "h1"]) == {
        "checked": 3,
        "inconsistent_inputs": [],
        "mismatches": 0,
    }
    rows.append({"input": 1, "output_sha256": "z"})
    out = load.check_outputs(rows, ref, ["h0", "h1"])
    assert out["mismatches"] == 1 and out["inconsistent_inputs"] == [1]
    assert load.check_outputs(rows, ref, ["h0", "other"])["mismatches"] is None  # wrong inputs


def test_accounting_separates_attempts_rejections_and_successes() -> None:
    def row(i: int, status: int, latency: float, **extra: object) -> dict:
        t = i / 10
        base = {"i": i, "status": status, "t_start_s": t, "t_end_s": t + latency / 1000}
        return {**base, "warmup": False, "latency_ms": latency, "server": {}, **extra}

    rows = [
        row(0, 200, 100.0, boxes=1),
        row(1, 503, 10.0, type="/errors/queue_full", retry_after="1"),
        row(2, 422, 100.0, type="/errors/image_corrupt"),
        row(3, 0, 100.0, error="ConnectError"),
    ]
    args = argparse.Namespace(
        label="t",
        url="u",
        concurrency=2,
        warmup=0,
        honor_retry_after=False,
        duration=5.0,
        stop_reason="duration",
    )
    s = load.summarize(rows, args, {})
    assert s["accounting"] == {
        "attempts": 4,
        "completed": 3,
        "rejected": 1,
        "accepted": 2,
        "success": 1,
    }
    assert s["mode"] == "duration" and s["stop_reason"] == "duration"
    assert s["latency_ms"]["n"] == 1 and s["latency_rejected_ms"]["n"] == 1
    assert s["queue_full_retry_after"] == ["1"]


def _samples(
    path: Path,
    minutes: float,
    slope_mib_h: float,
    noise: float,
    start: float = 1000.0,
    gpu_step_at: float | None = None,
) -> None:
    rng = random.Random(0)
    lines = ["t_unix,rss_mib,proc_gpu_mem_mib,tree_threads,threads"]
    for k in range(int(minutes * 60)):
        t = start + k
        rss = 2000 + slope_mib_h * k / 3600 + rng.gauss(0, noise)
        gpu = 1300 + (64 if gpu_step_at is not None and k >= gpu_step_at else 0)
        lines.append(f"{t},{rss:.2f},{gpu},{170 + (k % 3)},{150}")
    path.write_text("\n".join(lines) + "\n")


def _requests(
    path: Path, minutes: float, start: float = 1000.0, slow_after: float | None = None
) -> None:
    rows = []
    for k in range(int(minutes * 60)):
        latency = 40.0 * (1.5 if slow_after is not None and k >= slow_after else 1.0)
        rows.append(json.dumps({"status": 200, "t_unix": start + k, "latency_ms": latency}))
    path.write_text("\n".join(rows) + "\n")


def test_flat_memory_gives_a_tight_interval_around_zero(tmp_path: Path) -> None:
    _samples(tmp_path / "s.csv", 20, 0.0, 5.0)
    _requests(tmp_path / "r.jsonl", 20)
    out = sustained.analyze(tmp_path / "s.csv", [tmp_path / "r.jsonl"], 1000.0, 300, 60)
    tr = out["rss"]["trend_mib"]
    assert tr["bins"] == 15 and tr["ci95"][0] <= 0 <= tr["ci95"][1] and tr["ci95"][1] < 80
    assert out["threads"]["post_range"] == 2 and out["gpu"]["growth_after_warmup_mib"] == 0
    assert out["latency"]["drift_ratio"] == pytest.approx(1.0)


def test_steady_growth_is_detected_and_steps_after_warmup_are_reported(tmp_path: Path) -> None:
    _samples(tmp_path / "s.csv", 20, 400.0, 5.0, gpu_step_at=900)
    _requests(tmp_path / "r.jsonl", 20, slow_after=1000)
    out = sustained.analyze(tmp_path / "s.csv", [tmp_path / "r.jsonl"], 1000.0, 300, 60)
    tr = out["rss"]["trend_mib"]
    assert tr["slope_per_h"] == pytest.approx(400, rel=0.1) and tr["ci95"][0] > 80
    assert out["gpu"]["growth_after_warmup_mib"] == 64
    assert out["latency"]["drift_ratio"] == pytest.approx(1.5)


def test_too_few_bins_give_no_interval(tmp_path: Path) -> None:
    _samples(tmp_path / "s.csv", 8, 0.0, 5.0)
    _requests(tmp_path / "r.jsonl", 8)
    tr = sustained.analyze(tmp_path / "s.csv", [tmp_path / "r.jsonl"], 1000.0, 300, 60)["rss"][
        "trend_mib"
    ]
    assert tr["bins"] == 3 and tr["ci95"] is None


def test_autocorrelated_residuals_widen_the_interval() -> None:
    xs = [k / 60 for k in range(30)]
    wavy = [
        (x, 2000 + 50 * math.sin(k / 3)) for k, x in enumerate(xs)
    ]  # smooth: strongly autocorrelated
    rng = random.Random(1)
    white = [(x, 2000 + rng.gauss(0, 35)) for x in xs]
    a, b = sustained.trend(wavy), sustained.trend(white)
    assert a["lag1_autocorrelation"] > 0.5 and a["n_effective"] < b["n_effective"]
    half = lambda t: (t["ci95"][1] - t["ci95"][0]) / 2 if t["ci95"] else math.inf  # noqa: E731
    assert half(a) > half(b)


def _prior(d: Path, **over: object) -> Path:
    d.mkdir()
    meta = {
        "test_mode": False,
        "image": {"Id": ID},
        "params": {"mode": "pilot", "stages": ["rel-admission"], "rel_threads": 12},
    }
    for k, v in over.items():
        if k in ("mode", "stages", "rel_threads"):
            meta["params"][k] = v  # type: ignore[index]
        else:
            meta[k] = v
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "criteria.json").write_text(
        json.dumps(
            {
                "c": {"status": "pass", "counted": True},
                "i": {"status": "reported", "counted": False},
            }
        )
    )
    return d


def test_prior_gate(tmp_path: Path) -> None:
    assert gate.prior(_prior(tmp_path / "ok"), ID, "pilot", "rel-admission", 12) == []
    assert gate.prior(_prior(tmp_path / "t", test_mode=True), ID, "pilot", "rel-admission", 12)
    assert gate.prior(_prior(tmp_path / "old", test_mode=None), ID, "pilot", "rel-admission", 12)
    assert gate.prior(_prior(tmp_path / "m", mode="full"), ID, "pilot", "rel-admission", 12)
    assert gate.prior(
        _prior(tmp_path / "s", stages=["rel-admission", "ab"]), ID, "pilot", "rel-admission", 12
    )
    assert gate.prior(_prior(tmp_path / "th", rel_threads=6), ID, "pilot", "rel-admission", 12)
    assert gate.prior(tmp_path / "ok", "sha256:" + "0" * 64, "pilot", "rel-admission", 12)
    failed = _prior(tmp_path / "f")
    (failed / "criteria.json").write_text(
        json.dumps({"c": {"status": "insufficient_data", "counted": True}})
    )
    assert gate.prior(failed, ID, "pilot", "rel-admission", 12)


def test_fault_gate(tmp_path: Path) -> None:
    d = tmp_path / "fault"
    d.mkdir()
    (d / "run.json").write_text(
        json.dumps({"image_id": ID, "preprocess_threads": 12, "test_mode": False})
    )
    lines = [
        "PASS container runs with BEVSERVE_PREPROCESS_THREADS=12",
        "PASS process exited with code 70 (docker die event, then restart)",
        "PASS ready again after restart (3 s)",
        "PASS request after recovery -> 200",
        "PASS no further restarts (restart count 1 after 10 s)",
        "PASS clean shutdown with BEVSERVE_PREPROCESS_THREADS=12 (exit 0 in 1 s)",
    ]
    (d / "summary.txt").write_text("\n".join(lines))
    assert gate.fault(d, ID, 12) == []
    assert gate.fault(d, ID, 0)  # a serial retest does not authorize the parallel setting
    (d / "summary.txt").write_text("\n".join(lines[:-1]))  # no shutdown check
    assert gate.fault(d, ID, 12)


def _budget_args(d: Path, **over: object) -> argparse.Namespace:
    base = {
        "dir": str(d),
        "sampler": "s",
        "readiness": "s",
        "since": 0.0,
        "rss_max_mib": 8192,
        "gpu_max_mib": 4096,
        "threads_max": 400,
        "summaries": [],
    }
    return argparse.Namespace(**{**base, **over})


def test_budget_stop_conditions(tmp_path: Path) -> None:
    (tmp_path / "s.samples.csv").write_text(
        "t_unix,rss_mib,proc_gpu_mem_mib,tree_threads\n1,2000,1300,170\n2,2100,1316,180\n"
    )
    probes = ["t_unix,http_status,status,latency_ms"] + [f"{t},200,ready,3" for t in range(1, 10)]
    (tmp_path / "s.readiness.csv").write_text("\n".join(probes) + "\n")
    (tmp_path / "ok.summary.json").write_text(json.dumps({"outcomes": {"ok": 5, "rejected": 2}}))
    (tmp_path / "bad.summary.json").write_text(json.dumps({"outcomes": {"ok": 5, "http_error": 1}}))
    assert budget.check(_budget_args(tmp_path, summaries=["ok"])) == []
    assert budget.check(_budget_args(tmp_path, summaries=["bad"]))
    assert budget.check(_budget_args(tmp_path, threads_max=175))
    assert budget.check(_budget_args(tmp_path, rss_max_mib=2050))
    probes[3:10] = [f"{t},0,,2000" for t in range(3, 10)]  # 6+ s without a ready probe
    (tmp_path / "s.readiness.csv").write_text("\n".join(probes) + "\n")
    assert any("not ready" in r for r in budget.check(_budget_args(tmp_path)))
    assert budget.check(_budget_args(tmp_path, sampler="missing"))


def _timed(d: Path, n: int, stop: str = "duration", **over: object) -> None:
    rows = [{"i": i, "status": 200, "latency_ms": 1.0, "warmup": False} for i in range(n)]
    (d / "r.requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = {
        "label": "r",
        "mode": "duration",
        "stop_reason": stop,
        "attempts_total": n,
        "warmup": 0,
        "concurrency": 4,
        "accounting": {"attempts": n},
        "outcomes": {"ok": n},
        "outputs": {},
        **over,
    }
    (d / "r.summary.json").write_text(json.dumps(s))


def test_timed_run_validation(tmp_path: Path) -> None:
    _timed(tmp_path, 50)
    assert validate.check_run(tmp_path, "r", 100, 0, 4, 0) == []
    _timed(tmp_path, 50, stop="interrupted")
    assert validate.check_run(tmp_path, "r", 100, 0, 4, 0)
    _timed(tmp_path, 150)
    assert validate.check_run(tmp_path, "r", 100, 0, 4, 0)  # above the attempt cap
    _timed(tmp_path, 50, outputs={"reference_error": "reference was recorded for different inputs"})
    assert validate.check_run(tmp_path, "r", 100, 0, 4, 0)


def test_a_diagnostic_run_never_authorizes_c2(tmp_path: Path) -> None:
    d = _prior(tmp_path / "diag", stages=["rel-diag"])
    assert gate.prior(d, ID, "pilot", "rel-admission", 12)


def test_thread_cpu_reports_main_and_busiest_thread() -> None:
    sampler = _load("sampler")
    before, after = {100: 0, 101: 0, 102: 0}, {100: 50, 101: 150, 102: 10}
    main, busiest = sampler.thread_cpu(100, before, after, 2.0)
    clk = sampler.CLK
    assert float(main) == pytest.approx(100 * 50 / clk / 2.0, abs=0.1)
    assert float(busiest) == pytest.approx(100 * 150 / clk / 2.0, abs=0.1)
    assert sampler.thread_cpu(100, before, {}, 2.0) == ("", "")


def test_diag_keeps_success_and_rejection_apart_and_derives_per_request(tmp_path: Path) -> None:
    diag = _load("diag")

    def write(label: str, rows: list[dict]) -> None:
        (tmp_path / f"{label}.requests.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
        (tmp_path / f"{label}.summary.json").write_text(
            json.dumps({"outcomes": {}, "client_process": {"cpu_pct": 97.0}})
        )

    def ok(i: int, latency: float, queue: float, gpu: float, forward: float) -> dict:
        server = {
            "parse": 1.0,
            "validate": 1.0,
            "decode": 5.0,
            "preprocess": 15.0,
            "queue": queue,
            "gpu": gpu,
            "h2d": 1.0,
            "forward": forward,
            "serialize": 1.0,
            "total": latency - 3,
        }
        return {
            "i": i,
            "status": 200,
            "latency_ms": latency,
            "t_start_s": i / 10,
            "t_end_s": i / 10 + 0.1,
            "server": server,
        }

    rej = {
        "status": 503,
        "type": "/errors/queue_full",
        "latency_ms": 300.0,
        "t_start_s": 0.0,
        "t_end_s": 0.3,
    }
    write("rel-diag-a-base-c5", [ok(i, 100, 50, 13, 11) for i in range(10)])
    write(
        "rel-diag-a-c32-noretry",
        [ok(i, 400, 300, 90, 20) for i in range(10)] + [dict(rej, i=10 + k) for k in range(5)],
    )
    out = diag.analyze(tmp_path)["a"]
    burst = out["burst_c32_noretry"]
    assert burst["latency_success_ms"]["p99"] == 400 and burst["latency_rejected_ms"]["p50"] == 300
    assert burst["server_stages_ms"]["queue"]["p50"] == 300
    assert (
        burst["derived_per_request_ms"]["slot_host"]["p50"] == 90 - 1 - 20
    )  # per request, not from percentiles
    assert burst["derived_per_request_ms"]["outside"]["p50"] == 3
    assert burst["resources"]["client_cpu_pct"] == 97.0
    assert out["burst_over_own_baseline"]["gpu_p50"] == round(90 / 13, 2)
