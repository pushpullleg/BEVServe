"""M6 A/B analysis (tools/bench/ab.py): ABBA pairing, ratio intervals, per-trial metrics, gate."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
ID = "sha256:" + "c" * 64


def _load() -> ModuleType:
    sys.path.insert(0, str(ROOT / "tools/bench"))
    spec = importlib.util.spec_from_file_location("bench_ab", ROOT / "tools/bench/ab.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ab = _load()


def test_abba_labels_and_pairs_put_one_a_and_one_b_in_each_pair() -> None:
    assert ab.labels(8) == [
        (1, "a"),
        (2, "b"),
        (3, "b"),
        (4, "a"),
        (5, "a"),
        (6, "b"),
        (7, "b"),
        (8, "a"),
    ]
    arms = dict(ab.labels(12))
    for a, b in ab.pairs(12):
        assert (arms[a], arms[b]) == ("a", "b")


def test_ratio_interval_is_a_t_interval_on_log_ratios() -> None:
    r = ab.ratio_interval([0.5, 0.5, 0.5])
    assert r["ratio"] == pytest.approx(0.5) and r["ci95"] == pytest.approx([0.5, 0.5])
    r = ab.ratio_interval([0.4, 0.5])
    logs = [math.log(0.4), math.log(0.5)]
    mean, sd = sum(logs) / 2, abs(logs[0] - logs[1]) / math.sqrt(2)
    half = 12.706 * sd / math.sqrt(2)
    assert r["ci95"] == pytest.approx([math.exp(mean - half), math.exp(mean + half)])
    assert ab.ratio_interval([0.5]) is None and ab.ratio_interval([0.5, 0.0]) is None


def _trial(d: Path, k: int, arm: str, scale: float) -> None:
    for c, rps in ((1, 10.0), (8, 40.0)):
        label = f"ab-t{k}-{arm}-c{c}"
        rows = [
            {"i": i, "warmup": i < 1, "status": 200,
             "timing_ms": {"decode": 10 * scale, "preprocess": 70 * scale}}
            for i in range(4)
        ]  # fmt: skip
        (d / f"{label}.requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        summary = {
            "requests_measured": 3, "success": 3, "outcomes": {"ok": 3},
            "latency_ms": {"p50": 100 * scale, "p95": 110 * scale, "p99": 120 * scale},
            "throughput_rps": rps / scale,
            "server_stages_ms": {"gpu": {"p50": 12.0}, "queue": {"p50": 1.0}},
        }  # fmt: skip
        (d / f"{label}.summary.json").write_text(json.dumps(summary))
    rows = [
        "t_unix,rss_mib,cpu_pct,threads,gpu_util_pct,gpu_mem_used_mib,proc_gpu_mem_mib,gpu_status"
    ]
    rows += [f"{100 * k + t},{2000 + t},{500 * scale},1,{10},1,1300,ok" for t in range(6)]
    (d / f"ab-t{k}.samples.csv").write_text("\n".join(rows) + "\n")


def test_analyze_pairs_trials_and_splits_phases(tmp_path: Path) -> None:
    runs = []
    for k, arm in ab.labels(4):
        _trial(tmp_path, k, arm, 0.4 if arm == "b" else 1.0)
        runs.append({"kind": "phase", "label": f"ab-t{k}", "c1": [100 * k, 100 * k + 2],
                     "c8": [100 * k + 3, 100 * k + 5]})  # fmt: skip
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs))
    out = ab.analyze(tmp_path, 4)
    est = out["estimates"]
    assert est["c1_p50_ms"]["ratio"] == pytest.approx(0.4) and est["c1_p50_ms"]["pairs"] == 2
    assert est["c8_rps"]["ratio"] == pytest.approx(2.5)
    assert est["c1_prep_p50_ms"]["ratio"] == pytest.approx(0.4)  # body decode + preprocess
    assert est["peak_rss_mib"]["ratio"] == pytest.approx(1.0)
    assert est["c8_cpu_pct_mean"]["ratio"] == pytest.approx(0.4)
    assert out["trials"][0]["peak_rss_mib"] == 2005 and out["trials"][0]["samples"] == 6


def test_missing_trial_makes_estimates_incomplete(tmp_path: Path) -> None:
    for k, arm in ab.labels(4)[:3]:
        _trial(tmp_path, k, arm, 1.0)
    est = ab.analyze(tmp_path, 4)["estimates"]["c1_p50_ms"]
    assert not est["complete"] and "ci95" not in est


def _pilot(d: Path, crit_status: str = "pass", **meta_over: object) -> Path:
    d.mkdir()
    meta = {
        "image": {"Id": ID},
        "test_mode": False,
        "params": {"mode": "pilot", "stages": ["ab"], "ab_threads": 12},
    }
    meta.update(meta_over)
    (d / "meta.json").write_text(json.dumps(meta))
    crit = {
        "mode": "pilot",
        "ab: requests": {"status": crit_status, "counted": True},
        "ab: c1 ratio": {"status": "insufficient_data", "counted": False},
    }
    (d / "criteria.json").write_text(json.dumps(crit))
    return d


def test_pilot_gate(tmp_path: Path) -> None:
    assert ab.pilot_gate(_pilot(tmp_path / "ok"), ID, 12) == []
    assert ab.pilot_gate(_pilot(tmp_path / "failed", "fail"), ID, 12)
    assert ab.pilot_gate(tmp_path / "ok", "sha256:" + "d" * 64, 12)  # other image
    assert ab.pilot_gate(tmp_path / "ok", ID, 24)  # other thread count
    full = _pilot(tmp_path / "full", params={"mode": "full", "stages": ["ab"], "ab_threads": 12})
    assert ab.pilot_gate(full, ID, 12)
    assert ab.pilot_gate(tmp_path / "absent", ID, 12)
    assert ab.pilot_gate(_pilot(tmp_path / "test-mode", test_mode=True), ID, 12)
    assert ab.pilot_gate(_pilot(tmp_path / "old", test_mode=None), ID, 12)
