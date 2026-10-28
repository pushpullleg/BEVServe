"""Exit-code verification from Docker events (tools/fault_events.py) and source provenance."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
CID = "c" * 64


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fault_events = _load("fault_events")
provenance = _load("provenance")


def _ev(action: str, t: int, cid: str = CID, **attrs: str) -> dict:
    return {
        "Type": "container",
        "Action": action,
        "Actor": {"ID": cid, "Attributes": attrs},
        "timeNano": t,
    }


def test_one_die_with_70_then_start_passes() -> None:
    events = [_ev("start", 1), _ev("die", 2, exitCode="70"), _ev("start", 3)]
    r = fault_events.verify(events, CID, 70, 1)
    assert r["pass"] and r["die_exit_codes"] == ["70"]


def test_empty_window_fails_instead_of_passing() -> None:  # the M5 stage-A symptom
    r = fault_events.verify([], CID, 70, 1)
    assert not r["pass"] and "0 die event(s) for 1 restart(s)" in r["reasons"]


def test_wrong_exit_code_fails() -> None:
    r = fault_events.verify([_ev("die", 2, exitCode="1"), _ev("start", 3)], CID, 70, 1)
    assert not r["pass"]


def test_other_containers_are_ignored_and_restart_is_required() -> None:
    other = _ev("die", 2, cid="d" * 64, exitCode="70")
    assert not fault_events.verify([other, _ev("start", 3)], CID, 70, 1)["pass"]
    assert not fault_events.verify([_ev("die", 2, exitCode="70")], CID, 70, 1)["pass"]
    assert not fault_events.verify([], CID, 70, 0)["pass"]


def test_extra_die_events_fail() -> None:
    events = [
        _ev("die", 2, exitCode="70"),
        _ev("start", 3),
        _ev("die", 4, exitCode="70"),
        _ev("start", 5),
    ]
    assert not fault_events.verify(events, CID, 70, 1)["pass"]
    assert fault_events.verify(events, CID, 70, 2)["pass"]


def test_installed_package_matches_work_tree_source() -> None:
    host = provenance.source(ROOT)
    assert len(str(host["git_head"])) == 40
    assert host["git_dirty"] == bool(host["git_status"])
    script = (ROOT / "tools/provenance.py").read_text()
    image = json.loads(
        subprocess.run(
            [sys.executable, "-", "package"],
            input=script,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert provenance.compare(host, image)["match"]  # editable install: same files


def test_compare_reports_differences() -> None:
    a = {"package_files": {"x.py": "1", "y.py": "2"}}
    b = {"package_files": {"x.py": "1", "y.py": "3", "z.py": "4"}}
    a["package_fingerprint"] = provenance.fingerprint(a["package_files"])
    b["package_fingerprint"] = provenance.fingerprint(b["package_files"])
    r = provenance.compare(a, b)
    assert not r["match"] and r["content_differs"] == ["y.py"] and r["only_in_image"] == ["z.py"]
