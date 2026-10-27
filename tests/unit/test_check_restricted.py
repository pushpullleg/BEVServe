from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_restricted", ROOT / "tools/check_restricted.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load()


@pytest.mark.parametrize(
    ("relpath", "content", "reason"),
    [
        ("model.pth", b"x", "blocked file type"),
        ("infos_val.pkl", b"x", "blocked file type"),
        ("data/readme.txt", b"x", "blocked directory"),
        ("docs/samples/a.txt", b"x", "blocked directory"),
        ("big.txt", b"a" * (1024 * 1024 + 1), "limit"),
        ("photo.jpg", b"\xff" * (200 * 1024 + 1), "image of"),
        ("notes.md", b"see /" + b"home/someone/data for frames", "home-directory path"),
        ("key.txt", b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----", "private key"),
        ("t.txt", b"token ghp_" + b"a" * 36, "GitHub token"),
    ],
)
def test_flags_restricted_content(
    tmp_path: Path, relpath: str, content: bytes, reason: str
) -> None:
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    found = check.problems([path])
    assert any(reason in f for f in found), found


def test_clean_file_passes(tmp_path: Path) -> None:
    path = tmp_path / "ok.py"
    path.write_text("print('hello')\n")
    assert check.problems([path]) == []


def test_repository_is_clean() -> None:
    files = [ROOT / p for p in check.candidate_files()] if (ROOT / ".git").exists() else []
    assert check.problems(files) == []
