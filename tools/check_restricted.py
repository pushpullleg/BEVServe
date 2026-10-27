"""Fail if the repository would publish restricted artifacts or private identifiers.

Checks every file git would track (tracked + untracked-but-not-ignored), or the paths given:
- model weights / dataset files by extension;
- dataset-like directories (data/, samples/, sweeps/, maps/, checkpoints/, outputs/);
- files over 1 MiB, and images over 200 KiB;
- text containing absolute home-directory paths, private keys or common token formats.

    python tools/check_restricted.py            # scan the git working tree
    python tools/check_restricted.py FILE...    # scan specific files (e.g. from a pre-commit hook)
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

BLOCKED_SUFFIXES = {
    ".pth",
    ".pt",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
    ".safetensors",
    ".onnx",
    ".engine",
    ".plan",
    ".bin",
    ".tgz",
    ".tar",
    ".zip",
}
BLOCKED_DIRS = {"data", "samples", "sweeps", "maps", "checkpoints", "outputs"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
MAX_FILE_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 200 * 1024
TEXT_PATTERNS = {
    "home-directory path": re.compile(r"(/home/|/Users/)[A-Za-z0-9._-]+/"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
SELF = Path(__file__).resolve()


def candidate_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return [Path(p) for p in out.decode().split("\0") if p]


def problems(paths: Iterable[Path]) -> list[str]:
    found: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        name = path.as_posix()
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            found.append(f"{name}: blocked file type {path.suffix}")
        if BLOCKED_DIRS.intersection(path.parts[:-1]):
            found.append(f"{name}: inside a blocked directory")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            found.append(f"{name}: {size} bytes (limit {MAX_FILE_BYTES})")
        if path.suffix.lower() in IMAGE_SUFFIXES and size > MAX_IMAGE_BYTES:
            found.append(f"{name}: image of {size} bytes (limit {MAX_IMAGE_BYTES})")
        if path.resolve() == SELF or size > MAX_FILE_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in TEXT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                found.append(f"{name}:{line}: {label}")
    return found


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] if argv else candidate_files()
    found = problems(paths)
    for item in found:
        print(f"restricted: {item}")
    print(f"checked {len(paths)} files: {'FAIL' if found else 'OK'}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
