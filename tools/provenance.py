"""Source provenance for images and benchmark runs (stdlib only; runs on the host or in the image).

    python tools/provenance.py source --repo REPO     # git state + fingerprint of src/bevserve
    python - package < tools/provenance.py           # inside the image: installed bevserve package
    python tools/provenance.py compare HOST.json IMAGE.json

The package fingerprint is sha256 over sorted "relative path NUL sha256" lines of every file in the
bevserve package (no __pycache__ or .pyc), so the host source tree and the package installed in an
image can be compared by content, independent of git state or file times. `source` also records
whether the work tree is dirty (git status) and a hash of the uncommitted changes, because a short
commit hash alone does not identify a dirty tree.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def package_files(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def fingerprint(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(f"{path}\0{files[path]}\n".encode())
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def source(repo: Path) -> dict[str, object]:
    files = package_files(repo / "src" / "bevserve")
    status = git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
    changes = hashlib.sha256(git(repo, "diff", "HEAD", "--binary").encode())
    for line in status:  # untracked files are not in `git diff`; hash their content too
        if line.startswith("?? "):
            path = repo / line[3:]
            changes.update(f"{line[3:]}\0".encode() + hashlib.sha256(path.read_bytes()).digest())
    head = git(repo, "rev-parse", "HEAD").strip()
    return {
        "git_head": head,
        "git_describe": head[:7] + ("-dirty" if status else ""),
        "git_dirty": bool(status),
        "git_status": status,
        "uncommitted_changes_sha256": changes.hexdigest() if status else None,
        "package_fingerprint": fingerprint(files),
        "package_files": files,
    }


def installed_package() -> dict[str, object]:
    spec = importlib.util.find_spec("bevserve")  # locate without importing (no model stack needed)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("bevserve is not installed")
    files = package_files(Path(next(iter(spec.submodule_search_locations))))
    return {"package_fingerprint": fingerprint(files), "package_files": files}


def compare(host: dict[str, object], image: dict[str, object]) -> dict[str, object]:
    h, i = host["package_files"], image["package_files"]
    assert isinstance(h, dict) and isinstance(i, dict)
    return {
        "match": host["package_fingerprint"] == image["package_fingerprint"],
        "host_fingerprint": host["package_fingerprint"],
        "image_fingerprint": image["package_fingerprint"],
        "only_in_host": sorted(set(h) - set(i)),
        "only_in_image": sorted(set(i) - set(h)),
        "content_differs": sorted(p for p in set(h) & set(i) if h[p] != i[p]),
    }


def main(argv: list[str]) -> int:
    if argv[:1] == ["source"] and len(argv) == 3 and argv[1] == "--repo":
        result = source(Path(argv[2]).resolve())
    elif argv == ["package"]:
        result = installed_package()
    elif argv[:1] == ["compare"] and len(argv) == 3:
        result = compare(*(json.loads(Path(a).read_text()) for a in argv[1:]))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result.get("match", True) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
