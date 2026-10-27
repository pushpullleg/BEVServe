"""The image-layer scanner: no false positives on Python .pth files, real findings flagged,
failures never reported as a pass."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scan", ROOT / "tools/scan_image_layers.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan = _load()
WEIGHTS = b"w" * 5000


def _tar(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _image(*layers: dict[str, bytes]) -> bytes:
    """A docker-save style archive in OCI layout: layer tars under blobs/sha256/."""
    outer = {"index.json": b"{}" * 600, "oci-layout": b"{}"}
    for layer in layers:
        data = _tar(layer)
        outer[f"blobs/sha256/{hashlib.sha256(data).hexdigest()}"] = data
    return _tar(outer)


def _run(archive: bytes, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(archive)))
    code = scan.main(
        [
            "--checkpoint-size",
            str(len(WEIGHTS)),
            "--checkpoint-sha256",
            hashlib.sha256(WEIGHTS).hexdigest(),
        ]
    )
    return code, json.loads(capsys.readouterr().out)


def test_clean_image_with_python_path_files_passes(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    archive = _image(
        {"opt/venv/lib/python3.10/site-packages/distutils-precedence.pth": b"import os" * 20},
        {
            "opt/venv/lib/python3.10/site-packages/bevdet-sources.pth": b"/opt/src/bevdet\n" * 10,
            "usr/bin/tool": b"x" * 2048,
        },
    )
    code, out = _run(archive, monkeypatch, capsys)
    assert code == 0 and out["ok"] and out["layers"] == 2 and out["findings"] == []


@pytest.mark.parametrize(
    ("path", "data", "kind"),
    [
        ("models/renamed.bin", WEIGHTS, "checkpoint-bytes"),
        ("opt/x/model.pth", b"m" * (1024 * 1024 + 1), "large-weight-file"),
        ("data/nuscenes/samples/CAM_FRONT/a.jpg", b"j" * 2048, "nuscenes-camera-sample"),
        ("data/nuscenes/v1.0-trainval/sample.json", b"{}" * 1024, "nuscenes-tables"),
        ("x/bevdetv3-trainval_infos_val.pkl", b"p" * 2048, "dataset-info-pickle"),
        ("srv/n015-2018.pcd.bin", b"l" * 2048, "lidar-point-cloud"),
        ("home/someone/notes.txt", b"t" * 2048, "home-directory-path"),
    ],
)
def test_restricted_content_is_flagged_even_in_earlier_layers(
    path: str,
    data: bytes,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = _image({path: data}, {"usr/bin/tool": b"x" * 2048})  # artifact only in layer 1
    code, out = _run(archive, monkeypatch, capsys)
    assert code == 1 and not out["ok"]
    assert any(f["kind"] == kind for f in out["findings"]), out


def test_empty_or_broken_stream_never_passes(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out = _run(b"", monkeypatch, capsys)
    assert code != 0 and not out["ok"]
    code, out = _run(_tar({"index.json": b"{}" * 600}), monkeypatch, capsys)  # no layers
    assert code != 0 and not out["ok"]
