from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from bevserve.boxes import detections_from_arrays
from bevserve.checkpoint import read_verified_checkpoint, resolve_operator_file
from bevserve.detector import DetectorInitError


@pytest.fixture
def ckpt(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "trusted" / "model.bin"
    path.parent.mkdir()
    path.write_bytes(b"pretend weights")
    return path, hashlib.sha256(b"pretend weights").hexdigest()


def _reason(**kwargs: object) -> str:
    with pytest.raises(DetectorInitError) as err:
        read_verified_checkpoint(**kwargs)  # type: ignore[arg-type]
    return err.value.reason


def test_matching_hash_returns_the_verified_bytes(ckpt: tuple[Path, str]) -> None:
    path, sha = ckpt
    data, actual = read_verified_checkpoint(str(path), sha.upper())
    assert data == b"pretend weights" and actual == sha


def test_hash_must_be_configured_and_well_formed(ckpt: tuple[Path, str]) -> None:
    path, _ = ckpt
    assert _reason(path=str(path), expected_sha256=None) == "checkpoint_hash_missing"
    assert _reason(path=str(path), expected_sha256="abc") == "checkpoint_hash_invalid"


def test_wrong_hash_is_rejected(ckpt: tuple[Path, str]) -> None:
    path, _ = ckpt
    assert _reason(path=str(path), expected_sha256="0" * 64) == "checkpoint_hash_mismatch"


def test_path_rules(ckpt: tuple[Path, str], tmp_path: Path) -> None:
    path, sha = ckpt
    assert _reason(path=None, expected_sha256=sha) == "checkpoint_not_configured"
    assert _reason(path=str(tmp_path / "nope"), expected_sha256=sha) == "checkpoint_not_found"
    assert _reason(path=str(tmp_path), expected_sha256=sha) == "checkpoint_not_regular_file"
    assert (
        _reason(path=str(path), expected_sha256=sha, allowed_dirs=[str(tmp_path / "other")])
        == "checkpoint_outside_allowed_dirs"
    )
    assert _reason(path=str(path), expected_sha256=sha, max_bytes=3) == "checkpoint_too_large"
    read_verified_checkpoint(str(path), sha, allowed_dirs=[str(path.parent)])


def test_symlink_is_judged_by_its_target(ckpt: tuple[Path, str], tmp_path: Path) -> None:
    path, _ = ckpt
    link = tmp_path / "other" / "link.bin"
    link.parent.mkdir()
    os.symlink(path, link)
    with pytest.raises(DetectorInitError) as err:
        resolve_operator_file(str(link), "checkpoint", allowed_dirs=[str(link.parent)])
    assert err.value.reason == "checkpoint_outside_allowed_dirs"


def test_boxes_convert_bottom_center_to_geometric_center() -> None:
    raw = np.array([[10.0, -2.0, -1.5, 4.0, 2.0, 1.6, 0.3, 1.0, -0.5]], dtype=np.float32)
    (det,) = detections_from_arrays(raw, np.array([0.8]), np.array([3]))
    assert det.center == pytest.approx((10.0, -2.0, -1.5 + 0.8))
    assert det.size == pytest.approx((4.0, 2.0, 1.6))
    assert det.yaw == pytest.approx(0.3) and det.velocity == pytest.approx((1.0, -0.5))
    assert det.label_id == 3 and det.score == pytest.approx(0.8)


def test_boxes_validate_shapes_and_values() -> None:
    assert detections_from_arrays(np.zeros((0, 9)), np.zeros(0), np.zeros(0, int)) == []
    with pytest.raises(ValueError):
        detections_from_arrays(np.zeros((2, 7)), np.zeros(2), np.zeros(2, int))
    with pytest.raises(ValueError):
        detections_from_arrays(np.zeros((2, 9)), np.zeros(3), np.zeros(2, int))
    bad = np.zeros((1, 9))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        detections_from_arrays(bad, np.zeros(1), np.zeros(1, int))
