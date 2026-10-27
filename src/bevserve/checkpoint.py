"""Operator-supplied checkpoint handling (review correction 5).

Trust model: the checkpoint path and its expected sha256 come only from operator configuration,
never from a request. A matching sha256 proves the bytes are the ones the operator trusted; it
does not make arbitrary pickled content safe, so loading additionally uses
torch.load(weights_only=True) (see model/bevdet_detector.py).

The file is read once; the hash is computed over exactly the bytes that are later deserialized,
so the file cannot change between verification and loading.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

from bevserve.detector import DetectorInitError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def resolve_operator_file(
    path: str | None, what: str, allowed_dirs: Sequence[str] = (), max_bytes: int | None = None
) -> Path:
    """Resolve an operator-configured path to a regular file, enforcing optional directories."""
    if not path:
        raise DetectorInitError(f"{what}_not_configured", f"no {what} path configured")
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise DetectorInitError(f"{what}_not_found", f"{what} file does not exist")
    if not resolved.is_file():
        raise DetectorInitError(f"{what}_not_regular_file", f"{what} path is not a regular file")
    if allowed_dirs:
        roots = [Path(d).expanduser().resolve() for d in allowed_dirs]
        if not any(resolved.is_relative_to(root) for root in roots):
            raise DetectorInitError(
                f"{what}_outside_allowed_dirs", f"{what} is outside the configured directories"
            )
    if max_bytes is not None and resolved.stat().st_size > max_bytes:
        raise DetectorInitError(f"{what}_too_large", f"{what} exceeds {max_bytes} bytes")
    return resolved


def read_verified_checkpoint(
    path: str | None,
    expected_sha256: str | None,
    allowed_dirs: Sequence[str] = (),
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    """Return (bytes, sha256) of the checkpoint, or raise DetectorInitError with a reason."""
    expected = (expected_sha256 or "").strip().lower()
    if not expected:
        raise DetectorInitError(
            "checkpoint_hash_missing", "an expected checkpoint sha256 must be configured"
        )
    if not _SHA256.match(expected):
        raise DetectorInitError("checkpoint_hash_invalid", "expected sha256 is not 64 hex digits")
    file = resolve_operator_file(path, "checkpoint", allowed_dirs, max_bytes)
    data = file.read_bytes()
    actual = sha256_bytes(data)
    if actual != expected:
        raise DetectorInitError(
            "checkpoint_hash_mismatch",
            f"checkpoint sha256 {actual[:12]}... does not match the configured {expected[:12]}...",
        )
    return data, actual
