"""Shared helpers for the fake-Docker orchestration tests (fixtures live in conftest.py)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAKES = Path(__file__).parent / "fakes"
IMAGE_ID = "sha256:" + "f" * 64

SUPPORTED = sys.platform.startswith("linux") and all(
    shutil.which(t) for t in ("curl", "ss", "setsid")
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run_bench(env: dict[str, str], **extra: str) -> tuple[int, str, Path | None]:
    before = set((ROOT / "outputs").glob("m5-2*")) if (ROOT / "outputs").exists() else set()
    p = subprocess.run(
        ["bash", str(ROOT / "tools/m5_bench.sh")],
        env={**env, **extra},
        capture_output=True,
        text=True,
        timeout=600,
    )
    new = sorted(set((ROOT / "outputs").glob("m5-2*")) - before)
    return p.returncode, p.stdout + p.stderr, new[-1] if new else None


def records(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines() if x.strip()]


def make_env(tmp_path: Path) -> dict[str, str]:
    state = tmp_path / "state"
    state.mkdir()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "bevdet-r50-cbgs.pth").write_bytes(b"not a real checkpoint")
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "http_golden.json").write_text(json.dumps({"model": {}, "frames": [{}] * 10}))
    stage_a = tmp_path / "stage-a"
    stage_a.mkdir()
    (stage_a / "image.txt").write_text(f"id={IMAGE_ID} size_bytes=1 created=now\n")
    (stage_a / "summary.txt").write_text(
        "PASS build\nFAIL fault: process exited with code 70 (saw: )\nPASS clean-shutdown\n"
    )
    retest = tmp_path / "retest"
    retest.mkdir()
    (retest / "summary.txt").write_text(
        f"PASS image id is the verified {IMAGE_ID}\n"
        "PASS process exited with code 70 (docker die event, then restart; restart count 0 -> 1)\n"
        "PASS request after recovery -> 200\n"
    )
    stage_c = tmp_path / "stage-c"  # an all-PASS M6 verification run (no fault retest)
    stage_c.mkdir()
    (stage_c / "image.txt").write_text(f"id={IMAGE_ID} size_bytes=1 created=now\n")
    (stage_c / "summary.txt").write_text(
        "PASS build\nPASS http-golden-parity (10 real frames, byte-identical boxes)\n"
        "PASS parallel: recreated container (new id) runs with BEVSERVE_PREPROCESS_THREADS=12\n"
        "PASS http-golden-parity with preprocess threads 12 (10 real frames, byte-identical)\n"
        "PASS clean-shutdown\n"
    )
    bevdet = tmp_path / "bevdet"
    bevdet.mkdir()
    return {
        **os.environ,
        "FAKE_STATE": str(state),
        "FAKE_PY": sys.executable,
        "DOCKER_CMD": str(FAKES / "docker"),
        "NVIDIA_SMI": str(FAKES / "nvidia-smi"),
        "PORT": str(free_port()),
        "IMAGE": "fake:tag",
        "EXPECT_IMAGE_ID": IMAGE_ID,
        "STAGE_A_DIR": str(stage_a),
        "FAULT_RETEST_DIR": str(retest),
        "BEVDET_ROOT": str(bevdet),
        "CKPT_DIR": str(ckpt),
        "CKPT_SHA256": hashlib.sha256(b"not a real checkpoint").hexdigest(),
        "GOLDEN_DIR": str(golden),
        "HOST_PY": sys.executable,
        "INPUTS": "synthetic:2",
    }


def kill_fake(env: dict[str, str]) -> None:
    """Never leave a fake server behind (it runs in its own session)."""
    state = Path(env["FAKE_STATE"])
    pid = state / "pid"
    if pid.exists():
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.killpg(int(pid.read_text()), signal.SIGTERM)


def m6(env: dict[str, str], **extra: str) -> dict[str, str]:
    out = {k: v for k, v in env.items() if k != "FAULT_RETEST_DIR"}
    out["STAGE_A_DIR"] = str(Path(env["STAGE_A_DIR"]).parent / "stage-c")
    return {**out, "STAGES": "ab", **extra}


def record_fake_golden(path: Path) -> None:
    """Golden responses for 10 synthetic frames, recorded from an in-process FakeDetector app."""
    sys.path.insert(0, str(FAKES))
    from golden_http import frame_parts  # the same frames the fake comparison posts
    from tests.api.helpers import build_app, client_for

    async def record() -> dict:
        app, _ = build_app()
        async with client_for(app) as client:
            model = (await client.get("/v1/model")).json()
            frames = []
            for idx in range(10):
                files, data = frame_parts(idx)
                body = (await client.post("/v1/detect", files=files, data=data)).json()
                frames.append({"index": idx, "boxes": body["boxes"], "frame": body["frame"],
                               "units": body["units"]})  # fmt: skip
        return {"model": {k: model[k] for k in ("name", "checkpoint_sha256")}, "frames": frames}

    import asyncio

    path.write_text(json.dumps(asyncio.run(record())))
