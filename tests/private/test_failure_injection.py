"""Private failure-injection tests (review correction 2) with the real model stack.

Initialization failures are injected in isolated subprocesses: CUDA hidden, or one compiled
extension made unimportable by an import hook in that subprocess. No file is renamed, removed or
modified. The runtime-fatal case injects an error into the real detector's forward call.

    BEVSERVE_PRIVATE_GPU=1 BEVSERVE_BEVDET_CONFIG=... BEVSERVE_CHECKPOINT=... \\
    BEVSERVE_CHECKPOINT_SHA256=... python -m pytest -q tests/private
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

if os.environ.get("BEVSERVE_PRIVATE_GPU") != "1":
    pytest.skip("private GPU tests: set BEVSERVE_PRIVATE_GPU=1", allow_module_level=True)
pytest.importorskip("mmdet3d")

ROOT = Path(__file__).resolve().parents[2]
PROBE = r"""
import asyncio, importlib.abc, json, os, sys
block = os.environ.get("BLOCK_MODULE")
if block:
    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == block:
                raise ModuleNotFoundError(f"blocked {name}", name=name)
            return None
    sys.meta_path.insert(0, Blocker())
from bevserve.api import create_app
from bevserve.cli import make_factory
from bevserve.settings import Settings
from tests.api.helpers import client_for
from tests.synthetic import multipart_files

async def main():
    settings = Settings(detector="bevdet")
    exits = []
    app = create_app(settings, make_factory(settings), exit_fn=exits.append)
    async with client_for(app, wait_ready=False) as client:
        ready = await client.get("/readyz")
        health = await client.get("/healthz")
        files, data = multipart_files()
        detect = await client.post("/v1/detect", files=files, data=data)
    print("RESULT " + json.dumps({
        "readyz": ready.status_code, "readiness": ready.json(), "healthz": health.status_code,
        "detect": detect.status_code, "detect_type": detect.json()["type"], "exits": exits,
    }))

asyncio.run(main())
"""


def probe(**env: str) -> dict[str, Any]:
    full_env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", **env}
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=ROOT,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, proc.stderr[-2000:]
    result: dict[str, Any] = json.loads(lines[-1][len("RESULT ") :])
    return result


def assert_not_ready(result: dict[str, Any], reason: str) -> None:
    assert result["readyz"] == 503
    assert result["readiness"]["status"] == "init_failed"
    assert result["readiness"]["reason"] == reason, result["readiness"]
    assert result["healthz"] == 200
    assert (result["detect"], result["detect_type"]) == (503, "/errors/not_ready")
    assert result["exits"] == []  # an init failure never exits


def test_missing_gpu_reports_cuda_unavailable() -> None:
    assert_not_ready(probe(CUDA_VISIBLE_DEVICES=""), "cuda_unavailable")


@pytest.mark.parametrize(
    "module", ["mmdet3d.ops.bev_pool_v2.bev_pool_v2_ext", "mmcv._ext"], ids=["bev_pool_v2", "mmcv"]
)
def test_missing_custom_op_reports_model_stack_missing(module: str) -> None:
    assert_not_ready(probe(BLOCK_MODULE=module), "model_stack_missing")


@pytest.mark.anyio
async def test_fatal_cuda_error_in_real_detector_exits_for_restart() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from bevserve.api import create_app
    from bevserve.model.bevdet_detector import BEVDetDetector, load_bevdet
    from bevserve.settings import Settings
    from tests.api.helpers import ExitRecorder, client_for, wait_until
    from tests.synthetic import multipart_files

    settings = Settings(detector="bevdet", fatal_exit_grace_s=0.05)
    detector: BEVDetDetector = load_bevdet(settings)

    def broken_forward(*_: object, **__: object) -> None:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    detector._model.forward_test = broken_forward
    exits = ExitRecorder()
    app = create_app(settings, lambda: detector, exit_fn=exits)
    files, data = multipart_files()
    async with client_for(app) as client:
        first = await client.post("/v1/detect", files=files, data=data)
        assert (first.status_code, first.json()["type"]) == (500, "/errors/inference_fatal")
        ready = await client.get("/readyz")
        assert ready.status_code == 503 and ready.json()["status"] == "fatal"
        await wait_until(lambda: exits.codes)
    assert exits.codes == [70]
