"""Offline self-check for a BEVServe model environment (host or container).

    python -m bevserve.selfcheck                      # load, verify, warm up, run a fixed input
    python -m bevserve.selfcheck --record out.json    # store the fixed-input output as golden
    python -m bevserve.selfcheck --golden in.json     # compare the fixed-input output to golden
    python -m bevserve.selfcheck --readiness          # only report service readiness after init
    BEVSERVE_SELFCHECK_BLOCK_MODULE=mmcv._ext python -m bevserve.selfcheck --readiness

Meant to run with networking disabled (docker run --network none): it needs no dataset and no
network, only the operator-mounted checkpoint and the BEVSERVE_* settings. Prints one JSON object;
exit code 0 = all checks passed. The fixed input is seeded noise with valid camera geometry.
BEVSERVE_SELFCHECK_BLOCK_MODULE (test-only) makes one module unimportable in this process to
check how a missing compiled extension is reported; it is ignored by the server.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.abc
import importlib.machinery
import json
import os
import socket
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from bevserve.settings import Settings


class _Blocker(importlib.abc.MetaPathFinder):
    def __init__(self, name: str) -> None:
        self.name = name

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == self.name:
            raise ModuleNotFoundError(f"blocked {fullname}", name=fullname)
        return None


def fixed_inputs() -> Any:
    from bevserve.model.bevdet_detector import warmup_inputs
    from bevserve.preprocess import ModelInputs

    base = warmup_inputs()
    rng = np.random.default_rng(20261027)
    imgs = rng.standard_normal(base.imgs.shape).astype(np.float32)
    return ModelInputs(imgs, *base.as_tuple()[1:])


def heatmap_digest(detector: Any, inputs: Any) -> str:
    """sha256 of the raw CenterHead heatmap (10 x 128 x 128 float32) for the fixed input."""
    import hashlib

    import torch

    model, device = detector._model, detector._device
    tensors = [torch.from_numpy(a).unsqueeze(0).to(device) for a in inputs.as_tuple()]
    with torch.no_grad():
        feats, _, _ = model.extract_feat(points=None, img=tensors, img_metas=None)
        heatmap = model.pts_bbox_head(feats)[0][0]["heatmap"].float().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(heatmap).tobytes()).hexdigest()


def network_reachable() -> bool:
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=1).close()
        return True
    except OSError:
        return False


async def readiness() -> dict[str, Any]:
    from bevserve.cli import make_factory
    from bevserve.service import Service

    settings = Settings()
    service = Service(settings, make_factory(settings), exit_fn=lambda code: None)
    await service.initialize()
    state = service.readiness().model_dump()
    service.shutdown()
    return state


def run_model(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    import torch

    from bevserve.model.bevdet_detector import load_bevdet

    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "network_reachable": network_reachable(),
    }
    detector = load_bevdet(Settings(detector="bevdet"))
    report["model"] = {
        "name": detector.info.name,
        "checkpoint_sha256": detector.info.checkpoint_sha256,
        "config_sha256": detector.info.config_sha256,
    }
    inputs = fixed_inputs()
    report["heatmap_sha256"] = heatmap_digest(detector, inputs)
    boxes, scores, labels = detector.detect_native(inputs)
    again = detector.detect_native(fixed_inputs())
    output = {"boxes": boxes.tolist(), "scores": scores.tolist(), "labels": labels.tolist()}
    report["fixed_input"] = {
        "boxes": len(scores),
        "repeat_bit_identical": all(
            np.array_equal(a, b) for a, b in zip((boxes, scores, labels), again, strict=True)
        ),
    }
    ok = bool(report["fixed_input"]["repeat_bit_identical"])
    if args.record:
        with open(args.record, "w") as f:
            json.dump(
                {"model": report["model"], "heatmap_sha256": report["heatmap_sha256"], **output}, f
            )
        report["recorded"] = args.record
    if args.golden:
        with open(args.golden) as f:
            golden = json.load(f)
        same_shape = len(golden["scores"]) == len(scores)
        diff = None
        if same_shape and len(scores):
            diff = {
                "boxes": float(np.abs(np.asarray(golden["boxes"]) - boxes).max()),
                "scores": float(np.abs(np.asarray(golden["scores"]) - scores).max()),
            }
        labels_equal = same_shape and golden["labels"] == labels.tolist()
        identity = golden["model"] == report["model"]
        heatmap_equal = golden["heatmap_sha256"] == report["heatmap_sha256"]
        report["golden"] = {
            "box_count": [len(golden["scores"]), len(scores)],
            "labels_equal": labels_equal,
            "max_abs_diff": diff,
            "identity_equal": identity,
            "heatmap_bit_identical": heatmap_equal,
        }
        ok = (
            ok
            and labels_equal
            and identity
            and heatmap_equal
            and (diff is None or max(diff.values()) == 0.0)
        )
    return report, ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bevserve.selfcheck")
    parser.add_argument("--record", help="write the fixed-input output to this JSON file")
    parser.add_argument("--golden", help="compare the fixed-input output with this JSON file")
    parser.add_argument("--readiness", action="store_true", help="report readiness after init only")
    args = parser.parse_args(argv)
    blocked = os.environ.get("BEVSERVE_SELFCHECK_BLOCK_MODULE")
    if blocked:
        sys.meta_path.insert(0, _Blocker(blocked))
    if args.readiness:
        state = asyncio.run(readiness())
        print(json.dumps({"readiness": state, "blocked_module": blocked}))
        return 0 if state["status"] == "ready" else 1
    from bevserve.detector import DetectorInitError

    try:
        report, ok = run_model(args)
    except DetectorInitError as exc:
        print(json.dumps({"ok": False, "init_error": {"reason": exc.reason, "detail": exc.detail}}))
        return 1
    report["ok"] = ok
    print(json.dumps(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
