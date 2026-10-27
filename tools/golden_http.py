"""Record or compare real-frame API responses (golden reference across deployments).

Private tool: needs local nuScenes via a BEVDet checkout (run from its root). The golden file
contains model outputs on restricted data; keep it local.

    python tools/golden_http.py record --url http://127.0.0.1:8766 --out golden.json \
        --select even:10
    python tools/golden_http.py compare --url http://127.0.0.1:8000 --golden golden.json

compare checks, per frame: HTTP status, box JSON byte-identical, model identity (checkpoint and
config sha256), frame and units. Frame indices follow NuScenesDataset order (sorted timestamps).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import mmcv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_direct import request_parts, select

from bevserve.constants import CAMERA_ORDER


def post(client: httpx.Client, info: dict[str, Any], rid: str) -> httpx.Response:
    images, calibration = request_parts(info)
    files = [(c, (f"{c}.jpg", images[c], "image/jpeg")) for c in CAMERA_ORDER]
    return client.post(
        "/v1/detect",
        files=files,
        data={"calibration": json.dumps(calibration)},
        headers={"X-Request-ID": rid},
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["record", "compare"])
    p.add_argument("--url", required=True)
    p.add_argument("--ann-file", default="data/nuscenes/bevdetv3-trainval_infos_val.pkl")
    p.add_argument("--select", default="even:10")
    p.add_argument("--out")
    p.add_argument("--golden")
    args = p.parse_args()
    infos = sorted(mmcv.load(args.ann_file)["infos"], key=lambda e: e["timestamp"])
    client = httpx.Client(base_url=args.url, timeout=60)
    model = client.get("/v1/model").json()

    if args.mode == "record":
        indices = select(args.select, infos)
        frames = []
        for idx in indices:
            resp = post(client, infos[idx], f"golden-{idx}")
            resp.raise_for_status()
            body = resp.json()
            frames.append(
                {
                    "index": idx,
                    "boxes": body["boxes"],
                    "frame": body["frame"],
                    "units": body["units"],
                }
            )
        out = {
            "model": {k: model[k] for k in ("name", "checkpoint_sha256", "config_sha256")},
            "frames": frames,
        }
        Path(args.out).write_text(json.dumps(out))
        print(json.dumps({"recorded": len(frames), "boxes": sum(len(f["boxes"]) for f in frames)}))
        return 0

    golden = json.loads(Path(args.golden).read_text())
    identity_ok = all(model[k] == v for k, v in golden["model"].items())
    rows = []
    for frame in golden["frames"]:
        idx = frame["index"]
        resp = post(client, infos[idx], f"compare-{idx}")
        row: dict[str, Any] = {"index": idx, "status": resp.status_code}
        if resp.status_code == 200:
            body = resp.json()
            row["boxes_identical"] = json.dumps(body["boxes"]) == json.dumps(frame["boxes"])
            row["count"] = [len(frame["boxes"]), len(body["boxes"])]
            row["frame_units_ok"] = (
                body["frame"] == frame["frame"] and body["units"] == frame["units"]
            )
            row["identity_ok"] = (
                body["model"]["checkpoint_sha256"] == golden["model"]["checkpoint_sha256"]
                and body["model"]["config_sha256"] == golden["model"]["config_sha256"]
                and resp.headers.get("x-request-id") == f"compare-{idx}"
            )
        rows.append(row)
    passed = identity_ok and all(
        r["status"] == 200 and r["boxes_identical"] and r["frame_units_ok"] and r["identity_ok"]
        for r in rows
    )
    print(json.dumps({"pass": passed, "model_identity_ok": identity_ok, "frames": rows}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
