"""Fake golden comparison for orchestration tests (synthetic frames, no dataset, no GPU).

    python golden_http.py compare --url URL --golden FILE   # interface of tools/golden_http.py

Posts the synthetic frame recorded under each golden "index" to the running service and compares
the response byte for byte (boxes, frame, units, model identity, request id echo), producing the
same JSON as the real tool. A golden file is recorded with `record_golden` (in-process app).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tests.synthetic import calibration_payload, encode, multipart_files, synthetic_image  # noqa: E402

from bevserve.constants import CAMERA_ORDER  # noqa: E402


def frame_parts(index: int) -> tuple[list[Any], dict[str, str]]:
    images = {n: encode(synthetic_image(1000 * index + i)) for i, n in enumerate(CAMERA_ORDER)}
    return multipart_files(images, calibration_payload(with_ego_poses=True))


def compare(url: str, golden_path: str) -> int:
    import httpx

    golden = json.loads(Path(golden_path).read_text())
    with httpx.Client(base_url=url, timeout=30) as client:
        model = client.get("/v1/model").json()
        identity_ok = all(model.get(k) == v for k, v in golden["model"].items())
        rows = []
        for frame in golden["frames"]:
            idx = frame["index"]
            files, data = frame_parts(idx)
            resp = client.post(
                "/v1/detect", files=files, data=data, headers={"x-request-id": f"compare-{idx}"}
            )
            row: dict[str, Any] = {"index": idx, "status": resp.status_code}
            if resp.status_code == 200:
                body = resp.json()
                row["boxes_identical"] = json.dumps(body["boxes"]) == json.dumps(frame["boxes"])
                row["frame_units_ok"] = (
                    body["frame"] == frame["frame"] and body["units"] == frame["units"]
                )
                row["identity_ok"] = resp.headers.get("x-request-id") == f"compare-{idx}"
            rows.append(row)
    passed = identity_ok and all(
        r["status"] == 200 and r["boxes_identical"] and r["frame_units_ok"] and r["identity_ok"]
        for r in rows
    )
    print(json.dumps({"pass": passed, "model_identity_ok": identity_ok, "frames": rows}))
    return 0 if passed else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["compare"])
    p.add_argument("--url", required=True)
    p.add_argument("--golden", required=True)
    a = p.parse_args()
    return compare(a.url, a.golden)


if __name__ == "__main__":
    sys.exit(main())
