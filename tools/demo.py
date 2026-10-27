"""Exercise a running BEVServe instance with synthetic inputs.

    python tools/demo.py [--url http://127.0.0.1:8000]

Sends: health and readiness checks, one valid six-camera request, and three invalid requests.
All images and calibration values are generated; nothing comes from a dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.synthetic import encode, multipart_files, synthetic_image

from bevserve.constants import CAMERA_ORDER


def show(label: str, resp: httpx.Response) -> None:
    body = resp.json()
    summary = body.get("type") or body.get("status") or ""
    print(f"{label:34s} {resp.status_code}  {summary}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=30) as client:
        show("GET /healthz", client.get("/healthz"))
        show("GET /readyz", client.get("/readyz"))

        files, data = multipart_files()
        resp = client.post("/v1/detect", files=files, data=data, headers={"X-Request-ID": "demo-1"})
        show("POST /v1/detect (valid)", resp)
        if resp.status_code == 200:
            body = resp.json()
            print(
                f"  request_id={body['request_id']} model={resp.headers['x-model-version']} "
                f"frame={body['frame']} boxes={len(body['boxes'])}"
            )
            print(f"  first box: {json.dumps(body['boxes'][0])}")
            print(f"  timing_ms: {body['timing_ms']}")

        show("POST /v1/detect (5 cameras)", client.post("/v1/detect", files=files[1:], data=data))

        images = {n: encode(synthetic_image(i)) for i, n in enumerate(CAMERA_ORDER)}
        images["CAM_BACK"] = encode(synthetic_image(9, 1280, 720))
        bad_files, _ = multipart_files(images=images)
        show(
            "POST /v1/detect (1280x720 image)",
            client.post("/v1/detect", files=bad_files, data=data),
        )

        bad = json.loads(data["calibration"])
        bad["cameras"]["CAM_FRONT"]["sensor2ego"]["rotation_wxyz"] = [0.5, 0, 0, 0]
        show(
            "POST /v1/detect (bad quaternion)",
            client.post("/v1/detect", files=files, data={"calibration": json.dumps(bad)}),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
