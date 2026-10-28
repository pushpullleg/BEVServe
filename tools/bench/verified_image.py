"""Gate: the image to benchmark is the one verified in M5 Stage A (stdlib only).

    python tools/bench/verified_image.py --expect ID --actual ID --stage-a DIR --retest DIR

Stage A is tools/m4_container_checks.sh on the M5 image (DIR/image.txt, DIR/summary.txt). Its only
accepted failure is the fault-check exit-code query that tools/m5_fault_retest.sh superseded; that
retest must have passed every check on the same image id. Prints the evidence (with sha256 of both
summaries) as JSON; exit status 1 unless every condition holds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SUPERSEDED = "FAIL fault: process exited with code 70 (saw: )"  # stage A; replaced by the retest
RETEST_REQUIRED = ("process exited with code 70 (docker die event", "request after recovery -> 200")


def gate(expect: str, actual: str, stage_a: Path, retest: Path) -> dict[str, Any]:
    reasons: list[str] = []
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expect):
        reasons.append("EXPECT_IMAGE_ID is not a full sha256 image id")
    if actual != expect:
        reasons.append(f"image id {actual} != EXPECT_IMAGE_ID")
    try:
        image_txt = (stage_a / "image.txt").read_text()
        a_lines = (stage_a / "summary.txt").read_text().splitlines()
        r_lines = (retest / "summary.txt").read_text().splitlines()
    except OSError as exc:
        return {"pass": False, "reasons": [*reasons, f"verification record missing: {exc}"]}
    if f"id={expect} " not in image_txt + " ":
        reasons.append("stage A verified a different image id")
    a_fail = [x for x in a_lines if not x.startswith("PASS ")]
    if not a_lines or any(x != SUPERSEDED for x in a_fail):
        reasons.append(f"stage A has unaccepted results: {[x for x in a_fail if x != SUPERSEDED]}")
    r_fail = [x for x in r_lines if not x.startswith("PASS ")]
    if not r_lines or r_fail:
        reasons.append(f"fault retest did not pass every check: {r_fail}")
    if f"PASS image id is the verified {expect}" not in r_lines:
        reasons.append("fault retest was not run on this image id")
    for needed in RETEST_REQUIRED:
        if not any(needed in x for x in r_lines):
            reasons.append(f"fault retest lacks the check '{needed}'")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "image_id": actual,
        "stage_a_dir": stage_a.name,
        "stage_a_summary_sha256": hashlib.sha256("\n".join(a_lines).encode()).hexdigest(),
        "stage_a_superseded": [x for x in a_fail if x == SUPERSEDED],
        "retest_dir": retest.name,
        "retest_summary_sha256": hashlib.sha256("\n".join(r_lines).encode()).hexdigest(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--expect", required=True)
    p.add_argument("--actual", required=True)
    p.add_argument("--stage-a", required=True)
    p.add_argument("--retest", required=True)
    a = p.parse_args()
    result = gate(a.expect, a.actual, Path(a.stage_a), Path(a.retest))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
