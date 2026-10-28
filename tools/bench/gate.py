"""Gates between operator-run experiments (M7): a step starts only after a passing earlier one.

    python tools/bench/gate.py prior --dir RUN --image-id ID --mode pilot --stage rel-admission \\
        --threads 12
    python tools/bench/gate.py fault --dir RETEST --image-id ID --threads 12

`prior`: RUN is a tools/m5_bench.sh output whose meta.json records the same image id, mode,
stage (alone), thread setting and test_mode false, and whose criteria.json has every counted
criterion passing. `fault`: RETEST is a tools/m5_fault_retest.sh output whose run.json records the
image id, the thread setting and test_mode false, and whose summary is all PASS, including the
exit-70, restart, recovery, runtime-thread and clean-shutdown checks. Test-mode runs (fake Docker,
fake nvidia-smi or BENCH_TEST_MODE) never authorize anything. Prints JSON; exit 1 if refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

FAULT_REQUIRED = (
    "process exited with code 70 (docker die event",
    "ready again after restart",
    "request after recovery -> 200",
    "no further restarts",
)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def prior(run: Path, image_id: str, mode: str, stage: str, threads: int) -> list[str]:
    meta, crit = _load(run / "meta.json"), _load(run / "criteria.json")
    if not isinstance(meta, dict) or not isinstance(crit, dict):
        return [f"{run.name}: meta.json or criteria.json missing or unreadable"]
    params = meta.get("params") or {}
    reasons = []
    if meta.get("test_mode") is not False:
        reasons.append("prior run was a test-mode run (or predates test-mode recording)")
    if (meta.get("image") or {}).get("Id") != image_id:
        reasons.append("prior run used a different image")
    if params.get("mode") != mode or params.get("stages") != [stage]:
        reasons.append(f"prior run is not MODE={mode} STAGES={stage}")
    if params.get("rel_threads") != threads:
        reasons.append("prior run used a different preprocessing thread setting")
    counted = {k: v for k, v in crit.items() if isinstance(v, dict) and v.get("counted")}
    if not counted:
        reasons.append("prior run has no counted criteria")
    failed = [k for k, v in counted.items() if v.get("status") != "pass"]
    if failed:
        reasons.append(f"prior run has failed counted criteria: {failed}")
    return reasons


def fault(retest: Path, image_id: str, threads: int) -> list[str]:
    run = _load(retest / "run.json")
    try:
        lines = (retest / "summary.txt").read_text().splitlines()
    except OSError:
        lines = []
    if not isinstance(run, dict) or not lines:
        return [f"{retest.name}: run.json or summary.txt missing or unreadable"]
    reasons = []
    if run.get("test_mode") is not False:
        reasons.append("fault retest was a test-mode run")
    if run.get("image_id") != image_id:
        reasons.append("fault retest used a different image")
    if run.get("preprocess_threads") != threads:
        reasons.append("fault retest used a different preprocessing thread setting")
    failed = [x for x in lines if not x.startswith("PASS ")]
    if failed:
        reasons.append(f"fault retest has failed checks: {failed}")
    needed = (
        *FAULT_REQUIRED,
        f"container runs with BEVSERVE_PREPROCESS_THREADS={threads}",
        f"clean shutdown with BEVSERVE_PREPROCESS_THREADS={threads}",
    )
    for text in needed:
        if not any(x.startswith("PASS ") and text in x for x in lines):
            reasons.append(f"fault retest lacks the passed check '{text}'")
    return reasons


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prior")
    a.add_argument("--dir", required=True)
    a.add_argument("--image-id", required=True)
    a.add_argument("--mode", choices=["pilot", "full"], required=True)
    a.add_argument("--stage", required=True)
    a.add_argument("--threads", type=int, required=True)
    f = sub.add_parser("fault")
    f.add_argument("--dir", required=True)
    f.add_argument("--image-id", required=True)
    f.add_argument("--threads", type=int, required=True)
    args = p.parse_args()
    if args.cmd == "prior":
        reasons = prior(Path(args.dir), args.image_id, args.mode, args.stage, args.threads)
    else:
        reasons = fault(Path(args.dir), args.image_id, args.threads)
    print(json.dumps({"pass": not reasons, "reasons": reasons, "dir": Path(args.dir).name}))
    return 0 if not reasons else 1


if __name__ == "__main__":
    sys.exit(main())
