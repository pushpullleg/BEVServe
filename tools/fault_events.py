"""Verify a supervised fatal exit from recorded Docker events (stdlib only).

    docker events --since T0 --until T1 --filter type=container \
        --filter label=com.docker.compose.project=P --format '{{json .}}' > events.jsonl
    python tools/fault_events.py --events events.jsonl --container FULL_ID --exit-code 70 \
        --restarts-before 0 --restarts-after 1

Passes only if, for that container, the window holds exactly (restarts-after - restarts-before)
`die` events, every one with the expected exitCode attribute, each followed by a `start` event of
the same container (Docker's restart). The exit code comes from the daemon's die event, not from
application logs or the restart count. Prints the evidence as JSON; exit status 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def verify(
    events: list[dict[str, Any]], container: str, exit_code: int, restarts: int
) -> dict[str, Any]:
    own = sorted(
        (e for e in events if (e.get("Actor") or {}).get("ID", e.get("id")) == container),
        key=lambda e: int(e.get("timeNano", 0)),
    )
    dies = [e for e in own if e.get("Action", e.get("status")) == "die"]
    starts = [e for e in own if e.get("Action", e.get("status")) == "start"]
    codes = [(e.get("Actor") or {}).get("Attributes", {}).get("exitCode") for e in dies]
    restarted = all(any(int(s["timeNano"]) > int(d["timeNano"]) for s in starts) for d in dies)
    reasons = []
    if restarts < 1:
        reasons.append("restart count did not increase")
    if len(dies) != restarts:
        reasons.append(f"{len(dies)} die event(s) for {restarts} restart(s)")
    if any(c != str(exit_code) for c in codes):
        reasons.append(f"exit codes {codes}, expected {exit_code}")
    if dies and not restarted:
        reasons.append("no start event after a die event")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "container": container,
        "events_in_window": len(events),
        "container_events": [
            {"action": e.get("Action", e.get("status")), "time_nano": int(e.get("timeNano", 0))}
            for e in own
        ],
        "die_exit_codes": codes,
        "restarts": restarts,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--exit-code", type=int, required=True)
    p.add_argument("--restarts-before", type=int, required=True)
    p.add_argument("--restarts-after", type=int, required=True)
    a = p.parse_args()
    lines = Path(a.events).read_text().splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    result = verify(events, a.container, a.exit_code, a.restarts_after - a.restarts_before)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
