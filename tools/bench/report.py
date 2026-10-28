"""Assemble an M5 benchmark report (Markdown) from load.py summaries and sampler CSVs.

    python tools/bench/report.py --dir RUN_DIR > RUN_DIR/report.md

Reads RUN_DIR/meta.json (environment, provenance and plan written by tools/m5_bench.sh), every
*.summary.json, every *.samples.csv, runs.jsonl (execution records) and overhead.json. Reports
only what was measured, with sample counts and bootstrap intervals; it draws no conclusions beyond
the data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def fmt(stat: dict[str, Any], key: str) -> str:
    if not stat or stat.get("n", 0) == 0 or key not in stat:
        return "-"
    ci = stat.get(f"{key}_ci95")
    return f"{stat[key]:.1f}" + (f" [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "")


def samples(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {"n": 0}

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows if r.get(name) not in (None, "")]

    out: dict[str, Any] = {
        "n": len(rows),
        "duration_s": float(rows[-1]["t_unix"]) - float(rows[0]["t_unix"]),
    }
    for name in ("rss_mib", "cpu_pct", "gpu_util_pct", "proc_gpu_mem_mib", "threads"):
        values = col(name)
        if values:
            k = max(1, len(values) // 10)
            out[name] = {
                "first10pct_mean": sum(values[:k]) / k,
                "last10pct_mean": sum(values[-k:]) / k,
                "max": max(values),
                "mean": sum(values) / len(values),
            }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    args = p.parse_args()
    d = Path(args.dir)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    summaries = [json.loads(f.read_text()) for f in sorted(d.glob("*.summary.json"))]
    lines = ["# BEVServe M5 benchmark report", ""]
    if meta:
        lines += ["## Environment and configuration", ""]
        lines += [f"- **{k}**: {v}" for k, v in meta.items()]
        lines.append("")
    lines += [
        "## Method",
        "",
        "Closed-loop clients (tools/bench/load.py): each of C workers sends its next request",
        "when the previous one completes; warm-up requests are excluded. Latency is",
        "client-observed wall time (upload, server, download). Percentiles use nearest rank;",
        "brackets are 95% bootstrap intervals (2,000 resamples). Throughput counts successful",
        "responses only. Client and server share one host, so client CPU use is included in",
        "the host load.",
        "",
        "## Latency and throughput",
        "",
        "Outcomes: ok; rejected = 503 queue_full (admission control, expected under saturation);",
        "http_error, transport_error, malformed = failures.",
        "",
        "| Run | C | Measured | Outcomes | Statuses | Throughput (req/s) | p50 ms | p95 ms "
        "| p99 ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        lat = s["latency_ms"]
        statuses = "; ".join(f"{k}: {v}" for k, v in sorted(s["statuses"].items()))
        cells = [
            s["label"],
            s["concurrency"],
            s["requests_measured"],
            "; ".join(f"{k}: {v}" for k, v in sorted(s.get("outcomes", {}).items())),
            statuses,
            f"{s['throughput_rps']:.2f}",
            fmt(lat, "p50"),
            fmt(lat, "p95"),
            fmt(lat, "p99"),
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines += ["", "## Server stages (successful requests, ms; from Server-Timing)", ""]
    stage_names = sorted({k for s in summaries for k in s.get("server_stages_ms", {})})
    if stage_names:
        lines.append("| Run | " + " | ".join(f"{n} p50 / p95" for n in stage_names) + " |")
        lines.append("| --- |" + " --- |" * len(stage_names))
        for s in summaries:
            st = s.get("server_stages_ms", {})

            def point(stat: dict[str, Any], key: str) -> str:
                return fmt(stat, key).split(" [")[0]

            cells = [
                f"{point(st.get(n, {}), 'p50')} / {point(st.get(n, {}), 'p95')}"
                for n in stage_names
            ]
            lines.append(f"| {s['label']} | " + " | ".join(cells) + " |")
    else:
        lines.append("No Server-Timing data (instrumentation disabled).")
    ov_f = d / "overhead.json"
    if ov_f.exists():
        ov = json.loads(ov_f.read_text())
        lines += [
            "",
            "## Instrumentation overhead (balanced ABBA trials, C = 1, fresh service per trial)",
            "",
            "| Trial | Server-Timing | Measured | OK | p50 ms | mean ms | p95 ms |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for t in ov["trials"]:
            cells = [t["trial"], "on" if t["server_timing"] else "off", t["measured"], t["ok"]]
            cells += [
                f"{t[k]:.2f}" if t[k] is not None else "-" for k in ("p50_ms", "mean_ms", "p95_ms")
            ]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
        diffs = ", ".join(f"{x:+.2f}" for x in ov["paired_p50_diff_ms"])
        lines.append("")
        lines.append(f"Paired p50 differences (on - off, ms; pairs {ov['pairs']}): {diffs}.")
        if ov.get("ci95_ms"):
            lo, hi = ov["ci95_ms"]
            lines.append(
                f"Mean {ov['mean_diff_ms']:+.2f} ms, 95% t interval [{lo:+.2f}, {hi:+.2f}] ms "
                f"over {len(ov['paired_p50_diff_ms'])} pairs."
            )
        else:
            lines.append("Interval not available (incomplete trials).")
    ab_f = d / "ab.json"
    if ab_f.exists():
        ab = json.loads(ab_f.read_text())

        def num(v: Any, digits: int = 1) -> str:
            return "-" if v is None else f"{v:.{digits}f}"

        lines += [
            "",
            "## M6 A/B: serial (A) vs parallel preprocessing (B), fresh service per trial, ABBA",
            "",
            "Replicates are trial pairs (one A and one B container), not requests. Preprocessing",
            "= response-body decode + preprocess (wall time of the decode-and-build step).",
            "",
            "| Trial | Arm | C1 p50 | C1 p95 | C1 p99 | C1 prep p50 | C8 p50 | C8 p95 | C8 p99 "
            "| C8 req/s | C8 prep p50 | C8 gpu p50 | C8 queue p50 | peak RSS MiB | peak GPU MiB "
            "| C8 CPU % | C8 GPU util % |",
            "|" + " --- |" * 17,
        ]
        for t in ab["trials"]:
            keys = ("c1_p50_ms", "c1_p95_ms", "c1_p99_ms", "c1_prep_p50_ms", "c8_p50_ms",
                    "c8_p95_ms", "c8_p99_ms", "c8_rps", "c8_prep_p50_ms", "c8_gpu_p50_ms",
                    "c8_queue_p50_ms", "peak_rss_mib", "peak_gpu_mem_mib", "c8_cpu_pct_mean",
                    "c8_gpu_util_mean")  # fmt: skip
            cells = [str(t["trial"]), t["arm"].upper(), *(num(t[k]) for k in keys)]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "| Metric | ratio B/A | 95% CI | pairs |", "| --- | --- | --- | --- |"]
        for metric, e in ab["estimates"].items():
            ci = e.get("ci95")
            ci_txt = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
            lines.append(
                f"| {metric} | {num(e.get('ratio'), 3)} | {ci_txt} | {e.get('pairs', '-')} |"
            )
    runs_f = d / "runs.jsonl"
    if runs_f.exists():
        recs = [json.loads(x) for x in runs_f.read_text().splitlines() if x.strip()]
        bad = [r for r in recs if r.get("valid") is False or r.get("ok") is False]
        lines += ["", "## Execution records", ""]
        lines.append(
            f"{sum(r['kind'] == 'run' for r in recs)} client runs, "
            f"{sum(r['kind'] == 'sampler' for r in recs)} samplers, "
            f"{sum(r['kind'] == 'service' for r in recs)} services started, "
            f"{sum(r['kind'] == 'cleanup' and r.get('ok') for r in recs)} verified cleanups; "
            f"{len(bad)} invalid."
        )
        lines += [
            f"- INVALID {r['kind']} {r.get('label', r.get('seq'))}: "
            f"{r.get('problems', r.get('detail'))}"
            for r in bad
        ]
        for r in recs:
            if r["kind"] == "sampler":
                lines.append(
                    f"- sampler {r['label']}: {r.get('samples')} samples, "
                    f"max gap {r.get('max_gap_s')} s, "
                    f"server GPU memory in {r.get('gpu_attributed_fraction')} of samples, "
                    f"GPU pids {r.get('gpu_pids')}"
                )
    lines += ["", "## Resources (sampled ~1 Hz)", ""]
    for f in sorted(d.glob("*.samples.csv")):
        lines.append(f"- **{f.stem.replace('.samples', '')}**: `{json.dumps(samples(f))}`")
    extra = d / "notes.md"
    if extra.exists():
        lines += ["", extra.read_text()]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
