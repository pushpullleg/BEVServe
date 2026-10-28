# M5 plan: performance characterization and load testing

Goal: reproducible baselines for the verified GPU service. No optimization, no change to
inference behaviour. Fixed before any measurement; results: [m5-report.md](m5-report.md).

## Audit findings that shape the method

1. `timing_ms.inference` (response body) spans submit-to-result, so under concurrency it includes
   queue wait; `total` starts after middleware and excludes serialization. M5 adds an optional
   `Server-Timing` header with separate stages instead of changing the body.
2. Admission is taken before decode/preprocessing (CPU, ~90 ms) and released after inference
   (~12 ms GPU). With the production default `max_in_flight=5`, concurrency above ~5 is expected to
   be rejected (503 `queue_full`) even with an idle GPU. Measured, not changed, in M5.
3. GPU wall time is synchronized: the detector copies results to the host before returning. Pure
   forward time needs CUDA events (optional instrumentation).
4. Preprocessing runs in the default worker-thread pool; several requests' CPU stages overlap.
5. `Retry-After` is a fixed 1 s. Client and server share one host.

## Instrumentation (optional, default off)

`BEVSERVE_SERVER_TIMING=true` adds `Server-Timing: parse, validate, decode, preprocess, queue, gpu,
h2d, forward, serialize, total` (ms). `queue` = submit to slot start; `gpu` = detector wall time
(GPU-synchronized); `forward` = CUDA-event time of BEVDet's forward_test (incl. decode/NMS and its
own copy of results to the host); `h2d` = input upload. Body and outputs unchanged: tested
(`tests/api/test_server_timing.py`) and bit-identical with timing on/off on the real model
(`tests/private/test_bevdet_gpu.py`). Overhead is measured (stage `overhead`).

## Method

- Image: rebuilt from this commit and verified with `tools/m4_container_checks.sh` (code changed,
  so M4 parity is re-established before benchmarking). Image id, size, checkpoint sha256, GPU,
  driver, kernel, CPU, memory, configuration and inputs are recorded in `meta.json`.
- Inputs: 20 nuScenes val frames spread over the val set (6 JPEGs ~1600x900 + calibration each),
  read locally, pre-encoded once, cycled round-robin. Never written to outputs or images.
- Client: closed loop, C workers, warm-up excluded; nearest-rank p50/p95/p99 with 95% bootstrap
  intervals; throughput = successes / measurement window; every status, problem type and
  `Retry-After` recorded per request.
- Resources: `/proc` RSS and CPU of the server process, `nvidia-smi` GPU utilization and process
  GPU memory, ~1 Hz.

Two modes (`MODE=pilot|full`, required). The pilot validates the harness on the real GPU with
bounded requests; its statistical criteria are reported but not counted.

| Stage | Configuration | Pilot | Full |
| --- | --- | --- | --- |
| cold | timing off, max_in_flight 5 | 1 x (start -> ready, first request) | 3 x |
| levels | timing on, max_in_flight 16; sampled | C = 1, 2, 4, 8: 20 requests + 5 warm-up | 300 + 20 |
| overhead | max_in_flight 5, C = 1, fresh service per trial, ABBA order | 4 trials x (20 + 5) | 12 trials x (100 + 20) |
| admission | timing on, max_in_flight 5 (production); sampled | C = 8, 40 requests, with/without Retry-After; recovery | C = 8 and 16, 200 each |
| soak | timing on, max_in_flight 16, C = 2; sampled | 20 + 5, then readiness + golden parity | 100 + 20 (pilot soak); 1,000 only with `SOAK_FULL=1` |

Bounds are enforced (`tools/m5_bench.sh` header): request counts <= 1,000 (pilot <= 50), at most 16
overhead trials, admission concurrency 8 or 16, soak > 100 only with `SOAK_FULL=1`.

Overhead: per-trial client p50; pairs (A, B) are adjacent trials within an ABBA block, so linear
drift cancels; estimate = mean paired difference (on - off) with a 95% Student t interval. Every
trial is reported.

Execution validity (`tools/bench/validate.py`, `runs.jsonl`): a client run counts only if the client
exited 0 and wrote one well-formed record per planned request; a sampler only if it ran to the end
of its stage without gaps (> 3 intervals), its target process stayed alive, and the GPU memory it
recorded belongs to the serving process (nvidia-smi PID in the server's process tree, present in
>= 90% of samples, exactly one such process); unavailable GPU values are missing, never zero. The
golden comparison counts only if it ran over all 10 frames and passed. Each service removal is
verified (no containers or network of the run's project remain); a failed removal is reported and
the run exits 2. HTTP 503 `queue_full` responses are expected admission results (outcome
"rejected"), not execution failures; any other non-200, transport error or malformed 200 is a
failed request.

## Acceptance criteria (tools/bench/criteria.py)

| Criterion | Kind | Pass condition |
| --- | --- | --- |
| Execution | harness | every planned client run has a valid execution record |
| Samplers | harness | every sampled stage has a valid sampler record |
| Cleanup | harness | every started service was removed and the removal verified |
| Levels | correctness | every measured request at C = 1, 2, 4, 8 is ok |
| Admission correctness | correctness | only ok or rejected; >= 1 ok; every rejection carries `Retry-After: 1` |
| Admission recovery | correctness | `/readyz` status ready and 5/5 requests ok |
| Cold start | correctness | every start ready within 30 s and its first request ok |
| Soak requests | correctness | every measured request ok |
| Soak golden | correctness | ready afterwards; golden comparison ran over all frames and passed |
| Overhead | statistical | 95% CI upper bound of the paired p50 difference <= max(3 ms, 3% of off p50); fails if the estimate exceeds it; otherwise insufficient_data |
| Soak resources | statistical | RSS growth <= 200 MiB; serving-process GPU memory growth <= 100 MiB (mean of last vs first 10% of samples, >= 10 samples) |
| Soak drift | statistical | median latency last 10% / first 10% <= 1.25 (>= 5 requests per tenth) |

Harness and correctness criteria count in both modes; statistical criteria count in full mode only.
A missing measurement is `insufficient_data` and never passes; a stage that was requested but did
not produce valid results fails rather than showing as `not_run`. Latency and throughput values are
baselines; the provisional target (p95 server total <= 250 ms at C = 1) is information only.

## Provenance

`tools/provenance.py` records the work-tree state (git HEAD, dirty flag, `git status`, a sha256 of
the uncommitted changes) and compares the bevserve package installed in the image with
`src/bevserve` by content. `m5_bench.sh` aborts when they differ or when the image id is not
`EXPECT_IMAGE_ID`; containers report `git describe --dirty` (for example `3024183-dirty`) instead
of a bare commit.

## Stage A fault check (exit 70)

The M4 suite on the M5 image failed only "process exited with code 70" with an empty event
result, while the 500, the restart, the logged fatal exit and the recovery all passed. The query
ended its window at the current whole second (`--until $(date +%s)`). The daemon drops events
later than `--until`, so a die event earlier in that same second was cut off. The check now
starts at a fractional `--since`, ends 2 s in the future, records the raw events
(`fault_events.jsonl`), and `tools/fault_events.py` requires exactly one die event per restart
for that container, each with exitCode 70 and followed by a start. The re-check runs in
isolation with `tools/m5_fault_retest.sh` (no build).
