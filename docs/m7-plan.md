# M7 plan: production reliability and load management

Question: is `BEVSERVE_PREPROCESS_THREADS=12` safe as the default for sustained operation in the
tested deployment envelope (one GPU host, loopback clients, production `max_in_flight` 5)?
Performance is settled (M6); this milestone is about admission, failure handling and sustained
behaviour. Baseline: commit 7c6eaff, image `bevserve:m6-20261027-224240`
(`sha256:8d1a445473a106c8d55df04bfe7da2da92e4826fb8d678650b65dd4667d580fb`), whose bevserve package
fingerprint equals HEAD's (`1d6df56e…`).

## 1. Request lifecycle and resource ownership (code walkthrough)

| Step | Code | Thread | Holds | Released |
| --- | --- | --- | --- | --- |
| 1. Connection, body | uvicorn (h11), `middleware.RequestIdMiddleware`, `BodyLimitMiddleware` (`content-length` ≤ 50 MiB, streamed count) | event loop | socket, body chunks | response sent / disconnect |
| 2. Readiness gate | `api._require_ready` → `Service.readiness` | loop | — | — |
| 3. Multipart parse | `api._read_form` → Starlette `request.form` (parts ≤ 1 MiB in memory, larger spooled to `/tmp`, a 1 GiB tmpfs in the container) | loop | form, `UploadFile`s | `form.close()` in `finally` |
| 4. Part / calibration / header checks | `_collect_parts`, `_parse_calibration`, `_check_images` (reads all six parts to `bytes`, Pillow header parse, exact 1600x900) | loop | six JPEG `bytes` | end of request |
| 5. **Admission** | `executor.try_admit` (lock, `in_flight < max_in_flight` else `QueueFullError` → 503 `queue_full`, `Retry-After: 1`) | loop | one token | step 9 or `Admission.__exit__` if no job was submitted |
| 6a. Serial preprocessing | `anyio.to_thread.run_sync(_decode_and_build)` (anyio limiter, 40 threads) | anyio worker | decoded images, planes | function return |
| 6b. Parallel preprocessing | `run_sync(_decode_and_build_parallel)` → `CameraPool.build`: 6 tasks on the 12-worker pool; waits for **all** six; errors in serial precedence | anyio worker + pool workers | per-task image, plane | all six tasks done before return or raise |
| 7. GPU job | `executor.run` → `Admission.submit` → 1-worker slot pool `_job` → `detector.detect` | slot thread | GPU slot | job end |
| 8. Deadline | `asyncio.wait_for(shield(...), deadline_s=10)`: queued job cancelled (token released by done-callback), running job keeps the token until it finishes; result discarded | loop | — | — |
| 9. Token release | `future.add_done_callback(admission.release)` | slot thread | — | when the job completes, fails or is cancelled |
| 10. Response | `DetectResponse` → `JSONResponse` (after the token is released) | loop | body | sent |
| Fatal | `FatalDetectorError` → executor refuses all jobs → `Service._on_fatal` → `os._exit(70)` after 0.5 s; Docker `on-failure:5` restarts | slot thread, loop | — | process exit |
| Shutdown | lifespan `finally` → `executor.shutdown(wait=False, cancel_futures=True)`, `CameraPool.shutdown(wait=True, cancel_futures=True)`, OpenCV threads restored | loop | — | — |

Properties that hold by construction (code reading, plus tests where noted):

- The token covers preprocessing, queueing and GPU time, not parsing or serialization.
- Exceptions and cancellation cannot leak a token: preprocessing failures leave the `with admission`
  block (`__exit__` releases); `anyio.to_thread.run_sync` defers cancellation until the thread
  returns (anyio 4.15, `abandon_on_cancel=False`); once submitted, release is tied to the job future.
- Work is bounded: at most 5 admitted requests → ≤ 5 busy anyio threads and ≤ 30 queued camera
  tasks; the pool has 12 workers; the GPU runs one job at a time.
- A request whose client disconnects is not cancelled by uvicorn; it completes and frees its token.

## 2. Evidence: what is and is not established

| Behaviour | Serial (M5 image) | Parallel 12 (M6 image) |
| --- | --- | --- |
| Byte-identical inputs and outputs | — | **verified**: 6,019 frames inputs; GPU and 10-frame HTTP golden; golden in 12 A/B containers |
| Latency/throughput at `max_in_flight` 16 | verified (M5 levels) | **verified** (M6 A/B, C = 1 and 8) |
| Admission at production `max_in_flight` 5 (C = 8, 16; with and without Retry-After; recovery) | **verified** (M5) | **not tested** (A/B used 16) |
| 100-request soak, memory/drift, golden after load | verified (M5; GPU growth 88 MiB of a 100 MiB limit) | **not tested** |
| Fatal error → exit 70 → restart → ready | verified (M4 suite, retest) | **not tested in a container** (C2 ran the fault step with threads 0) |
| Clean shutdown | verified | **not tested in a container** |
| Init failures (hash, no GPU, missing op) | verified | not tested (pool is created before init; expected identical) |
| Malformed/corrupt images, error precedence | verified (API tests) | **verified by tests** (unit + API, incl. 1–6 corrupt cameras) |
| Concurrent requests get their own inputs | — | verified with synthetic data in tests (8 concurrent); **outputs under concurrent real load not checked** (the A/B checked golden sequentially after load) |
| Thread lifecycle, OpenCV setting restore | — | verified by unit tests |
| Sustained memory and thread counts | — | **not tested** (A/B trials ~30 s; RSS rose ~380 MiB within each, peak 2.6–2.7 GiB) |

Stage A probes (CPU only, in-process FakeDetector app, nothing committed): a client disconnect
during upload raises an **unhandled `ClientDisconnect`, logged as an error with a traceback**
(the 500 the error handler produces cannot reach the disconnected client); the exception escapes
to the ASGI server; the token count stays 0 (it happens before admission). A burst of 192 requests (64
concurrent, a quarter with a truncated camera, `max_in_flight` 5, 12 threads) returned only 200,
503 `queue_full` and 422; admission returned to 0; the next request succeeded; threads stayed
bounded and the pool's workers were joined at shutdown.

## 3. Risk register

Likelihood and impact are for the stated deployment envelope.

| # | Risk | Likelihood | Impact | Existing evidence | Test gap / action |
| --- | --- | --- | --- | --- | --- |
| R1 | Admission at limit 5 behaves differently in parallel mode (shorter hold time → different rejection rate, goodput, latency of admitted requests) | high (by design) | medium | none for parallel | **GPU test** (C2) |
| R2 | Memory not stable under sustained parallel load (RSS from allocator/arena growth, GPU caching allocator) | unknown | high (OOM restart) | short trials only; M5 serial GPU +88 MiB in 100 requests | **GPU test** with warm-up separated and slope bounded (C3) |
| R3 | Outputs wrong under concurrent load (thread-safety of Pillow/OpenCV/NumPy in parallel tasks) | low (functions share no state; libraries thread-safe for separate objects) | high | synthetic concurrent tests; sequential golden | **harness**: per-response output hash, identical for every repeat of each input, during load |
| R4 | Fatal restart / shutdown differ with the pool (threads blocking exit, CancelledError at shutdown) | low (`os._exit`; uvicorn drains requests before lifespan shutdown) | medium | serial only | **GPU test**: fault retest and shutdown with threads 12 (harness option) |
| R5 | Client disconnect during upload → unhandled `ClientDisconnect`, logged with a traceback | **confirmed** (probe) | low (log noise, misleading error counts; no leak; pre-existing in M5) | probe | fix in the next image (catch `ClientDisconnect` in `_read_form`, log at info); not a blocker for the default |
| R6 | Pre-admission cost: rejected requests are fully uploaded, parsed and header-checked; concurrent parsing is not bounded (no uvicorn `limit_concurrency`); parts > 1 MiB spool to a RAM-backed 1 GiB tmpfs | low with real clients (6 parts ≈ 0.9 MB) | medium under floods | M5 C = 16 bursts fine | **GPU test**: C = 32 burst, measure RSS/CPU and rejection latency; config-only mitigation (`limit_concurrency`) only if evidence demands |
| R7 | `Retry-After: 1` is mis-scaled (a token frees every ~20–30 ms in parallel mode), so clients that honour it lose goodput | high | low (safe, conservative) | M5: honour 37 vs 42 req/s | measure goodput; no change proposed |
| R8 | Threads approach the Compose `pids: 512` limit (counts threads): idle server ~162, peaks 264 serial / 213 parallel at in-flight 16 | low | high (thread creation failure) | M6 samples | track thread count in all GPU tests |
| R9 | Preprocessing has no deadline and is not covered by stall detection: a hung decode would hold a token while `/readyz` stays ready | very low (inputs bounded to 1600x900 before decode) | medium | same in serial | hypothetical: no code |
| R10 | Log volume under overload (one line per rejected request) | medium | low | — | observe in C2; no change |
| R11 | OpenCV process-wide single-thread setting affects other code | very low (no OpenCV in the inference path; parity verified) | low | tests, parity | none |

## 4. Recommendation on source changes

**No service source change is required before GPU testing.** Every open question (R1–R4, R6–R8)
can be answered on the verified M6 image with harness-only changes, which keeps the tested binary
identical to the one already verified for parity. R5 is a confirmed, pre-existing, minor defect
that does not depend on the preprocessing mode; fix it together with any later image change (it
needs a rebuild and re-verification).

## 5. Stage B: harness built (CPU only; service code unchanged)

The serving code (`src/`, `docker/`) is unchanged; the tested binary stays the verified M6 image
(package fingerprint `1d6df56e…` equals HEAD's).

| Capability | Where |
| --- | --- |
| Canonical output hash per 200 response (model identity, frame, units, boxes; sorted keys, fixed separators; request id, timings and deployment metadata excluded); serial reference per input from a threads-0 container of the same image; per-run check: every 200 equals the reference, one output per input; input identity checked by request-body hashes | `tools/bench/load.py` (`--write-reference`, `--reference`) |
| Time-bounded load with **both** a duration and an attempt cap; in-flight requests finish within the client timeout, a hard stop at duration + timeout + 5 s cancels anything left; SIGINT/SIGTERM stop new requests, mark the run interrupted and exit 3 | `load.py` (`--duration`, `--max-attempts`) |
| Accounting: attempts, completed, rejected (503 `queue_full`), accepted (completed − rejected), success; goodput and attempt rate; admitted and rejected latency reported separately | `load.py` summary |
| Process-specific resources: server RSS, CPU %, process-tree threads, the container's cgroup `pids.current` / `pids.max`, server-attributed GPU memory, GPU utilization | `tools/bench/sampler.py` |
| Readiness poller (1 s, 2 s timeout; a timeout counts as not ready) | `tools/bench/readiness.py` |
| Stop conditions between phases (RSS, GPU memory, threads, not ready > 5 s, unexpected statuses); a violation aborts with project-scoped cleanup | `tools/bench/budget.py` |
| Sustained analysis: warm-up excluded; 60 s bins; least-squares RSS slope with a 95% t interval widened for residual lag-1 autocorrelation (effective sample size); raw peaks reported apart from trends; GPU-memory plateau vs warm-up peak; thread range; latency drift | `tools/bench/sustained.py` |
| Stages `rel-admission` (C1/C2) and `rel-sustained` (C3/C4) | `tools/m5_bench.sh` |
| Gates: a step runs only after a passing, non-test-mode earlier step on the same image and thread setting; fake Docker, fake nvidia-smi or `BENCH_TEST_MODE` make a run test-mode | `tools/bench/gate.py`, `m5_bench.sh` |
| Fault recovery with `PREPROCESS_THREADS` (runtime setting verified from the container's `Config.Env`), existing exit-70 / restart / recovery assertions, plus a clean-shutdown check; `run.json` for the gate | `tools/m5_fault_retest.sh` |
| Criteria for both stages | `tools/bench/criteria.py` |

CPU-only verification: ingress tests (`tests/api/test_ingress.py`: 32 concurrent uploads in serial
and parallel mode release tokens, threads and memory; truncated, incomplete and garbage bodies are
4xx before admission; disconnects during upload and after admission); unit tests for hashing,
accounting, trend statistics, gates, budgets and validators; fake-Docker orchestration tests for
a passing C1 and C3, and failing runs for each failure class (wrong outputs, unexpected 5xx,
thread budget, RSS budget, lost readiness, failed golden, interruption, gates, fault retest
with a wrong runtime setting).

The fault-retest fake reproduces PID-1 signal semantics: uvicorn re-raises SIGTERM after its
graceful shutdown, which the kernel ignores for a container's PID 1 (hence exit 0 in M4); a
plain process would report 143. The fake starts the server with SIGTERM ignored, which has the
same effect without hiding a server that does not shut down.

## 6. Clarifications (fixed before the GPU runs)

- **Readiness during ordinary overload: must stay ready.** `queue_full` is admission control, not
  unreadiness; a probe that fails or exceeds 2 s during overload would make an orchestrator
  restart a healthy service. Any not-ready probe while the parallel service runs fails the
  criterion; more than 5 s not ready stops the run.
- **Admitted vs rejected latency**: admitted = latency of 200 responses; rejected = latency of 503
  `queue_full` responses; reported separately per phase and never mixed.
- **Reference p99**: the p99 of 200 responses in a C = 5 phase (= `max_in_flight`, so admission
  never refuses), honouring Retry-After, in the same parallel container, right before the overload
  phases.
- **Memory warm-up and stabilization**: a fixed, documented warm-up (C3: 120 s, C4: 300 s) is
  excluded; stabilization is then judged by the trend over 60 s bin means (slope interval),
  next to the raw post-warm-up maximum and 99th percentile, and for GPU memory by the post-warm-up
  maximum against the warm-up peak.
- **Thread variation under stable load**: in the sustained phase (constant C, no overload), the
  post-warm-up range of process-tree threads; in overload phases only the maximum (anyio may add
  idle workers up to the admitted concurrency).
- **Client disconnects vs server failures**: a disconnect is client-side; it produces no
  client-visible response and is not counted as a server error. Server failures are 5xx other than
  `queue_full`, transport errors on connected clients, and malformed 200 bodies. The harness's
  own clients never disconnect early except on interruption, which aborts the run.
- **Ingress vs GPU admission**: ingress (parsing, part and header checks) is before admission and
  bounded only by the number of client connections; admission bounds preprocessing + GPU work to 5
  requests. Ingress cost shows up as rejected-request latency, CPU and RSS during the C = 32
  no-retry phases; it is reported, and judged only through the readiness, admitted-latency and
  resource criteria.

## 7. GPU experiments (operator-run; each step only after review of the previous one)

Common inputs: `IMAGE=bevserve:m6-20261027-224240`,
`EXPECT_IMAGE_ID=sha256:8d1a445473a106c8d55df04bfe7da2da92e4826fb8d678650b65dd4667d580fb`,
`STAGE_A_DIR=outputs/m4-20261027-224240` (must contain the 12-thread golden and recreation checks),
`REL_THREADS=12`, production `max_in_flight` 5. Every run: verified image and package
fingerprint, port free, no project leftovers, no other GPU process above 1 GiB, project-scoped
verified cleanup, private outputs under `outputs/`.

| Step | Command (see section 10) | Content | Bounds |
| --- | --- | --- | --- |
| C1a | `m5_fault_retest.sh`, `PREPROCESS_THREADS=12` | fatal error → exit 70 → restart → ready → 200; no further restarts; clean shutdown | ~1 min, ≤ 8 requests |
| C1b | `m5_bench.sh MODE=pilot STAGES=rel-admission` | reference (20 inputs, serial); parallel: baseline C = 5 10 s; C = 8 honour 10 s; C = 32 no-retry 10 s; recovery after each; golden | ~2 min; ≤ 1,000 attempts per phase |
| C2 | `MODE=full STAGES=rel-admission PILOT_DIR=<C1b> FAULT_DIR=<C1a>` | baseline 30 s; C = 8, 16, 32 × no-retry / honour, 30 s each, recovery after each; golden; serial context run (C = 16 honour) | ~6 min; ≤ 3,000 attempts per phase |
| C3 | `MODE=pilot STAGES=rel-sustained PRIOR_DIR=<C2>` | C = 4 honouring, 10 min in 2-min blocks (golden after each), 120 s warm-up | 10 min (max 15) |
| C4 | `MODE=full STAGES=rel-sustained PILOT_DIR=<C3> SUSTAINED_AUTH=1` | C = 4 honouring, default 30 min in 5-min blocks, 300 s warm-up | needs separate authorization; max 60 min |

Request counts are bounded by both time and attempt caps; C3/C4 caps are 100 attempts per second
of block. The C4 duration is decided after C3, from its results, not assumed.

Stop conditions (checked after every phase or block; violation aborts with cleanup): server
RSS > 8 GiB, server GPU memory > 4 GiB, server threads > 400, not ready > 5 s, any transport
error, non-admission 5xx or malformed response. Budgets can only be lowered.

## 8. Acceptance criteria (frozen; constants in `tools/bench/criteria.py`)

Counted in every mode (correctness, resource, harness):

| Criterion | Pass condition |
| --- | --- |
| Configuration | reference container threads 0; parallel container threads 12; `max_in_flight` 5; fresh containers (verified from the running container) |
| Outputs | every 200 equals the serial reference for its input; one output per input within each run |
| Golden | 10/10 frames byte-identical after the admission stage / after every sustained block |
| Readiness | every probe ready while the parallel service runs |
| Calm phases (rel-admission) | baseline C = 5 and every recovery run: every request 200 |
| Overload phases | only 200 or 503 `queue_full`; every 503 carries `Retry-After: 1`; ≥ 1 success per phase |
| Recovery | `/readyz` 200 within 2 s after every overload phase, then 25 requests at C = 5 all 200 (proves all 5 tokens returned) |
| Admitted latency | p99 of 200s in every overload phase ≤ 2 × baseline C = 5 p99 |
| Resources (both stages) | server threads ≤ 300, RSS ≤ 8 GiB, GPU memory ≤ 4 GiB throughout |
| Sustained requests | every request 200 (C = 4 < 5, so no rejection is possible) |
| Sustained threads | after warm-up: range ≤ 8, maximum ≤ 300 |
| Sustained GPU memory | after warm-up ≤ warm-up peak + 128 MiB and ≤ 2,048 MiB |
| Harness | every planned run, sampler, readiness poller valid; every cleanup verified |
| Fault (C1a, gate for C2) | all checks PASS with threads 12, including exit 70, restart, recovery, clean shutdown (exit 0) |

Statistical, counted in C4 only (reported in C3):

| Criterion | Pass condition |
| --- | --- |
| RSS trend | 95% CI upper bound of the post-warm-up slope ≤ 80 MiB/h (16 GiB limit − ~2.7 GiB peak ≈ 13 GiB, about 7 days at 80 MiB/h); fail if the estimate exceeds it; otherwise insufficient data |
| Latency drift | median of the last window / first window after warm-up ≤ 1.10 |

A 30–60-minute run can bound the RSS slope; it cannot show the absence of slow leaks, and the report
will say so.

## 9. Decision rule and remaining defects

- **APPROVE** the 12-thread default for this envelope: C1–C4 pass every counted criterion.
- **CONDITIONAL** (stay opt-in): C1–C3 pass, C4 not run or a statistical bound undecided; named
  limitations listed.
- **REJECT** (keep 0; fix first): any wrong output, unexpected 5xx, readiness loss, failed
  recovery, leaked token or thread, or a violated memory or thread bound.

Remaining genuine defect: **R5** (client disconnect during upload → unhandled server-side
`ClientDisconnect`, an error-level log entry with a traceback per event; no leak; pre-existing in M5; independent of the
preprocessing mode). Recommended fix, together with the next image change: catch
`ClientDisconnect` in `api._read_form` and log it at info level without a 500. It is not a blocker
for the default decision. No other code defect was found; the API is unchanged.

Rollback is the setting itself (`BEVSERVE_PREPROCESS_THREADS=0`, restart); no image change.

## 10. Operator commands

Common variables (C1a needs only the first four):

```bash
cd <bevserve checkout>
export IMAGE=bevserve:m6-20261027-224240
export EXPECT_IMAGE_ID=sha256:8d1a445473a106c8d55df04bfe7da2da92e4826fb8d678650b65dd4667d580fb
export CKPT_DIR=<directory with bevdet-r50-cbgs.pth>
export CKPT_SHA256=8b67e60de54e76fe64195816d90eb2ec1d73aac93ed7d834873107c509a13443
export HOST_PY=<model-environment python>
export STAGE_A_DIR=outputs/m4-20261027-224240 BEVDET_ROOT=<BEVDet checkout>
export GOLDEN_DIR=outputs/m4-golden REL_THREADS=12
```

- C1a: `PREPROCESS_THREADS=12 bash tools/m5_fault_retest.sh`
- C1b: `MODE=pilot STAGES=rel-admission bash tools/m5_bench.sh`
- C2: `MODE=full STAGES=rel-admission PILOT_DIR=outputs/m5-<C1b> FAULT_DIR=outputs/m5-fault-<C1a> bash tools/m5_bench.sh`
- C3: `MODE=pilot STAGES=rel-sustained PRIOR_DIR=outputs/m5-<C2> bash tools/m5_bench.sh`
- C4: `MODE=full STAGES=rel-sustained PILOT_DIR=outputs/m5-<C3> SUSTAINED_AUTH=1 REL_SUSTAINED_S=<agreed> bash tools/m5_bench.sh`

## 11. Definition of done

A deployment decision (APPROVE / CONDITIONAL / REJECT) recorded in `docs/m7-report.md` with the
evidence above, reproducible from committed harness code on the verified image; any default change
made only after that decision and in its own reviewed commit.

## 12. Outcome

Executed: C1a (PASS), C1b (FAIL: admitted p99 657.1 ms at C = 32 no-retry vs 2 × 107.1 ms;
recorded and unchanged), diagnostic 1 (`rel-diag`, single client process, confounded by client CPU
saturation) and diagnostic 2 (`rel-diagmp`, four client processes, rate-matched at ~245/s; the
12-thread arm's burst p99 was 2.31 × its own baseline, serial 1.37 ×; the 12-thread absolute p99
was lower, 254 vs 270 ms; one container per arm, no statistical claim; C1b's FAIL stands). C2, C3
and C4 were not run.
Decision: **CONDITIONAL**, default stays `BEVSERVE_PREPROCESS_THREADS=0`. Evidence, limitations
and the (unimplemented) ingress-protection proposal: [m7-report.md](m7-report.md).
