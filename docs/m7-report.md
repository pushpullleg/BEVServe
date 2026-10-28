# M7 report: production reliability and load management

**Decision: CONDITIONAL.** Parallel six-camera preprocessing (`BEVSERVE_PREPROCESS_THREADS=12`)
remains opt-in; the deployment default stays serial (`0`). Correctness, readiness, admission,
recovery, fault handling and resource bounds held in every test, in both modes. One frozen
acceptance criterion failed (C1b: admitted-request latency under overload) and, in the controlled
diagnostic, the 12-thread mode still exceeded it relative to its own, faster baseline. Sustained
operation (C3/C4) was not run. Plan and frozen criteria: [m7-plan.md](m7-plan.md). Raw evidence is
kept privately, outside git.

Image under test, unchanged throughout: `bevserve:m6-20261027-224240`,
`sha256:8d1a445473a106c8d55df04bfe7da2da92e4826fb8d678650b65dd4667d580fb` (bevserve package
byte-identical to commit 7c6eaff's source; checkpoint sha256 `8b67e60d…3a13443`). Production
admission limit `max_in_flight` 5 in every M7 run. No serving code was changed in M7.

## 1. Stage A: architecture audit

- Admission is taken after multipart parsing, part and calibration checks and the six image-header
  checks (all on the event loop) and covers preprocessing, GPU queue and GPU execution; it is
  released when the GPU job finishes, or on leaving the request block if no job was submitted. By
  construction, exceptions and cancellation cannot leak a token (verified again by tests).
- Gaps confirmed before any GPU run: admission at the production limit, sustained memory, fatal
  restart and shutdown were untested with the pool; outputs under concurrent real load were
  unchecked; pre-admission work is unbounded except by connections; `Retry-After: 1` is coarse for
  a ~20–30 ms slot turnover.
- Confirmed pre-existing defect (CPU probe, both modes): a client disconnect during upload raises
  `ClientDisconnect`, which BEVServe does not handle: it is logged server-side as an unhandled
  error with a traceback and propagates to the ASGI server (the 500 that the error handler produces
  cannot reach the client, which has gone). No token or resource leaks.
  Not fixed (no serving-code changes in M7); documented by a test.

## 2. Stage B: harness (CPU-only verified)

Harness-only additions: canonical output hashes compared with a serial reference from the same
image; time- and attempt-bounded load with interruption handling; accounting of attempts,
completions, rejections and successes with admitted and rejected latency kept apart; server
process-tree threads, cgroup PIDs, per-thread CPU, server-attributed GPU memory; a 1 s readiness
probe; stop conditions between phases; sustained-trend analysis with autocorrelation-adjusted
intervals; gates so that only a passing, non-test-mode run authorizes the next step; the fault
retest with the thread setting and a clean-shutdown check; and a multi-process, rate-paced load
generator. 313 public tests pass, including fake-Docker orchestration tests for every failure class
(wrong outputs, unexpected 5xx, thread and memory budgets, lost readiness, failed golden, worker
failure, interruption, gates).

## 3. GPU results

### C1a, fault recovery with 12 threads: PASS (12/12)

Runtime setting verified from the container; call 2 → 500 `inference_fatal`; exit 70 from the
Docker die event; one restart; ready again; request after recovery 200; no further restarts;
clean shutdown exit 0.

### C1b, admission pilot (12 threads, single load-generator process): FAIL (recorded, unchanged)

| Phase | Success / 503 | Successful p99 | Result |
| --- | --- | --- | --- |
| C = 5 baseline, 10 s | 512 / 0 | 107.1 ms | reference |
| C = 8 honouring Retry-After, 10 s | 505 / 30 | 113.7 ms | ≤ 2 × 107.1 |
| C = 32 no retry, 1,000 attempts in 2.41 s | 50 / 950 | **657.1 ms** | **> 214.3 ms: FAIL** |

Everything else passed: outputs equal the reference (1,117 successes), golden 10/10, readiness
0 failures (readiness stayed healthy throughout; after each overload phase the recovery check's
first `/readyz` probe answered ready within 0.015 s), 25/25 at C = 5 after each phase (all five
admission tokens free again), threads ≤ 210
(cgroup PIDs ≤ 216 of 512), RSS ≤ 2.7 GiB, GPU memory 1,316 MiB, cleanup verified. The burst was a 2.4 s
transient (attempt cap reached), so its p99 was the maximum of 50 values.

### Diagnostic 1 (serial vs 12 threads, Server-Timing on, single client process)

Both arms degraded severely: successful p99 811.0 vs own baseline 180.3 ms (serial) and 762.8 vs
111.6 ms (12 threads). The load generator ran at 88% and 97% of one core, and time outside the
server handler (p99 ~614 ms in both arms) dominated the tail; the measurement was confounded.

### Diagnostic 2 (final): multi-process, rate-matched, Server-Timing on

Four client processes per arm, aggregate C = 32, attempts paced to 250/s, at most 1,000 attempts
and 10 s. Achieved attempt rates **245.0 (serial) and 246.1/s (12 threads)**, within 0.5%; in-flight
mean 6.9 / 7.2, maximum 13 / 26; every worker 16.0–16.5% CPU of one core; schedule lag p95
1.9–2.0 ms; no attempt waited for a client slot. **Client saturation was removed**; both arms
received the same offered load.

Successful requests only, p50 / p99 (ms); Server-Timing stages are distributions over successful
requests (never subtracted from each other); "outside" and "slot host" are computed per request.

| | Serial C = 5 | Serial C = 32 | 12 threads C = 5 | 12 threads C = 32 |
| --- | --- | --- | --- | --- |
| Successes / 503s (goodput) | 361 / 0 (35.7/s) | 111 / 889 (27.2/s) | 507 / 0 (50.4/s) | 132 / 868 (32.5/s) |
| Client latency | 137.1 / 197.5 | 177.6 / 270.2 | 97.6 / 110.0 | 153.9 / 254.5 |
| Rejected latency | – | 8.4 / 17.2 | – | 7.9 / 67.3 |
| Server total | 133.9 / 177.5 | 174.5 / 248.6 | 94.9 / 106.5 | 150.6 / 229.2 |
| Parse | 1.2 / 4.4 | 3.2 / 8.0 | 1.2 / 2.6 | 3.0 / 18.5 |
| Decode + preprocess | 109.9 / 128.7 | 118.3 / 133.9 | 23.2 / 30.9 | 24.0 / 67.2 |
| Queue for the GPU slot | 0.1 / 28.6 | 14.5 / 74.7 | 48.3 / 58.0 | 87.8 / 116.5 |
| GPU slot time | 17.4 / 29.1 | 33.3 / 47.7 | 19.3 / 24.2 | 30.5 / 44.7 |
| CUDA event forward span | 15.4 / 26.6 | 30.8 / 46.1 | 16.5 / 21.9 | 27.6 / 42.7 |
| Host time in the slot outside the span | 0.4 / 1.5 | 0.4 / 1.6 | 0.4 / 1.3 | 0.4 / 2.4 |
| Outside the handler | 2.9 / 20.0 | 2.9 / 21.6 | 2.7 / 3.7 | 3.0 / 28.7 |

Burst-to-own-baseline ratios of successful p99 (the C1b rule, applied here for context; this
diagnostic does not replace or revise the C1b result, which remains FAIL):

| Arm | Client p99 ratio | Server-total p99 ratio | Against 2× |
| --- | --- | --- | --- |
| Serial | 270.19 / 197.51 = **1.37** | 1.40 | within |
| 12 threads | 254.48 / 109.99 = **2.31** | 2.15 | **exceeds** |

The 12-thread mode's burst p99 (254 ms) is lower in absolute terms than serial's (270 ms), but its
relative criterion still fails (2.31×): its baseline is faster and dominated by GPU queueing, which
is what grows. One container per arm, no replicates: descriptive only; it does not establish that
either mode is statistically superior under overload.

Resources in the bursts: server threads ≤ 261 (serial) / 210 (12 threads), PIDs equal, of 512;
RSS ≤ 2.57 / 2.65 GiB; GPU memory 1,316 MiB; GPU utilization mean 11.6 / 10.7%; event-loop thread
mean 20.7 / 20.2% (max 54.6 / 53.4%); readiness 0 failures (max 4.6 ms). Golden 10/10 in both arms;
all 1,111 successful responses equal the serial reference; cleanup verified.

## 4. What is established and what is not

Observed:
- Under a no-retry burst at about 5–7× capacity, GPU-slot time and the CUDA event forward span
  rose about 1.6–1.9× at p50 in both modes, while host time in the slot outside the span did not
  change and GPU utilization stayed low. GPU-queue time grew accordingly (bounded by the 5 tokens).
- Decode + preprocessing did not grow materially; parse time rose (p50 about 2.6×) but stayed
  small in absolute terms.
- With an unsaturated client, time outside the handler stayed near baseline; the large outside
  times of C1b and diagnostic 1 came with a saturated single-process client.
- Correctness, readiness, admission (only 200 and 503 `queue_full` with `Retry-After: 1`),
  recovery and resource bounds held in every run.

Not established (hypotheses): why the forward span grows while the GPU is mostly idle (host-side
launch delays from CPU contention, Python scheduling or the GIL are consistent with it but were
not measured; the event span includes stream idle gaps and is not kernel time); whether
pre-admission parsing of rejected requests is the dominant source of that contention (the event
loop was not saturated on average); behaviour over longer runs.

## 5. Decision

**CONDITIONAL: keep `BEVSERVE_PREPROCESS_THREADS=0` as the default; 12 threads remains a verified,
opt-in setting.**

Tested envelope for the 12-thread setting: one GPU host, loopback clients, the M6 image,
`max_in_flight` 5. Within it: byte-identical outputs (M6, and under load here), fault recovery and
clean shutdown, readiness under overload, correct admission and fast recovery, bounded threads and
memory in short runs, lower latency at every load level tested.

Conditions that block APPROVE:
1. C1b's frozen latency criterion failed and the 12-thread mode exceeds it (2.31×) even with a
   controlled client: under overload by clients that ignore `Retry-After`, admitted latency roughly
   doubles relative to the fast baseline.
2. Sustained reliability (C3/C4: memory trend, thread stability, drift) was not measured; the
   short-run memory and thread figures do not establish leak-freedom.
3. Admission at the production limit beyond the C1b pilot (C2) was not run.

Serial stays the default because it is the configuration verified since M5, and the decision rule
requires every M7 criterion to pass before changing it. Both modes share the overload behaviour;
the serial mode was not shown to be better under overload, only to have a slower baseline.

Rollback is unchanged: the setting itself (`BEVSERVE_PREPROCESS_THREADS=0` and a restart).

## 6. Remaining limitations and proposed future work (not implemented)

- **Ingress protection (proposal, unverified):** a fast path that, when all admission tokens are
  taken, drains the request body without multipart parsing or image checks and returns the same
  503 `queue_full` with `Retry-After: 1`, keeping the connection usable; the normal admission check
  still runs afterwards. It would reduce the work done for rejected requests; whether it reduces
  the observed slot-time inflation is unknown. Needs a code change, a new image, the M4 checks and
  this M7 harness.
- `ClientDisconnect` during upload is an unhandled server-side exception, logged at error level
  with a traceback (pre-existing; minor; no leak).
- Sustained operation (C3, then a separately authorized C4) and C2 remain to be run before any
  default change.
- Clean shutdown reports exit 0 because the server runs as the container's PID 1; under an init
  process (`docker --init`) a clean SIGTERM shutdown would report 143.
- All results are from one workstation, one GPU, 20 repeated frames and short phases.
