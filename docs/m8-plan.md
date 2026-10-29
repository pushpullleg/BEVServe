# M8 plan: ingress protection and overload hardening

> **Final outcome: Option A** (see section 10 and [m8-report.md](m8-report.md)). The pre-registered
> A/B ran once. P0 (reproduction) failed and P1–P6 passed, so fast rejection is not accepted, and
> its effectiveness was not established. The experimental code described in sections 8 and 9 has
> been removed; the service is the M7 code (dac825d). Sections 1–7 are kept as pre-registered.

Problem (from M7, [m7-report.md](m7-report.md), unchanged): under a no-retry burst at about 5–7×
capacity, admitted-request p99 rose to 2.31× its own baseline with 12 preprocessing threads (1.37×
serial) at `max_in_flight` 5; C1b failed the frozen ≤ 2× criterion. Correctness, readiness,
admission and recovery held. Baseline: commit dac825d, image `bevserve:m6-20261027-224240`
(`sha256:8d1a4454…80fb`, package fingerprint `1d6df56e…` = HEAD's), default
`BEVSERVE_PREPROCESS_THREADS=0`. M8 addresses only this limitation.

## 1. Request lifecycle around admission (code as of dac825d)

| Step | Code | Thread | Cost for a rejected request |
| --- | --- | --- | --- |
| HTTP parsing of headers and every body chunk | uvicorn h11 (pure Python; httptools is not installed), `BodyLimitMiddleware` | event loop | whole ~0.87 MB body through Python |
| Readiness gate | `api._require_ready` | event loop | small |
| Multipart parsing; parts > 1 MiB spooled to `/tmp` | `api._read_form` → Starlette / python-multipart 0.0.32 | event loop | whole body parsed |
| Part, calibration and image-header checks (6 × Pillow `Image.open` header parse) | `_collect_parts`, `_parse_calibration`, `_check_images` | event loop | all six parts read to `bytes` |
| **Admission** (token owner: `InferenceExecutor`, `try_admit` → `Admission`) | `executor.try_admit` | event loop | → 503 `queue_full`, `Retry-After: 1` |
| Decode + preprocessing, GPU queue, GPU job, release on job end | unchanged (M7 §1) | worker / slot threads | — |

## 2. What the M7 data supports

Observed (final multi-process, rate-matched diagnostic `m5-20261028-030910`):

- Admission is capped at 5 in both the C = 5 baseline and the burst. The client-side in-flight mean
  of 4.98 during the baseline shows the five clients were almost always waiting for a response; it
  does **not** prove that five server tokens were continuously held (a token is released when the
  GPU job ends, before the response is serialized and delivered). What changed between baseline
  and burst is that the burst added about 214–218 rejected requests per second (each a full
  ~0.87 MB upload, multipart parse and header checks), extra client processes and loopback traffic
  on the same host, while successful throughput fell (50.4 → 32.5/s with 12 threads) and GPU
  execution behaviour changed. **Rejected-request ingress work is a plausible cause of the
  slowdown, not an established one.**
- During the burst, GPU-slot time and the CUDA-event forward span rose 1.6–1.9× (p50) in both modes
  while host time in the slot outside the span did not change and GPU utilization fell
  (26.9 → 10.7% with 12 threads). Aggregate server CPU fell (796 → 340%), so the server was not
  CPU-saturated; the event-loop thread rose from 10–13% to about 20% of a core (≈ 0.3–0.5 ms of
  main-thread CPU per rejection, a rough estimate).
- Rejections were fast (p50 about 8 ms); the client was not saturated (4 × 16% CPU, lag p95 2 ms).

Not supported: that multipart parsing (rather than h11 body handling, header checks, client
processes or kernel networking) is what slows the GPU slot; that less parsing would speed CUDA
work; that the mechanism is the GIL. Reducing rejected-request work is a plausible lever, not a
demonstrated one. Hence one bounded experiment, not a rollout.

## 3. Alternatives

| | A. Keep, document the envelope | B. Pre-admission capacity check, drain body, then reject | C. Acquire admission before multipart parsing | D. Ingress / server concurrency limit (uvicorn `limit_concurrency`) |
| --- | --- | --- | --- | --- |
| What changes | nothing (M7 docs) | if all 5 tokens are taken at handler entry: read and discard the body without multipart parsing or image checks, return the same 503 `queue_full` | token taken first; parsing happens while holding it | uvicorn refuses new connections/tasks above N |
| Keep-alive, body consumption | unchanged | body fully consumed → connection stays usable (rejecting *without* draining would close the connection and make clients that send the whole body first see transport errors instead of 503) | unchanged | uvicorn answers `Connection: close` |
| Validation and error precedence | unchanged | **changes when at capacity**: 503 precedes 415/413/422 (a malformed request gets 503, then its 4xx on retry); below capacity unchanged | changes: a malformed request consumes a token until rejected | uvicorn's 503 precedes everything |
| Races when a slot frees | n/a | the check is advisory and takes no token; the normal `try_admit` still runs after parsing, so no request is admitted beyond the limit. A request arriving while all 5 are taken is rejected even if a slot frees during what would have been its parse (today it might have been admitted): possible **goodput loss**, to be measured | no race, but tokens idle during upload | n/a |
| Disconnects, slow uploads | unchanged (R5 logging defect) | a drain holds a coroutine, not a token; disconnect during drain must be handled quietly | **slow uploads hold tokens: 5 slow clients starve the GPU**; would need upload timeouts | connections counted, slow clients consume the limit |
| DoS amplification | full parse per rejection | lower: no multipart parse, no `/tmp` spooling, no Pillow work; body still read (bounded by the 50 MiB limit) | lower parse cost, but starvation risk is worse | cheapest, but rejects everything |
| CPU / GPU effect | — | less Python work per rejection; effect on the GPU slot **unknown** (the hypothesis under test) | similar parse savings for rejections, but admitted work unchanged | — |
| Readiness, contract | unchanged | unchanged (`/readyz` untouched; same problem+json body and `Retry-After: 1`) | unchanged | **breaks both**: plain-text 503 without `Retry-After`, and `/readyz` probes are refused too (orchestrator restarts) |
| Serial / 12 threads | both | both (before preprocessing) | both | both |
| Rollback | — | a setting, default off; off = today's code path | a code revert | config |
| Complexity / risk | none | small (one check, one drain loop, tests) | medium–high (token lifetime, timeouts) | small code, unacceptable behaviour |

D is rejected (contract and readiness). C is rejected (slow-upload starvation of GPU tokens; new
timeout machinery). The choice is between A and B.

## 4. Recommendation

**B, behind a new opt-in setting `BEVSERVE_FAST_REJECT` (default false), evaluated by one A/B.**
If B does not meet the pre-registered criteria, M8 ends with **A**: the code is removed, the M7
envelope documentation stands.

Design (Stage B, not yet implemented):
- In `_detect`, after `_require_ready` and before `_read_form`: if the setting is on and
  `executor.in_flight >= max_in_flight`, consume `request.stream()` to the end (still bounded by
  `BodyLimitMiddleware`), then raise the existing `ApiError(503, "queue_full", …,
  Retry-After: 1)`. A `ClientDisconnect` during the drain ends the request without an error log.
- No token is taken by the check; everything after it is unchanged. Readiness, model,
  preprocessing, admission algorithm and error schema are unchanged.
- Tests: precedence (readiness 503 first; at capacity 503 before 4xx; below capacity unchanged);
  keep-alive preserved after a fast reject (second request on the same connection); body fully
  consumed; disconnect during drain (no token, no error log); exactly-5-in-flight race (never more
  than 5 admitted); setting off = byte-identical behaviour to today on the existing suites.
- New image, verified with the M4 container checks (both preprocessing settings, both
  fast-reject settings for the golden check).

## 5. Pre-registered experiment and acceptance

One bounded A/B on the new image, `BEVSERVE_PREPROCESS_THREADS=12`, `max_in_flight` 5,
Server-Timing on (as in the M7 diagnostic that produced 2.31×): arms OFF / ON = `FAST_REJECT`
false / true; 8 fresh containers in ABBA order (4 pairs); in each: C = 1 baseline (10 s),
C = 5 baseline (10 s), C = 32 no-retry burst with the M7 multi-process client (4 workers,
250 attempts/s, ≤ 1,000 attempts, ≤ 10 s), golden check. A pilot of one pair precedes it.

Mandatory (every container, both arms): every 200 equals the serial reference; golden 10/10; only
200 and 503 `queue_full` with `Retry-After: 1` (identical body schema); zero transport errors and
zero other 5xx; every readiness probe ready; resource budgets (M7); verified cleanup; the running
container's `BEVSERVE_FAST_REJECT` and `BEVSERVE_PREPROCESS_THREADS` verified; no client worker
saturated (≥ 80% of one core); in each pair, achieved burst attempt rates within ±10%; the run
used the registered design (12 threads, `max_in_flight` 5, Server-Timing on, 4 load workers,
250 attempts/s, 10 s C = 1 and C = 5 baselines, C = 32 burst ≤ 10 s and ≤ 1,000 attempts; pilot
one OFF/ON pair, full 4 ABBA pairs).

Data adequacy (mandatory, pilot and full; added before any M8 GPU measurement): every container
has at least **50** successful C = 1, **200** successful C = 5 and **100** successful C = 32 burst
requests. The actual counts of every phase are reported. If adequacy fails, the run has
insufficient data: P0–P6 are not decided (reported as insufficient data, never a latency pass and
never P0 "not reproduced"), and the outcome is a failed mandatory gate.

Pilot-to-full authorization (`tools/bench/gate.py m8-pilot`, before any full-run measurement): the
pilot must be a non-test-mode MODE=pilot STAGES=m8-ab run on the same verified image ID, from the
same source identity as the full run (git head, uncommitted-changes hash, package fingerprint;
image package matching it), with the same checkpoint sha256, the registered design above with
one OFF/ON pair (two fresh containers, fast reject false then true as read from the running
containers, 12 threads, `max_in_flight` 5, Server-Timing on), and every counted check passing,
including execution, the response/output correctness checks, golden parity, readiness, design,
data adequacy, client saturation and rate matching. P0–P6 are not judged on the pilot.

Ratios below use each container's own baselines; "pairs" are the ABBA pairs (1,2), (4,3), (5,6),
(8,7); geometric means are over the 4 pairs and need a ratio from every pair (a missing ratio
fails the criterion; an ON goodput of 0 gives a ratio and a geometric mean of 0). Latency figures
are over successful requests only; the success count behind each is reported, not a criterion.

| Criterion | Pass condition |
| --- | --- |
| P0 reproduction | OFF burst p99 / OFF C = 5 p99 > 2.0 in at least 3 of 4 OFF containers (otherwise "not reproduced" → A) |
| P1 primary | ON burst p99 / ON C = 5 p99 ≤ 2.0 in **all 4** ON containers (the M7 rule) |
| P2 direction | ON ratio < OFF ratio in **all 4** pairs |
| P3 goodput | burst successful throughput ON / OFF, geometric mean ≥ 0.90 |
| P4 normal-load latency | C = 1 and C = 5 client p50 ON / OFF, each geometric mean ≤ 1.05; zero rejections in every C = 5 run |
| P5 baseline tail | C = 5 client p99 ON / OFF, geometric mean ≤ 1.10 |
| P6 absolute burst tail | ON burst p99 < OFF burst p99 in at least 3 of 4 pairs |

All criteria and thresholds above are fixed before any M8 GPU result exists; they are encoded in
`tools/bench/m8ab.py` and `tools/bench/criteria.py` and are not changed after results are seen.

Outcome: all mandatory + P0–P6 → B accepted as a verified opt-in mitigation (its default is a
separate later decision). P0 fails → not reproduced → A. Any of P1–P6 fails → A. Any mandatory
failure (including data adequacy) → stop, no P0–P6 decision, revert, A. The M7 C1b FAIL is not
re-adjudicated by M8.

## 6. Stopping rule

Exactly one implementation of B and one A/B (after its one-pair pilot). No tuning of the drain,
limit, rate or thresholds after seeing results; no further profiling experiments in M8. Whatever
the result, M8 closes with A or B as defined above.

## 7. Scope, risks, rollback, definition of done

In scope: the fast-reject check and drain, its setting, tests, harness stage for the A/B, docs.
Out of scope: model, checkpoint, preprocessing arithmetic, admission algorithm, deployment defaults,
new infrastructure, the R5 `ClientDisconnect` logging outside the fast path.

Risks: goodput loss under overload (P3); changed error precedence at capacity (documented, tested);
drain still costs h11 work, so the effect may be small (P1/P2 decide); new image must pass M4.

Rollback: `BEVSERVE_FAST_REJECT=false` (default) restores today's path; a git revert removes it.

Done: `docs/m8-report.md` records the A/B and the A-or-B outcome; README updated only for that
outcome; one reviewed local commit.

## 8. Stage B: implementation (CPU-only verified; removed after the Option A outcome)

- `src/bevserve/settings.py`: `fast_reject: bool = False` (`BEVSERVE_FAST_REJECT`; unknown or
  invalid values are rejected at startup; an older image ignores the variable).
- `src/bevserve/api.py`: `_fast_reject_if_full` runs in `_detect` after `_require_ready` and
  before `_read_form`, only when the setting is on. If `executor.in_flight >= max_in_flight` it
  reads `request.stream()` to the end (bounded by `BodyLimitMiddleware`) and raises the existing
  `ApiError(503, "queue_full", "Service busy", "<n> jobs in flight (limit <m>)", Retry-After: 1)`.
  It takes no token; `try_admit` still decides for every request that passes. A
  `ClientDisconnect` during the drain is absorbed (same 503, unsent, no error log). Setting off:
  the only change on the request path is one boolean check.
- `docker/compose.yaml`, `docker/.env.example`: pass `BEVSERVE_FAST_REJECT` (default false).
- `tools/m4_container_checks.sh` step 4c: 10-frame golden with `BEVSERVE_FAST_REJECT=true` and the
  parallel thread setting, recreation and both settings verified from the running container.
- Harness: stage `m8-ab` in `tools/m5_bench.sh` (fast-reject setting written to the env file and
  verified from `Config.Env` for every container; pilot = one pair, full = 8 ABBA containers,
  gated on a passing non-test pilot; bursts limited to 10 s and 1,000 attempts),
  `tools/bench/m8ab.py` (P0–P6 and the mandatory rate and client checks), `tools/bench/criteria.py`
  (gates; P0–P6 and the pre-registered design parameters counted in the full run only).

Tests (CPU only): fast path skips multipart parsing and Pillow header checks (OFF does both), same
503 body, headers and detail as OFF; below capacity identical responses; precedence (not_ready
first; at capacity 503 before 415/400/422 with ON, 4xx with OFF); chunked bodies drained to the
end; declared and streamed oversized bodies still 413; disconnect during drain quiet, no token;
exactly-at-limit race never admits more than the limit; keep-alive on a real local uvicorn server
(two sequential 503s on one TCP connection, same local port, both arms); criteria logic on
synthetic results (each criterion can fail; missing data never passes); fake-Docker runs of the
pilot and the full ABBA design, wrong-setting fail-closed, gates and bounds; step 4c of the
container checks (the script's own 4b/4c text against the fake Docker): passes with fast reject
and 12 threads read back from the running container, fails closed when the container reports
fast reject off, omits it, reports a wrong thread count or is not recreated (no parity check is
run then); data adequacy (exactly the minimum passes, one fewer in any phase of any container,
zero burst successes or a missing count leaves P0–P6 undecided, never "not reproduced"); the
pilot gate (a registered pilot authorizes; a different image, source, checkpoint, mode, stage,
thread count, `max_in_flight`, worker count, rate, durations, attempt cap or trial count, swapped
or incomplete arms, Server-Timing off, a reused container, test mode, or any required check
missing, uncounted or not passing is refused, and the full run stops before any measurement);
a fake pilot with another design fails its counted design check.

Limitations: at capacity, ON changes error precedence (documented above); an oversized body at
capacity ends in 413 and closes the connection; keep-alive is shown for normal-size bodies only;
the drain still passes the body through h11 in Python, so savings are partial; a slot that frees
just after the check does not help the rejected request (goodput, P3); the step 4c fake test runs
that step only (not the full M4 script with build, GPU and sudo), and its golden comparison posts
one frame at a time, so it checks output parity and startup configuration, never the fast-reject
path; burst p99 rests on few samples even above the adequacy minimum (M7 bursts had 111–132 successes
per container, so p99 is about the second-largest value); the pilot gate needs an unchanged work
tree between pilot and full run (any edit changes the source identity); the fake server cannot reproduce
the overload, so P0–P6 are decided only by the GPU run; Server-Timing is on in the A/B, as in the
M7 diagnostic, not as in production.

## 9. Operator procedure (executed once; the harness stage has since been removed)

1. Build and verify a new image (the M6 image stays as is):
   `IMAGE=bevserve:m8-<date> PARALLEL_THREADS=12 ... bash tools/m4_container_checks.sh`: all checks
   PASS, including 4b (12 threads) and 4c (12 threads + fast reject).
2. Pilot (one pair): `MODE=pilot STAGES=m8-ab IMAGE=bevserve:m8-<date> EXPECT_IMAGE_ID=<new id>
   STAGE_A_DIR=outputs/m4-<step 1 run> REL_THREADS=12 LOAD_WORKERS=4 REL_DIAG_RATE=250 REL_BASE_S=10
   REL_PHASE_S=10 REL_MAX_ATTEMPTS=1000 bash tools/m5_bench.sh`: every counted check passes.
   The pilot's own counted checks include the design and data-adequacy gates.
3. Full A/B: the same command with `MODE=full PILOT_DIR=outputs/m5-<pilot run>`, with the work tree
   unchanged since the pilot; it stops before measuring unless `gate-m8-pilot.json` passes; outcome
   A or B by section 5, then stop (section 6).

## 10. Outcome

**Runs:**

| Run | Directory | Result |
| --- | --- | --- |
| M4 verification of the M8 image (`bevserve:m8-20261028-204931`, `sha256:8d32f159…5af`) | `outputs/m4-20261028-204931` | 29/29 PASS |
| Pilot | `outputs/m5-20261028-205851` | all counted checks pass |
| Full A/B | `outputs/m5-20261028-211317` | exit 1 |

**Full-run results:**
- **Mandatory checks:** all passed — execution, correctness, golden, readiness, budgets, design,
  data adequacy, client saturation, rate matching, cleanup and provenance.
- **P0 failed:** the OFF burst/baseline ratio exceeded 2.0 in 1 of 4 OFF containers (1.698, 1.932,
  2.425, 1.989).
- **P1–P6 passed:**
  - ON ratios 1.375–1.679;
  - ON below OFF in all pairs, both in ratio and in absolute burst p99;
  - goodput ON/OFF 1.151;
  - C = 1 and C = 5 p50 at 1.017 and 0.995;
  - C = 5 p99 at 1.008.

By the frozen rule (section 5), a P0 failure means **Option A**.

**Rollback:** the fast-reject code, its setting, the compose passthrough, step 4c and the `m8-ab`
harness were removed, restoring commit dac825d. A gitignored local source snapshot and manifest
are kept for reproducibility. The M7 overload limitation stands (see [m7-report.md](m7-report.md)
and [m8-report.md](m8-report.md)).
