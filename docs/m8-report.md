# M8 report: ingress protection under overload

**Decision: Option A.** Fast rejection (Option B, `BEVSERVE_FAST_REJECT`) is **not accepted** as a
verified mitigation. Its effectiveness was **not established** by the pre-registered experiment.
That is not a finding that it is ineffective. The experimental code has been removed, so the
service is again exactly the M7 code (commit dac825d), and the M7 overload limitation stands. The
plan and pre-registration are in [m8-plan.md](m8-plan.md).

## 1. Problem

In M7 ([m7-report.md](m7-report.md)), clients ignored `Retry-After` and sent a no-retry burst of
about 5–7× capacity at the production admission limit of 5. With 12 preprocessing threads,
admitted-request p99 rose to 2.31× its own C = 5 baseline (one container), which failed the
frozen ≤ 2× criterion (C1b). Correctness, readiness, admission and recovery held.

The M7 data made one cause plausible, without establishing it: work done on rejected requests,
since each rejection was a full ~0.87 MB upload plus multipart parsing and image-header checks.

## 2. Alternatives and choice

| Option | Summary | Verdict |
| --- | --- | --- |
| A | Keep the M7 code; document the overload envelope | Fallback |
| B | Before multipart parsing, if all admission tokens are taken, drain and discard the body and return the same 503 `queue_full` (`Retry-After: 1`). Advisory check: no token is taken, and the atomic admission check still decides. | **Selected for one bounded test** |
| C | Take the admission token before parsing | Rejected: slow uploads would hold GPU tokens (starvation), and timeout machinery would be needed |
| D | Ingress concurrency limit (uvicorn `limit_concurrency`) | Rejected: breaks the error contract (plain-text 503 without `Retry-After`) and refuses `/readyz` probes |

B was chosen to test for five reasons:
- it targets the plausible cause directly;
- it keeps the response contract and readiness unchanged;
- it cannot admit more than the limit;
- it was opt-in, with default off;
- it was the smallest change, so it was easy to remove if the test failed.

## 3. Verification before any GPU measurement

**CPU-only implementation verification (final suite: 430 passed, 4 GPU/torch skips; ruff, format,
mypy, `bash -n`, restricted-content guard, `git diff --check` clean):**
- **Fast path:** when on, it skipped multipart parsing and Pillow header checks and returned the
  same 503 body and headers as when off; below capacity, responses were identical.
- **Error precedence:** not-ready still came first. At capacity, ON answered 503 before
  415/400/422.
- **Bodies:** chunked bodies were drained to the end, and oversized bodies still got 413.
- **Disconnects:** a disconnect during the drain was quiet and took no token.
- **Admission limit:** at exactly the limit, no more than the limit was ever admitted.
- **Keep-alive:** two sequential 503s were served on one TCP connection of a real local server.
  This was shown for normal-size bodies only.
- **Harness:** analysis and gate logic were tested on synthetic data and with fake-Docker
  orchestration. Bounded runs, ABBA order, fail-closed setting checks, data adequacy and the
  pilot gate were all covered.

**Two analysis and gate fixes, made after review and before any GPU result. Neither changed a
threshold:**
- **Geometric means:** a missing pair, or an ON goodput of 0, now fails a geometric-mean criterion
  instead of being dropped from it.
- **Gates added:**
  - per-container minimum success counts (C = 1 ≥ 50, C = 5 ≥ 200, burst ≥ 100); insufficient
    data leaves P0–P6 undecided;
  - a stronger pilot-to-full gate (same image, source identity, checkpoint, registered design,
    OFF then ON, all checks passing).

**New-image M4 verification** (`outputs/m4-20261028-204931`, image `bevserve:m8-20261028-204931`,
`sha256:8d32f159…5af`): **29/29 checks PASS.**
- **Build and safety:** layer scans, non-root user, offline self-check.
- **Deployment:** loopback-only publishing, restart policy, model identity.
- **Golden parity:** 10/10 byte-identical frames in each of three configurations — serial, 12
  threads, and 12 threads with fast reject on. For the last two, the setting was read back from a
  recreated container.
- **Failure handling:** exit-70 fault restart, initialization failures, clean shutdown.

This verifies output parity and startup configuration only. Its sequential golden requests never
reach capacity, so it says nothing about overload.

## 4. Experiment (pre-registered, one pilot and one full run)

**Fixed design:**
- **Image and source:** the M8 image, built from source `dac825d-dirty`; the image package
  matched the source (fingerprint `20538614…`).
- **Model input:** checkpoint sha256 `8b67e60d…3443`, inputs `nuscenes:even:20`.
- **Service settings:** 12 preprocessing threads, `max_in_flight` 5, Server-Timing on.
- **Per container:** a fresh container, then:
  - C = 1 and C = 5 baselines of 10 s each;
  - a C = 32 no-retry burst from the M7 multi-process client (4 workers, 250 attempts/s, at most
    1,000 attempts). Each burst reached the 1,000-attempt cap after about 4.06 s.
  - a golden check.

**Pilot** (`outputs/m5-20261028-205851`, one OFF/ON pair):
- All 11 counted execution, correctness, design and adequacy checks passed.
- Burst/baseline ratios were OFF 2.083 and ON 1.633. Goodput was 32.6 and 37.9/s.
- P0–P6 are not judged on a pilot. The pilot gate later passed (`gate-m8-pilot.json`).

**Full run** (`outputs/m5-20261028-211317`, 8 containers in ABBA order, pairs (1,2), (4,3), (5,6),
(8,7)). Latency is client-side and over successful requests only, using nearest-rank percentiles:

| Pair | Arm | C = 5 p99 ms | Burst p99 ms (95% CI) | Burst / C = 5 | Burst successes | Goodput /s | Attempts /s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (1,2) | OFF t1 | 112.2 | 190.5 (176.8–245.3) | 1.698 | 133 | 32.79 | 246.5 |
|  | ON t2 | 108.0 | 181.3 (146.0–203.9) | 1.679 | 152 | 37.53 | 246.9 |
| (4,3) | OFF t4 | 106.1 | 205.0 (174.4–235.5) | 1.932 | 133 | 32.79 | 246.6 |
|  | ON t3 | 106.2 | 177.7 (145.2–210.8) | 1.674 | 153 | 37.80 | 247.1 |
| (5,6) | OFF t5 | 105.0 | 254.5 (174.6–258.9) | **2.425** | 133 | 32.71 | 246.0 |
|  | ON t6 | 110.4 | 176.3 (145.9–200.5) | 1.597 | 151 | 37.29 | 247.0 |
| (8,7) | OFF t8 | 106.3 | 211.4 (171.6–241.2) | 1.989 | 133 | 32.75 | 246.2 |
|  | ON t7 | 108.4 | 149.0 (142.5–188.5) | 1.375 | 155 | 38.17 | 246.3 |

ON − OFF burst p99, per pair:

| Pair | Difference |
| --- | --- |
| (1,2) | −9.1 ms (−4.8%) |
| (4,3) | −27.2 ms (−13.3%) |
| (5,6) | −78.3 ms (−30.8%) |
| (8,7) | −62.4 ms (−29.5%) |

**Totals:**
- **Burst successes:** OFF 532, ON 611.
- **Mean burst p99:** OFF 215.4 ms, ON 171.1 ms.
- **Burst p50:** OFF 149–154 ms, ON 129–130 ms.
- **Rejection latency (p50):** OFF 7.8–8.0 ms, ON 6.2–6.5 ms.

**Mandatory checks (all pass):**
- **Containers and settings:** every container was fresh and ran the intended setting, read back
  from the running container.
- **Responses:** only 200 or 503 `queue_full` with `Retry-After: 1`, with zero transport errors or
  other 5xx. Every 200 matched the serial reference.
- **Golden and readiness:** golden parity in all 8 containers; no not-ready probe in any
  container.
- **Resource budgets:** no stop check fired. Peaks were RSS 2,683 MiB, server GPU 1,316 MiB and
  210 threads, against budgets of 8,192 / 4,096 / 400.
- **Load and data:**
  - burst rates within 0.4% in every pair;
  - no client worker saturated;
  - registered design used;
  - adequacy met: minimum successes 277 / 508 / 133.
- **Cleanup and provenance:** all 9 project-scoped cleanups ok; verified image and provenance
  match.

## 5. Outcome against the pre-registered criteria

These outcomes were recomputed independently from the raw request records, and they match the
harness exactly.

| Criterion | Rule | Observed | Result |
| --- | --- | --- | --- |
| P0 reproduction | OFF ratio > 2.0 in ≥ 3 of 4 OFF containers | 1 of 4 (1.698, 1.932, **2.425**, 1.989) | **FAIL** |
| P1 primary | ON ratio ≤ 2.0 in all ON containers | 1.679, 1.674, 1.597, 1.375 | pass |
| P2 direction | ON ratio < OFF ratio in all pairs | 4 of 4 | pass |
| P3 goodput | geometric mean ON / OFF ≥ 0.90 | 1.151 (per pair 1.140–1.166) | pass |
| P4 normal load | C = 1 and C = 5 p50 geometric means ≤ 1.05; no C = 5 rejection | 1.017 and 0.995; 0 rejections | pass |
| P5 baseline tail | C = 5 p99 geometric mean ≤ 1.10 | 1.008 | pass |
| P6 absolute tail | ON burst p99 < OFF in ≥ 3 of 4 pairs | 4 of 4 | pass |

**Why P0 prevents acceptance.** P0 is a validity condition, fixed before any result. It asks
whether the problem B is meant to fix actually occurred in this experiment. The M7 2.31× came from
one container. Here, only one of the four OFF containers exceeded 2×, and the OFF ratios ranged
from 1.70 to 2.43. Without the failure being reproduced, an ON ratio below 2× cannot be credited
to B: ON is below 2× in all four pairs, but OFF itself is also below 2× in three of them. The
frozen outcome rule maps a P0 failure to Option A. The results are not reinterpreted, the
thresholds are not relaxed, and the experiment is not repeated.

**What the data does show, within this experiment and this design only:**
- **Direction:** every pair moved in B's favour:
  - higher burst goodput (+14–17%, about 20 more successes per burst);
  - lower burst p99 (−5% to −31%);
  - lower burst/baseline ratio (P2, 4 of 4);
  - lower burst p50;
  - slightly faster rejections.
- **Cost:** no measurable cost at normal load (P4, P5).

These are consistent with a real effect of reducing rejected-request work. They are not
established:
- **Precision:** burst p99 rests on 133–155 successes per container, so it is about the
  second-largest value, and its bootstrap intervals overlap between arms in every pair.
- **Sample size:** four pairs.
- **Shape of the overload:** the bursts were about 4 s long. The criteria were designed to decide
  only together with P0, and P0 failed.

## 6. What was rolled back

All M8 experimental functionality was removed. Every source file is restored to commit dac825d:

| Area | Removed |
| --- | --- |
| Serving | the `BEVSERVE_FAST_REJECT` setting (`settings.py`) and the pre-admission check and drain (`api.py`) |
| Deployment | the compose and `.env.example` passthrough |
| Container checks | step 4c of `tools/m4_container_checks.sh` |
| A/B harness | stage `m8-ab` in `tools/m5_bench.sh`, `tools/bench/m8ab.py`, the M8 criteria and gate code in `tools/bench/criteria.py` and `tools/bench/gate.py` |
| Tests | the M8 tests and fake-Docker extensions |

Model inference, preprocessing, admission and deployment defaults were never changed by M8, and
remain as in M7. The default is still `BEVSERVE_PREPROCESS_THREADS=0`.

A gitignored local snapshot of the M8 source, with its manifest, is kept outside version control
for reproducibility. The manifest records base commit, image tag and ID, evidence directories and
file hashes. The M8 image, the M6 image and all M4–M8 evidence directories are kept unchanged.

## 7. Remaining limitation

The M7 limitation is unchanged. When clients ignore `Retry-After` and offer about 5–7× capacity,
admitted-request latency rises above its own baseline. The size of the rise varies by container:
- in M7, 1.37× serial and 2.31× with 12 threads, one container each;
- in M8's unmodified (OFF) arm, 1.70–2.43× with 12 threads across four containers.

Correctness, the 503 contract, readiness and recovery hold throughout. Clients should honour
`Retry-After: 1`. Sustained-load reliability has not been measured.

## 8. Possible future work (not implemented)

- A separately pre-registered replication of B with a design powered for the tail: longer or
  repeated bursts, more successes per container, and a reproduction check fixed beforehand.
- Measuring the cost of ingress for rejected requests directly (h11 body handling versus multipart
  parsing), for example with a C-accelerated HTTP parser, or by rejecting on headers before the
  upload where clients send `Expect: 100-continue`.
- An ingress proxy or rate limiter in front of the service, with the error contract and the
  readiness probes kept intact.
- The R5 `ClientDisconnect` logging defect from M7.
