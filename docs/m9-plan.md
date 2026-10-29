# M9 plan: portfolio release, reproducibility and publication safety

Stage A (audit and plan, read-only) on 2026-10-28.
- **Repository state:** HEAD `9f9edf4`, branch `main`, clean working tree, no remote.
- **History:** 7 commits.
- **Tracked files:** 129.

The only file this stage changes is this plan.

## 1. Objective and non-goals

**Objective:** make BEVServe an honest, reproducible and publication-safe ML-systems portfolio
project. That means:
- every public claim traces to retained evidence;
- each reproducibility level states exactly what runs, on what, and with which assets;
- the tree and history are cleared for publication.

**Non-goals:**
- no change to serving code, preprocessing, model arithmetic, admission or deployment defaults;
- no new benchmarks or GPU runs, no rebuild, no new dependencies, CI changes, demo apps or
  infrastructure;
- no reinterpretation of M5–M8 results;
- no history rewrite and no publication without the separate Stage E authorization (section 11).

## 2. Repository inventory

| Area | What exists (verified by reading the tree) |
| --- | --- |
| Service (`src/bevserve`, ~2,400 lines) | FastAPI app: `POST /v1/detect` (six 1600×900 JPEG/PNG parts + calibration JSON), `/healthz`, `/readyz` (with reason), `/v1/model`; FastAPI's default `/docs` and `/openapi.json` (the multipart body is read from the raw request, so it is not described in the schema); RFC 9457 problem+json errors; upload limits before decoding; loopback-only bind policy; one GPU execution slot with bounded admission (`max_in_flight` 5), deadlines, init-failure vs fatal-exit (code 70) lifecycle; hash-verified `weights_only` checkpoint loading; optional `Server-Timing`; opt-in parallel six-camera preprocessing (`BEVSERVE_PREPROCESS_THREADS`, default 0); `FakeDetector` (default) and `BEVDetDetector`; offline self-check; test-only fault injector |
| Containers (`docker/`) | multi-stage CUDA 12.8 GPU image (pinned base digests, mmcv `4c01b026` + one patch, BEVDet `a0142cc`, 104 exact model-stack pins, sm_120 only, non-root, whitelist `.dockerignore`); hardened Compose service (read-only rootfs, caps dropped, loopback publish, `on-failure:5`); CPU smoke image and `build_cpu_stack.sh` |
| Tools (`tools/`) | parity tools (`reference_check_bevdet.py`, `compare_direct.py`, `full_val_parity.py`, `repeatability.py`, `m6_input_parity.py`, `golden_http.py`); container verification (`m4_container_checks.sh`, `m4_parity_retest.sh`, `m5_fault_retest.sh`, layer scanner); benchmark harness (`m5_bench.sh` + `tools/bench/`, 14 modules: closed-loop and multi-process load, sampler, validation, gates, criteria, reports); restricted-content guard; provenance |
| Tests | 313 collected public tests (unit 224, API 47, fake-Docker orchestration 42) + T1 CPU smoke (random weights, needs the CPU model stack) + 3 private GPU test files. Final M8 run: 313 passed, 4 skipped |
| CI (`.github/workflows`) | `ci.yml` (T0: guard, ruff, format, mypy, pytest; read-only token, no secrets) and `smoke-cpu.yml` (T1). **Never executed: the repository has no remote.** |
| Docs | README; `deploy.md`, `host-setup.md`, `env-m2.md`, `frames.md`, `verification-m4.md`; plans and reports for M5–M8. **No M1–M3 report** (M3 claims exist only in the README and a commit message) |
| Packaging | `pyproject.toml`: Python 3.10 only, exact pins; the description still says "M1" |
| Licensing | Apache-2.0 `LICENSE`; `NOTICE` (attribution for the copied inference config and the reimplemented conventions; data and weights not distributed) |

**External dependencies:**
- the nuScenes dataset (CC BY-NC-SA 4.0 plus its Terms of Use);
- the BEVDet-R50-CBGS checkpoint (sha256 `8b67e60d…3443`; terms set by its publisher);
- the BEVDet source: the pinned commit `a0142cc` lives on the author's public fork
  `pushpullleg/BEVDet`, on work-package branches only;
- mmcv v1.7.2;
- NVIDIA CUDA/cuDNN base images, an NVIDIA GPU (only sm_120 built and validated), Docker with the
  NVIDIA Container Toolkit, `sudo` for Docker on the workstation.

## 3. Release-readiness gaps (ranked)

| # | Severity | Gap | Kind |
| --- | --- | --- | --- |
| G1 | **Blocker, time-sensitive** | The M2/M3 evidence behind the headline parity claims survives only in a volatile session scratch directory under `/tmp`:<br>- full validation (6,019 frames): mAP 0.308580, NDS 0.394126, 0 round-trip mismatches, checkpoint sha matches;<br>- 50- and 200-frame HTTP vs BEVDet comparisons: all max abs diffs 0.0;<br>- repeatability;<br>- the M2 comparisons.<br>A reboot loses it. | evidence preservation |
| G2 | **Blocker** | No M3 (or M1/M2) report: the README's headline claim has no document behind it | documentation |
| G3 | **Blocker** | The README presents CI as running ("runs in CI"), but no CI run has ever happened | wording; verification in Stage C (local equivalent) and E (first real run) |
| G4 | High | README structure: milestone-chronological; "Status: milestone M8"; internal process detail; long. The M5 and M6 tables omit that they were measured at `max_in_flight` 16 with Server-Timing on (stated in the reports) | documentation |
| G5 | High | Workstation-bound steps:<br>- `tools/m4_container_checks.sh` aborts unless the kernel is exactly `6.14.0-37-generic`;<br>- the image builds sm_120 only;<br>- verification and benchmark scripts assume `sudo docker`, the nuScenes val info file and a host model environment.<br>`deploy.md` gives no build command, and says results go to `outputs/m4/` (actually `outputs/m4-<run>/`) | docs now; any script change is out of M9 scope (proposed only) |
| G6 | High | Build depends on BEVDet `a0142cc` on a personal fork's work-package branches; deleting them breaks the GPU and T1 builds | external verification + disclosure |
| G7 | Medium | Licensing: the checkpoint's redistribution terms are unverified, so it is never distributed. The GPU image contains NVIDIA CUDA/cuDNN and mmcv ops under the NVIDIA license (upfirdn2d, fused_leaky_relu), so **images must not be published**. `bev_pool_v2` sources carry "All rights reserved" and are covered only by BEVDet's top-level Apache-2.0 | disclosure; external verification |
| G8 | Medium | Git refs beyond `main`: three `refs/agents/…/checkpoints/turn/*` refs, created by an agent host tool and content-identical to HEAD. `git push --all/--mirror` would publish them | Stage E push scope |
| G9 | Low | Three historical commit messages (M1–M4) contain a `Claude-Session` URL (private session link, not a credential) | Stage E decision: keep, or rewrite history (separate approval) |
| G10 | Low | Minor:<br>- `host-setup.md` names the workstation's vendor apt repository and the kernel pin (workstation-specific, not secret);<br>- docstrings say "thesis path";<br>- `pyproject` description is stale;<br>- NOTICE does not mention the reimplemented `bev_pool_v2` CPU forward (`model/cpu_ops.py`);<br>- `tools/demo.py` imports from `tests/` (needs a source checkout) | documentation and metadata |
| G11 | Low (disclose) | Known defect: an unhandled `ClientDisconnect` during upload (M7 R5), logged server-side with a traceback | disclosure only |

The history scan (all 248 reachable objects, all revisions of `main`) was clean:
- no private keys, tokens or credential assignments;
- no home paths, user names or e-mail addresses in content;
- no non-loopback IPs (only test fixtures in RFC 1918 ranges and `1.1.1.1` in the offline check);
- no weights, dataset, image, `outputs/` or binary files ever committed (largest blob 40 KB);
- commit identities use the GitHub noreply address only;
- the current tree passes `tools/check_restricted.py`.

## 4. Required documentation and proposed file changes (Stage B)

Prefer linking to existing reports; add only what a reader cannot find.

| File | Change |
| --- | --- |
| `README.md` | Rewrite to roughly 150–200 lines:<br>- what it is and what it is not (single-GPU, single-host, no batching, not for safety use);<br>- the architecture diagram (below);<br>- quick start without assets (FakeDetector + demo);<br>- results summary table with conditions and links (section 5);<br>- reproducibility levels (section 6);<br>- limitations;<br>- data and licensing;<br>- a docs index.<br>Milestone narrative moves to the reports. No new numbers. |
| `docs/architecture.md` (new) | Components, the two diagrams below as Mermaid text (renders on GitHub; no image files), design trade-offs (reject-fast admission vs queueing; one GPU slot, no batching; serial default vs opt-in pool; exit-on-fatal for supervised restart vs in-process recovery; loopback-only by default), failure semantics table |
| `docs/reproducibility.md` (new) | Levels L0–L4 with exact commands, assets, versions, hardware, run times and known failure modes; a claims-to-evidence table (section 5) |
| `docs/verification-m3.md` (new) | M2/M3 numerical-parity record, written from the preserved summaries only (aggregates; no frame, scene or box data), in the style of `verification-m4.md` |
| `docs/deploy.md` | Add the build command; fix the outputs path; state that sm_120 only is validated and that images are not for redistribution |
| `docs/host-setup.md` | Frame as "the setup used on the one test workstation (example)", not as a requirement |
| `NOTICE` | One line for the reimplemented `bev_pool_v2` CPU forward; a note that container images are not distributed (NVIDIA base-image and mmcv op licenses) |
| `pyproject.toml` | Description only (metadata; no dependency or build change) |
| `docs/m9-report.md` (new, Stage D) | Short record of the audit, the gates and their results |

Not proposed:
- `docs/performance.md`, `docs/reliability.md`, `docs/demo.md`: the M5–M8 reports already serve
  these, and the README results table links to them;
- a mock or simulated GPU service: the FakeDetector already provides API exploration and is
  labelled as fake (`model.name = "fake-detector"`). Nothing further is needed.

**Before Stage B (G1, on approval):** copy the aggregate M2/M3 evidence from the scratch directory
to a gitignored, private `outputs/m3-evidence-20261027/`, with a manifest and hashes:
- `fullval/summary.json`, `fullval/metrics.json`, `fullval_time.txt`;
- the `m3_50` and `m3_200` `summary.json` files;
- `repeat.json`;
- the three `m2_compare_*.json` files.

Frame-level files (`frames.jsonl` with scene tokens, the 242 MB prediction results) stay out of any
document. Whether they are also retained privately is your decision. The copy is not an edit of
evidence; nothing is committed.

### Diagrams (Mermaid text, written in Stage B)

**System architecture.** Components, each mapped to a source file:
- client;
- uvicorn/FastAPI app (`api.py`) with `BodyLimitMiddleware` (`middleware.py`);
- request validation (`schemas.py`, `calibration.py`, `errors.py`);
- admission and GPU slot (`executor.py`);
- decode and preprocessing (`preprocess.py`, `camera_pool.py`);
- detector interface (`detector.py`: `FakeDetector` | `model/bevdet_detector.py` →
  mmdet3d/BEVDet, mmcv ops, `bev_pool_v2`, CUDA);
- service lifecycle and readiness (`service.py`);
- settings (`settings.py`);
- the container boundary (`docker/compose.yaml`: read-only checkpoint mount, loopback publish,
  restart on exit 70).

**Request lifecycle**, from `api._detect` and `executor.py`:
1. HTTP and body limit (413).
2. Readiness gate (503 `not_ready`).
3. Multipart parse (400/415).
4. Part, calibration and image-header checks (422/413).
5. Admission `try_admit` (503 `queue_full` + `Retry-After: 1`).
6. Decode + preprocess (serial or pool).
7. GPU queue and job (single slot; deadline → 504; failure → 500; fatal → 500 + exit 70).
8. Token released at GPU-job end.
9. Response serialization (boxes in `ego@CAM_FRONT_LEFT`, stage timings).

## 5. Public claims-to-evidence matrix

Location ✓ = README, R = technical report only, ✗ = avoid.

| Claim (exact) | Conditions | Evidence | Limitations | Where |
| --- | --- | --- | --- | --- |
| Full nuScenes val (6,019 frames) through BEVServe's input path and adapter gives mAP 0.3086, NDS 0.3941, identical to BEVDet's own evaluation of the same checkpoint | checkpoint `8b67e60d…`; BEVDet dev3.0 evaluator; one GPU | M3 run (`summary.json`, `metrics.json`; to be preserved, G1), thesis baseline 0.3086/0.3941 | adapter-level run, not over HTTP; one checkpoint; published BEVDet reference is 0.313/0.398 (different environment) | ✓ (after G1/G2) |
| HTTP responses on 200 val frames (150 scenes) equal BEVDet's evaluation path: counts, labels, identity equal; max abs diff 0.0 for score, centre, z, size, yaw, velocity; preprocessing arrays max abs diff 0.0 | same checkpoint and host | M3 `m3_200/summary.json` (+ 50-frame run) | 200 of 6,019 frames | ✓ |
| Preprocessing is bit-identical to BEVDet's own code on synthetic inputs | CPU | `tools/reference_check_bevdet.py` | synthetic inputs | R |
| Container: 10 golden frames, 1,209/1,209 boxes byte-identical to M3; offline self-check bit-identical heatmap | M4 image | `verification-m4.md`, `outputs/m4-*` | 10 frames | R (✓ as one line) |
| M5 baseline: C = 1 9.23 req/s, p50 107.7 ms; C = 8 42.23 req/s, p50 185.7 ms; at C = 1 89 of 104 ms server time is CPU decode + preprocessing | one workstation (RTX PRO 6000 Blackwell, TR PRO 7975WX), client on the same host, 20 val frames cycled, closed loop, `max_in_flight` 16, Server-Timing on | `m5-report.md`, run `m5-20261027-160635` | one run, one host; Server-Timing overhead not established (+1.12 ms, CI −8.11…+10.35) | ✓ with conditions |
| M5: 1,200/1,200 requests at C = 1–8 succeeded; at the production limit only 503 `queue_full` + `Retry-After: 1`, then recovery; 100-request soak within limits | same | `m5-report.md` | short soak; 1,000-request soak never run | ✓ |
| M6: 12-thread preprocessing, C = 1 p50 0.302× (CI 0.298–0.305), C = 8 throughput 1.173× (1.159–1.187), C = 8 p95 0.849× | 12 containers ABBA, 6 pairs, same image, `max_in_flight` 16, Server-Timing on | `m6-report.md` | one host; production limit and soak not re-run in M6 | ✓ with conditions |
| M6: model inputs byte-identical serial vs pooled on all 6,019 val frames; golden parity in every A/B container | CPU input parity + GPU golden | `m6-report.md`, `outputs/m6-parity-*` | output parity on 10 golden frames per container | ✓ |
| M7: fault recovery (exit 70 → restart → ready) and clean shutdown with 12 threads; under overload only 200/503 + `Retry-After: 1`, `/readyz` healthy, all slots free afterwards, every 200 equal to the serial reference | production limit 5, M6 image | `m7-report.md` | sustained operation (C3/C4) not run | ✓ |
| M7: under a no-retry burst of about 5–7× capacity, admitted p99 rises 1.37× (serial) and 2.31× (12 threads) over its own baseline; the ≤ 2× criterion failed; decision CONDITIONAL | one container per arm | `m7-report.md`, `outputs/m5-20261028-030910` | no statistical claim | ✓ (as a limitation) |
| M8: early rejection tested in one pre-registered A/B (4 ABBA pairs + pilot): P0 failed (OFF > 2× in 1 of 4), P1–P6 passed; effectiveness not established; Option A; code removed | M8 image, 12 threads, limit 5, Server-Timing on, ~4 s bursts | `m8-report.md`, `outputs/m5-20261028-211317` | small burst samples (133–155 successes); bootstrap CIs overlap; descriptive only | ✓ (one line) / R (details) |
| "Production-ready", "scales", "real-time", "robust under overload", "fast rejection improves tail latency", any comparison with other serving systems | — | none | — | ✗ |

## 6. Reproducibility levels

| Level | What genuinely runs | Needs | Status |
| --- | --- | --- | --- |
| L0 | install, guard, ruff, format, mypy, 313 tests (~6–10 min), FakeDetector server, `tools/demo.py`, `/docs` | Linux, Python 3.10, `curl`, `ss`, `setsid` (orchestration tests); network for pip | verified on the workstation venv; **fresh-clone run still to verify (Stage C)** |
| L1 | T1 CPU smoke: real BEVDet model code with random weights, pure-PyTorch `bev_pool_v2` | builds mmcv-full 1.7.2 from source (slow), fetches BEVDet `a0142cc` from the fork | stack built locally; CI never run |
| L2 | real GPU inference with the operator's checkpoint | NVIDIA GPU (sm_120 built; others need `CUDA_ARCH_LIST`, unvalidated), Docker + NVIDIA toolkit, checkpoint + trusted sha256 | verified on one workstation only |
| L3 | M4 container checks; M5–M8 benchmark procedures | L2 + nuScenes val (info pkl and images), host model env, `sudo docker`; M4 script requires kernel 6.14.0-37 | workstation-bound; documented, not portable as is |
| L4 | numerical parity (`compare_direct.py`, `full_val_parity.py`) | L2 + nuScenes + BEVDet checkout and its environment | verified once (M3); results to be preserved (G1) |

Known failure modes to document:
- Ubuntu runtime images lack `lib2to3` and `distutils` (fixed in the image, M4);
- relative path inputs (fixed);
- GPUs other than sm_120 need a rebuild;
- pip pins require Python 3.10;
- `opencv-python` vs `-headless` coexistence in model environments (`env-m2.md`).

## 7. Privacy, history and licensing risks

- **Data:**
  - no dataset frames, annotations, predictions, goldens or checkpoints exist anywhere in history;
  - `outputs/` (all evidence, the goldens, the M8 source snapshot) is gitignored and stays local;
  - aggregate metrics (mAP/NDS, box counts) appear in docs, and no per-frame or scene data does.
- **Personal and infrastructure details:**
  - noreply identity only;
  - workstation hardware and OS versions are described (normal for benchmarks);
  - `host-setup.md` vendor repository (low).
- **History:**
  - 3 session URLs in commit messages (G9);
  - non-`main` agent refs (G8).

  Neither is a secret. Remediation, if any, belongs to Stage E: push `main` only, and optionally
  rewrite the three messages with explicit approval.
- **Licensing (external verification required before Stage E):**
  - the BEVDet and mmcv Apache-2.0 terms, and the attribution of the copied config;
  - the `bev_pool_v2` per-file notice;
  - the checkpoint's terms: link to the source, never redistribute;
  - nuScenes CC BY-NC-SA 4.0: no derived assets are published;
  - NVIDIA base-image and NVIDIA-licensed mmcv ops: never publish images;
  - the availability of the fork commit `a0142cc`.

  Direct Python dependencies are permissive: BSD-3, MIT, Apache-2.0 and MIT-CMU (per installed
  metadata); mmcv-full declares none in its metadata, but its source tree is Apache-2.0.

## 8. Stage B: documentation scope

Covers only the changes in section 4, plus the G1 evidence copy. Allowed paths: `README.md`,
`NOTICE`, `docs/**`, and the `description` field in `pyproject.toml`. Every number is copied from
an existing report or preserved summary, and none is recomputed. Stage B ends with a review, and
there is no commit.

## 9. Stage C: verification and acceptance criteria

Each criterion is a pass or fail with recorded output:

1. **Scope:** `git diff --name-only 9f9edf4` lists only the allowed paths; `src/`, `docker/`,
   `tools/`, `tests/`, `configs/` and `.github/` are byte-identical to `9f9edf4`.
2. **Fresh-clone L0:**
   - `git clone` the working tree to a scratch directory (no `.venv`, `.cache` or `outputs`);
   - create a new Python 3.10 venv and run `pip install -e ".[dev]"`;
   - run the `ci.yml` steps in order (guard, ruff, format, mypy, pytest): all pass, with 313
     passed and the expected skips;
   - the FakeDetector demo prints 200 and the three expected 422s.
3. **Claims:** every number in `README.md`, `architecture.md` and `reproducibility.md` appears in
   its cited report or preserved summary, checked against a script-extracted list of numbers. No
   claim from the ✗ row.
4. **Links:** every relative link and anchor in the changed docs resolves; external links are
   listed for manual review.
5. **Restricted content:**
   - `tools/check_restricted.py` passes on the tree;
   - the section 3 history scan is re-run with the same result;
   - no file under `outputs/` is tracked.
6. **Evidence integrity:** M3–M8 evidence directories, the M8 snapshot (manifest hashes) and the
   M6/M8 images are unchanged.
7. **Optional, on approval:** T1 smoke with the existing `.venv-cpu` (CPU only).

## 10. Stage D: local release-candidate gate

- One local docs commit after review.
- Then re-run gate 2 from a clean clone of that commit, and record the results in
  `docs/m9-report.md` (second local commit, or amend before it is shared).
- Optionally, a local annotated tag `v0.1.0-rc1`.
- No remote, no push.

## 11. Stage E: public publication (separate authorization)

Not part of M9 Stages A–D. It requires an explicit decision on:
- repository host and visibility;
- whether to rewrite the three commit messages (G9);
- deletion or exclusion of `refs/agents/*` (push `main` and tags only, never `--all` or
  `--mirror`);
- the results of the external license checks (section 7);
- confirmation that the BEVDet fork commit stays reachable;
- observing the first real CI runs (T0, then T1), with README wording corrected if they fail.

## 12. Definition of done and rollback

**Done:**
- Stages B–D complete, with gates 1–6 passing and recorded;
- README claims limited to the ✓ rows with their conditions;
- the reproducibility levels state truthfully what runs;
- G1–G4 closed, and G5–G11 closed or disclosed;
- a clean local release candidate exists.

**Rollback:**
- M9 commits touch documentation and metadata only, so `git revert` restores `9f9edf4` exactly;
- the evidence copy (G1) is additive and gitignored;
- nothing is published before Stage E.

## 13. Status (Stage B drafted, 2026-10-28)

- **G1 closed:** 21 original M2/M3 files copied byte-for-byte to the private, gitignored
  `outputs/m3-evidence-20261027/original/`:
  - every SHA-256 equals its source, and modification times are preserved;
  - the manifest and an explanatory note are kept separately (`MANIFEST.json`, `notes/`);
  - the 253 MB per-sample prediction file was deliberately not copied;
  - the scratch originals are untouched.
- **Drafted:**
  - `README.md` (rewritten);
  - `docs/architecture.md`, `docs/reproducibility.md` and `docs/verification-m3.md` (new);
  - `docs/deploy.md` (build commands, fork, no published images, outputs path, kernel check);
  - `docs/host-setup.md` (framed as the one workstation's record);
  - `NOTICE` (`cpu_ops.py` and the mmcv patch as derived work; images not distributed);
  - `pyproject.toml` (description only).
- **Gap status:**
  - **G2 closed** by `verification-m3.md`; **G3 and G4 closed** in the README;
  - **G5, G6 and G11 disclosed**;
  - **G7 disclosed**: licence verification is still open and remains a Stage E gate;
  - **G8 and G9 open** (Stage E);
  - **G10 closed** for the docs and metadata items. The "thesis path" docstrings in
    `src/bevserve/model/bevdet_detector.py` are left unchanged, because they are code.
- No code, tests, Docker, tool or CI file changed. Stage C (fresh-clone verification) has not
  started.

## 14. Stage C results (2026-10-28)

- **Fresh clone:**
  - a local clone of `9f9edf4` in a temporary directory outside the repository, with the nine
    approved files copied in;
  - SHA-256 of source and clone match for all nine;
  - the clone holds the 129 tracked files plus the 4 new docs, and no `outputs/`, venv, cache,
    env file, image or weights.
- **L0 in a new, isolated Python 3.10.21 venv:**
  - `pip install -e ".[dev]"` with pip 23.0.1 and no cache: success in 20 s, all pins resolved
    from PyPI;
  - guard, ruff, format and mypy pass;
  - pytest: **313 passed, 4 skipped** in 6 min 17 s;
  - FakeDetector demo: 200 plus three 422s; `/docs` returns 200; no process left afterwards.
- **Undocumented prerequisite found:** no `python3.10` on `PATH` (Ubuntu 24.04 ships 3.12). The
  README and `reproducibility.md` now say so; the check used a conda-provided CPython 3.10 as the
  venv base.
- **Documentation fixes made during verification:**
  - the M3 reference-baseline limitation is made explicit (README, `verification-m3.md`);
  - the `cpu_ops.py` attribution is made precise (`NOTICE`);
  - a 413 is added to the multipart-parse step of the request diagram;
  - the Python 3.10 note.
- **Checks:** links and anchors resolve; Mermaid was structurally linted but **not rendered** (no
  local renderer); every number traces to a report or the preserved originals; M3 claims match
  the originals exactly.
- **Agent checkpoint refs:** the refs under `refs/agents/` now number five and include snapshots
  of the M9 draft docs. There is still nothing under `outputs/` and nothing restricted. Push
  `main` only (G8).
