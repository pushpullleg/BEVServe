# M9 report: portfolio release preparation and reproducibility verification

**Outcome:** release candidate prepared locally; **not published**. M9 changed documentation and
package metadata only. Serving code, model arithmetic, preprocessing, admission, deployment
defaults, Docker files, tools, tests and CI workflows are byte-identical to the M8 commit
`9f9edf4`. No GPU work, benchmarks or image builds were run. Plan, gaps and gates:
[m9-plan.md](m9-plan.md).

## 1. Evidence preservation

**What was preserved.** The M2/M3 numerical-parity evidence (2026-10-27) survived only in a
volatile temporary directory. 21 original files were copied byte-for-byte into a private,
gitignored archive (`outputs/m3-evidence-20261027/`, owner-only access). The originals sit in
`original/`; a SHA-256 manifest and an explanatory note are kept separately:
- **Full validation:** summary, metrics, evaluator summaries, run log, timing.
- **HTTP comparisons:** the 50- and 200-frame comparisons.
- **Repeatability, M2 and provenance:** repeatability, the three M2 comparisons, and environment
  and provenance records.

**Verification.** Every hash matched its source after copying and again at the end of Stages C
and D, and modification times were preserved.

**Deliberately not copied:** the 253 MB per-sample prediction file (nuScenes-derived).

**Missing, and not reconstructed:**
- the separate reference-baseline artifact (mAP 0.3086 / NDS 0.3941 from BEVDet's own test path,
  kept only in the author's research records);
- M1 run artifacts;
- output of the synthetic preprocessing reference check;
- any HTTP comparison of all 6,019 frames (never run).

The record is [verification-m3.md](verification-m3.md). The M8 source snapshot (18 files and a
patch) still matches its manifest.

## 2. Fresh-clone installation and tests (level L0)

| Step | Result |
| --- | --- |
| Clone | `git clone --no-hardlinks` of the local repository at `9f9edf4` into a temporary directory outside the repository, then exactly the nine approved M9 files copied in. Source and clone SHA-256 match for all nine, re-checked byte-for-byte (`cmp`) in Stage D. The clone holds the 129 tracked files plus 4 new docs, and no `outputs/`, virtual environment, cache, env file, image or weights. The `refs/agents/*` refs were not transferred |
| Interpreter | No `python3.10` on `PATH` (Ubuntu 24.04 ships 3.12). A conda-provided CPython 3.10.21 was used as the base of a new, isolated venv (no inherited site-packages). This prerequisite is now documented |
| Install | `pip install -e ".[dev]"` with pip 23.0.1 and the cache disabled: success in 20 s, every exact pin resolved from PyPI |
| CI checks (order of `ci.yml`) | restricted-content guard OK (133 files); ruff check pass; ruff format check pass (91 files); mypy pass (23 files) |
| Tests | `pytest -q`: **313 passed, 4 skipped, 0 failed**, 6 min 17 s, exit 0. The skips are the 3 private GPU test files and the CPU smoke test without torch. No test or server process remained |

The test run used earlier versions of four documentation files (README, `architecture.md`,
`reproducibility.md`, `m9-plan.md`). Their final versions differ only in documentation text. They
were re-checked in the clone with the guard, the link checker and `git diff --check`.
`pyproject.toml`, the only installed metadata, had the same hash at install time and at the end.

## 3. Demo and documentation validation

**FakeDetector demo** (README commands, in the clone, port 8000 on loopback):
- `/v1/model` reported `fake-detector` with no checkpoint, and `/docs` and `/openapi.json`
  returned 200.
- The valid six-camera request returned 200 with 3 synthetic boxes; the three invalid requests
  returned 422 (`camera_set`, `image_size`, `rotation`).
- The venv contained no torch, and no weights, dataset or GPU were used.
- After shutdown, no listener or process remained.

**Record of the demo:**
- The retained server log shows the fake-detector warning, the request sequence (one 200 and
  three 422s on `/v1/detect`) and a clean shutdown.
- The demo client's printed output, including the error type names, was observed in the
  verification session but not saved to a file.

**Documentation checks:**
- **Links:** all local links and anchors in the README and `docs/` resolve; the only external link
  is BEVDet's repository.
- **Mermaid diagrams:** structurally linted: balanced blocks; all 12 flowchart edges and 14
  sequence messages use declared nodes; balanced quotes and brackets. They were checked against
  the code:
  - all 17 source files they name exist;
  - middleware order, request step order, status codes and admission-token rules match
    `api.py`, `middleware.py`, `executor.py` and `service.py`.

  **Rendering was not verified** (no local renderer).

## 4. Claims-to-evidence verification

- **M3 claims:** every one matches the preserved originals exactly:
  - frames; mAP and NDS at both precisions; the true-positive error metrics;
  - box counts and round-trip mismatches;
  - the 50- and 200-frame results, repeatability and the M2 counts;
  - checkpoint hash and wall time.
- **Every number** in the public documents traces to an existing report, a source file or a
  preserved original. The one non-literal value, NDS 0.394126, is the rounding of 0.3941255…
- **Conditions shown with the results:**
  - M5 and M6: `max_in_flight` 16, Server-Timing on, 20 frames, one host, same-host client;
  - M7: a conditional decision, sustained stages not run, the overload limitation;
  - M8: P0 failed, P1–P6 passed, effectiveness not established, Option A, code removed.
- **No unsupported framing:** no production-readiness, real-time, scaling, "robust under
  overload" or CI-passing claims. CI is described as configured but never observed to run.

## 5. Privacy and history audit

**Tree.** The restricted-content guard passes on the tree.

**Reachable history** (266 objects): no private keys or tokens, no home paths or user names, no
weights, dataset, image or `outputs/` files ever committed (largest blob 40 KB). Commit identities
use the GitHub noreply address only.

**Unresolved in history** (not rewritten):
- **Session URLs:** three commit messages (`784c227`, `4655ad7`, `3024183`) contain a
  `Claude-Session` URL. It is a private session link, not a credential.
- **Agent checkpoint refs:** five refs under `refs/agents/…/checkpoints/` were created by an agent
  host tool. They hold snapshots of the working tree, including M9 draft documents, but nothing
  under `outputs/` and no restricted files. Any push must be limited to `main` (and approved
  tags), never `--all` or `--mirror`.

## 6. Reproducibility limitations

- **L0 (CPU, no assets):** verified from a fresh clone. It needs a CPython 3.10 interpreter and
  network access to PyPI.
- **L1 (CPU smoke, random weights):** not run in M9. It builds mmcv-full from source and fetches
  BEVDet commit `a0142cc` from the author's public fork. Its CI workflow has never run.
- **L2–L4 (GPU, real model, verification suite, parity):** need a GPU, a legally obtained
  checkpoint and nuScenes. They were verified on one workstation only, and only sm_120 is built.
  The container verification script stops unless the kernel is `6.14.0-37-generic`.
- **Parity:** the M3 full-validation parity can be checked only by re-running both sides (level
  L4); the reference-run artifact is not preserved.

## 7. Outstanding public-release blockers (Stage E, separate authorization)

1. **Licensing, needs external verification:**
   - redistribution terms of the third-party software that container images would contain
     (NVIDIA CUDA/cuDNN base images, mmcv operators); images stay unpublished until then;
   - the checkpoint's terms (never redistributed);
   - BEVDet's `bev_pool_v2` per-file notice ("All rights reserved", under the repository's
     Apache-2.0).
2. **External build input:** confirm that BEVDet commit `a0142cc` stays publicly reachable on the
   fork.
3. **History decision:** keep or rewrite the three session-URL commit messages; delete the agent
   refs or exclude them from any push.
4. **First real CI runs (T0, then T1):** observe them after publication and correct the
   documentation if they fail.
5. **Hosting:** choose the repository host and visibility.

Known and disclosed, not blockers: the unhandled `ClientDisconnect` logging defect (M7), and the
"thesis path" wording in two source docstrings, left unchanged because they are code.
