# M2/M3 verification record: numerical parity with BEVDet (2026-10-27)

The original runs took place on 2026-10-27, at commit `784c227` (M1+M2) and `4655ad7` (M3). This
record was written on 2026-10-28, during M9, from the surviving original result files.

**No new validation run was made for it.** The originals are kept byte-for-byte in a private,
gitignored archive, with a SHA-256 manifest. They are not published: some files contain local
paths and truncated nuScenes scene identifiers.

## Setup

| Item | Value |
| --- | --- |
| Model | BEVDet-R50-CBGS, checkpoint sha256 `8b67e60de54e76fe64195816d90eb2ec1d73aac93ed7d834873107c509a13443`; config sha256 `aef7c119…cb440` |
| Reference code | BEVDet dev3.0 checkout, used read-only (its `git status` was recorded before and after) |
| Environment | the `bevserve-m2` host environment ([env-m2.md](env-m2.md)), one RTX PRO 6000 Blackwell GPU |
| Data | nuScenes val split (6,019 frames), prepared locally; not part of this repository |
| Tools (unchanged since `4655ad7`) | `tools/full_val_parity.py`, `tools/compare_direct.py`, `tools/repeatability.py`; the M2 comparison tool `tools/m2_compare_direct.py` was replaced by `compare_direct.py` in `4655ad7` |

## Results

| Check | Method | Result |
| --- | --- | --- |
| Full validation | `full_val_parity.py`, all 6,019 val frames. Raw JPEG bytes and calibration from the info file go through BEVServe's own input path (header checks, decode, calibration validation, input building) and `BEVDetDetector.detect_native`. They then go through the API's box conversion to geometric-center z and back (the inverse is asserted bit-identical to the native output). Finally, BEVDet's `NuScenesDataset.evaluate` runs unchanged. In-process, not over HTTP | **mAP 0.308580, NDS 0.394126**; mATE 0.7148, mASE 0.2800, mAOE 0.5979, mAVE 0.7789, mAAE 0.2300; 545,868 boxes; **0 frames** with a round-trip mismatch; wall time 4 min 06 s |
| HTTP vs BEVDet, 50 frames | `compare_direct.py --select scenes:50` (the middle frame of 50 val scenes) against BEVDet's `single_gpu_test` per-sample path, through a running BEVServe | 50/50 HTTP 200; box counts, labels and model identity equal for all 50; max abs diff **0.0** for score, center xy, bottom z, size, yaw and velocity; all seven preprocessing arrays max abs diff **0.0**; 4,831 boxes |
| HTTP vs BEVDet, 200 frames | `compare_direct.py --select even:200` (frames evenly spaced over val; 150 distinct scenes) | 200/200 HTTP 200; counts, labels and identity equal; every box-field and preprocessing max abs diff **0.0**; 18,176 boxes |
| Repeatability | `repeatability.py --index 0 --repeats 20`: 20 repeats over HTTP, 20 in process, and 20 in process with other frames interleaved | bit-identical in all three (46 boxes each time, labels identical, every max abs diff 0.0) |
| M2 spot checks | three single frames (val indices 0, 2500, 6000), HTTP vs BEVDet path | box counts 46/46, 87/87, 41/41; labels equal; every box-field and preprocessing max abs diff 0.0 |

**Reference baseline (not preserved here).** The full-validation metrics are equal, to every
printed digit, to the author's earlier reference evaluation of the same checkpoint with BEVDet's
own test path: mAP 0.3086, NDS 0.3941.

The two sides of that comparison have different support:
- **BEVServe's side** has surviving original evidence (`fullval/summary.json`, `metrics.json` and
  the run log, archived privately).
- **The reference side** is documented only by the author's separate research records. Its
  artifact is **not preserved** in this repository or in the archive, and was not re-run for M9.

Neither side can be reproduced from the public repository alone: both need nuScenes and the
checkpoint (level L4 in [reproducibility.md](reproducibility.md)). Anyone re-running the
comparison must also produce the reference side themselves.

BEVDet's README publishes 0.313 / 0.398 for this model. That figure came from a different
environment, and BEVServe does not claim to match it.

## Scope and limitations

- **Over HTTP:** the full-validation run exercises BEVServe's input path and adapter in process;
  the HTTP comparisons cover 50 + 200 frames, not all 6,019.
- **One run:** each check ran once, with one checkpoint, on one GPU type (sm_120) and in one
  environment. A different environment can change floating-point results.
- **Synthetic check:** the preprocessing bit-identity check on synthetic inputs
  (`tools/reference_check_bevdet.py`) left no retained output; it can be re-run where a BEVDet
  checkout is available ([reproducibility.md](reproducibility.md), level L4).
- **Not retained:**
  - the per-sample prediction file of the full-validation run (dataset-derived, 253 MB);
  - the M1 run artifacts, beyond the test suite.
- **Later checks:**
  - container results were compared with these host results in M4 ([verification-m4.md](verification-m4.md));
  - the inputs of parallel preprocessing were compared on all 6,019 frames in M6
    ([m6-report.md](m6-report.md)).
