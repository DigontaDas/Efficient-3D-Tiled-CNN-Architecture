# Persistent thesis rules — RASNet coronary segmentation (ImageCAS)

These rules apply to every AI agent working in this repo (Claude Code here, Gemini/Antigravity via `GEMINI.md`). Keep `CLAUDE.md` and `GEMINI.md` consistent: when one changes, update the other in the same change.

## 1. The thesis in one paragraph

RASNet is a SegResNet backbone (MONAI) with 3D attention gates, deep supervision and `StenosisAwareLoss`, at 4.71 M parameters. It is compared with nnU-Net V2 (genuine `nnunetv2 2.8.1`), SegResNet, V-Net and a residual 3D U-Net on ImageCAS.

The goal is to show RASNet is an accurate **and** efficient alternative. The aim is to beat the baselines, but only under protocols that are identical for every model. A win obtained by giving RASNet extra tuning, extra TTA or different settings is worthless to the thesis and must never be engineered. Efficiency (parameters, GFLOPs, latency, VRAM) is always a valid claim.

## 2. Two machines — know which one you are on

| | Home PC (`H:\Thesis_Trainings`) | Lab PC (`C:\Thesis_RASNET`, RTX 4080 SUPER 16 GB) |
| :--- | :--- | :--- |
| Data, checkpoints, patient data | **Not available** | Available |
| What may run | Synthetic / CPU tests and code reading only | Real training and evaluation |
| Env | Whatever is installed (MONAI may differ, `nnunetv2` absent) | `C:\Thesis_RASNET_env\bench` (Python 3.12, MONAI 1.6.1, nnunetv2 2.8.1) |

- Never invent paths, results or numbers. If something can only be verified on the lab PC, mark it **LAB-CHECK**.
- Paths such as `C:\Thesis_RASNET\...` in transcripts and configs map to the same repo-relative path here.
- `Rasnet-v2/configs/fair_v2.yaml` and the `Rasnet-v2/tests/*` files hard-code `C:/Thesis_RASNET`. On the home PC those tests fail for environment reasons only; say so, do not "fix" the paths.
- **Everything that runs for real runs on the lab PC.** Write all commands, settings and performance tuning for the lab PC: PowerShell, `$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"`, repo root `C:\Thesis_RASNET`, RTX 4080 SUPER 16 GB. Never tune for the home PC's hardware.
- v1 synthetic suite status: `tests/` collects 33 tests, all passing on the home PC on 2026-11-13.

## 3. Hard rules

1. **Data paths:** the dataset root lives only in config (`config.lab.yaml`, `Rasnet-v2/configs/fair_v2.yaml`), never in code.
2. **Frozen science:** do not change model architectures, losses, hyperparameters, epoch counts or checkpoints.
   - The only exception is a **dated pre-registration amendment the user has explicitly approved**, committed before any affected result exists.
   - The amendment goes in `BENCHMARK_PREREGISTRATION.md` (v1) or `Rasnet-v2/PREREGISTRATION_V2.md` (v2).
3. **Locked test set:** test cases 851–1000 are locked. No threshold, recipe, LR, seed, design or checkpoint selection may use them; select on validation (Val-94) only.
   - Test inference happens once per frozen recipe, after the validation lock file is written and committed.
   - Because v1 Arms A/B/C and Phase H already inspected 851–1000, every v1 test result is reported as **retrospective**.
4. **Symmetry for every comparison:**
   - Every compared model gets the same inference (overlap, blending), the same TTA definition, the same threshold grid and the same selection rule.
   - Symmetric TTA means 8-pass mirroring: identity + all 7 flip-axis combinations, logits averaged before softmax. This is what nnU-Net's predictor does.
   - Never tune RASNet alone and then compare it with untuned baselines.
5. **Evidence:** every reported number must come from a file produced by a run with a manifest (git commit, source hashes, checkpoint SHA-256, env).
   - Pilot numbers (Stage 2A `val20_dice`, 50-epoch, single seed) are for LR selection only and must never be cited as results.
6. **Tests first:** write the test, run it and show it failing, then implement and show it passing.
7. **Git:**
   - Never commit data or checkpoints. Record checkpoint SHA-256 in `CHECKPOINTS.md`.
   - Commit only when the user asks.
   - Never push without explicit authorisation.
8. **GPU:** one GPU job at a time on the lab PC. Before launching anything, check `Rasnet-v2/logs/queue_status.json` and `Rasnet-v2/STATE_HANDOFF.md`.
9. **Protected results:** never overwrite `results/benchmark_runs/val_full_94/`, `results/benchmark_runs/test_retrospective_150/`, `results/rasnet_fixed_weight_optimization/` or `results/thesis_outputs/`. New analyses write to new folders.
10. **Claims:**
    - Choose thesis wording from the pre-written outcome tables (`docs-for-next-v2-gen/01_…` §7, `docs-for-next-v2-gen/02_…` §5) that match the numbers. Never write a stronger claim.
    - "Not significant" is not "equal".
    - A Phase H RASNet-only result is not a fair comparison with nnU-Net.

## 4. Where things are

| What | Where |
| :--- | :--- |
| Canonical evidence index (paths + SHA-256) | `CURRENT_EVIDENCE_INDEX.md` |
| Checkpoints + hashes | `CHECKPOINTS.md` (§2 loss description corrected 2026-11-13: SegResNet and V-Net used `DiceFocalLoss(lambda_dice=1.0, lambda_focal=2.0)`; the archived ablation variants used `DiceCELoss`) |
| v1 benchmark code | `benchmark/` (`run_benchmark.py`, `metrics.py` surface-v1, `postprocess.py`, `thresholds.py`, `stats.py`, `adapters/`) |
| v1 models / training | `Phase3_Local_Integration/rasnet_model.py`, `rasnet_loss.py`; `Q1_Publication_Package/matched_200ep_benchmark/train_*_200ep.py` |
| Splits | `Phase3_Local_Integration/splits_final.json` (690 / 94 / 150) |
| v1 locked results | `results/benchmark_runs/{val_full_94,test_retrospective_150}/`, `results/thresholds.json`, `results/thesis_outputs/Table1–4` |
| Phase H (RASNet-only inference exploration) | `results/rasnet_fixed_weight_optimization/` |
| Rasnet-v2 (controlled multi-seed study) | `Rasnet-v2/` (`PREREGISTRATION_V2.md`, `configs/fair_v2.yaml`, `src/run_queue.py`) |
| **Lab PC manual — start here** | `docs-for-next-v2-gen/00_LAB_PC_OPERATOR_MANUAL.md` (step-by-step for the lab PC, including instructions for the IDE agent) |
| Next-step plans (2026-11-12) | `docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md`, `docs-for-next-v2-gen/02_RASNET_V2_FAIRNESS_FIXES_AND_WIN_STRATEGY.md`, `docs-for-next-v2-gen/03_NEXT_STEPS_RUNBOOK.md` |
| Archived / superseded material | `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/` (`ARCHIVE_INDEX.csv`) |

## 5. Current state (update this section when it changes)

- **2026-11-10:** all v1 LAB-CHECK items were verified on the lab PC (`CURRENT_EVIDENCE_INDEX.md`, `results/audits/2026-11-10_verification_audit/`).
- **2026-11-12 (lab-reported, LAB-CHECK):** Rasnet-v2 Stage 2A is paused at 14/16 pilots. Job 15's resume state is saved at epoch 30, step 200; see `Rasnet-v2/STATE_HANDOFF.md` on the lab PC.
- **Open work** (not yet run, LAB-CHECK), in this order. On the lab PC follow `docs-for-next-v2-gen/00_LAB_PC_OPERATOR_MANUAL.md`; the full plan is in `docs-for-next-v2-gen/03_…`:
  1. **Option A (next):** symmetric re-evaluation of the frozen v1 RASNet / SegResNet / V-Net / nnU-Net. Stage 2A stays paused until Option A is done.
     - Create the Option A code only with the manual's extractor (§5), never by retyping.
     - Manifests record `git.dirty_code`, which must be `false`, and `git.dirty_any`, which is informational: training logs are git-tracked.
  2. **Rasnet-v2 Amendment A1:** matched clDice weight, binding Stage 2B gates, 8-pass TTA, F32 LR grid incl. 1e-4, 3-seed DS ablation. It needs the user's sign-off and must be committed before Stage 2A resumes.

## 6. Known technical pitfalls

- **SimpleITK `.img.nii.gz` quirk:** SimpleITK writes `*.img.nii.gz` as Analyze (`.hdr`/`.img`). Write to a temp `.nii.gz` name and rename, as `benchmark/adapters/nnunet_adapter.py::export_mask_like` does.
- **nnU-Net training-log numbers are not comparable:** `Pseudo dice` is computed on foreground-oversampled patches, not full volumes.
- **Empty predictions:** precision and HD95 are NaN for an empty prediction. Count empties explicitly; never let NaN-dropping help a candidate.
- **v1 3D U-Net:** it was trained on 128³ resized volumes. It is excluded from sliding-window comparisons and reported with that caveat.
