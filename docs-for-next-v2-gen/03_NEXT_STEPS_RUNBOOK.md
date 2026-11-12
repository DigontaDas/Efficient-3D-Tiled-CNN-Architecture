# 03 — Next-Steps Runbook (start at Option A, end at a defensible thesis)

**Machine:** lab PC `C:\Thesis_RASNET`, RTX 4080 SUPER.
**Python:** `C:\Thesis_RASNET_env\bench\Scripts\python.exe` (called `PY` below).
**Companion docs:**

- `01_OPTION_A_SYMMETRIC_V1_EVAL.md` — Option A code and protocol
- `02_RASNET_V2_FAIRNESS_FIXES_AND_WIN_STRATEGY.md` — Rasnet-v2 fixes

In PowerShell, set the shortcut once per session:

```powershell
$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"; Set-Location C:\Thesis_RASNET
```

### Ground rules (from `CLAUDE.md`, enforced at every step)

- Test IDs 851–1000 are never used to pick anything.
- Write tests first and keep the failing output.
- Every reported number must come from a file that has a manifest.
- Commit no data or checkpoints. Do not push without deciding to.
- Only one GPU job at a time.

---

## Overview (order and why)

| Phase | Runs on | Purpose |
| :--- | :--- | :--- |
| P0 Housekeeping | CPU | Correct a doc error; freeze both protocol amendments **before** any new result exists |
| P1 Option A code | CPU | Tests first, then the library and CLI from doc 01 |
| P2 Rasnet-v2 fixes | CPU | Tests first, then the patches from doc 02. Can be done while P3 runs on the GPU. |
| P3 Option A run | **GPU** | Parity, smoke, Val-94, lock, Test-150, stats. Stage 2A is paused, so the GPU is free. |
| P4 Resume Rasnet-v2 | **GPU** | Stage 2A (now 18 jobs), LR lock, 2B with binding gates, Stage 3, 4A/4B/4C, Stage 5 |
| P5 Thesis integration | CPU | Tables, figures and wording chosen from the pre-written outcome tables |

Rasnet-v2 code fixes (P2) **must** be committed before Stage 2A resumes (P4). The F32 LR grid change and the gate logic affect the LR lock and Stage 2B.

---

## P0 — Housekeeping (about 30 min, no GPU)

1. **Fix the documentation error in `CHECKPOINTS.md` §2.** It says SegResNet and V-Net were trained with `DiceCELoss(...)`. The training scripts use `DiceFocalLoss(lambda_dice=1.0, lambda_focal=2.0)` (`train_segresnet_200ep.py:153`, `train_vnet_200ep.py:154`). Correct the two bullets. This is a doc fix only; no number changes.
2. **Write the Option A pre-registration.** Copy §1 of doc 01 into `BENCHMARK_PREREGISTRATION.md` as "Amendment: Option A (retrospective inference-parity supplement)".
3. **Write the Rasnet-v2 amendment.** Copy the Amendment A1 block from doc 02 §2 into `Rasnet-v2/PREREGISTRATION_V2.md`. Decide item by item; skipped items become stated limitations.
4. **Commit P0 alone,** so the protocol timestamp precedes all code and results:

```powershell
git add CHECKPOINTS.md BENCHMARK_PREREGISTRATION.md Rasnet-v2/PREREGISTRATION_V2.md docs; git commit -m "Pre-register Option A and Rasnet-v2 Amendment A1; fix CHECKPOINTS loss description"
```

Write the commit hash into your lab notebook. Both manifests will reference it.

---

## P1 — Option A code (about 1–2 h, no GPU)

1. **Create the test file and run it to see it fail.** Create `tests/test_optionA_symmetric_eval.py` from doc 01 §3, then run:

   ```powershell
   & $PY -m pytest tests/test_optionA_symmetric_eval.py -q 2>&1 | Tee-Object results/audits/optionA_tests_FAILING_before_impl.txt
   ```

   Expected: `ModuleNotFoundError: No module named 'benchmark.symmetric_eval'`. Keep that file.
2. **Implement the library and CLI.** Create `benchmark/symmetric_eval.py` (doc 01 §4) and `scripts/run_optionA_symmetric_v1_eval.py` (doc 01 §5).
3. **Run the full test suite.** Everything must pass, including the old tests:

   ```powershell
   & $PY -m pytest tests -q 2>&1 | Tee-Object results/audits/optionA_tests_PASSING_after_impl.txt
   ```

4. **Commit the code.** Do this before any GPU run, so the manifest's `git.dirty` is `false`:

   ```powershell
   git add benchmark/symmetric_eval.py scripts/run_optionA_symmetric_v1_eval.py tests/test_optionA_symmetric_eval.py results/audits/optionA_tests_*.txt; git commit -m "Option A: symmetric 8-pass TTA evaluation (tests first)"
   ```

---

## P2 — Rasnet-v2 fixes (about 2–3 h, no GPU; can overlap with P3)

Follow doc 02 §3 exactly:

1. Add `tests/test_amendment_a1.py`. Run it and save the failing output to `Rasnet-v2/logs/amendment_a1_tests_FAILING.txt`.
2. Apply patches 3.2–3.6. Update the 16 → 18 and 20 → 22 assertions in `test_v2_pipeline_cpu.py`.
3. `& $PY -m pytest Rasnet-v2/tests -q` → all pass. Save the output to `..._PASSING.txt`.
4. `& $PY Rasnet-v2/src/run_stage1_gates.py`: the parity gates must still pass. 1-pass inference is unchanged.
5. Commit: "Rasnet-v2 Amendment A1 …" (see doc 02 §3.7).

> **Do not edit while a job runs.** If P3 is running on the GPU while you do P2, edit only `Rasnet-v2/` files. Option A imports only `benchmark/`, so the two never touch the same code.

---

## P3 — Option A run on the GPU

Confirm nothing else is using the GPU (no `python.exe` processes in `nvidia-smi`), then proceed.

| Step | Command | Pass criterion | If it fails |
| :--- | :--- | :--- | :--- |
| 3.1 Parity | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage parity` | Prints `[parity] OK`; `parity_check.csv` has max \|ΔDice\| ≤ 1e-4 for every model and check | **Stop.** The new path does not reproduce locked numbers. Check the MONAI version, AMP and `sw_batch_size` (must be 8). Do not continue until it passes. |
| 3.2 Smoke | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage val --smoke 2` | Runs; prints per-case seconds | Read the error; most likely VRAM. Retry with `--sw-batch-size 4`. |
| 3.3 Estimate time | Read seconds per case per model from 3.2, × 94 (val) and × 150 (test) | Fits in your GPU window | Run model by model with `--models rasnet` etc. Resume is automatic. |
| 3.4 Val-94 | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage val --resume` | `val_grid_scores.csv` has 94 × 4 × 22 = 8,272 rows | Re-run the same command; it resumes per model and case |
| 3.5 Lock | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage lock` | `locked_recipes.json` + `.sha256` written | Lock is immutable. If wrong, delete the whole output folder and restart from 3.4, documenting why. |
| 3.6 Commit the lock | `git add -f results/optionA_v1_symmetric_sw050/locked_recipes.* results/optionA_v1_symmetric_sw050/manifest_*; git commit -m "Option A: Val-94 locked recipes"` | Lock committed **before** test inference | — |
| 3.7 Test-150 (once) | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage test --resume` | `test_per_case.csv` has 150 × 4 × 4 = 2,400 rows | Resume after a crash is fine. **Never** re-lock after seeing test rows. |
| 3.8 Stats | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage stats` | `summary_test.csv`, `paired_stats_families.csv`, `noninferiority_*.csv`, `OPTION_A_RESULTS.md` | — |
| 3.9 Record | Add the folder, its manifests and SHA-256s to `CURRENT_EVIDENCE_INDEX.md` as a new "Supplement: Option A" section | Every Option A number traceable | — |

The smoke run writes to `results/optionA_v1_symmetric_sw050_SMOKE/`. Delete it after 3.3; it is never reported.

**Reading the result.** Open `OPTION_A_RESULTS.md`:

- Find the R4 rows.
- Find family `FA_primary_rasnet_vs_nnunet`, family `FB_rasnet_vs_segresnet`, and the non-inferiority table.
- Pick the matching row in doc 01 §7.

That row is your thesis sentence for v1.

---

## P4 — Resume Rasnet-v2 on the GPU

The order below is the existing pipeline. The new parts come from Amendment A1.

1. **Resume Stage 2A.** Job 15 resumes from `mid_epoch_state.pt` (epoch 30, step 200). Then Job 16 runs, then the two new F32 lr = 1e-4 pilots.

   ```powershell
   & $PY Rasnet-v2/src/run_queue.py --stage stage2a
   ```

   Check `STATE_HANDOFF.md` first for any flag your parent session used, such as `--auto-cache`.
2. **LR lock on Val-94** (epoch-50 checkpoints) → `results/pilots_val94/locked_lrs.json`:

   ```powershell
   & $PY Rasnet-v2/src/run_queue.py --stage stage2a_eval
   ```

   Check the edge rule: if any locked LR is 1e-4 or 1e-3 (a grid edge), record it in the thesis limitations. Do **not** extend the grid after seeing it.
3. **Stage 2B design pilots and gated lock:**

   ```powershell
   & $PY Rasnet-v2/src/run_queue.py --stage stage2b; & $PY Rasnet-v2/src/run_queue.py --stage stage2b_eval
   ```

   Open `pilot_lock_decision.json`:
   - `all_gates_passed: true` → continue.
   - `false` → Stage 3 now refuses to start (Patch D2). Write a dated amendment explaining the failure (e.g. precision 0.79). Only then use `--allow-gate-failure`. Report the gate failure in the thesis.
4. **Stage 3:** 22 runs × 200 epochs plus 3 nnU-Net seeds. This is the long block (many days):

   ```powershell
   & $PY Rasnet-v2/src/run_queue.py --stage stage3
   ```

   - Before starting, check disk headroom: the queue requires more than 100 GB free.
   - Disable Windows sleep and updates for the duration.
   - The queue is resumable via `DONE.flag`, `latest_state.pt` and `mid_epoch_state.pt`.
   - For a clean pause, stop at an epoch boundary as your parent session did, then resume with the same command.
5. **Stage 4A/4B:** val20 curves, then the Val-94 thresholds lock. **Commit `thresholds_v2.json` before 4C.**

   ```powershell
   & $PY Rasnet-v2/src/run_queue.py --stage stage4a; & $PY Rasnet-v2/src/run_queue.py --stage stage4b
   ```

6. **Stage 4C:** Test-150, once:

   ```powershell
   & $PY Rasnet-v2/src/run_queue.py --stage stage4c
   ```

7. **Stage 5:** families, non-inferiority, efficiency on the 4080 SUPER, Pareto plot:

   ```powershell
   & $PY Rasnet-v2/src/run_queue.py --stage stage5
   ```

8. **Stage 6 / Phase I:** the prospective clinical cohort, if ethics and data are ready. Same frozen recipes; no re-tuning.

Checkpoints along the way. Record each in `CHECKPOINTS.md` with its SHA-256, never commit the files, and keep the zip archives as for v1:

- after 2A, `locked_lrs.json`
- after 2B, `pilot_lock_decision.json`
- after Stage 3, every `best_model.pth` / `checkpoint_best.pth`

---

## P5 — Thesis integration (how the two studies fit together)

| Chapter / section | Content | Source files |
| :--- | :--- | :--- |
| Methods: v1 benchmark | Arms A/B/C as already locked; state the inference asymmetry openly | `results/thesis_outputs/Table1–4` |
| Results: v1 symmetric supplement | Option A R1–R4 table, families F-A to F-D, non-inferiority | `results/optionA_v1_symmetric_sw050/OPTION_A_RESULTS.md` |
| Results: Phase H | Present as a *RASNet-only inference exploration*. Its comparison to nnU-Net is superseded by Option A, which tuned all models equally. | `results/rasnet_fixed_weight_optimization/` |
| Results: Rasnet-v2 (main controlled study) | Family 1 (architecture, 3 seeds), Family 2 (vs nnU-Net + non-inferiority), ablations, efficiency Pareto | Stage 5 outputs |
| Limitations | Test set inspected in v1 (retrospective); val20 ⊂ Val-94 checkpoint selection; nnU-Net trained with its own recipe (framework vs framework); V-Net BN artefact if found; any gate failure or skipped amendment item | this runbook + manifests |
| Discussion: "why use RASNet" | Efficiency numbers (parameters, GFLOPs, latency, VRAM on the 4080 SUPER) plus whichever accuracy claims the families support | Table 4, Stage 5 efficiency CSV |

**Headline-wording rule.** Choose the sentence from the pre-written tables (doc 01 §7; doc 02 §5) that matches the numbers. Never write a stronger one.

Most likely realistic headline, if the numbers support it: "RASNet matches nnU-Net V2 within 0.005 Dice (non-inferior) while using 6.5× fewer parameters, and outperforms the identical-size SegResNet backbone." That is a strong, publishable, adoptable result.

---

## Quick reference — stop conditions

Stop and decide before continuing if any of these happen:

- Option A parity fails (> 1e-4).
- Any test file fails after a patch.
- Stage 2B `all_gates_passed` is `false`.
- Any manifest shows `git.dirty: true` for a result you intend to report.
- Disk free drops below 100 GB during Stage 3.
- You notice you want to change a threshold, LR, seed or recipe **after** seeing Val-94 or Test-150 numbers. That is the moment to write it down as an exploratory note, not to change the protocol.
