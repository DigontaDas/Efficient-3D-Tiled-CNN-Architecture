# 03 — Next-Steps Runbook (start at Option A, end at a defensible thesis)

**Machine:** lab PC `C:\Thesis_RASNET`, RTX 4080 SUPER. Every command below runs there.
**Python:** `C:\Thesis_RASNET_env\bench\Scripts\python.exe` (called `PY` below).

**Home PC vs lab PC:**

- The home PC (`H:\Thesis_Trainings`) has no data or checkpoints. Use it only to write and edit code and docs and to run the synthetic tests.
- To move work to the lab PC, either:
  - commit on the home PC, push yourself, then on the lab PC run `git pull`, then check `git log -1`; or
  - copy the changed files across.
- Never start a GPU stage on a lab tree that differs from the commit you intend to report: the manifest records `git.dirty_code`.
**Companion docs:**

- `00_LAB_PC_OPERATOR_MANUAL.md` — **start here on the lab PC**: the step-by-step for Option A, with IDE-agent instructions
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

## P0 — Housekeeping (no GPU)

1. **`CHECKPOINTS.md` §2 loss description — done 2026-11-13** on the home PC. SegResNet and V-Net are now correctly described as `DiceFocalLoss(…, lambda_dice=1.0, lambda_focal=2.0)`. Make sure that commit is on the lab PC.
2. **Write the Option A pre-registration.** Copy §1 of doc 01 into `BENCHMARK_PREREGISTRATION.md` as "Amendment: Option A (retrospective inference-parity supplement)".
3. **Write the Rasnet-v2 amendment.** Copy the Amendment A1 block from doc 02 §2 into `Rasnet-v2/PREREGISTRATION_V2.md`. Decide item by item; skipped items become stated limitations.
4. **Commit P0 alone,** so the protocol timestamp precedes all code and results:

```powershell
git add CHECKPOINTS.md BENCHMARK_PREREGISTRATION.md Rasnet-v2/PREREGISTRATION_V2.md docs-for-next-v2-gen; git commit -m "Pre-register Option A and Rasnet-v2 Amendment A1; fix CHECKPOINTS loss description"
```

Write the commit hash into your lab notebook. Both manifests will reference it.

---

## P1 — Option A code (no GPU; can be done on the home PC)

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

4. **Commit the code.** Do this before any GPU run, so the manifest's `git.dirty_code` is `false`:

   ```powershell
   git add benchmark/symmetric_eval.py scripts/run_optionA_symmetric_v1_eval.py tests/test_optionA_symmetric_eval.py results/audits/optionA_tests_*.txt; git commit -m "Option A: symmetric 8-pass TTA evaluation (tests first)"
   ```

---

## P2 — Rasnet-v2 fixes (no GPU; can overlap with P3)

Follow doc 02 §3 exactly. The Rasnet-v2 tests hard-code `C:\Thesis_RASNET`, so run them on the **lab PC**; on the home PC 3 of them fail for environment reasons only.

1. Add `Rasnet-v2/tests/test_amendment_a1.py`. Run it and save the failing output to `Rasnet-v2/logs/amendment_a1_tests_FAILING.txt`.
2. Apply patches 3.2–3.6, including the Stage 5 3-seed DS block in §3.2. Create `Rasnet-v2/src/diag_vnet_bn.py` from §4. Update the 16 → 18 and 20 → 22 assertions in `test_v2_pipeline_cpu.py`.
3. `& $PY -m pytest Rasnet-v2/tests -q` → all pass on the lab PC. Save the output to `..._PASSING.txt`.
4. `& $PY Rasnet-v2/src/run_stage1_gates.py`: the parity gates must still pass. 1-pass inference is unchanged.
5. Commit: "Rasnet-v2 Amendment A1 …" (see doc 02 §3.7).

> **Do not edit while a job runs.** If P3 is running on the GPU while you do P2, edit only `Rasnet-v2/` files. Option A imports only `benchmark/`, so the two never touch the same code.

---

## P3 — Option A run on the GPU

Do the machine preparation in doc 01 §6 first: High-performance power plan, `nvidia-smi` showing no other GPU process, sleep disabled. Replace `<B>` below with the `--sw-batch-size` chosen in step 3.1b. Use the **same** value for 3.2, 3.4 and 3.7.

| Step | Command | Pass criterion | If it fails |
| :--- | :--- | :--- | :--- |
| 3.1 Parity | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage parity` | Prints `[parity] OK`; `parity_check.csv` has max \|ΔDice\| ≤ 1e-4 for every model and check | **Stop.** The new path does not reproduce locked numbers. Check the MONAI version, AMP and `sw_batch_size` (must be 8 for parity). Do not continue until it passes. |
| 3.1b Tune `sw_batch_size` | The `foreach ($b in 4, 8, 16)` loop in doc 01 §6 | A stable, fastest value `<B>`; no 3–5× paging jumps | Use 8 (Phase H's value) if unsure |
| 3.2 Smoke | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage val --smoke 2 --sw-batch-size <B>` | Runs; prints seconds | Read the error; most likely VRAM. Use a smaller `<B>`. |
| 3.3 Estimate time | Read seconds per case per model from 3.2, × 94 (val) and × 150 (test) | Fits in your GPU window | Run model by model with `--models rasnet` etc. Resume is automatic. |
| 3.4 Val-94 | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage val --resume --sw-batch-size <B>` | `val_grid_scores.csv` has 94 × 4 × 22 = 8,272 rows | Re-run the same command; it resumes per model and case |
| 3.5 Lock | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage lock` | `locked_recipes.json` + `.sha256` written | Lock is immutable. If wrong, delete the whole output folder and restart from 3.4, documenting why. |
| 3.6 Commit the lock | `git add -f results/optionA_v1_symmetric_sw050/locked_recipes.* results/optionA_v1_symmetric_sw050/manifest_*; git commit -m "Option A: Val-94 locked recipes"` | Lock committed **before** test inference | — |
| 3.7 Test-150 (once) | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage test --resume --sw-batch-size <B>` | `test_per_case.csv` has 150 × 4 × 4 = 2,400 rows | Resume after a crash is fine. **Never** re-lock after seeing test rows. |
| 3.8 Timing (efficiency evidence) | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage timing --sw-batch-size <B>` with **nothing else on the GPU** | `timing_val.csv` (10 val cases × 4 models × 2 modes, after 1 warm-up) and `timing_summary.csv` | Re-run; timings taken while anything else used the GPU are invalid |
| 3.9 Stats | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage stats` | `summary_test.csv`, `paired_stats_families.csv`, `noninferiority_*.csv`, `OPTION_A_RESULTS.md` (now including the timing table) | — |
| 3.10 Figures | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage figures` | `fig_optionA_dice_vs_params`, `fig_optionA_dice_distributions` and `fig_optionA_paired_diff_rasnet_vs_nnunet`, each as `.png` (300 dpi) and `.pdf` | Open each PNG and check labels don't overlap with the real numbers |
| 3.11 Record | Add the folder, its manifests and SHA-256s to `CURRENT_EVIDENCE_INDEX.md` as a new "Supplement: Option A" section | Every Option A number and figure traceable | — |

The smoke run writes to `results/optionA_v1_symmetric_sw050_SMOKE/`, and the tuning runs to `results/optionA_tune_bs*`. Delete both after 3.3; they are never reported.

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

   Optional, while the GPU is free: run the V-Net BatchNorm diagnostic (doc 02 §4). It is diagnostic only and never used for any selection.

   ```powershell
   & $PY Rasnet-v2/src/diag_vnet_bn.py --run pilot2a_vnet_f16_lr1e-4_seed42 --n-cases 3
   ```
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

8. **Stage 6 / Phase I:** the prospective clinical cohort, only when ethics approval and the doctor-annotated scans are ready.
   - This stage is **not** part of `run_queue.py`; it is its own script.
   - It reads the frozen Stage 5 lock `Rasnet-v2/results/stats_and_efficiency/MANIFEST_V2_LOCKED.json` and the Val-94 thresholds. It does no tuning of any kind.
   - Point it at the cohort either by folder (`--cohort-dir`) or with a CSV of `case_id,image_path,label_path` (`--manifest-csv`).
   - Keep patient data out of git and out of this repo's tracked folders. Put the cohort folder path only on the command line or in config, never in code.

   ```powershell
   & $PY Rasnet-v2/src/stage6_phaseI_clinical.py --config Rasnet-v2/configs/fair_v2.yaml --manifest-csv <path-to-your-cohort-manifest.csv>
   ```

   - `--manifest-locked <path>` overrides the default Stage 5 lock location.
   - Re-running resumes by default. `--no-resume` starts over, but only before any cohort result has been looked at.

Checkpoints along the way. Record each in `CHECKPOINTS.md` with its SHA-256, never commit the files, and keep the zip archives as for v1:

- after 2A, `locked_lrs.json`
- after 2B, `pilot_lock_decision.json`
- after Stage 3, every `best_model.pth` / `checkpoint_best.pth`

---

## P5 — Thesis integration (how the two studies fit together)

| Chapter / section | Content | Source files |
| :--- | :--- | :--- |
| Methods: v1 benchmark | Arms A/B/C as already locked; state the inference asymmetry openly | `results/thesis_outputs/Table1–4` |
| Results: v1 symmetric supplement | Option A R1–R4 table, families F-A to F-D, non-inferiority, end-to-end timing, the three Option A figures | `results/optionA_v1_symmetric_sw050/OPTION_A_RESULTS.md`, `fig_optionA_*.pdf` |
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
- Any Option A manifest shows `git.dirty_code: true` for a result you intend to report. `dirty_any` can be `true` on the lab PC because training logs are git-tracked; that is expected and harmless.
- Disk free drops below 100 GB during Stage 3.
- You notice you want to change a threshold, LR, seed or recipe **after** seeing Val-94 or Test-150 numbers. That is the moment to write it down as an exploratory note, not to change the protocol.
