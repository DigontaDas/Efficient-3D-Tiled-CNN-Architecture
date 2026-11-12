# 00 — Lab PC Operator Manual (read this first)

**Where this runs:** lab PC, repo `C:\Thesis_RASNET`, RTX 4080 SUPER 16 GB, Python env `C:\Thesis_RASNET_env\bench`.
**Who reads it:** you, and the AI agent in your IDE (Antigravity reads `GEMINI.md`, Claude Code reads `CLAUDE.md`; both point here).
**Today's goal:** run **Option A**, the symmetric re-evaluation of the frozen v1 RASNet / SegResNet / V-Net / nnU-Net. Rasnet-v2 Stage 2A stays **paused** until Option A is finished.

| Doc | Use it for |
| :--- | :--- |
| `00_LAB_PC_OPERATOR_MANUAL.md` (this file) | What to do, in order, on the lab PC |
| `01_OPTION_A_SYMMETRIC_V1_EVAL.md` | Option A protocol (§1), code (§3–§5), lab-PC tuning (§6), thesis wording (§7) |
| `02_RASNET_V2_FAIRNESS_FIXES_AND_WIN_STRATEGY.md` | Rasnet-v2 Amendment A1 and its patches, needed **before** Stage 2A resumes |
| `03_NEXT_STEPS_RUNBOOK.md` | The full long-term plan (P0–P5), incl. Stages 2A → 6 |

---

## 0. Instructions for the AI agent in the IDE

1. **Rules.** Follow `CLAUDE.md` / `GEMINI.md` §3 at all times. The ones that matter most today:
   - Test IDs 851–1000 are never used for any selection.
   - Never write into `results/benchmark_runs/val_full_94/`, `results/benchmark_runs/test_retrospective_150/`, `results/rasnet_fixed_weight_optimization/` or `results/thesis_outputs/`.
   - One GPU job at a time.
   - Every number you report must come from a file produced by a run with a manifest. Do not quote numbers from chat or transcripts.
2. **Do not write or rewrite core code.** All core code is already committed:
   - **Option A:** `benchmark/symmetric_eval.py`, `scripts/run_optionA_symmetric_v1_eval.py`, `tests/test_optionA_symmetric_eval.py` (commit `aecf60e`).
   - **Rasnet-v2 Amendment A1:** commit `79e0458`.

   You may fix only trivial environment problems, such as a missing package or a path in a config file, and only after showing the user the exact diff and getting approval. Never change logic, protocol constants, thresholds, grids, seeds, losses or architectures. If code seems wrong, **STOP** and report it.
3. **Work in order.** Do the steps in §2–§8 in order. At every **STOP** condition, stop and report to the user; never work around it.
4. **Git.** Run `git commit` only at the steps that say so, and only after the user says go. Never `git push`.
5. **Reporting.** Do not post repeated status-only messages. Report when a step finishes, fails, or hits a STOP.
6. **Hands off Rasnet-v2.** Do not touch `Rasnet-v2/` during Option A. In particular, never delete or move `Rasnet-v2/checkpoints/pilot2a_rasnet_v2_f32_lr3e-4_seed42/mid_epoch_state.pt` or `latest_state.pt`: they hold the paused Job 15.

---

## 1. Current state (2026-11-13)

| Item | State |
| :--- | :--- |
| v1 benchmark (5 models, Arms A/B/C), Phase H | Locked and verified (2026-11-10). Do not modify. |
| Rasnet-v2 Stage 2A | **Paused** at 14/16 (lab-reported 2026-11-12). Job 15 can resume from epoch 30, step 200. Resume flags are in `Rasnet-v2/STATE_HANDOFF.md`. |
| Pre-registrations | Option A amendment (`BENCHMARK_PREREGISTRATION.md` §6) and Rasnet-v2 Amendment A1 (`Rasnet-v2/PREREGISTRATION_V2.md`, all 5 items signed off) committed 2026-11-13 (`34ee6ff`) |
| Option A code | Committed (`aecf60e`); tests first; 51/51 passing on the home PC. Verified on the lab PC in step §5. |
| Rasnet-v2 Amendment A1 code | Committed (`79e0458`); 11/11 contract tests on the home PC. The full Rasnet-v2 suite must be re-run on the lab PC (§8). |
| `CHECKPOINTS.md` loss description | Fixed 2026-11-13 (SegResNet and V-Net used `DiceFocalLoss`) |

---

## 2. Step A — pull safely

Open **PowerShell** on the lab PC:

```powershell
$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"; Set-Location C:\Thesis_RASNET
```

```powershell
git status --short; git log -1 --oneline; git diff --stat HEAD -- Rasnet-v2/src Rasnet-v2/tests benchmark scripts tests
```

Read the output:

| What you see | What to do |
| :--- | :--- |
| Only `Rasnet-v2/logs/*`, `Rasnet-v2/checkpoints/*/DONE.flag` or `queue_status.json` modified | Normal: training updates them. Pull. |
| `?? Rasnet-v2/STATE_HANDOFF.md` or other untracked files | Normal. Pull. |
| Any file under `Rasnet-v2/src`, `Rasnet-v2/tests`, `benchmark`, `scripts` or `tests` modified | **STOP.** These are local code edits that were never committed. Show them to the user (`git diff -- <file>`) and commit or discard them, as the user decides. Only then pull. |

```powershell
git pull
```

```powershell
git log -1 --oneline; Get-ChildItem docs-for-next-v2-gen
```

All four docs (`00_`–`03_`) must be listed. `git log --oneline -5` must include `79e0458`, `aecf60e` and `34ee6ff`. If `git pull` reports a conflict: **STOP**, do not resolve it automatically, and show it to the user.

---

## 3. Step B — environment and GPU check

```powershell
& $PY -c "import sys, torch, monai, nnunetv2, SimpleITK, cc3d; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), monai.__version__, nnunetv2.__version__)"
```

Expected: Python 3.12.x, CUDA `True`, `NVIDIA GeForce RTX 4080 SUPER`, MONAI 1.6.1, nnunetv2 2.8.1. Anything different: **STOP** and report.

```powershell
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv; Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, StartTime, Path
```

- Expected: no training process running, and the GPU close to idle.
- If any `python.exe` from `C:\Thesis_RASNET_env` is running, Stage 2A or another job is active. **STOP**: do not kill it without the user's say-so.

Checkpoint SHA-256s are verified automatically by the Option A script against `CHECKPOINTS.md`. A mismatch aborts the run.

---

## 4. Step C — pre-registration (already done; just confirm)

Both amendments were committed on 2026-11-13 in `34ee6ff`, before any code or result:

- Option A: `BENCHMARK_PREREGISTRATION.md` §6;
- Rasnet-v2 Amendment A1, all 5 items signed off: end of `Rasnet-v2/PREREGISTRATION_V2.md`.

Confirm with:

```powershell
git log --oneline -1 34ee6ff; Select-String -Path BENCHMARK_PREREGISTRATION.md -Pattern "Amendment: Option A"; Select-String -Path Rasnet-v2/PREREGISTRATION_V2.md -Pattern "Amendment A1"
```

All three must print something. Nothing to commit here.

---

## 5. Step D — verify the committed code (no writing)

The code was written tests-first on the home PC. The failing-before and passing-after outputs are committed:

- `results/audits/optionA_tests_FAILING_before_impl.txt`, `results/audits/optionA_tests_PASSING_after_impl.txt`
- `Rasnet-v2/logs/amendment_a1_tests_FAILING.txt`, `Rasnet-v2/logs/amendment_a1_tests_PASSING.txt`

On the lab PC you only re-run and confirm.

**D1. Check that the code in the repo is exactly the code in the guide.** This catches accidental edits. It must print `OK` three times:

```powershell
@'
import pathlib, re
doc = pathlib.Path("docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md").read_text(encoding="utf-8")
blocks = re.findall(r"```python\n(.*?)```", doc, re.S)
for i, path in enumerate(["tests/test_optionA_symmetric_eval.py", "benchmark/symmetric_eval.py",
                          "scripts/run_optionA_symmetric_v1_eval.py"]):
    code = pathlib.Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    print("OK      " if code == blocks[i] else "DIFFERS ", path)
'@ | Set-Content -Encoding utf8 $env:TEMP\verify_optionA.py
& $PY $env:TEMP\verify_optionA.py
```

Leave the `'@` line at column 0. If anything prints `DIFFERS`: **STOP**. Show `git diff` and `git log -3 -- <file>` to the user.

**D2. Run the v1 + Option A test suite on the lab PC:**

```powershell
& $PY -m pytest tests -q -p no:cacheprovider 2>&1 | Tee-Object results/audits/optionA_tests_lab_pc.txt
```

**Expected: `51 passed`** (33 existing + 18 Option A). Anything less: **STOP** and report the failing test names with their `E ` lines. Do not edit code to make a test pass.

**D3. Run the Rasnet-v2 suite on the lab PC.** It needs the lab paths and `nnunetv2`, so this is its real check. It runs on the CPU and does not touch Job 15's saved state.

```powershell
& $PY -m pytest Rasnet-v2/tests -q -p no:cacheprovider 2>&1 | Tee-Object Rasnet-v2/logs/amendment_a1_tests_lab_pc.txt
```

**Expected: all pass, 43 tests.** On the home PC 3 tests failed only because of missing lab paths, `nnunetv2` and MONAI 1.6.1; they must pass here. Any failure: **STOP** and report.

**D4. Commit the two lab test logs,** after the user says go:

```powershell
git add -f results/audits/optionA_tests_lab_pc.txt Rasnet-v2/logs/amendment_a1_tests_lab_pc.txt; git commit -m "Lab PC: Option A and Amendment A1 test runs"
```

After this commit, the Option A manifests will record `git.dirty_code: false`.

---

## 6. Step E — run Option A on the GPU

**E0. Prepare the machine** (doc 01 §6):

```powershell
powercfg /setactive SCHEME_MIN
```

- This switches to Windows' built-in High-performance power plan.
- Disable sleep, close browsers and other GPU apps.
- Confirm `nvidia-smi` shows no other process using the GPU.

Then run the steps below in order. `<B>` is the `--sw-batch-size` chosen in E2. Use the **same** `<B>` for E3, E5 and E8.

| # | Command | Expected / pass | If not |
| :--- | :--- | :--- | :--- |
| E1 Parity | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage parity` | `[parity] OK: max \|dDice\| = … <= 1e-04`; `parity_check.csv` written | **STOP.** The new code path does not reproduce locked numbers. Report `parity_check.csv`. |
| E2 Tune | `foreach ($b in 4, 8, 16) { & $PY scripts/run_optionA_symmetric_v1_eval.py --stage val --smoke 2 --models rasnet --sw-batch-size $b --out results/optionA_tune_bs$b }` | Note the seconds printed for each `$b`. `<B>` = the fastest stable value (no 3–5× jumps). | Unsure: `<B>` = 8 |
| E2b Clean | `Remove-Item -Recurse -Force results/optionA_tune_bs*` | Folders gone | — |
| E3 Smoke | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage val --smoke 2 --sw-batch-size <B>` | Runs all 4 models; prints seconds | Read the error. Out of memory → smaller `<B>`. |
| E4 Estimate | From E3: seconds per case × 94 (val) + × 150 (test), per model | Tell the user the estimate before E5 | — |
| E4b Clean | `Remove-Item -Recurse -Force results/optionA_v1_symmetric_sw050_SMOKE` | Smoke output deleted; never reported | — |
| E5 Val-94 | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage val --resume --sw-batch-size <B>` | `val_grid_scores.csv` = 94 × 4 × 22 = **8,272 rows** | Interrupted? Re-run the same command; it resumes. |
| E6 Lock | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage lock` | `locked_recipes.json` + `locked_recipes.sha256`; the locked τ per model is printed | Lock is immutable. Never edit or re-create it. If it is wrong: **STOP** and report. |
| E7 Commit lock | `git add -f results/optionA_v1_symmetric_sw050/locked_recipes.* results/optionA_v1_symmetric_sw050/manifest_*; git commit -m "Option A: Val-94 locked recipes"` (after the user says go) | Lock committed **before** any test inference | — |
| E8 Test-150 (once) | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage test --resume --sw-batch-size <B>` | `test_per_case.csv` = 150 × 4 × 4 = **2,400 rows** | Re-run resumes. Never re-lock after test rows exist. |
| E9 Timing | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage timing --sw-batch-size <B>` (nothing else on the GPU) | `timing_val.csv`, `timing_summary.csv` | Re-run if anything else used the GPU |
| E10 Stats | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage stats` | `summary_test.csv`, `paired_stats_families.csv`, `noninferiority_rasnet_vs_nnunet_dice.csv`, `OPTION_A_RESULTS.md` | "Incomplete test results" → finish E8 first |
| E11 Figures | `& $PY scripts/run_optionA_symmetric_v1_eval.py --stage figures` | 3 figures × (`.png` + `.pdf`) in `results/optionA_v1_symmetric_sw050/` | Open each PNG; report any overlapping labels |

Outputs go only to `results/optionA_v1_symmetric_sw050/`. Each stage also writes a `manifest_<stage>_<UTC>.json` recording:

- the git commit and `dirty_code` / `dirty_any`;
- source and checkpoint SHA-256s;
- the package versions;
- the protocol settings.

On the lab PC `dirty_any` may be `true`, because training logs are git-tracked. That is fine. `dirty_code` must be `false`.

---

## 7. Step F — record and report

1. **Evidence index.** Add a section "Supplement: Option A (2026-11-xx)" to `CURRENT_EVIDENCE_INDEX.md`. List:
   - the output folder;
   - `locked_recipes.json` and its SHA-256;
   - every manifest file;
   - `OPTION_A_RESULTS.md`;
   - the three figures.

   Get the hashes with:

   ```powershell
   Get-FileHash -Algorithm SHA256 results/optionA_v1_symmetric_sw050/*.json, results/optionA_v1_symmetric_sw050/*.csv, results/optionA_v1_symmetric_sw050/*.md | Format-Table -AutoSize
   ```

2. **Pick the thesis wording.** Open `OPTION_A_RESULTS.md` and, for recipe **R4** (the primary recipe), read:
   - family `FA_primary_rasnet_vs_nnunet`;
   - family `FB_rasnet_vs_segresnet`;
   - the non-inferiority table.

   Choose the matching row in doc 01 §7. Never write a stronger sentence than that row.
3. **Report to the user.**
   - The R4 table, the FA/FB rows and the NI row, copied from the files.
   - The timing table.
   - The figure paths.
   - Whether any STOP happened.
4. **Commit.** After the user says go, commit the CSVs, manifests, report and figures (`git add -f results/optionA_v1_symmetric_sw050/...`). Never commit probability maps or checkpoints. Option A writes no probability maps.

---

## 8. After Option A — getting back to Rasnet-v2 (do not start until §7 is done)

1. **Amendment A1 is already signed and committed** (`34ee6ff` text, `79e0458` code). D3 already re-ran the Rasnet-v2 tests on the lab PC.
2. **Gate check before resuming:**

   ```powershell
   & $PY Rasnet-v2/src/run_stage1_gates.py
   ```

   It must pass; any failure is a **STOP**. Then confirm Job 15's saved state is present:

   ```powershell
   Get-Item Rasnet-v2/checkpoints/pilot2a_rasnet_v2_f32_lr3e-4_seed42/mid_epoch_state.pt, Rasnet-v2/checkpoints/pilot2a_rasnet_v2_f32_lr3e-4_seed42/latest_state.pt
   ```

   Patch D5 only **appends** two pilots at the end of the queue. Jobs 1–16 keep their positions, and Job 15's resume check (`config_sha256`) hashes only Job 15's own settings, which are unchanged.
3. **Resume Stage 2A.** Check `Rasnet-v2/STATE_HANDOFF.md` for extra flags first (for example `--auto-cache`), then run:

   ```powershell
   & $PY Rasnet-v2/src/run_queue.py --stage stage2a
   ```

   Expected order: Job 15 resumes from `mid_epoch_state.pt`, then Job 16, then the two new F32 lr 1e-4 pilots (jobs 17–18).
4. **Continue** with runbook doc 03 **P4**: `stage2a_eval`, `stage2b`, `stage2b_eval` (gates now binding), `stage3`, `stage4a`, `stage4b`, `stage4c`, `stage5`, then Stage 6 when the clinical cohort is ready.

---

## 9. Troubleshooting

| Symptom | Likely cause | Action |
| :--- | :--- | :--- |
| `RuntimeError: <model> checkpoint SHA-256 … != CHECKPOINTS.md` | Wrong or corrupted checkpoint file at the `config.lab.yaml` path | **STOP.** Do not edit the expected hash. Re-extract the checkpoint from the archive (`CHECKPOINTS.md` §1). |
| `PARITY FAILED` | Different MONAI/torch version, AMP state, or `sw_batch_size` ≠ 8 during parity | **STOP.** Report `parity_check.csv` and the versions from step B. |
| `FileExistsError … not empty; pass --resume` | Output folder already has files | For val/test/stats this is implied; for parity, use a new `--out` or delete an aborted parity folder |
| `FileExistsError: Lock already exists` | Lock was created before | Correct behaviour: the lock is immutable. Never delete it to re-lock after test rows exist. |
| `ValueError: Refusing to write into protected results directory` | `--out` points at a locked v1 folder | Use the default `--out` |
| `Lock requires the entire validation split` | Val stage incomplete | Finish E5 (re-run it; it resumes) |
| nnU-Net: `Missing required nnU-Net artifact` | `nnUNet_results/…/fold_0/checkpoint_best.pth`, `plans.json` or `dataset.json` missing | Restore the trained-model folder from the archive (`CHECKPOINTS.md`) |
| Sudden 3–5× slowdown | WDDM GPU memory paging | Smaller `--sw-batch-size`; close other GPU apps |
| A `.img.nii.gz` written as `.hdr`/`.img` | SimpleITK naming quirk | Write a temp `.nii.gz` and rename (as `export_mask_like` does) |
| `git pull` conflict | Local uncommitted code edits on the lab PC | **STOP.** Show the user; never auto-resolve. |

---

## 10. Checklist (tick as you go)

- [ ] A. `git status` reviewed; pulled; `00_`–`03_` docs present
- [ ] B. Versions match; GPU idle; no training process
- [ ] C. Pre-registrations present (`34ee6ff`)
- [ ] D1. Code identical to guide 01 (3 × `OK`)
- [ ] D2. `tests/`: `51 passed` on the lab PC
- [ ] D3. `Rasnet-v2/tests`: all 43 passed on the lab PC
- [ ] D4. Lab test logs committed (hash: ________)
- [ ] E1. Parity OK
- [ ] E2. `<B>` chosen = ____; tuning folders deleted
- [ ] E3/E4. Smoke OK; time estimate given to user; smoke folder deleted
- [ ] E5. Val: 8,272 rows
- [ ] E6/E7. Lock written and committed **before** test
- [ ] E8. Test: 2,400 rows
- [ ] E9. Timing done with the GPU otherwise idle
- [ ] E10. Stats + `OPTION_A_RESULTS.md`
- [ ] E11. Figures checked
- [ ] F. Evidence index updated; wording picked from doc 01 §7; results committed
- [ ] Then (§8): `run_stage1_gates.py` passes → Job 15 state present → user approves → Stage 2A resumed

---

## 11. Prompts to paste into Antigravity (one at a time)

Paste them in order. Do not paste the next one until the previous phase has finished and you have answered its questions. Each phase ends at a decision only you can make: commits, the immutable threshold lock, test inference, resuming training.

**Prompt 1 — pull, verify, test (no GPU stages)**

```text
Read GEMINI.md, then docs-for-next-v2-gen/00_LAB_PC_OPERATOR_MANUAL.md, then docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md. Follow the manual exactly. All core code is already committed; do NOT write, rewrite or "improve" any code. You may only fix trivial environment problems, and only after showing me the exact diff and getting my OK.
Do manual §2 (Step A, safe git pull), §3 (Step B, environment + GPU check), §4 (confirm pre-registrations) and §5 D1–D3 (verify code identical to guide 01, run tests/ and Rasnet-v2/tests). Do NOT run any GPU stage and do NOT commit yet.
Report: git log --oneline -5, git status --short, the versions line, GPU state, the D1 output, and the pass counts of D2 (expect 51 passed) and D3 (expect 43 passed) with any failing test names and their E lines. Stop at every STOP condition in the manual and ask me.
```

**Prompt 2 — Option A up to the threshold lock**

```text
Continue with the manual: §5 D4 (commit the two lab test logs) after I confirm, then §6 E0–E6. Run parity (E1) first; if it fails, STOP and show parity_check.csv. Do the --sw-batch-size tuning (E2, E2b), the smoke run (E3), then give me the time estimate (E4) and WAIT for my OK. After my OK, delete the smoke folder (E4b), run Val-94 (E5) and the lock (E6). Show me the printed locked thresholds and the git.dirty_code value from the newest manifest, then WAIT for my OK before committing the lock (E7). Never edit, delete or re-create the lock. No status-only messages; report when a step finishes or fails.
```

**Prompt 3 — test once, timing, stats, figures, record**

```text
The lock is committed. Continue with manual §6 E8–E11 using the same --sw-batch-size as E5: Test-150 (E8, once), timing with nothing else on the GPU (E9), stats (E10), figures (E11; open each PNG and tell me if any labels overlap). Then do §7 Step F: add the "Supplement: Option A" section to CURRENT_EVIDENCE_INDEX.md with SHA-256s, and report to me the R4 table, families FA_primary_rasnet_vs_nnunet and FB_rasnet_vs_segresnet, the non-inferiority row and the timing table — copied from the files, never from memory — and which row of doc 01 §7 matches. Do not commit until I say so.
```

**Prompt 4 — later: resume Rasnet-v2 Stage 2A**

```text
Option A is finished and committed. Follow manual §8. Run Rasnet-v2/src/run_stage1_gates.py (must pass), confirm Job 15's mid_epoch_state.pt and latest_state.pt exist, read Rasnet-v2/STATE_HANDOFF.md for any extra resume flags, and show me the exact resume command. WAIT for my OK. After my OK, run it and report only at milestones (a job finished, a validation epoch with its val20_dice, or an error). The expected order is Job 15 (resume), Job 16, then the two new F32 lr=1e-4 pilots. When Stage 2A is done, STOP and tell me; do not start stage2a_eval without my OK.
```
