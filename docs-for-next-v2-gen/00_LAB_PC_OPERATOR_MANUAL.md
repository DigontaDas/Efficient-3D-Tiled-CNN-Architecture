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
2. **Do not rewrite the Option A code.** Create the three files with the extractor in §5. It copies the code blocks from doc 01 byte for byte. The code was tested as written (51/51 tests on the home PC). Retyping or "improving" it breaks that guarantee.
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
| Option A code | Exists only inside doc 01. Created on the lab PC in step §5. |
| Rasnet-v2 Amendment A1 | Written in doc 02 §2. Needs the user's sign-off. Must be applied and committed **before** Stage 2A resumes. |
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

All four docs (`00_`–`03_`) must be listed. If `git pull` reports a conflict: **STOP**, do not resolve it automatically, and show it to the user.

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

## 4. Step C — pre-register Option A (before any code or result)

1. Open `docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md` §1 ("Pre-registered protocol").
2. Copy that whole section to the end of `BENCHMARK_PREREGISTRATION.md`, under the heading `## Amendment: Option A (retrospective inference-parity supplement) — <today's date>`.
3. **Commit, after the user says go:**

```powershell
git add BENCHMARK_PREREGISTRATION.md; git commit -m "Pre-register Option A (symmetric v1 inference supplement)"
```

Write the commit hash in the lab notebook.

---

## 5. Step D — create the Option A code, tests first

The extractor below copies code blocks from doc 01 into files. It:

- checks that each block starts with the expected header;
- refuses to overwrite an existing file;
- prints a short hash of what it wrote.

**D1. Create only the test file, and watch it fail:**

```powershell
@'
import hashlib, pathlib, re, sys
doc = pathlib.Path("docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md").read_text(encoding="utf-8")
blocks = re.findall(r"```python\n(.*?)```", doc, re.S)
if len(blocks) != 3:
    raise SystemExit(f"Expected 3 python blocks in doc 01, found {len(blocks)}; doc changed - stop.")
targets = {
    "tests": (0, "tests/test_optionA_symmetric_eval.py", '"""Option A symmetric-evaluation contract tests'),
    "lib":   (1, "benchmark/symmetric_eval.py", '"""Option A: symmetric post-hoc inference'),
    "cli":   (2, "scripts/run_optionA_symmetric_v1_eval.py", "#!/usr/bin/env python"),
}
for key in sys.argv[1:]:
    i, path, head = targets[key]
    code = blocks[i]
    if not code.startswith(head):
        raise SystemExit(f"Block {i} does not start with {head!r}; doc changed - stop.")
    p = pathlib.Path(path)
    if p.exists():
        raise SystemExit(f"{p} already exists - not overwriting.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code, encoding="utf-8", newline="\n")
    print("wrote", p, "sha256:", hashlib.sha256(code.encode("utf-8")).hexdigest()[:16])
'@ | Set-Content -Encoding utf8 $env:TEMP\extract_optionA.py
& $PY $env:TEMP\extract_optionA.py tests
```

Leave the `'@` line at column 0: PowerShell requires it.

```powershell
New-Item -ItemType Directory -Force results/audits | Out-Null; & $PY -m pytest tests/test_optionA_symmetric_eval.py -q -p no:cacheprovider 2>&1 | Tee-Object results/audits/optionA_tests_FAILING_before_impl.txt
```

**Expected:** a collection error with `ModuleNotFoundError: No module named 'benchmark.symmetric_eval'`. That is the required "failing test first" evidence; keep the file.

**D2. Create the library and the CLI, then run the whole suite:**

```powershell
& $PY $env:TEMP\extract_optionA.py lib cli
```

```powershell
& $PY -m pytest tests -q -p no:cacheprovider 2>&1 | Tee-Object results/audits/optionA_tests_PASSING_after_impl.txt
```

**Expected: `51 passed`** (33 existing + 18 Option A). Anything less: **STOP** and report the failing test names and their `E ` lines.

**D3. Commit the code before any GPU run,** so every Option A manifest records `git.dirty_code: false`. After the user says go:

```powershell
git add benchmark/symmetric_eval.py scripts/run_optionA_symmetric_v1_eval.py tests/test_optionA_symmetric_eval.py results/audits/optionA_tests_FAILING_before_impl.txt results/audits/optionA_tests_PASSING_after_impl.txt; git commit -m "Option A: symmetric 8-pass TTA evaluation (tests first)"
```

If `results/` is git-ignored, add `-f` to the `git add` command.

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

1. **User decision.** The user signs off on Amendment A1 (doc 02 §2), item by item.
   - Append the signed block to `Rasnet-v2/PREREGISTRATION_V2.md` and commit it.
   - Unsigned items become stated limitations.
2. **Patches, tests first.** Apply doc 02 §3 on the lab PC:
   - add `Rasnet-v2/tests/test_amendment_a1.py` and run it; it must fail;
   - apply patches 3.2–3.6;
   - create `Rasnet-v2/src/diag_vnet_bn.py` (§4);
   - update the 16→18 and 20→22 assertions;
   - run `& $PY -m pytest Rasnet-v2/tests -q`; all must pass;
   - run `& $PY Rasnet-v2/src/run_stage1_gates.py`; it must pass.

   Then commit. Patch D5 only **appends** two pilots at the end of the queue. Jobs 1–16 keep their positions, and Job 15's resume check (`config_sha256`) is unaffected.
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
- [ ] C. Option A pre-registration committed (hash: ________)
- [ ] D1. Test file extracted; failing output saved
- [ ] D2. Library + CLI extracted; `51 passed`; passing output saved
- [ ] D3. Code committed (hash: ________)
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
- [ ] Then: Amendment A1 signed → doc 02 patches (tests first) → Stage 2A resumed
