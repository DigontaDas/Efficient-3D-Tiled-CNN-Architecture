You are running on the lab PC (`C:\Thesis_RASNET`, PowerShell, `$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"`).
Read `GEMINI.md`, `RASNET_MASTER_FIX_AND_RETRAIN_PLAN.md`, `RASNET_CRITICAL_AUDIT.md`, `RASNET_CRITICAL_audit done1T.md`, and `docs-for-next-v2-gen/04_CODE_REVIEW_FIXES.md` first.

Execute **PART I (Stages 1–4)** of `RASNET_MASTER_FIX_AND_RETRAIN_PLAN.md` completely and autonomously, following every rule in `GEMINI.md` §3.

### Hard Guardrails
1. **Do NOT run Git history purge / filter-repo** — I am handling the tracked CT zip and checkpoint purge separately.
2. **Do NOT modify frozen v1 checkpoints** (`rasnet_best.pth`, `nnunet_best.pt`, etc.) or **overwrite protected result directories** (`results/benchmark_runs/val_full_94/`, `results/benchmark_runs/test_retrospective_150/`, `results/rasnet_fixed_weight_optimization/`, `results/thesis_outputs/`, `results/optionA_v1_symmetric_sw050/`).
3. **Do NOT start any GPU training or unpause `Rasnet-v2` Stage 3+.**
4. **Tests first (Rule 6):** Write/add tests and verify they fail BEFORE applying code patches, then show them passing.
5. **Do NOT `git commit` or `git push`** — stop at the end with `git status`, `git diff --stat`, and verification outputs so I can review and authorize the commit.

---

### Exact Steps to Execute Sequentially

#### Step 1 — Defense, Thesis & README Wording Fixes (Stage 1)
Update these exact files/lines to remove overclaims and add required disclosures:
- `results/v1_reporting_correction_20261117_105000Z/DEFENSE_QA_V1.md` (lines 38, 39, 73), `DEFENSE_SLIDE_OUTLINE.md` (lines 80, 81, 101), `THESIS_REPLACEMENT_TEXT.md` (lines 27, 66–68, 86), and `README.md` (lines 31–36, 203):
  1. Replace "consistent precision advantage" with recipe/threshold-qualified wording (citing Arm C Holm `p = 0.160` n.s. and R3 CI `[-0.0027, +0.0136]` crossing zero), and explicitly disclose nnU-Net's R4 `tau* = 0.20` grid-edge clamp next to every R4 precision claim.
  2. Remove causal attribution of gains to attention gates, deep supervision, or `StenosisAwareLoss` (disclose v1 LR/loss/calibration confounders; describe `AttentionGate3D` strictly as "soft per-patch spatial re-weighting", never "background suppression").
  3. For `RASNet vs SegResNet` R4 (`FB`), state the rank-test vs. mean-CI distinction (`112/150` wins, Wilcoxon Holm `p = 4.27e-8`; small mean gain `+0.0050` with 95% CI `[-0.0024, +0.0121]` including 0).
  4. Lock `RASNet vs nnU-Net` phrasing to comparable Dice (`-0.0021`, 95% CI `[-0.0088, +0.0030]`, not non-inferior at `-0.005`; median `+0.0002`, `76/150` wins) at `6.54x` fewer parameters and `9.07x` fewer GFLOPs/patch, with a precision–recall trade-off; Epoch-200 (`p = 0.24`) establishes neither superiority nor equivalence.
  5. Explicitly note that v1 ablations (`ablation_{attngate,deepsup}_best.pth`) are invalid due to the `PersistentDataset` LAS-train / RAS-val cache mismatch (`train_ablation_models_200ep.py`) and must not be cited.

#### Step 2 — Apply `docs-for-next-v2-gen/04_CODE_REVIEW_FIXES.md` + `benchmark/` Provenance Fixes (Stage 2, Tests First)
1. **Tests first:**
   - Create `tests/test_optionA_review_fixes.py` (11 tests) and `Rasnet-v2/tests/test_review_fixes_v2.py` (3 tests) exactly from `04_CODE_REVIEW_FIXES.md` §5.1–§5.2.
   - Add unit tests covering:
     - `benchmark/symmetric_eval.py::env_info()` resolving `nnunetv2` and `cc3d` via `importlib.metadata.version` (not `"missing"`),
     - `benchmark/symmetric_eval.py::done_keys()` deduplicating `(model, case_id, recipe)` before counting,
     - `benchmark/thresholds.py::_fast_dice()` returning `np.nan` on `both_empty`,
     - `RASNet(init_filters=16)` loading frozen `rasnet_best.pth` with `strict=True` (and matching logits bit-for-bit) plus `RASNet(init_filters=32)` forward pass succeeding.
   - Run pytest on the new tests and confirm they FAIL before patching.
2. **Apply patches:**
   - Apply `04_CODE_REVIEW_FIXES.md` §5.3–§5.7 (`R1–R10`, including re-syncing `docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md` byte-identity check).
   - Patch `benchmark/symmetric_eval.py` (`env_info` using `importlib.metadata.version`, `done_keys` deduplication) and `benchmark/thresholds.py` (`_fast_dice` `both_empty -> np.nan`).
   - Add `.gitattributes` pinning `* text=auto eol=lf` for `.json`, `.csv`, `.py`, `.md`, `.yaml`.

#### Step 3 — Non-Retraining Hardcodes, Legacy Quarantine & Doc Sync (Stage 3)
1. **`Phase3_Local_Integration/rasnet_model.py`:**
   - Derive `attention_gates` (`[4*f, 2*f, f]`) and `ds_heads` (`[4*f, 2*f]`) dynamically from `f = kwargs["init_filters"]` (keeping `GroupNorm` safe when `F_int < 8`, while staying 100% state-dict-identical for `init_filters=16` — verify strict load + output equality against `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/rasnet_best.pth`).
   - Fix the inverted comment on line 73 (`Level 0 (1/4 size): 64 ch, Level 1 (1/2 size): 32 ch`). Do NOT change `GroupNorm(1,1)` or any weight/layer math in frozen v1 `rasnet_model.py`.
2. **Remove hardcoded paths (Rule 1):**
   - Replace `ROOT = Path(r"C:\Thesis_RASNET")` with `Path(__file__).resolve().parents[1]` in `scripts/run_phaseH_rasnet_optimization.py:43` and `scripts/backfill_phaseH_val94_and_report.py:31`, and remove hardcoded `C:\Thesis_RASNET` in `recovered_checkpoints/v1_epoch200_20261117_041500Z/recover_and_verify.py:1156,1166`.
   - Update `Phase3_Local_Integration/dataset_paths.py:24-30` to read the dataset root from `config.lab.yaml` or `IMAGECAS_DATA_ROOT` env var (preserving import compatibility).
3. **Quarantine stale legacy files** into `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-19/` (and log them in an index CSV):
   - `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py` and `evaluation_results/`
   - `Phase3_Local_Integration/eval_utils.py`
   - The 7 intermediate/aborted recovery CSVs in `Q1_Publication_Package/matched_200ep_benchmark/logs/` (keep the 5 canonical `*_200ep_log.csv` files live).
4. **Fix doc drift:**
   - `CHECKPOINTS.md:13` (`nnunet_best.pt` archive timestamp -> `2026-09-08 09:19:08`), `:40` (clarify `nnUNetTrainer_200epochs` is a stock `nnunetv2` trainer; add short v1 training non-determinism & weights-only resume note).
   - `README.md:31-33,203` (correct symbol names `self.up_samples`, `self.attention_gates`, `self.ds_heads`; remove nonexistent `config.yaml`).
   - `Q1_Publication_Package/matched_200ep_benchmark/run_all_200ep.py:10` (fix stale `Epoch 71 -> 200` docstring).
   - Re-sync `GEMINI.md` §3 and `CLAUDE.md` §3 word-for-word (all 11 rules + retrospective & manifest bullets).

#### Step 4 — Read-Only Lab PC Verifications (`LAB-CHECK`, Stage 4)
1. Export pinned lab environment to `requirements.lab.txt` from `$PY -m pip freeze`.
2. Run a read-only script across all 934 ImageCAS cases in `splits_final.json` (using `config.lab.yaml` data root) verifying whether `CropForegroundd(source_key="image")` (`HU > 0`) clips ANY ground-truth coronary voxel. Save the summary to `results/audits/2026-11-19_lab_checks/crop_foreground_gt_audit.json`.
3. Inspect `Q1_Publication_Package/matched_200ep_benchmark/cache/segresnet` (if present on disk) to confirm whether cached tensors existed for the training split during the archived ablation run, and log the finding to `results/audits/2026-11-19_lab_checks/ablation_cache_check.txt`.
4. Inspect Option A predicted component volume lists for Cases `930`, `948`, `861`, `978` (if masks exist on disk) and save a short summary to `results/audits/2026-11-19_lab_checks/top2cc_outlier_components.md` (read-only prep in case I want Stage 5C later).
5. Run full test suites:
   - `& $PY -m pytest tests/ -q` (verify ALL tests pass, expected 83+ passed)
   - `& $PY -m pytest Rasnet-v2/tests/ -q` (verify ALL 46 tests pass)

### Final Deliverable
Stop without committing and report:
1. Checklist of every completed item + exact pytest counts (`tests/` and `Rasnet-v2/tests/`).
2. Results of the `CropForegroundd` GT-clipping check, `rasnet_best.pth` strict-load identity check, and ablation cache check.
3. `git status -sb` and `git diff --stat`, then wait for my approval to commit.