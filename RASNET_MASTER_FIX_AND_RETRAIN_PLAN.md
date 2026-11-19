# RASNet Master Fix & 200-Epoch Retrain Plan

**Date:** `2026-11-19`  
**Synthesized From:** `RASNET_CRITICAL_AUDIT.md` and `RASNET_CRITICAL_audit done1T.md` (both independently verified against source code, checkpoints, and result CSVs)  
**Scope Note:** Excludes the Git / patient-CT-zip / tracked-checkpoint purge you are already executing. Respects **Rule 2** (frozen v1 science), **Rule 4** (strict symmetry), **Rule 6** (tests first), and **Rule 9** (never overwrite protected `results/` directories).

---

## Executive Decision Map

Your existing v1 Option A (`R4`) and Epoch-200 numbers are **mathematically solid and reproducible** — no silent bug corrupts them. Because of that, this plan is split into **two clean tracks**:

1. **PART I — Mandatory Fixes (Zero Retraining):** Fixes every defense/thesis overclaim, applies the unapplied `04_CODE_REVIEW_FIXES.md` guards, removes hardcoded paths, fixes non-retraining `rasnet_model.py` hardcoding, quarantines legacy/invalid artifacts, and runs quick read-only `LAB-CHECK`s. **Completing Part I makes your current v1 thesis & repository 100% defensible as-is.**
2. **PART II — Full Core RASNet Bug-Fix & 200-Epoch Retrain Track:** Fixes the baked-in architecture/loss/pipeline defects (`AttentionGate3D.psi` `GroupNorm(1,1)`, inverted Deep Supervision, `StenosisAwareLoss`, `0.5 mm` spacing, `96³` patch, flip-only augmentations) and specifies **exactly which models must be retrained for 200 epochs** so every comparison stays scientifically fair and publication-grade.

---

# PART I — Mandatory Fixes (Zero Retraining, Do First)

*(User already in progress: removing tracked clinical CT zips `CT4`/`CT61`, `checkpoints_200ep.zip.001..010`, and invalid `ablation_*.pth` from Git history, checking GitHub repo visibility, and updating supervisor/ethics contact if applicable.)*

## Stage 1 (P0) — Fix Defense, Thesis & README Wording Overclaims

Fix these exact lines in `results/v1_reporting_correction_20261117_105000Z/` and `README.md`, then propagate them into your external `.tex` / `.docx` / `.pptx` manuscript and defense deck (**P5**):

| # | Exact File & Lines | Current Issue / Overclaim | Exact Required Fix |
| :--- | :--- | :--- | :--- |
| **1.1** | `DEFENSE_QA_V1.md:38`<br>`THESIS_REPLACEMENT_TEXT.md:66-68,86`<br>`DEFENSE_SLIDE_OUTLINE.md:80` | Claims *"consistent precision advantage"* and omits nnU-Net's R4 $\tau^* = 0.20$ grid-edge clamp, Arm C n.s. precision (`p = 0.160`), and R3 CI crossing zero (`[-0.0027, +0.0136]`). | Rewrite as: *"higher precision in Arms A/B and Option A R4 (recipe- and operating-point-dependent; Arm C was non-significant at Holm `p = 0.160`, and R3 CI `[-0.0027, +0.0136]` crossed zero)."* **Explicitly disclose** next to every R4 precision claim that nnU-Net's Val-94 locked $\tau^* = 0.20$ hit the lower grid boundary (nearly flat Val curve `0.8067–0.8098`), shifting nnU-Net toward higher recall / lower precision. |
| **1.2** | `DEFENSE_QA_V1.md:39,73`<br>`README.md:36` | Credits gains over SegResNet/nnU-Net causally to attention gates, deep supervision, and `StenosisAwareLoss` (*"suppresses background false positives"*). | Remove all causal attribution. State: (a) `RASNet vs SegResNet` in v1 is **confounded** by LR (`2e-4` vs `1e-4`), loss function, and historical $\alpha/\gamma$ calibration; (b) `AttentionGate3D` acts as **soft per-patch spatial re-weighting**, NOT background suppression (verified on real `rasnet_best.pth`: `GroupNorm(1,1)` leaves `38.5–59.3%` of pure-background voxels with $\psi > 0.50$). |
| **1.3** | `DEFENSE_SLIDE_OUTLINE.md:81`<br>`THESIS_REPLACEMENT_TEXT.md:68` | States *"RASNet beats SegResNet"* / *"higher Dice (`+0.0050`, Holm `p = 4.27e-8`)"* without showing the mean bootstrap CI crosses zero. | Replace with: *"RASNet outperforms SegResNet on most cases (`112/150`, Wilcoxon Holm `p = 4.27e-8`), though the mean Dice gain (`+0.0050`) is small and its 95% bootstrap CI (`[-0.0024, +0.0121]`) includes 0."* Never write *"significantly higher mean Dice."* |
| **1.4** | `README.md`, `DEFENSE_SLIDE_OUTLINE.md`, `THESIS_REPLACEMENT_TEXT.md` | Casual *"comparable Dice"* phrasing risks violating Rule 10 (*"not significant ≠ equal"*). | Lock headline wording to: *"Comparable Dice to nnU-Net V2 (`-0.0021`, 95% CI `[-0.0088, +0.0030]`; not non-inferior at `-0.005`; median `+0.0002`, `76/150` wins) at `6.54×` fewer parameters and `9.07×` fewer GFLOPs/patch, with a precision–recall trade-off. Epoch-200 (`p = 0.24`) was not significantly different and establishes neither superiority nor equivalence."* |
| **1.5** | Thesis / Defense Ablation Section | Legacy v1 ablation checkpoints (`ablation_attngate_best.pth`, `ablation_deepsup_best.pth`) are scientifically invalid. | **Do not cite the v1 ablation table.** Explicitly note that v1 ablations are excluded due to a `PersistentDataset` cache reuse bug (`train_ablation_models_200ep.py:89,149,174,183`: LAS cached tensors during training vs `Orientationd("RAS")` during validation, plus switched loss to `DiceCELoss` and `num_samples=1`). |
| **1.6** | External Manuscript & Slides (**P5**) | No `.tex` / `.docx` / `.pptx` in repo (`OUTSTANDING`). | Paste corrected tables, `fig3_corrected_*`, `fig_v1_epoch200_*`, and updated claims above into your external thesis and defense slides. |

---

## Stage 2 (P0/P1) — Apply Unapplied Code-Review Fixes & Provenance Patches

*(Do on the home PC using the **tests-first** workflow from Rule 6 before running any future evaluation.)*

1. **Apply All 10 Tested Fixes (`R1–R10`) from `docs-for-next-v2-gen/04_CODE_REVIEW_FIXES.md`:**
   - **Step 2.1 (Tests first):** Create `tests/test_optionA_review_fixes.py` (11 tests) and `Rasnet-v2/tests/test_review_fixes_v2.py` (3 tests) from `04_CODE_REVIEW_FIXES.md` §5.1–§5.2; verify they fail before patching.
   - **Step 2.2 (Apply patches):** Run patch scripts from §5.3–§5.7 updating `benchmark/symmetric_eval.py`, `scripts/run_optionA_symmetric_v1_eval.py`, `tests/test_optionA_symmetric_eval.py`, `Rasnet-v2/src/stage2_pilot_eval.py`, `Rasnet-v2/src/stage5_stats_and_efficiency.py`, and re-syncing `docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md`. This adds:
     - `R1`: Explicit `n_empty_pred` and `precision_strict_mean` in summary & report.
     - `R2`: `run_fingerprint.json` enforcing code + `--sw-batch-size` consistency across val/lock/test.
     - `R3`: `assert_lock_committed()` blocking test inference on uncommitted Val-94 locks.
     - `R4`: Primary `R4` banner clearly separated from descriptive `R1–R3`.
     - `R5`: Crash-safe `fsync` writes + `repair_csv_tail()`.
     - `R6`: Mandatory `.sha256` sidecar check in `load_lock()`.
     - `R7–R8`: Read-only reporting tags (`val94_precision_strict`, `ablation_label_valid`) in paused v2 reporting scripts (does NOT touch paused v2 training or gates).
     - `R10`: PyTorch default TF32 numerics matching locked runs.
2. **Fix 3 Small Remaining Bugs in `benchmark/`:**
   - **`benchmark/symmetric_eval.py:86-93` (`env_info`):** Use `importlib.metadata.version(pkg)` fallback (`connected-components-3d` for `cc3d`, `nnunetv2` for `nnunetv2`) so manifests no longer log `"missing"`.
   - **`benchmark/symmetric_eval.py:365-372` (`done_keys`):** Deduplicate on `("model", "case_id", "recipe")` (or `("model", "case_id", "mode", "tau")` for val grid) before checking `>= rows_per_case`.
   - **`benchmark/thresholds.py:16-17` (`_fast_dice`):** Return `np.nan` instead of `1.0` when `ps + gs == 0`, matching `benchmark/metrics.py:53`.
3. **Add `.gitattributes` for Cross-OS Hash Determinism:**
   - Pin `* text=auto eol=lf` for `.json`, `.csv`, `.py`, `.md`, `.yaml` so `locked_recipes.json` and manifest SHA-256 digests match identically between Windows and Linux checkouts.

---

## Stage 3 (P0/P2) — Non-Retraining Hardcoded Fixes, Legacy Quarantine & Doc Sync

1. **Fix Non-Retraining Hardcoded Model Widths (`Phase3_Local_Integration/rasnet_model.py:66-77`):**
   - Derive channel widths from `f = kwargs["init_filters"]` (`[4*f, 2*f, f]` for `attention_gates`, `[4*f, 2*f]` for `ds_heads`, with `F_int = max(c // 2, 1)` and safe `GroupNorm` group guard `min(8, F_int)` when `F_int < 8`).
   - **Zero retraining needed:** For `init_filters=16`, this creates the **exact same tensors, shapes, and `114` state-dict keys**, so frozen `rasnet_best.pth` (`SHA-256 8d4c25cd...`) still loads with `strict=True` and produces bit-identical logits. Add a unit test in `tests/` verifying both `RASNet(init_filters=16)` strict-load compatibility and `RASNet(init_filters=32)` forward pass.
   - Fix the inverted comment at `rasnet_model.py:73` (`Level 0 (1/4 size): 64 ch, Level 1 (1/2 size): 32 ch`).
2. **Remove Hardcoded Machine Paths (Rule 1):**
   - `scripts/run_phaseH_rasnet_optimization.py:43` and `scripts/backfill_phaseH_val94_and_report.py:31`: replace `ROOT = Path(r"C:\Thesis_RASNET")` with `Path(__file__).resolve().parents[1]`.
   - `recovered_checkpoints/v1_epoch200_20261117_041500Z/recover_and_verify.py:1156,1166`: replace hardcoded `C:\Thesis_RASNET` with relative repo-root resolution.
   - `Phase3_Local_Integration/dataset_paths.py:24-30`: read dataset root from `config.lab.yaml` or `IMAGECAS_DATA_ROOT` env var (keeps frozen v1 trainer imports working without hardcoded paths).
3. **Quarantine Stale Legacy Evaluators & Aborted Logs** (move into `_Archive_Historical_Experiments_and_Logs/`):
   - `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py` and `evaluation_results/` (hardcoded $\tau = 0.60$, broken `_surface_distances`).
   - `Phase3_Local_Integration/eval_utils.py` (`HD95 = 0.0` bug at `:131-132`).
   - The 7 intermediate/aborted recovery CSVs in `Q1_Publication_Package/matched_200ep_benchmark/logs/` (keep only the 5 canonical `*_200ep_log.csv` logs live).
4. **Fix Remaining Documentation Drift:**
   - `CHECKPOINTS.md:13`: fix `nnunet_best.pt` archive timestamp to `2026-09-08 09:19:08`.
   - `CHECKPOINTS.md:40`: clarify `nnUNetTrainer_200epochs` is a stock `nnunetv2` trainer variant, and add a short limitation note on v1 training non-determinism (no cuDNN determinism, no training manifests, weights-only resume).
   - `README.md:31-33,203`: fix symbol names (`self.up_samples`, `self.attention_gates`, `self.ds_heads`) and remove nonexistent `config.yaml`.
   - `Q1_Publication_Package/matched_200ep_benchmark/run_all_200ep.py:10`: fix stale contradictory docstring (`Epoch 71 -> 200`).
   - Sync `GEMINI.md` §3 and `CLAUDE.md` §3 word-for-word (reconcile Rule 11, retrospective bullet, and manifest bullet), and keep §2 accurate regarding where checkpoints live after your Git cleanup.

---

## Stage 4 (`LAB-CHECK`) — Fast Read-Only Checks on `C:\Thesis_RASNET`

After committing and pulling Stages 1–3 onto the lab PC (`git.dirty_code = false`):

1. **Export `requirements.lab.txt`** from `C:\Thesis_RASNET_env\bench` (`torch 2.14.1+cu126`, `monai 1.6.1`, `nnunetv2 2.8.1`, `SimpleITK 2.5.7`, `cc3d 4.1.1`) and commit it.
2. **Verify `CropForegroundd(x > 0)` Safety:** Run a read-only script over all 934 ImageCAS scans checking whether raw `HU > 0` cropping (`train_rasnet_200ep.py:110`, `monai_models.py:37`) ever clips a single ground-truth coronary voxel. Log the result (`0` clipped or exact count).
3. **Confirm Ablation Cache Hit:** Check `cache/segresnet` on `C:\Thesis_RASNET` to confirm cached training tensors existed when `train_ablation_models_200ep.py` ran (closes the invalid-ablation footnote).
4. **Run Full `pytest` Suite:** Execute `C:\Thesis_RASNET_env\bench\Scripts\python.exe -m pytest tests/` and confirm all **`83` tests** (`72` existing + `11` from Stage 2) pass, plus `Rasnet-v2/tests/` (`46` tests).

---

## Stage 5 (Optional, Zero Retraining) — High-ROI Protocol-Safe Eval Add-Ons

Only run these **if you want extra strength before the defense without retraining**. Each requires a short dated amendment in `BENCHMARK_PREREGISTRATION.md` and a fresh `results/` folder:

1. **5A — Precision–Recall & Dice-vs-$\tau$ Operating Curves (Highest Defense ROI):**
   - Inference-only re-run saving 8-pass probabilities on Val-94 & Test-150 for RASNet and nnU-Net.
   - Compare **Precision at matched Recall** across $\tau \in [0.10, 0.80]$. This cleanly eliminates the $\tau^* = 0.20$ edge-clamp criticism.
2. **5B — Symmetric `best + last` Checkpoint Ensembling:**
   - Average 8-pass probabilities of `best` + `last` checkpoints **symmetrically for all 4 models**, re-lock $\tau^*$ on Val-94, and run once on Test-150.
3. **5C — Symmetric Physical/Relative CC Volume Floor (Post-Processing Upgrade):**
   - First inspect Cases `930`, `948`, `861`, `978` component volume lists on `C:\Thesis_RASNET` (hard `keep_top_k=2` dropped `8` cases `> 0.05` on *both* RASNet and nnU-Net, and Case `930` alone dragged RASNet's R4 mean by `-0.00237`).
   - Test a symmetric rule on Val-94 across all 4 models: keep all 26-connected components $\ge V_{\min} \in \{200, 350, 500\}\text{ mm}^3$ or $\ge 10\%$ of largest CC, capped at top-3. *(Do NOT retest naive `top3cc` or 50-vox hysteresis — Phase H already proved they lose on Val-94.)*

---

# PART II — Full RASNet Bug-Fix & 200-Epoch Retrain Plan

*(Execute Part II **only** if you want a newly trained, bug-free RASNet that fixes the architecture/loss flaws and closes the ~80–90% pipeline engineering gap to nnU-Net. Never overwrite v1 checkpoints or protected `results/` directories.)*

## 1. Exact Core RASNet Code & Training Fixes to Implement

Create a clean versioned model/trainer (e.g., `rasnet_v1_1` or via `Rasnet-v2/` — see §2 below) incorporating **all 6 fixes**:

1. **Fix `AttentionGate3D` (`rasnet_model.py:20-32`):**
   - Replace `nn.GroupNorm(1, 1)` in `self.psi` with `nn.Conv3d(F_int, 1, kernel_size=1, bias=True)` + `nn.Sigmoid()` so gates can genuinely close on empty/background patches. *(Validate `conv_bias` vs `groupnorm_orig` on Val-94 if scaling to `F=32`, as 50-ep engineering pilots showed width-sensitivity.)*
   - Remove redundant dual bias in `W_g` + `W_x` (only one bias or post-sum `GroupNorm` is needed before `ReLU`).
   - Guard `GroupNorm` group count when `F_int <= 8` (use `num_groups = min(4, F_int)` so Gate 2 doesn't degenerate to 1-channel `InstanceNorm3d`).
   - Derive all gate and DS channel widths dynamically from `init_filters`.
2. **Fix Inverted Deep Supervision (`train_rasnet_200ep.py:211-215`):**
   - Weight finer `1/2`-res (`ds_outputs[1]`) **higher** than coarser `1/4`-res (`ds_outputs[0]`):
     $$\mathcal{L}_{\text{total}} = \frac{1.0\,\mathcal{L}_{\text{main}} + 0.5\,\mathcal{L}_{\text{half (1/2)}} + 0.25\,\mathcal{L}_{\text{quarter (1/4)}}}{1.75}$$
   - Downsample ground-truth labels (nearest / max-pool) to `48³` and `24³` (or `48×80×80` and `24×40×40`) instead of trilinearly upsampling coarse `24³` logits to full patch size (sharper distal vessel gradients + lower VRAM).
3. **Fix Loss Function (`rasnet_loss.py:59-84`):**
   - Replace `StenosisAwareLoss` (`include_background=True`, `use_softmax=False`, `smooth_nr=0.0`) with **`DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, batch=True)`** (matching nnU-Net's `BatchDice + CE`, eliminating ~2× FG Dice dilution, activation mismatch, and `3×` weaker empty-crop FP gradients).
   - *(Optional Val-94 ablation only)*: compare against foreground-only softmax `DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, use_softmax=True, batch=True)` if you want to keep a Focal variant for precision tuning.
4. **Adopt nnU-Net's Preprocessing & Augmentation Pipeline (Closes ~80–90% of the Gap):**
   - **Target spacing:** native ImageCAS median `[0.5, 0.349609375, 0.349609375] mm` (recovers the `51%` lost in-plane voxels on thin distal vessels).
   - **Patch size:** `96 × 160 × 160` (`batch_size = 2`, fits comfortably on RTX 4080 SUPER 16 GB).
   - **Normalization:** fingerprint `CTNormalization` — clip `[-166, 727]` HU, then z-score `(x - 147.97) / 179.68`.
   - **Cropping:** remove raw-HU `CropForegroundd(x > 0)`.
   - **3D Augmentations:** 3-axis random flips + random 3D rotation/scaling (`RandAffined`), Gaussian noise (`RandGaussianNoised`), Gaussian blur (`RandGaussianSmoothd`), brightness/contrast/gamma (`RandScaleIntensityd`, `RandAdjustContrastd`).
5. **Fix Training Loop & Reproducibility Hygiene (`T1`, `T3`):**
   - Add gradient clipping (`clip_grad_norm_(model.parameters(), 1.0)`).
   - Save full resume checkpoints: `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `scheduler_state_dict`, `epoch`, `best_val_dice`, and Python/NumPy/Torch RNG states.
   - Use a **unique `PersistentDataset` cache directory keyed by the SHA-256 of the deterministic transform list** (makes the LAS/RAS cache-reuse bug impossible).
   - Standardize **identical validation cadence** (e.g., every 10 epochs + epoch 200 on full Val-94) and emit a `train_manifest.json` per run (git commit, dirty flag, config hash, seed, environment versions, epoch timings).

---

## 2. Exactly WHICH Models Must Be Retrained (200 Epochs)?

Pick **one** of the two retrain strategies below — do **not** mix upgraded RASNet with old `0.5 mm / 96³` baselines (violates Rule 4):

### Strategy A — Minimal Fair Retrain (1 Seed, 200 Epochs, Upgraded Pipeline)
*Best if you have limited GPU time (~1–2 days on RTX 4080 SUPER) and want a clean, single-seed updated benchmark.*

| Model | Retrain for 200 Epochs? | Why / Exact Configuration |
| :--- | :---: | :--- |
| **`RASNet` (Fixed)** | **YES (Mandatory)** | `init_filters=16` (`4.71M`), fixed `conv_bias` gate, fixed DS weights, `DiceCE(batch=True)`, upgraded `0.35 mm` / `96×160×160` / CT z-score / 3D augs, seed `42`. |
| **`SegResNet` (Matched)** | **YES (Mandatory)** | `init_filters=16` (`4.70M`), **identical** spacing, patch, CT z-score, 3D augs, `DiceCE(batch=True)`, LR, optimizer, gradient clipping, val cadence, seed `42`. *Required to prove RASNet's architecture helps over its backbone without confounders.* |
| **`V-Net` (Matched)** | **YES (if kept in table)** | **Identical** upgraded pipeline, `num_samples` matched to RASNet/SegResNet, seed `42`. *(Drop V-Net from the new table if you don't retrain it — never compare upgraded RASNet against old `0.5 mm` V-Net.)* |
| **`Ablation: No-Attn` & `No-DS`** | **YES (if citing ablations)** | 200 epochs each on the **exact same** upgraded pipeline + fresh transform-hashed cache. Replaces the invalid v1 `ablation_*.pth` checkpoints. |
| **`nnU-Net V2` (fold 0)** | **NO** | Existing 200-epoch `nnunet_best.pt` (`SHA-256 9accddf2...`) **already** trained for 200 epochs on `[0.5, 0.3496, 0.3496] mm`, `96×160×160`, `CTNormalization`, full 3D augs, and `BatchDice + CE`. Reuse it directly as the framework-vs-framework reference. |
| **`3D U-Net` (`128³`)** | **NO** | Already excluded from sliding-window comparisons. |

### Strategy B — Recommended Publication-Grade Retrain: Unpause `Rasnet-v2` via Written Amendment (3 Seeds × 200 Epochs)
*Best if you are targeting a Q1 journal paper or want indisputable multi-seed proof.*

- **Why Strategy B beats hacking v1:** `Rasnet-v2/` **already implements** every fix in §1 above (`conv_bias` gate, fixed DS, `0.3496 mm` XY spacing, `160×160×96` patch, CT normalization, full 3D augmentations, matched SegResNet/V-Net/UNet3D, 3 seeds `{42, 123, 456}`, 3-seed ablations, and `nnU-Net V2` × 3 seeds).
- **What is blocking `Rasnet-v2` right now:** Stage 2B (`Rasnet-v2/results/pilots_val94/pilot_lock_decision.md`) hit `FALLBACK` because the pre-registered 50-epoch unpruned gate (`Val Precision >= 0.8000` at $\tau = 0.50$, no CC, after only 50 epochs) was too strict — **no** candidate reached `0.8000` at epoch 50 (lead candidate `F=16, conv_bias, w_cldice=0.0` got `0.7340` Dice / `0.6966` Precision, beating `w_cldice=0.20` by `+0.0245` Dice).
- **Exact step to unpause cleanly (respecting Rule 2):**
  1. Write **Amendment A2** in `Rasnet-v2/PREREGISTRATION_V2.md` explaining that the 50-epoch unpruned `Precision >= 0.8000` gate is unachievable at 25% of training without post-processing, locking `F=16, gate_mode=conv_bias, w_cldice=0.0` (the highest-Dice Stage 2B design) — and optionally evaluating the fragmentation gate with Top-2 CC or at 200 epochs — then get explicit sign-off and commit before running Stage 3.
  2. Run `Rasnet-v2/src/run_queue.py --stage stage3` → `stage4a` → `stage4b` → commit Val-94 lock → `stage4c` → `stage5` (after applying Stage 2 `04_CODE_REVIEW_FIXES.md` first!).

---

## 3. Mandatory Protocol Sequence for ANY Retrain (Strategy A or B)

1. **Before Epoch 1:** Write and commit the pre-registration amendment (`BENCHMARK_PREREGISTRATION.md` for Strategy A; `Rasnet-v2/PREREGISTRATION_V2.md` for Strategy B). Verify `git.dirty_code = false` and check GPU availability (`Rasnet-v2/logs/queue_status.json`).
2. **Training (200 epochs, 1 GPU job at a time per Rule 8):** Save all new checkpoints to a **new** directory (never overwrite `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/`), record every new checkpoint's SHA-256 in `CHECKPOINTS.md`.
3. **Val-94 Grid & Lock:** Run symmetric `0.50` Gaussian SW + 8-pass logit-averaged TTA over $\tau \in [0.10, 0.70]$ (widened down to `0.10` so no model hits a `0.20` floor!) + locked symmetric CC post-processing on **Val-94 only**. Write and **commit** `locked_recipes.json` + `.sha256` sidecar.
4. **One-Shot Test-150 Evaluation:** Run once per frozen model into a fresh `results/` directory (still explicitly labeled **retrospective** on ImageCAS `851–1000`, since this split was inspected during v1).
5. **Statistical & Efficiency Reporting:** Run paired Wilcoxon + Holm-Bonferroni + 10,000-resample bootstrap 95% CIs + Dice non-inferiority (`margin = -0.005`), measure single-patch GFLOPs/VRAM and full-volume runtime with matched/disclosed `sw_batch_size`, and verify `0` empty predictions/NaNs.

---

# Master Execution Checklist

### Track I — Lock Current V1 for Defense (No Retraining)
- [ ] **Git / Data Governance (User In Progress):** Purge tracked `CT4`/`CT61` patient CT zips, `checkpoints_200ep.zip.*`, and invalid `ablation_*.pth`; verify GitHub visibility.
- [ ] **Stage 1 (Defense & Thesis Text):** Fix `DEFENSE_QA_V1.md:38,39,73`, `DEFENSE_SLIDE_OUTLINE.md:80,81,101`, `THESIS_REPLACEMENT_TEXT.md:27,66-68,86`, `README.md:31-36,203`; mark v1 ablations invalid; integrate into external `.tex`/`.docx`/`.pptx` (**P5**).
- [ ] **Stage 2 (Code Review Fixes — Tests First):** Apply `04_CODE_REVIEW_FIXES.md` (`R1–R10` + 14 tests), fix `env_info()` (`importlib.metadata`), `done_keys()` deduplication, `_fast_dice()` `both_empty → np.nan`, and add `.gitattributes`.
- [ ] **Stage 3 (Hygiene & Non-Retrain Hardcodes):** Make `rasnet_model.py:66-77` dynamic on `init_filters` (strict-load verified on `F=16`) + fix comment `:73`; fix hardcoded paths; quarantine `evaluate_matched_200ep_models.py`, `evaluation_results/`, `eval_utils.py`, and 7 aborted logs; fix `CHECKPOINTS.md:13,40`, `run_all_200ep.py:10`, and sync `GEMINI.md`/`CLAUDE.md` §3.
- [ ] **Stage 4 (`LAB-CHECK`s on `C:\Thesis_RASNET`):** Export `requirements.lab.txt`; verify `CropForegroundd(x > 0)` clips `0` GT voxels; confirm ablation cache hit; pass all `83` tests with `dirty_code = false`.
- [ ] **Stage 5 (Optional Zero-Retrain Add-Ons):** Pre-register & run 5A (Precision–Recall operating curves), 5B (symmetric `best+last` ensemble), or 5C (symmetric CC volume floor after Case 930/948/861 inspection).

### Track II — Retrain Core RASNet + Matched Baselines (200 Epochs)
- [ ] **Choose Strategy:** Strategy A (1-seed fair upgraded retrain: `RASNet` + `SegResNet` [+ `V-Net` + 2 ablations]; reuse `nnU-Net V2`) **OR** Strategy B (Amendment A2 to unpause `Rasnet-v2` 3-seed study).
- [ ] **Pre-Register & Commit Amendment** before running Epoch 1 (`dirty_code = false`).
- [ ] **Implement & Unit-Test** `conv_bias` gate, fixed DS weights/downsampled GT, `DiceCELoss(batch=True)`, native `0.35 mm` / `96×160×160` / CT z-score / 3D augs, full resume checkpoints, and transform-hashed `PersistentDataset` caches.
- [ ] **Train 200 Epochs** into new checkpoint directory; record SHA-256s in `CHECKPOINTS.md`.
- [ ] **Symmetric Val-94 Lock → Commit Lock → One-Shot Test-150 → Paired Stats & Figures** into a new `results/` directory.
