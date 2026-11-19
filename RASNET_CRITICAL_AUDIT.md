# RASNet Critical Audit & nnU-Net Gap Analysis

**Audit Date:** `2026-11-19`  
**Repository Audited:** `H:\Thesis_Trainings` (mirrors lab workstation `C:\Thesis_RASNET`, branch `main` clean)  
**Scope:** Full read-only forensic code, checkpoint, config, pipeline, and result audit of core **RASNet (v1)**, active `benchmark/` evaluation infrastructure, baseline parity, and nnU-Net V2 engineering gap. (`Rasnet-v2/` is intentionally paused and audited solely for isolation and cross-contamination.)  
**Environment Constraint:** ImageCAS NIfTI volumes and exported Option A prediction masks reside only on `C:\Thesis_RASNET` and are absent on `H:\Thesis_Trainings`. Items requiring dataset access are explicitly flagged **LAB-CHECK**.

---

## 1. Executive Verdict

### Overall Health Rating: **Needs Minor-to-Moderate Fixes**
- **Active Evaluation & Statistical Engine (`benchmark/` + `results/optionA_v1_symmetric_sw050/` + `results/v1_reporting_correction_20261117_105000Z/`):** **Publication-Grade** (fail-closed geometry checks, disjoint splits, exact `surface-v1` metrics, strict paired Wilcoxon + Holm-Bonferroni + bootstrap 95% CIs, byte-verified checkpoint SHA-256 locks).
- **Core v1 `RASNet` Architecture, Loss & Training Pipeline:** **Contains Real Math/Design Defects & Training Asymmetries** (documented below; frozen weights remain evaluable as-is, but any next training cycle must fix them).
- **Repository Hygiene & Data Governance:** **P0 Violations Found** (`1.057 GiB` of git-tracked binaries including **two full 3D clinical CT volumes** in tracked `.zip` files, tracked checkpoints, and a live-tracked broken legacy evaluator alongside invalid ablation checkpoints).

### Direct Answer: *"Is everything actually great right now, and can I trust my current RASNet results?"*

1. **Yes — you can trust the active `benchmark/` numbers**, strictly within their documented caveats:
   - `Phase3_Local_Integration/splits_final.json` (`SHA-256 efe3f06cfe9c...`) is strictly disjoint (`690 train / 94 val / 150 test`, `0` overlaps; `66` IDs in `3..199` excluded). Test cases `851–1000` are honestly disclosed as **retrospective** (`BENCHMARK_PREREGISTRATION.md`, `CURRENT_EVIDENCE_INDEX.md`).
   - Across all `4,838` active evaluated case-rows (`val_full_94` Arm A: `470`; `test_retrospective_150` Arms A–C: `2,250`; Option A `test_per_case.csv`: `2,400` with `0` duplicates; Val-94 `final_epoch200`: `188`), there are **`0` empty predictions, `0` NaN exclusions, and 100% physical header verification**.
   - All 7 canonical checkpoints (`rasnet_best.pth`, `segresnet_best.pt`, `vnet_best.pt`, `3dunet_best.pth`, `nnunet_best.pt`, `rasnet_last.pth`, `nnunet_last.pt`) match their SHA-256 digests in `CHECKPOINTS.md` byte-for-byte.
   - Under symmetric inference (`results/optionA_v1_symmetric_sw050/`, primary recipe `R4_tuned_tta8_top2`), frozen v1 `RASNet` (`4.707M` params, `123.39` GFLOPs/patch) achieves **`0.8039 ± 0.0691` Dice**, decisively beating `SegResNet` (`0.7989`, Holm `p = 4.27e-8`) and `V-Net` (`0.7915`, Holm `p = 3.58e-12`), **beating `nnU-Net V2` on `76/150` cases** (median paired Dice diff **`+0.00021`**), and winning `FA_primary` Precision (`0.8302` vs `0.8084`, Holm `p = 7.23e-17`) — while trailing `nnU-Net V2` (`0.8059 ± 0.0630`) in mean Dice by `-0.0021` (`95% CI [-0.0088, +0.0030]`, **non-inferiority `False` at `-0.005` margin**; nnU-Net retains Recall `+0.0244`, Holm `p = 2.40e-17`).
2. **No — the repository and core v1 `RASNet` implementation are NOT bug-free or cleanup-ready:**
   - **Real defects inside `RASNet` (`rasnet_model.py`, `rasnet_loss.py`, `train_rasnet_200ep.py`):** Direct forward-hooking of the actual trained `rasnet_best.pth` weights proves `AttentionGate3D.psi`'s `nn.GroupNorm(1, 1)` breaks sparse background gating (`38.5–59.3%` of pure-background voxels still pass `psi > 0.50`); `W_g.1.bias` and `W_x.1.bias` learned bit-for-bit identical redundant values; `init_filters != 16` crashes on hardcoded `[64, 32, 16]` widths; Deep Supervision weights are **inverted** (`0.4` on 1/4-res `24³` vs `0.2` on 1/2-res `48³`); and `StenosisAwareLoss` dilutes foreground Dice by ~2× (`include_background=True`), mixes `softmax=True` Dice with `use_softmax=False` sigmoid Focal, and yields `~3×` weaker empty-crop false-positive gradients than unweighted CE.
   - **Severe legacy ablation bug:** The two tracked ablation checkpoints (`ablation_{attngate,deepsup}_best.pth`) were trained by `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/Q1_Publication_Package/ablation/train_ablation_models_200ep.py`, which reused `cache/segresnet` (LAS tensors) under `PersistentDataset` while validating on `Orientationd("RAS")` volumes — a **silent LAS-train / RAS-val orientation mismatch** that invalidates both checkpoints.
   - **P0 Git & Data Governance Violation:** Two git-tracked `.zip` files (`Phase3_Local_Integration/local_data/CT4/CT4_Stenosis_Annotation_Package.zip` and `CT61/CT61_Stenosis_Annotation_Package.zip`) contain **full 3D clinical CT volumes (`image.nii.gz`, `79.9 MB` & `74.0 MB`)** alongside `895 MiB` of tracked checkpoints (`checkpoints_200ep.zip.001..010` and `ablation_*.pth`), violating Rule 7. In addition, the broken legacy evaluator (`evaluate_matched_200ep_models.py` + `evaluation_results/`) is still tracked live in `Q1_Publication_Package/matched_200ep_benchmark/`.

---

## 2. Repository Map & Structural Health

### 2.1 Active vs. Legacy vs. Paused (`Rasnet-v2`) Breakdown

| Component | Exact Paths | Status | Audit Health & Notes |
| :--- | :--- | :--- | :--- |
| **Core RASNet Model, Loss & Splits** | `Phase3_Local_Integration/rasnet_model.py`, `rasnet_loss.py`, `splits_final.json` | **Active (Frozen v1)** | Contains verified defects (`M1–M3`, `L1` in §3). `splits_final.json` (`SHA-256 efe3f06c...`) verified strictly disjoint (`690 / 94 / 150`). |
| **Fail-Closed Evaluation Engine** | `benchmark/` (`run_benchmark.py`, `symmetric_eval.py`, `endpoint_eval.py`, `metrics.py`, `postprocess.py`, `thresholds.py`, `stats.py`, `dataset.py`, `preflight.py`, `comparison.py`, `adapters/{monai_models,nnunet_adapter}.py`) | **Active** | **Strong.** Consistent native-grid inversion, exact 6-connected `surface-v1` distances, immutable SHA-256 lock verification, case-ID-aligned paired statistics. Two negligible edge-case discrepancies noted (§4.1). |
| **Active Orchestration & Reporting Scripts** | `scripts/run_optionA_symmetric_v1_eval.py`, `run_v1_epoch200_endpoint_eval.py`, `generate_v1_epoch200_report_and_figures.py`, `generate_v1_reporting_corrections.py`, `run_phaseH_rasnet_optimization.py`, `backfill_phaseH_val94_and_report.py` | **Active** | Functional; however, `scripts/run_phaseH_rasnet_optimization.py:43` and `scripts/backfill_phaseH_val94_and_report.py:31` hardcode `ROOT = Path(r"C:\Thesis_RASNET")`. |
| **Automated Test Suite** | `tests/` (`13` test modules, `66` `test_*` functions → `72` collected pytest cases on lab PC `2026-11-17`) | **Active** | Verified `72/72` passing on `C:\Thesis_RASNET` (`2026-11-17`). Local `H:\Thesis_Trainings` environment lacks `pytest` (`LAB-CHECK` for fresh execution). |
| **Frozen V1 Trainers & Logs** | `Q1_Publication_Package/matched_200ep_benchmark/train_{rasnet,segresnet,vnet,3dunet}_200ep.py`, `prepare_and_run_nnunet_200ep.py`, `resume_rasnet_200ep.py`, `run_all_200ep.py`, `logs/*.csv` | **Frozen Provenance** | Must stay unchanged per Rule 2 (`T1–T4` open). `run_all_200ep.py:10` has stale docstring (`Epoch 71 -> 200`) and `:44` hardcodes `C:\`. `logs/` mixes canonical 200-ep CSVs with 7 aborted/intermediate recovery CSVs (e.g., `segresnet_run1_133ep_recovered.csv` showing `0.8030` Val Dice). |
| **Stale Legacy Code & Clinical Data Still Live** | `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py`, `evaluation_results/*`, `checkpoints/ablation_{attngate,deepsup}_best.pth`, `Phase3_Local_Integration/eval_utils.py`, `dataset_paths.py`, `local_data/CT{4,61}/*` | **HAZARD (P0)** | **All tracked live in Git** despite `AUDIT_FINDINGS.md:91` stating `evaluate_matched_200ep_models.py` was archived. Contains broken `_surface_distances`, `HD95 = 0.0` on empty masks (`eval_utils.py:131-132`), hardcoded paths (`dataset_paths.py:25-30`, `local_data/CT{4,61}/load_case_in_slicer.py:9`), invalid ablation checkpoints, and **tracked clinical CT volumes**. |
| **Paused Study (`Rasnet-v2`)** | `Rasnet-v2/` (`PREREGISTRATION_V2.md`, `configs/fair_v2.yaml`, `src/`, `results/pilots_val94/pilot_lock_decision.md`) | **Paused (Isolated)** | **Cleanly isolated.** Zero active v1 modules import from `Rasnet-v2/`. Only referenced in protected-directory safeguards (`benchmark/endpoint_eval.py:96`, `scripts/generate_v1_reporting_corrections.py:46`) and comments (`benchmark/symmetric_eval.py:63`). Stage 2B ended in `FALLBACK (NO CANDIDATE PASSED BOTH GATES)`. |
| **Historical Archive** | `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/` (`306` indexed files) | **Archived** | Properly quarantined with `ARCHIVE_INDEX.csv`. |

### 2.2 Reproducibility, Git Hygiene & Config-to-Results Alignment

1. **Rule 7 Violations (`1.057 GiB` / `1.135 GB` across `29` git-tracked files `> 1 MB`):**
   - **Clinical CT Data Committed to Git (P0 Privacy/Governance):**
     - `Phase3_Local_Integration/local_data/CT4/CT4_Stenosis_Annotation_Package.zip` (`76.88 MiB`, tracked) — contains full 3D CT `image.nii.gz` (`79.9 MB`), `pred_mask.nii.gz`, `stenosis_mask.nii.gz`, and `stenosis_annotation_preview.png`.
     - `Phase3_Local_Integration/local_data/CT61/CT61_Stenosis_Annotation_Package.zip` (`71.22 MiB`, tracked) — contains full 3D CT `image.nii.gz` (`74.0 MB`), `pred_mask.nii.gz`, `stenosis_mask.nii.gz`, and preview PNG.
   - **Checkpoints Committed to Git:**
     - `Q1_Publication_Package/matched_200ep_benchmark/checkpoints_best_and_last/checkpoints_200ep.zip.001`..`.010` (`859.43 MiB` total).
     - `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/ablation_attngate_best.pth` (`18.00 MiB`) and `ablation_deepsup_best.pth` (`18.00 MiB`) — both scientifically invalid (§3.3).
2. **Manifest & Config Provenance Audit:**
   - **Option A (`results/optionA_v1_symmetric_sw050/manifest_{parity,val,lock,test,timing,stats,figures}_*.json`):** **Clean** (`git.dirty_code: False`; Val-94 lock committed at `558e7eb` prior to Test-150 execution at `22d4581`).
   - **V1 3-Arm Runs (`results/benchmark_runs/{val_full_94,test_retrospective_150}/manifest.json`):** Record `git: {'branch': 'main', 'commit': '8eb28bb...', 'dirty': True}` and **do NOT contain a `source_tree_sha256` key** (`source_tree_sha256` was introduced later in `smoke_verified_v2`, commit `d81485d5`). Provenance is externally anchored via `results/audits/2026-11-10_verification_audit/source_inventory*.csv`.
   - **Epoch-200 Endpoint (`results/v1_epoch200_comparison_20261117_041500Z/manifest.json`):** Records `git.dirty_code: True` on `b3ca1f1` (executed before commit `8e9b303`; mitigated by embedded `source_tree_sha256` and 72 passing tests) and `timing_seconds.rasnet: 0.0` (an `--resume` skip artifact from `benchmark/endpoint_eval.py:1020-1066`, properly excluded from latency claims).
   - **Phase H (`results/rasnet_fixed_weight_optimization/manifest_phaseH.json`):** Omits the `git` metadata key entirely (`None`).
3. **Environment Pinning & Documentation Drift:**
   - `requirements_cuda.txt` pins the home workstation (`torch 2.6.0+cu124`, `monai 1.5.2`), whereas all verified lab runs executed on `Python 3.12.10`, `torch 2.14.1+cu126`, `monai 1.6.1`, `nnunetv2 2.8.1`, `SimpleITK 2.5.7`, `cc3d 4.1.1` (`CHECKPOINTS.md:59-60`). No `requirements.lab.txt` exists.
   - `CHECKPOINTS.md:13` lists `nnunet_best.pt` Archive Timestamp as `2026-09-09 20:51:38`; the actual entry timestamp inside `checkpoints_200ep.zip` is `2026-09-08 09:19:08` (`nnunet_last.pt`: `2026-09-08 09:20:06`; SHA-256 `9accddf2...` matches byte-for-byte).
   - `README.md:203` lists a nonexistent `config.yaml` alongside `config.lab.yaml`; `README.md:31-33` cites nonexistent symbols `up_conv[i](x)`, `attn_gate(g=x, x=down_x[i])`, `ds_head2`, `ds_head3` instead of actual `self.up_samples`, `self.attention_gates[i](g=g, x=down_x[i + 1])`, `self.ds_heads[0..1]`.
   - `GEMINI.md` §3 contains 11 rules while `CLAUDE.md` §3 stops at 10 (missing Rule 11 `Status polling`, breaking the stated §3 sync guarantee).

---

## 3. Core RASNet Code & Pipeline Audit

### 3.1 Architecture Review (`Phase3_Local_Integration/rasnet_model.py`)

- **Parameter & Efficiency Profile (Verified):** `RASNet` (`4,706,772` params, `114` state-dict keys) adds only **`+5,858` parameters (`+0.125%`)** over MONAI `SegResNet` (`4,700,914` params, `83` keys): `5,662` in `attention_gates[0..2]` and `196` in `ds_heads[0..1]`. Single-patch compute is `123.39` GFLOPs (`390.7 MB` peak patch VRAM, `13.71 ms` AMP on RTX 4080 SUPER) — `6.54×` fewer parameters and `9.07×` fewer GFLOPs/native patch than `nnU-Net V2` (`30.786M` params, `1,119.34` GFLOPs at `96×160×160`; `3.27×` fewer at equal `96³`).
- **Train/Eval & Activation Integrity:** Clean. `decode()` (`rasnet_model.py:101-103`) returns `(x, ds_outputs)` only when `self.training` is `True`; `benchmark/adapters/monai_models.py:30,68-69,95-96` and `benchmark/symmetric_eval.py:140,148-149` enforce `net.eval()` and defensively unpack tuples. `conv_final` outputs raw logits (`softmax` is never double-applied).

#### Verified Architectural Bugs & Bottlenecks
1. **Defect M1 — `AttentionGate3D.psi` `nn.GroupNorm(1, 1)` Prevents Sparse Background Suppression (`rasnet_model.py:28-32`):**
   - `self.psi` applies `Conv3d(F_int, 1, 1, bias=False)` → `nn.GroupNorm(1, 1)` → `nn.Sigmoid()`.
   - Because `GroupNorm(1, 1)` computes the mean and standard deviation across **all spatial voxels `(D, H, W)`** of the single attention channel per patch, it forces the pre-sigmoid logit distribution to zero mean and unit variance (`~50%` of voxels `> 0`) even on empty background patches where `>99.7%` of voxels are non-coronary tissue — unless the learned affine bias collapses strongly negative.
   - **Direct proof on real trained `rasnet_best.pth` (`SHA-256 8d4c25cd...`):** Inspecting the actual checkpoint weights shows `psi.1` affine parameters remained near identity/zero:
     - Gate 0 (`64 ch`, `1/4` res): `weight = 1.0611`, `bias = +0.0387`
     - Gate 1 (`32 ch`, `1/2` res): `weight = 1.1533`, `bias = -0.1024`
     - Gate 2 (`16 ch`, `1×` res): `weight = 1.6098`, `bias = -0.2191`
   - Forward-hooking **real `rasnet_best.pth`** on a pure-background `96³` patch (where final `fg_prob max = 0.0069`) yields `psi > 0.50` on **`59.28%` (gate 0, `mean = 0.5182`)**, **`40.91%` (gate 1, `mean = 0.4686`)**, and **`38.54%` (gate 2, `mean = 0.4449`)** of all voxels. Rather than spatially suppressing background skips, the gates act primarily as mild global contrast scalers.
   - **Secondary gate redundancies:**
     - At Gate 2 (`rasnet_model.py:69`, `AttentionGate3D(16, 16, F_int=8)`), `nn.GroupNorm(8, 8)` (`:22, :26`) degenerates into single-channel `InstanceNorm3d`.
     - In `rasnet_best.pth`, `W_g.1.bias` and `W_x.1.bias` converged to **bit-for-bit identical vectors** across all 3 gates because `g1 + x1` (`:39`) sums them linearly prior to `ReLU`, creating a redundant bias degree of freedom.
2. **Defect M2 — Hardcoded Skip/DS Channel Widths Crash `init_filters != 16` (`rasnet_model.py:66-77`):**
   - `self.attention_gates` hardcodes `[64, 32, 16]` and `self.ds_heads` hardcodes `[64, 32]` regardless of `kwargs["init_filters"]`.
   - Verified directly: `RASNet(init_filters=32)(torch.randn(1, 1, 64, 64, 64))` throws `RuntimeError: expected input[1, 128, 16, 16, 16] to have 64 channels, but got 128 channels instead`.
3. **Defect M3 — Inverted Deep Supervision Comment & Loss Weighting (`rasnet_model.py:73,85-96`; `train_rasnet_200ep.py:211-215`):**
   - Verified forward pass on `(1, 1, 96, 96, 96)`: `ds_outputs[0]` (`i = 0`, 64 ch) has shape **`(1, 2, 24, 24, 24)` (`1/4` resolution, `2.0 mm` voxel — wider than distal coronary arteries)**, and `ds_outputs[1]` (`i = 1`, 32 ch) has shape **`(1, 2, 48, 48, 48)` (`1/2` resolution)**.
   - The comment at `rasnet_model.py:73` is inverted (`# Level 0 (1/2 size): 64 channels, Level 1 (1/4 size): 32 channels`).
   - More importantly, `train_rasnet_200ep.py:215` computes `total_loss = loss_main + 0.4 * aux2_loss + 0.2 * aux3_loss`, assigning **2× higher weight (`0.4`) to the coarser `24³` map than to the finer `48³` map (`0.2`)** — the exact opposite of standard decaying deep supervision (`1, 0.5, 0.25`) — unnormalized (`sum = 1.6`), and trilinearly upsamples `24³` logits to `96³` (`:211-212`) instead of downsampling GT (blurring thin vessels and increasing VRAM).

### 3.2 Loss Function Audit (`Phase3_Local_Integration/rasnet_loss.py`)

- **Defect L1 — `StenosisAwareLoss` (`rasnet_loss.py:59-84`):**
  - Instantiates `0.4 * DiceLoss(smooth_nr=0.0, smooth_dr=1e-5, to_onehot_y=True, softmax=True, include_background=True)` + `0.6 * FocalLoss(gamma=2.5, to_onehot_y=True, use_softmax=False, include_background=True)`.
  - **Four verified mathematical weaknesses (benchmarked directly against `DiceFocalLoss(1.0, 2.0)` and `DiceCELoss(batch=True)` on matched `32³` tensors):**
    1. **Activation mismatch:** `DiceLoss` normalizes via `softmax=True`, while `FocalLoss` leaves `use_softmax=False` (independent per-channel sigmoid BCE), allowing logit offsets to drift between Dice and Focal gradients. *(Note: MONAI's `DiceFocalLoss` used by SegResNet and V-Net also sets `use_softmax=False` internally.)*
    2. **Background Dice dilution:** Leaving `include_background=True` (default) averages foreground vessel Dice with background Dice (`≈ 0.998`), cutting foreground Dice gradient magnitude roughly in half.
    3. **Empty-crop false-positive gradient weakness (`1/3` of crops under `pos=2, neg=1`):** On an empty GT patch containing a 1,000-voxel FP blob (logits `+2 / -2`), `0.4 * DiceLoss` with `smooth_nr=0.0` stalls (`grad_norm = 0.000002`), while `0.6 * FocalLoss` **does** fire (`grad_norm = 0.000319`, total `0.000321`). Thus `StenosisAwareLoss` does *not* completely stall on empty-crop FPs, but its FP suppression gradient is **`3.0×` weaker than `DiceCELoss(batch=True)` (`0.000948`)** and **`4.6×` weaker than SegResNet/V-Net's `DiceFocalLoss(1.0, 2.0)` (`0.001469`)**.
    4. **Realistic `0.293%` FG mid-training patch:** The `0.4*Dice` / `0.6*Focal` gradient-norm ratio is `193×` (`0.001202` vs `0.000006`) — comparable to `DiceCELoss` (`168×`) and actually *tighter* than SegResNet/V-Net's `DiceFocalLoss` (`824×`), because voxel-averaged Focal/CE over `99.7%` easy background is inherently small.
  - `BoundaryLoss` (`rasnet_loss.py:95-131`) is unused dead code (explicitly disclosed in module docstring `:19`).

### 3.3 Preprocessing, Augmentation, Training Loop & Ablation Forensics

1. **Defect P1 — Raw-HU `CropForegroundd(x > 0)` (`train_rasnet_200ep.py:110,127`; `benchmark/adapters/monai_models.py:37`):**
   - Runs *before* `ScaleIntensityRanged(-100, 800 -> 0, 1)`, thresholding raw HU at `> 0`. Because `benchmark/adapters/monai_models.py:37` mirrors this exact order, train and inference are consistent; however, whether `HU > 0` ever clips a peripheral low-attenuation vessel voxel across the 934/1,000 ImageCAS scans requires a **LAB-CHECK**.
2. **Defect P2 — In-Plane Resolution & Intensity Loss vs. ImageCAS Fingerprint (`train_rasnet_200ep.py:111-112`):**
   - Verified `nnunet_best.pt` `plans` show ImageCAS median spacing is `[0.5, 0.349609375, 0.349609375] mm` and FG HU `[0.5%, 99.5%]` = `[-166, 727]` (`mean = 147.97, std = 179.68`).
   - RASNet resamples XY to `0.50 mm` isotropic (**discarding `51%` of in-plane voxels** on sub-millimeter distal branches), clips `[-166, -100]` HU, and linearly scales to `[0, 1]`.
3. **Defect P3 — Flip-Only Augmentation (`train_rasnet_200ep.py:117-119`):**
   - Only 3-axis `RandFlipd(prob=0.5)` is applied across all 200 epochs — zero 3D rotation, scaling, elastic deformation, Gaussian noise/blur, or brightness/contrast/gamma jitter.
4. **Defect P4 — `PersistentDataset` Cache Hazard & Catastrophic Legacy Ablation Bug (`train_rasnet_200ep.py:80,134`):**
   - Sharing `cache/segresnet` across v1 `RASNet`, `SegResNet`, and `V-Net` was safe because all three shared identical deterministic pre-crop transforms (`LoadImaged` → `EnsureChannelFirstd` → `CropForegroundd` → `Spacingd(0.5)` → `ScaleIntensityRanged(-100, 800)`).
   - **Critical Forensic Finding on `ablation_{attngate,deepsup}_best.pth`:** Tracing `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_2026-11-10/Q1_Publication_Package/ablation/train_ablation_models_200ep.py` reveals it reused the same `cache/segresnet` directory (`:89`, non-RAS LAS cached tensors) via `PersistentDataset` (`:174`) while adding `Orientationd(axcodes="RAS")` (`:149, :167`). Because MONAI `PersistentDataset` keys cache hashes on input filenames rather than transform pipelines, **training silently loaded LAS cached tensors while `val_ds = Dataset(...)` (`:183`) evaluated on RAS-flipped volumes** — and it also switched the loss to `DiceCELoss` (`:196`) and halved patches/epoch (`num_samples=1`, `:155`). **Both legacy ablation checkpoints are scientifically invalid and must never be cited.**
5. **Defect P5 — Incomplete Auto-Resume (`T1`, `train_*_200ep.py:167-189`) & V1 Training Asymmetries:**
   - Auto-resume reloads only `model.state_dict()` and steps the LR scheduler without restoring AdamW moments, GradScaler, or RNG state (`rasnet` and `3dunet` ran uninterrupted; `segresnet` resumed at epochs `86, 109, 133` and `vnet` at `18–19`).
   - Across the v1 MONAI models, learning rates (`2e-4` RASNet vs `1e-4` SegResNet/V-Net/3D U-Net), losses (`StenosisAwareLoss` vs `DiceFocalLoss(1.0, 2.0)` vs `DiceCELoss`), patches/epoch (`1,380` vs `690` for V-Net), gradient clipping (V-Net only: `clip_grad_norm_(1.0)`), and validation cadences (`12 / 20 / 13 / 21 / 200` evals) were not strictly matched — a known v1 caveat already noted in `CHECKPOINTS.md` §2.1 and `V1_REPORTING_CORRECTION_REPORT.md` §4.

---

## 4. Benchmark & Evaluation Integrity Check

### 4.1 Integrity Audit of `benchmark/` vs. Legacy Evaluators

- **Active `benchmark/` Engine (`benchmark/run_benchmark.py`, `symmetric_eval.py`, `endpoint_eval.py`, `metrics.py`, `stats.py`):**
  - **Verified clean:** Strict split validation (`preflight.py`), identical native-grid inversion and physical header checks (`check_geometry`, `atol=1e-5`), exact 6-connected boundary extraction + `distance_transform_edt` (`metrics.py:30-42`), explicit `status` tagging for empty masks/skeletons (`metrics.py:58,65-66`), case-ID-aligned paired Wilcoxon + Holm-Bonferroni + 10,000-resample paired bootstrap CIs (`stats.py`), and immutable `.sha256` sidecar verification on Val-94 threshold locks (`symmetric_eval.py:330-361`).
  - **Two minor non-blocking code edge cases found:**
    1. `benchmark/thresholds.py:16-17` (`_fast_dice`) returns `1.0` when `ps + gs == 0` (`both_empty`), whereas `benchmark/metrics.py:53` returns `np.nan` (`0` impact on ImageCAS because `0` GT masks are empty).
    2. `benchmark/symmetric_eval.py:371` (`done_keys`) checks `df.groupby(list(key_cols)).size() >= rows_per_case` without deduplicating `(model, case_id, recipe)` first (verified `0` duplicate rows exist in `val_grid_scores.csv` [`8,272` rows] or `test_per_case.csv` [`2,400` rows]).
- **Fairness Caveats Disclosed in V1 & Option A:**
  1. **V1 Arms A/B/C (`test_retrospective_150/`):** MONAI models used `0.25` overlap constant-blend SW and 4-pass probability-averaged TTA (Arm B) vs. nnU-Net's `0.50` Gaussian SW and 8-pass logit-averaged TTA; the Arm C threshold grid floor `0.30` clamped RASNet, V-Net, and 3D U-Net (`results/thresholds.json`).
  2. **Option A (`results/optionA_v1_symmetric_sw050/`):** Eliminated all SW overlap, blending, TTA, and `0.30`-clamping asymmetries across the 4 SW models (`3dunet` excluded due to `128³` resize). Three residual nuances remain:
     - `nnunet` R4 hit the expanded grid floor `tau* = 0.20` (`locked_recipes.json:234`).
     - `nnUNetPredictor` resamples logits before softmax whereas `infer_monai_symmetric` applies softmax before `Invertd` (required so `CropForegroundd` zero-padding maps to `0.0` foreground probability rather than `0.5`).
     - Option A full-volume timing (`timing_summary.csv`, `n = 10`) ran MONAI at `sw_batch_size=8` (`4,782–5,929 MB` VRAM; `5.4s` 1-pass / `22.9s` TTA-8 for RASNet) vs. nnU-Net at batch `1` (`2,243–2,261 MB`; `29.8s` / `74.4s`), explaining why MONAI shows higher full-volume peak VRAM despite `4.9×` lower single-patch VRAM (`390.7` vs `1,926.5 MB`).
  3. **Live Legacy Evaluator Hazard (P0):** `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py` and `Phase3_Local_Integration/eval_utils.py:131-132` (returns `HD95 = 0.0` on empty masks) remain tracked in active directories and must be quarantined so nobody accidentally invokes or cites them.

### 4.2 Verified Metrics Summary (Pulled Directly from Repo Result CSVs)

| Regime & Cohort | Model | Params (M) | $\tau$ / CC | Dice | Precision | Recall | HD95 (mm) | clDice | Verified Source File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Test-150 Arm A** *(asymm SW, 1-pass)* | **RASNet** | `4.71` | `0.50 / None` | **`0.7736 ± 0.0616`** | **`0.8032`** | `0.7521` | `14.83` | **`0.8389`** | `test_retrospective_150/per_case_arm_A.csv` |
| | nnU-Net V2 | `30.79` | `0.50 / None` | `0.7687 ± 0.0668` | `0.7391` | **`0.8106`** | `21.10` | `0.8202` | |
| | SegResNet | `4.70` | `0.50 / None` | `0.7481 ± 0.0654` | `0.7253` | `0.7811` | `30.41` | `0.7941` | |
| | V-Net | `45.60` | `0.50 / None` | `0.7492 ± 0.0693` | `0.7524` | `0.7555` | `22.10` | `0.8164` | |
| | 3D U-Net (`128³`) | `4.81` | `0.50 / None` | `0.5517 ± 0.0480` | `0.7269` | `0.4484` | **`10.88`** | `0.7103` | |
| **Test-150 Arm C** *(asymm SW, 1-pass, Top-2)* | **nnU-Net V2** | `30.79` | `0.35 / Top2` | **`0.8055 ± 0.0633`** | `0.8207` | **`0.7977`** | **`8.26`** | **`0.8739`** | `test_retrospective_150/per_case_arm_C.csv` |
| | RASNet | `4.71` | `0.30* / Top2` | `0.7899 ± 0.0677` | **`0.8252`** | `0.7649` | `9.94` | `0.8646` | *(Arm C vs nnU-Net Holm p: Prec `0.160` n.s., HD95 `4.2e-4` nnU-Net)* |
| **Option A `R1`** *(symm 0.50G, 1-pass)* | **RASNet** | `4.71` | `0.50 / None` | **`0.7788 ± 0.0623`** | **`0.7764`** | `0.7888` | **`15.97`** | **`0.8394`** | `optionA_v1_symmetric_sw050/test_per_case.csv` |
| | nnU-Net V2 | `30.79` | `0.50 / None` | `0.7687 ± 0.0668` | `0.7391` | **`0.8106`** | `21.10` | `0.8201` | *(RASNet Dice `+0.0101`, `p = 2.92e-4`, Superior = `True`)* |
| **Option A `R2`** *(symm 0.50G, TTA-8)* | **RASNet** | `4.71` | `0.50 / None` | **`0.7863 ± 0.0619`** | **`0.7876`** | `0.7923` | **`14.88`** | **`0.8469`** | `optionA_v1_symmetric_sw050/test_per_case.csv` |
| | nnU-Net V2 | `30.79` | `0.50 / None` | `0.7760 ± 0.0641` | `0.7536` | **`0.8090`** | `19.25` | `0.8264` | *(RASNet Dice `+0.0103`, `p = 8.24e-6`, Superior = `True`)* |
| **Option A `R4` (PRIMARY)** *(symm 0.50G, TTA-8, Top-2)* | **nnU-Net V2** | `30.79` | `0.20* / Top2` | **`0.8059 ± 0.0630`** | `0.8084` | **`0.8104`** | **`8.17`** | `0.8744` | `optionA_v1_symmetric_sw050/test_per_case.csv` |
| | **RASNet** | `4.71` | `0.40 / Top2` | `0.8039 ± 0.0691` | **`0.8302`** | `0.7861` | `8.93` | **`0.8750`** | *(NI `-0.005`: **`False`**, CI `[-0.0088, +0.0030]`; `76W/0T/74L`; Prec Holm `7.23e-17`)* |
| | SegResNet | `4.70` | `0.45 / Top2` | `0.7989 ± 0.0692` | `0.8200` | `0.7855` | `9.12` | `0.8728` | *(RASNet vs SegResNet Dice `+0.0050`, Holm `p = 4.27e-8`)* |
| | V-Net | `45.60` | `0.45 / Top2` | `0.7915 ± 0.0677` | `0.8216` | `0.7711` | `9.50` | `0.8678` | *(RASNet vs V-Net Dice `+0.0124`, Holm `p = 3.58e-12`)* |
| **Val-94 `final_epoch200`** *(symm 0.50G, 1-pass)* | **RASNet** (`last`) | `4.71` | `0.50 / None` | `0.7792 ± 0.0569` | **`0.7586`** | `0.8065` | `17.32` | `0.8413` | `v1_epoch200_comparison.../per_case_metrics_combined.csv` |
| | nnU-Net (`last`) | `30.79` | `0.50 / None` | `0.7767 ± 0.0562` | `0.7428` | **`0.8203`** | `18.67` | `0.8347` | *(exploratory unadj. Dice `p = 0.2401`, `54W/40L`; neither sup. nor equiv.)* |

*\*Hit threshold grid floor (`0.30` in Arm C; `0.20` in Option A R4).*

### 4.3 Forensic Finding: Hard `keep_top_k=2` CC Pruning Hurts All 4 Models — and Case `930` Swings R4

Inspection of all `2,400` rows in `results/optionA_v1_symmetric_sw050/test_per_case.csv` reveals why RASNet leads nnU-Net by `+0.0103` in `R2_raw_tta8` (`0.7863` vs `0.7760`) yet trails by `-0.00208` in `R4_tuned_tta8_top2` (`0.8039` vs `0.8059`) despite winning **`76 / 150` cases** (median paired diff **`+0.00021`**, Wilcoxon `p = 0.99`):

- **Dataset-Wide `keep_top_k=2` Vulnerability (R2 → R4 Dice drops `< -0.05` across all 4 models):**
  - `rasnet`: `8` cases (`930: -0.2112`, `948: -0.1079`, `861: -0.0951`, `894: -0.0818`, `978: -0.0784`)
  - `nnunet`: `8` cases (`880: -0.1445`, `931: -0.1096`, `948: -0.1073`, `861: -0.1047`, `866: -0.0747`)
  - `segresnet`: `6` cases (`941: -0.1932`, `948: -0.1060`, `861: -0.0929`)
  - `vnet`: `6` cases (`948: -0.1369`, `861: -0.1129`, `978: -0.0850`)
- All four models simultaneously collapse `-0.09` to `-0.14` on **Cases `948` and `861`**, proving multi-component coronary anatomy where hard Top-2 CC inevitably amputates a true third branch (e.g., separate conus/ramus/LCA-RCA split).
- On **Case `930`**, RASNet beat nnU-Net by `+0.0331` in R2 (`0.6814` vs `0.6483`), but under R4 Top-2 CC pruned `4,023.7 mm³` from RASNet (`0.4703`) while boosting nnU-Net to `0.8260` (paired diff `-0.3557`). That **single case** contributes `-0.00237` to the 150-case mean — larger than the entire `-0.00208` R4 gap. *(Note: on Case `868`, RASNet actually improved from `0.7177` in R2 to `0.7418` in R4 [`+0.0241`]; nnU-Net simply gained more [`+0.1125`].)*

---

## 5. The nnU-Net Gap Analysis (Architecture vs. Engineering)

| Dimension | Core RASNet Setup (v1) | nnU-Net V2 Setup (`nnunet_best.pt` `plans` + Trainer) | Category | Estimated Impact | Effort to Adopt in Core RASNet |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Target Spacing** | `[0.5, 0.5, 0.5] mm` isotropic (`-51%` XY voxels) | `[0.5, 0.349609375, 0.349609375] mm` native median | **Engineering** | **Very High** (`+0.01–0.02` distal recall & clDice) | Low (config + retrain) |
| **Patch Size & FOV** | `96×96×96` (`0.88M` vox; `48³ mm`) | `96×160×160` (`2.46M` vox; `48×56×56 mm`) | **Engineering** | **High** (fewer ostial/branch splits) | Low–Medium (`bs=2` fits 16 GB) |
| **Intensity Norm** | Clip `[-100, 800]` → linear `[0, 1]` | `CTNormalization`: clip `[-166, 727]`, z-score `(x - 147.97) / 179.68` | **Engineering** | **Medium** | Trivial |
| **Foreground Cropping** | Raw-HU `CropForegroundd(x > 0)` before normalization | Non-zero / bounding-box safe CT pipeline | **Engineering** | **Low–Medium** (eliminates peripheral clipping risk) | Trivial |
| **Augmentation** | 3-axis `RandFlipd` **only** | 3D rotation, scaling, elastic/gamma, Gaussian noise/blur, brightness/contrast, low-res simulation | **Engineering** | **High** (recall & domain generalization) | Low (standard MONAI transforms) |
| **Loss Function** | `0.4*Dice(inc_bg=T, sm_nr=0) + 0.6*Focal(g=2.5, sigmoid)` (`3×` weaker empty-FP grad than CE) | `BatchDice(inc_bg=F, batch_dice=True) + Unweighted CE` | **Engineering** | **High** (fixes ~2× FG Dice dilution & recall gap) | Trivial |
| **Deep Supervision** | 2 heads (**inverted**: `0.4` on `1/4`-res, `0.2` on `1/2`-res), trilinear upsample to `96³` | 5 decoder scales, decaying normalized weights (`1, 0.5, 0.25...`), downsampled GT | **Engineering** | **Medium** (sharper thin-vessel gradients, lower VRAM) | Trivial |
| **Post-Processing** | Hard `keep_top_k=2` (drops `8` cases `>0.05` on both RASNet & nnU-Net; Case `930` `-0.21`) | Native none (Top-2 CC applied symmetrically in Option A) | **Engineering** | **High** (must beat `top2cc` on Val-94) | Low (test `200–600 mm³` or relative-size floor) |
| **Optimizer & Schedule** | `AdamW(lr=2e-4)`, Cosine, no grad clip, `1,380` patches/ep | `SGD(lr=1e-2, mom=0.99, nesterov)`, PolyLR, `clip_grad_norm(12)`, `500` patches/ep | **Engineering** | **Low–Medium** | Trivial |
| **Backbone & Capacity** | 4-stage `SegResNet`, `[16, 32, 64, 128]`, `4.71M` params, `123` GFLOPs, `GN(8)+ReLU+Drop(0.1)` | 6-stage `PlainConvUNet` (`[1,2,2]` stage-5 stride), `[32..320]`, `30.79M` params, `1,119` GFLOPs, `IN+LeakyReLU` | **Architecture** | **Medium** (receptive field & capacity) | Deferred to v2 (preserve `4.71M` efficiency identity) |
| **Attention Gate** | `AttentionGate3D` with `GroupNorm(1,1)` before `Sigmoid` (broken sparsity on real `rasnet_best.pth`) | None | **Architecture** | **Low–Medium** (validate `conv_bias` vs `GN` on Val-94; width-dependent in 50-ep engineering signal) | Trivial (1-line change) |
| **Ensembling** | Single fold / single seed | Single fold here (fold 0); 5-fold standard | **Engineering** | **Medium** (`+0.005–0.01` Dice for any model) | Medium (`5×` training cost) |

### Honest Assessment: Engineering vs. Model Capacity
Frozen v1 `RASNet` already **beats nnU-Net by `+0.0101` (R1) and `+0.0103` (R2) in raw full-volume Dice**, wins **`76 / 150` cases in R4**, and trails R4 mean Dice by only **`-0.0021`** while running at **`6.54×` fewer parameters**, **`9.07×` fewer GFLOPs/native patch**, and **`3.2×–5.5×` faster full-volume inference** (`5.4s` vs `29.8s` 1-pass; `22.9s` vs `74.4s` TTA-8). Consequently, **~80–90% of nnU-Net's remaining lead is pure pipeline engineering** (native `0.35 mm` XY resolution, `96×160×160` FOV, rich augmentations, `BatchDice + CE`, proper DS weighting, and Top-2 CC interaction), **not** its 30.8M-parameter backbone.

---

## 6. Prioritized Action Plan for Core RASNet

*(Per Rules 2 & 9, do NOT modify frozen v1 checkpoints or overwrite protected `results/` directories. Any retraining or re-evaluation requires a dated pre-registration amendment and fresh output directories.)*

### P0 — Must Fix Immediately (Data Governance, Legacy Hazards & Reproducibility)
1. **Remove Tracked Clinical CT Data & Checkpoints from Git (Rule 7):**
   - Untrack (and evaluate `git filter-repo` history scrub after checking repo visibility/PHI headers) `Phase3_Local_Integration/local_data/CT4/CT4_Stenosis_Annotation_Package.zip` (`76.88 MiB`) and `CT61/CT61_Stenosis_Annotation_Package.zip` (`71.22 MiB`).
   - Untrack `Q1_Publication_Package/matched_200ep_benchmark/checkpoints_best_and_last/checkpoints_200ep.zip.001`..`.010` (`859.43 MiB`) and invalid `checkpoints/ablation_{attngate,deepsup}_best.pth` (`36.00 MiB`).
2. **Quarantine Stale Legacy Evaluator & Invalid Artifacts:**
   - Move still-tracked `Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py`, its stale `evaluation_results/` folder, the 7 intermediate/aborted recovery CSVs in `logs/`, and `Phase3_Local_Integration/eval_utils.py` (`HD95=0.0` bug) into `_Archive_Historical_Experiments_and_Logs/`.
3. **Eliminate Hardcoded Paths & Close Documentation/Env Drift:**
   - Replace `ROOT = Path(r"C:\Thesis_RASNET")` in `scripts/run_phaseH_rasnet_optimization.py:43` and `scripts/backfill_phaseH_val94_and_report.py:31` with `Path(__file__).resolve().parents[1]`; deprecate `Phase3_Local_Integration/dataset_paths.py`.
   - Export `requirements.lab.txt` from `C:\Thesis_RASNET_env\bench` (`torch 2.14.1+cu126`, `monai 1.6.1`, `nnunetv2 2.8.1`, `SimpleITK 2.5.7`, `cc3d 4.1.1`).
   - Fix `CHECKPOINTS.md:13` nnU-Net archive timestamp (`2026-09-08 09:19:08`), update `README.md:31-33,203` (correct `up_samples`/`attention_gates`/`ds_heads` names and remove nonexistent `config.yaml`), and sync Rule 11 into `CLAUDE.md` §3.
4. **Complete External Thesis & Defense Integration (P5):**
   - Apply verified tables, figures, and claim wording from `results/v1_reporting_correction_20261117_105000Z/` (`THESIS_REPLACEMENT_TEXT.md`, `DEFENSE_QA_V1.md`, `HISTORICAL_ERRATA.md`) into external `.tex`/`.docx`/`.pptx` manuscripts.

### P1 — High-ROI Upgrades for Core RASNet (Clean Wins Without Building `Rasnet-v2`)
1. **Zero-Retraining Symmetric Post-Processing Rule (Val-94 Locked Across All 4 Models):**
   - Phase H (`results/rasnet_fixed_weight_optimization/val_94_all_candidates_summary.csv`) already demonstrated that naive `top3cc` (`0.8000`, precision `-0.040`) and small-seed hysteresis (`hyst_lo0.30_hi0.55_seed50_nocc`, `0.7836`) lose to `single_tau0.40_top2cc` (`0.8023`) because cardiac FP islands exceed `50–150 mm³`.
   - Instead, evaluate a **larger physical/relative component floor** on Val-94 symmetrically across all 4 models (e.g., retain 26-connected components `>= V_min` in `{200, 350, 500} mm³` or `>= 10%` of largest CC, capped at top-3), preserving genuine `1,100–4,024 mm³` third branches (`Cases 930, 948, 861`) while rejecting `<150 mm³` FP islands.
2. **Fix the 3 Core Math/Architecture Bugs in `RASNet` + Loss (for Next Pre-Registered Training Run):**
   - **Attention Gate (`rasnet_model.py:20-32,66-77`):** Replace `GroupNorm(1, 1)` in `psi` with `Conv3d(F_int, 1, 1, bias=True) + Sigmoid` *(validate against `GN` on Val-94: non-citable 50-ep engineering pilots in `Rasnet-v2/results/pilots_val94/pilot_lock_decision.md` showed `conv_bias` gained at `F=16` but regressed at `F=32`, so do not assume width-invariance without validation)*; remove redundant `W_g`/`W_x` dual biases; guard `GroupNorm` groups when `F_int <= 8`; derive `attention_gates` and `ds_heads` channels dynamically from `init_filters`.
   - **Deep Supervision (`train_rasnet_200ep.py:211-215`):** Reverse inverted weights so finer `1/2`-res receives higher weight than coarser `1/4`-res (normalized `(1.0 * main + 0.5 * ds_half + 0.25 * ds_quarter) / 1.75`) and downsample GT masks rather than trilinearly upsampling `24³` logits.
   - **Compound Loss (`rasnet_loss.py:59-84`):** Switch to `DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, batch=True)` to eliminate ~2× background Dice dilution, `use_softmax=False` mismatch, and `3×` weaker empty-crop FP gradients.
3. **Adopt nnU-Net's Preprocessing, Augmentation & Training Hygiene:**
   - Resample to native median `[0.5, 0.3496, 0.3496] mm`, train on `96×160×160` patches (`bs=2`), use fingerprint `CTNormalization` (`[-166, 727]` z-score), drop raw-HU `CropForegroundd(x > 0)`, add 3D affine/noise/blur/contrast/gamma augmentations, add `clip_grad_norm_(1.0)`, save full optimizer/scaler/RNG state (`T1`/`T3`), and isolate `PersistentDataset` caches by transform hash.

### P2 — Polish & Cleanup
- Align `benchmark/thresholds.py:16-17` (`_fast_dice` returning `1.0` on `both_empty`) with `benchmark/metrics.py:53` (`np.nan`).
- Harden `benchmark/symmetric_eval.py:371` (`done_keys`) to deduplicate `(model, case_id, recipe)` before counting completed rows.
- Add regression tests in `tests/test_model_shapes.py` for `RASNet(init_filters=32)` and `ds_outputs` spatial resolution ordering once P1 architecture fixes are adopted in a future version.
- Note `v1_epoch200_comparison_20261117_041500Z/manifest.json` `git.dirty_code: True` (run prior to `8e9b303`; verified via `source_tree_sha256` and 72 passing tests) in any appendix provenance table.

### Explicitly Deferred to `RASNetv2`
- Width scaling (`F=32`, `18.8M` params) and 5–6 stage anisotropic backbone redesign (`PlainConvUNet`-style `[1, 2, 2]` stride).
- Multi-seed (`{42, 123, 456}`) controlled architecture-vs-architecture study with matched LR grids across all baselines.
- Topology-aware losses (`soft-clDice`), which showed strong regression in 50-ep engineering screening (`Rasnet-v2/results/pilots_val94/pilot_lock_decision.md`, non-citable per Rule 5).
- 5-fold cross-validation ensembling and prospective external multi-center / clinical QCA validation.

---

## 7. Final Go / No-Go Checklist

### Already Verified (Green)
- [x] **Split Integrity & Leakage Disclosure:** `splits_final.json` verified `690 / 94 / 150`, `0` overlaps; Test-150 explicitly framed as **retrospective**.
- [x] **Checkpoint & Result Reconciliation:** All 7 canonical checkpoints match `CHECKPOINTS.md` SHA-256 byte-for-byte; `0` empty predictions/NaNs across `4,838` rows; `0` duplicate rows in Option A CSVs; Table 2, Option A, and Epoch-200 numbers match raw CSVs.
- [x] **Claim Discipline & `Rasnet-v2` Isolation:** Thesis claims bounded by `V1_REPORTING_CORRECTION_REPORT.md` (`NI = False` on R4; Epoch-200 `p = 0.2401` neither superiority nor equivalence; `unseen_66`, hospital QCA, and invalid v1 ablations excluded); zero imports from paused `Rasnet-v2/`.

### Required Before Final Lock (Action Items)
- [ ] **P0 Data/Git Cleanup:** Remove git-tracked clinical CT zips (`CT4`/`CT61`), `checkpoints_200ep.zip.001..010`, and invalid `ablation_{attngate,deepsup}_best.pth` (and verify remote visibility / PHI headers for history-scrub decision).
- [ ] **P0 Legacy Quarantine:** Archive still-tracked `evaluate_matched_200ep_models.py`, `evaluation_results/`, `Phase3_Local_Integration/eval_utils.py`, and the 7 aborted recovery CSVs in `logs/`.
- [ ] **P0 Reproducibility & Doc Sync:** Export `requirements.lab.txt`, remove hardcoded `C:\Thesis_RASNET` in Phase H scripts, fix `CHECKPOINTS.md:13` timestamp, `README.md:31-33,203`, and sync `CLAUDE.md` Rule 11.
- [ ] **LAB-CHECK — `CropForegroundd(x > 0)` Audit:** Run a read-only script on `C:\Thesis_RASNET` verifying `0` ground-truth coronary voxels are clipped across all 934/1,000 ImageCAS scans.
- [ ] **LAB-CHECK — Case `930`/`948`/`861`/`978` Component Inspection:** Inspect Option A predicted component volume lists and 3D overlays on `C:\Thesis_RASNET` before locking any P1 symmetric volume-floor post-processing rule on Val-94.
- [ ] **LAB-CHECK — Clean `pytest` Verification:** Re-run `C:\Thesis_RASNET_env\bench\Scripts\python.exe -m pytest tests/` (`72/72`) after P0 hygiene edits with `git.dirty_code = false`.
- [ ] **External Manuscript & Slide Integration (P5):** Paste verified tables, corrected figures, and claim wording from `results/v1_reporting_correction_20261117_105000Z/` into external `.tex`/`.docx`/`.pptx` files.
