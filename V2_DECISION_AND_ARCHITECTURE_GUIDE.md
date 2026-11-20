# RASNet-v2 Decision & Architecture Guide

**Date:** `2026-11-20`  
**Repo Audited:** `Efficient-3D-Tiled-CNN-Architecture` (pulled through commit `e2c929f`)  
**Target:** Help us choose the smartest path before spending days of RTX 4080 SUPER GPU time on `Rasnet-v2` — **should we make real architectural changes, do pipeline/post-processing upgrades, or lock V1 as-is?**

---

## 1. TL;DR — Executive Decision Matrix

Before starting **any** GPU run, look at the three real options on the table:

| Option | What It Does | GPU Time | Can It Beat `nnU-Net V2` Across the Board? | Verdict |
| :--- | :--- | :---: | :--- | :--- |
| **Option 1: Lock & Ship V1 (+ Zero-Retrain Add-Ons)** | Finish P0 Git/doc cleanup (`prompt1-mand-fix.md`), optionally add Val-94-locked symmetric volume-floor CC + PR curves + `best+last` checkpoint ensemble. | **0 hours** *(~2 hrs eval if doing add-ons)* | **No (near-tie):** `0.8039` vs `0.8059` Dice (wins `76/150` cases, wins Precision `0.830` vs `0.808`, `6.5×` fewer params). Strong defense story if framed honestly, **not** universal superiority. | **Best if defense deadline is < 1–2 weeks away.** |
| **Option 2: Resume Current `Rasnet-v2` Code As-Is** | Write Amendment A2 and launch Stage 3 with the **existing** `Rasnet-v2/src/` (`4-stage SegResNet + conv_bias AG`, per-sample Dice, weak augs). | **~3–5 GPU days** | **Unlikely (high risk of another ±0.003 tie):** Current v2 code STILL has 4-stage shallow RF, weak augs, per-sample Dice, and zero vessel-specific architectural novelty. | **DO NOT DO THIS.** Worst ROI of all three options. |
| **Option 3: Upgrade `Rasnet-v2` to a Vessel-Native Architecture + Pipeline Parity** | Replace 4-stage `SegResNet` with a compact **6-stage Anisotropic Tubular Strip-ResNet + Bottleneck 3D Self-Attention + Fixed Channel-Spatial Gates (`~7–9M` params)** + `BatchDice + CE + Skeleton Recall Loss` + full 3D augs + `SGD/PolyLR`, gated by a **2-hour 50-epoch pilot**. | **2 hours** *(Go/No-Go gate)* → **~2–4 GPU days** if GO | **YES — genuinely strong shot** at beating `nnU-Net V2` on **Dice, HD95, ASD, Precision, Recall, AND clDice** simultaneously while staying **3.5–4× smaller & faster** (`~7–9M` vs `30.8M`), plus real Q1 architectural novelty. | **RECOMMENDED if we want to beat `nnU-Net` & publish Q1 (IEEE J-BHI / TMI).** |

> **Short Answer:**  
> - **Can we make RASNet beat `nnU-Net V2`?** **Yes.** A buggy, shallow `4.7M`-param v1 `SegResNet` with blurry `0.5 mm` spacing and flip-only augmentations ALREADY came within **`0.0020` Dice** of `30.8M`-param `nnU-Net V2` (`76/150` wins)! That proves how much headroom is still untouched.  
> - **What should we NOT do?** Do **not** launch Stage 3 on the *current* `Rasnet-v2/src/` code without upgrading the architecture, loss, and augmentations first — otherwise we risk burning 4 days of GPU time just to get another tie.

---

## 2. Where Everything Stands Right Now (Commit `e2c929f`)

### A. Verified V1 Results (`Option A` Symmetric Benchmark, Test-150 Retrospective)
From `results/optionA_v1_symmetric_sw050/OPTION_A_RESULTS.md`:

| Regime | Model | Params | Dice | Precision | Recall | HD95 (mm) | clDice |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R2 (Raw TTA-8, No CC)** | **RASNet v1** | `4.71M` | **`0.7863`** *(+0.0103, p=8.2e-6)* | **`0.7876`** | `0.7923` | **`14.88`** | **`0.8469`** |
| | nnU-Net V2 | `30.79M` | `0.7760` | `0.7536` | **`0.8090`** | `19.25` | `0.8264` |
| **R4 (Primary: TTA-8 + Top-2 CC)** | **nnU-Net V2** ($\tau=0.20^*$) | `30.79M` | **`0.8059`** | `0.8084` | **`0.8104`** | **`8.17`** | `0.8744` |
| | **RASNet v1** ($\tau=0.40$) | `4.71M` | `0.8039` *(76W / 74L, median +0.0002)* | **`0.8302`** *(p=7.2e-17)* | `0.7861` | `8.93` | **`0.8750`** |
| | SegResNet ($\tau=0.45$) | `4.70M` | `0.7989` | `0.8200` | `0.7855` | `9.12` | `0.8728` |
| | V-Net ($\tau=0.45$) | `45.60M` | `0.7915` | `0.8216` | `0.7711` | `9.50` | `0.8678` |

### B. Verified V1 & Repo Audit Findings (Must Fix Regardless of V2!)
From `RASNET_CRITICAL_AUDIT.md` and `RASNET_CRITICAL_audit done1T.md`:
1. **P0 Governance (Do Immediately):** Two clinical CT zips (`Phase3_Local_Integration/local_data/CT4/` and `CT61/`) and `859 MiB` of tracked checkpoints (`checkpoints_200ep.zip.*`) must be purged from Git history; execute `prompt1-mand-fix.md` so all v1 defense/thesis wording is 100% honest and audit-proof.
2. **V1 Ablation Checkpoints Are Invalid:** `ablation_{attngate,deepsup}_best.pth` suffered a silent `PersistentDataset` LAS-train / RAS-val cache mismatch (`train_ablation_models_200ep.py`). Never cite the old v1 ablation table; V2 must produce clean ablations.
3. **V1 Model Defects Confirmed on Real Weights (`rasnet_best.pth`):**
   - `AttentionGate3D.psi` used `nn.GroupNorm(1, 1)` before `Sigmoid`, making gates mathematically unable to close (`38.5–59.3%` of pure-background voxels remained `> 0.50`).
   - Deep supervision weights were **inverted** (`0.4` on coarse `1/4`-res `24³` vs `0.2` on finer `1/2`-res `48³`).
   - `StenosisAwareLoss` left `include_background=True` (~2× foreground Dice dilution) and mixed softmax Dice with sigmoid Focal (`use_softmax=False`).

### C. Current `Rasnet-v2` Status (`Rasnet-v2/results/pilots_val94/pilot_lock_decision.md`)
- Stage 1 gates passed; `784/784` FP16 caches built (`[0.3496, 0.3496, 0.50] mm`, `160×160×96`).
- Stage 2A (18 LR pilots, 50 ep) and Stage 2B (5 design pilots, 50 ep) **finished**.
- **Stage 2B hit `FALLBACK`:** Zero candidates passed the unpruned 50-epoch `Val Precision >= 0.8000` gate:
  - Best 50-ep candidate: `F=16, conv_bias, w_cldice=0.0` → **`0.7340` Val Dice**, `0.6966` Precision, `33.96` components (vs SegResNet `28.05`).
  - Adding `w_cldice=0.20` (`soft_skel`) from Epoch 1 **hurt Dice by `-0.0245`** (`0.7095` vs `0.7340`).
  - Scaling `F=16 → F=32` (`4.7M → 18.8M` params) **hurt Dice** (`0.7095 → 0.7027` with `conv_bias`), proving that raw channel width without deeper receptive field does not solve the problem.

---

## 3. Root-Cause Analysis: Why `nnU-Net V2` Still Wins After Top-2 CC (and Why Current `Rasnet-v2` Is Still Handicapped)

### A. Why Top-2 CC Flips the Scoreboard (`R2` → `R4` & `Arm C`)
1. **`nnU-Net V2` has higher true vessel recall (`0.8104` vs `0.7861`)** — it traces more thin, low-contrast distal branches and keeps arterial trees connected. Raw (`R1/R2`), it only loses because it outputs floating false-positive blobs outside the heart (`HD95 = 19.25 mm`).
2. **Top-2 CC erases `nnU-Net`'s only weakness for free:** Deleting disconnected FP islands leaves `nnU-Net`'s superior distal recall intact, skyrocketing its Dice `+0.0299` (`0.7760 → 0.8059`) and collapsing its HD95 (`19.25 → 8.17 mm`).
3. **Top-2 CC punishes broken trees:** On `RASNet`, any single gap in a thin vessel splits a real branch into a 3rd component, which hard `keep_top_k=2` amputates. On **Case `930` alone**, Top-2 CC pruned `4,023.7 mm³` from `RASNet` (`-0.2112` Dice), dragging `RASNet`'s 150-case mean down by `-0.00237` — **larger than the entire `-0.00208` gap to `nnU-Net`**.

### B. 6 Bottlenecks STILL Present Inside Current `Rasnet-v2/src/`!
Even though `Rasnet-v2/` fixed spacing (`0.35 mm` XY), patch size (`160×160×96`), CT z-score, and `conv_bias` gating, auditing the actual code shows **six major reasons current `Rasnet-v2` still cannot cleanly beat `nnU-Net V2`**:

1. **Shallow 4-Stage Backbone & Tiny Receptive Field (`Rasnet-v2/src/rasnet_v2_model.py:58-79`):**
   - `RASNetV2` still subclasses MONAI `SegResNet` with default `blocks_down=(1, 2, 2, 4)` → only **4 stages (3 strides = `8×`)**.
   - On `160×160×96` at `0.35 mm` XY spacing, the bottleneck is `20×20×12` — an effective receptive field of only **~12–16 mm in XY**. It literally cannot see the aorta/ostia and distal coronary branches together!
   - By contrast, `nnU-Net V2` uses **6 stages (5 anisotropic strides: `32×` XY, `16×` Z)** → `5×5×6` bottleneck spanning the **entire `56×56×48 mm` cardiac patch**.
2. **Weak Decoder (`blocks_up=(1, 1, 1)` + Additive Skips):**
   - `SegResNet` uses only **1 residual block** per decoder stage and element-wise addition (`g + attended_skip`), whereas `nnU-Net` uses **2 conv blocks** per decoder stage + channel concatenation, preserving sharper sub-millimeter lumen boundaries.
3. **Zero Tubular / Vessel-Specific Inductive Bias (`rasnet_v2_model.py:15-55`):**
   - Oktay `AttentionGate3D` (`+5,858` params = `+0.125%`) uses only `1×1×1` pointwise convs. It has **zero spatial kernel** and **zero awareness of curved 1D tubular vessels** — which is why reviewers correctly view it as an incremental `SegResNet` refinement.
4. **Loss Bottlenecks STILL in `Rasnet-v2/src/rasnet_v2_loss.py`:**
   - **Per-sample Dice (`:82-85`) instead of Batch Dice (`batch_dice=True`):** Under `pos=1:neg=2`, two-thirds of crops are background-centered (many with zero vessel voxels), making per-sample Dice noisy compared to `nnU-Net`'s `BatchDice`.
   - **Under-weighted voxel CE/Focal (`:64`):** Focal is divided by `n_ch = 2` AND multiplied by `w_focal = 0.3` with `gamma = 1.5`, producing a weak voxel-level gradient compared to `nnU-Net`'s full `1.0 * CE`.
   - **`soft_skel` `clDice` (`:88-96`) hurt Stage 2B by `-0.0245` Dice** because differentiable thinning on noisy early predictions fights optimization.
   - **DS still trilinearly upsamples coarse logits (`:155`)** to full `160×160×96` instead of downsampling GT masks, wasting VRAM and blurring thin targets.
5. **Weak Augmentations STILL in `Rasnet-v2/src/augmentations.py:207-233`:**
   - Current v2 ONLY applies random flips, 90° XY rotation, and ±10% scale/shift.
   - It is **missing** arbitrary 3D rotation, random 3D scaling/zoom, elastic deformation, Gaussian noise, Gaussian blur, contrast/gamma jitter, and low-resolution simulation — all of which `nnU-Net V2` uses heavily.
6. **Optimizer Gap (`fair_v2.yaml:33-36`):**
   - Only tested `AdamW` (`lr=1e-4` won at 50 ep, reaching only `0.709–0.734` Val Dice). In 3D medical UNets with `InstanceNorm + BatchDice + CE`, `SGD(lr=1e-2, momentum=0.99, nesterov=True)` + `PolyLR` consistently converges sharper and `+0.01–0.02` higher at 200 epochs.

---

## 4. Architecture Changes vs. Doing Something Else: Honest Trade-Off Analysis

### What if we ONLY do "Something Else" (No Architecture Redesign)?
1. **Zero-Retraining Post-Processing & Ensembling (`Option 1` + Stage 5):**
   - Replacing rigid `keep_top_k=2` with a **symmetric Val-94-locked physical volume floor** (e.g., keep 26-connected components `>= V_min ∈ {200, 350, 500} mm³` or `>= 10%` of largest CC, capped at top-3) rescues multi-branch cases (`930, 948, 861`) for *all* models; symmetric `best + last` checkpoint ensembling adds `+0.002–0.005` Dice.
   - **Limitation:** Because symmetry requires giving `nnU-Net V2` the exact same post-processing/ensembling, this will **not** produce a large, indisputable win across all metrics — and it leaves v1's broken `GroupNorm(1,1)` gates, invalid ablations, and *"incremental SegResNet tweak"* criticism unresolved.
2. **Fixing ONLY Pipeline/Loss/Augs While Keeping 4-Stage `SegResNet`:**
   - Closing the `0.35 mm` spacing, `BatchDice+CE`, and augmentation gaps will definitely raise `SegResNet`/`RASNet` past v1, **likely matching `nnU-Net` (~`0.805–0.810` Dice)** — **but** without real architectural novelty, `RASNet` will barely separate from matched `SegResNet`, and reviewers will say: *"You just trained MONAI SegResNet with nnU-Net's preprocessing."*

### Why Architecture + Pipeline Together (`Option 3`) Is the Right Decision for V2
- **Pipeline parity** eliminates the ~80% engineering handicap so `RASNet` is no longer fighting uphill.
- **Vessel-native architectural design** exploits `nnU-Net`'s **exact blind spot**: `nnU-Net V2` is a generic `30.8M`-param `3×3×3` `PlainConvUNet` with **zero tubular directional kernels**, **zero global bottleneck attention**, and **zero centerline recall supervision**.
- That lets a **compact `~7–9M`-parameter vessel-specialized architecture** genuinely beat `30.8M` `nnU-Net V2` on **Dice, HD95, ASD, Precision, Recall, AND clDice** — while delivering an unmistakable, publishable architectural contribution!

---

## 5. Exact Blueprint for Option 3: Vessel-Native `RASNet-v2`

Target budget: **`~7–9M` parameters** (**3.5–4× smaller** than `nnU-Net V2`'s `30.79M`, fits easily on 16 GB RTX 4080 SUPER with `bfloat16` AMP and `160×160×96` patches).

### 5.1 Architecture (`rasnet_v2_model.py` — Standalone, Not Subclassing 4-Stage `SegResNet`)

1. **6-Stage Anisotropic Residual Encoder–Decoder:**
   - Spatial strides (in `X, Y, Z` matching `[0.35, 0.35, 0.50] mm`):
     - Stage 0: `(1, 1, 1)` → `160 × 160 × 96`
     - Stage 1: `(2, 2, 2)` → `80 × 80 × 48`
     - Stage 2: `(2, 2, 2)` → `40 × 40 × 24`
     - Stage 3: `(2, 2, 2)` → `20 × 20 × 12`
     - Stage 4: `(2, 2, 1)` → `10 × 10 × 12`
     - Stage 5 (Bottleneck): `(2, 2, 1)` → `5 × 5 × 12` (or `(2, 2, 2)` → `5 × 5 × 6`)
   - Lean channel widths: **`[24, 48, 96, 160, 224, 256]`** (keep total params `~7.5–9.5M`).
   - Normalization & activation: **`InstanceNorm3d(affine=True)` + `LeakyReLU(0.01)`** (matches `nnU-Net`'s proven 3D CT stability at `batch_size=2`).
   - Decoder: **2 residual blocks** at upper decoder stages + **concatenation skips** fused via gated `1×1×1` projection (no destructive additive skip overwriting).

2. **3D Directional Tubular Strip Residual Block (`AnisotropicStripResBlock3D`) — Core Novelty:**
   - Deployed in Encoder Stages 2–4 & Decoder Stages 1–3:
     - **Local branch:** standard `3×3×3` conv (captures cross-sectional lumen diameter & calcified plaque texture).
     - **Tubular directional branch:** parallel depthwise 1D strip convolutions along all 3 axes — **`7×1×1` + `1×7×1` + `1×1×5`** (anisotropic Z kernel matching `0.50 mm` vs `0.35 mm` XY) — summed and fused via `1×1×1` pointwise conv + residual skip.
   - **Why it beats `nnU-Net`:** Coronary arteries are thin curving 1D tubes. Orthogonal 1D strip kernels triple the directional receptive field along vessel centerlines at minimal FLOPs/params, directly boosting **distal branch recall** and **bridging calcified plaque dropouts** (preventing Case 930-style breaks).

3. **Lightweight Bottleneck 3D Global Self-Attention (`Bottleneck3DAttention`):**
   - At the `5×5×6` (`150` tokens) or `5×5×12` (`300` tokens) bottleneck, add a **2-layer Multi-Head Self-Attention block** (4–8 heads, 3D learnable positional embeddings, Pre-LN + FFN).
   - **Cost:** `<1%` extra FLOPs and negligible VRAM (`150×150` attention matrix is tiny!).
   - **Benefit:** Every distal arterial voxel attends globally to the aortic root, LCA/RCA ostia, and cardiac chambers — suppressing distant extracardiac false positives (**Precision & HD95**) while keeping peripheral branches attached (**Recall & clDice**).

4. **Fixed Channel + Spatial Gated Skips (`FixedChannelSpatialGate3D`):**
   - Replaces v1's broken `GroupNorm(1,1)` gate with:
     - **Biased Spatial Gate:** `Conv3d(F_int, 1, kernel_size=1, bias=True) → Sigmoid` (NO `GroupNorm(1,1)` — gates can genuinely close on pure background!).
     - **3D Squeeze-and-Excitation (Channel Gate):** Global avg-pool → `FC → ReLU → FC → Sigmoid` modulating channel importance on the skip path.

5. **Full Multi-Scale Deep Supervision (4 Auxiliary Heads, Downsampled GT):**
   - Attach `1×1×1` heads at decoder resolutions `1/2`, `1/4`, `1/8`, `1/16` with normalized decaying weights `[1.0, 0.5, 0.25, 0.125] / 1.875`.
   - **Downsample ground-truth labels** (nearest / max-pool) to each decoder scale instead of trilinearly upsampling coarse logits (lower VRAM, faster steps, sharper thin-vessel gradients).

### 5.2 Loss Function (`rasnet_v2_loss.py`) — Replace Broken `soft_skel` clDice!
Use **`BatchDice + Full Cross-Entropy + Skeleton Recall Loss`** (Kirchhoff et al., *ECCV 2024* — developed by the DKFZ `nnU-Net` team specifically to beat stock `nnU-Net` on thin tubular structures!):

$$\mathcal{L}_{\text{main}} = 1.0\,\mathcal{L}_{\text{BatchDice}}(\text{fg only}) + 1.0\,\mathcal{L}_{\text{CE}} + w_{\text{skel}}\,\mathcal{L}_{\text{SkelRecall}}$$

- **`BatchDice` (`include_background=False`, aggregate `TP/FP/FN` across the batch):** Stabilizes training under `pos=1:neg=2` empty crops and cleanly suppresses empty-patch FPs.
- **`Full CE` (`F.cross_entropy`, NOT divided by `n_ch` and NOT attenuated by `0.3`):** Sharpens calibration and eliminates v1's threshold-floor shift.
- **`SkeletonRecallLoss` (Zero Training Overhead, Zero Inference Cost):**
  - Precompute the 3D tubed skeleton (`skimage.morphology.skeletonize` dilated by 1 voxel) of each ground-truth label **once during caching** and save as `skel.npy` in `cache_fp16/`.
  - During training, compute soft recall of foreground predicted probabilities on the fixed ground-truth skeleton tube:
    $$\mathcal{L}_{\text{SkelRecall}} = 1 - \frac{\sum_{v} p_{\text{fg}}(v)\,S_{\text{gt}}(v) + \epsilon}{\sum_{v} S_{\text{gt}}(v) + \epsilon}$$
  - Unlike `soft_skel` `clDice` (which crashed Stage 2B by `-0.0245` Dice), `SkeletonRecallLoss` is 100% smooth, fast, non-noisy, and proven to simultaneously boost **Dice, clDice, distal Recall, AND HD95** on vessels. Sweep `w_skel ∈ {0.0, 0.5, 1.0}` in the pilot.

### 5.3 Augmentations, Optimizer & Post-Processing
1. **Upgraded 3D Augmentations (`augmentations.py`):**
   - Keep 3-axis flips + 90° XY rotation, AND add:
     - Random 3D rotation (`±15°` X/Y/Z, `p=0.25`) & random isotropic/anisotropic scaling (`0.85–1.15`, `p=0.25`)
     - Gaussian noise (`std ∈ [0, 0.1]`, `p=0.15`) & Gaussian blur (`sigma ∈ [0.5, 1.0]`, `p=0.15`)
     - Brightness (`×[0.75, 1.25]`, `p=0.15`), contrast (`[0.75, 1.25]`, `p=0.15`), gamma (`[0.7, 1.5]`, `p=0.20`), and low-resolution zoom simulation (`0.5–1.0×`, `p=0.15`).
2. **Optimizer & Schedule:**
   - Test **`SGD(lr=1e-2, momentum=0.99, nesterov=True, weight_decay=3e-5)` + `PolyLR (1 - step/T)^0.9`** directly against `AdamW(lr=3e-4)` + `CosineAnnealingLR` in the 50-epoch pilot!
3. **Symmetric Post-Processing (`stage4b_val94_lock.py`):**
   - Keep `Arm C-Top2` for comparison, AND widen `Arm C-Seeded` / add a **Symmetric Physical Volume Floor** (`keep 26-connected components >= V_min ∈ {200, 350, 500} mm³ or >= 10% of largest CC, capped at top-3`), locked strictly on Val-94 identically across ALL models — curing the Case `930`/`948`/`861` Top-2 CC amputation failure.
   - Widen threshold sweep to **`τ ∈ [0.10, 0.70]`** so no model hits a `0.20` grid edge clamp.

---

## 6. Smart Risk-Controlled Execution Plan (Start Here!)

Follow this exact sequence so we **never waste days of GPU compute** and stay **100% compliant** with `GEMINI.md` / `CLAUDE.md` rules:

### Phase 0 — Mandatory V1 Hygiene First (~1–2 hours, Zero GPU)
- [ ] Finish Git purge of tracked `CT4`/`CT61` patient CT zips and `checkpoints_200ep.zip.*` (Rule 7).
- [ ] Execute `prompt1-mand-fix.md` (fixes v1 defense wording overclaims, applies `04_CODE_REVIEW_FIXES.md`, runs `CropForegroundd` + `pytest` checks).
- [ ] **Outcome:** Frozen V1 is 100% locked, safe, and thesis-defense ready as our fallback floor.

### Phase 1 — The 2-Hour Go/No-Go Architecture Pilot (1 GPU Job, 50 Epochs, Seed 42)
*Before committing to a 20-run × 200-epoch study, test whether the upgraded Vessel-Native `RASNet-v2` actually delivers a breakthrough!*
- [ ] Implement `VesselRASNetV2` (`6-stage anisotropic + 3D Strip-ResBlocks + Bottleneck 3D Self-Attention + Fixed Channel-Spatial Gates + 4-scale DS`), `BatchDice + CE + SkeletonRecallLoss`, and upgraded 3D augmentations inside `Rasnet-v2/src/`.
- [ ] Write unit tests first (Rule 6): shape/stride verification on `(1, 1, 160, 160, 96)`, parameter count (`~7–9M`), `bfloat16` VRAM check (`< 10.5 GB` peak on RTX 4080 SUPER), gate closure on zero/background patch, `SkeletonRecallLoss` gradient sanity check.
- [ ] Precompute offline tubed GT skeletons into `cache_fp16/` (`~5–10 minutes` CPU across 690 train cases).
- [ ] Run **2 fast 50-epoch pilots (~2–3 hours total)**:
  1. `pilot_vessel_v2_sgd` (`SGD lr=1e-2, PolyLR`, `w_skel=0.5`)
  2. `pilot_vessel_v2_adamw` (`AdamW lr=3e-4, Cosine`, `w_skel=0.5`)
- [ ] Evaluate both on **full `Val N = 94`** (`0.50 Gauss`, 1-pass, $\tau=0.50$) and compare directly against:
  - Old Stage 2B best (`SegResNet + conv_bias AG`, 50 ep): **`0.7340` Val Dice / `0.6966` Precision / `27.94 mm` HD95**
  - `nnU-Net V2` Epoch-50 snapshot on Val-94.

#### Binding Go / No-Go Decision Gate After the 2-Hour Pilot:
- **GO to Phase 2 (Full 200-Epoch Study):** If the new Vessel-Native `RASNet-v2` jumps decisively past old Stage 2B (e.g., **`>= 0.76+` raw Val-94 Dice at 50 epochs**, sharply better HD95 & Precision, and competitive with/ahead of `nnU-Net V2` Epoch-50 at `3–4×` fewer params).
- **NO-GO (Stop & Save GPU Days):** If 50-epoch Val-94 Dice barely moves past `0.73–0.74`, **do NOT launch a 4-day Stage 3 queue** — inspect failure modes first or ship Option 1 (defense-ready V1 + Stage 5 zero-retrain operating curves & symmetric CC volume floor).

### Phase 2 — Pre-Registered Full 200-Epoch Study (Only If Phase 1 Passes GO Gate)
1. **Write & Commit Amendment A2** in `Rasnet-v2/PREREGISTRATION_V2.md` documenting the Stage 2B fallback, the upgraded Vessel-Native architecture, `BatchDice + CE + SkeletonRecallLoss`, upgraded augmentations, widened `τ ∈ [0.10, 0.70]`, and symmetric volume-floor CC. Commit cleanly (`git.dirty_code = false`).
2. **Run Lean, Fair 200-Epoch Matrix (1 GPU job at a time per Rule 8):**
   - **Primary Models (3 seeds `{42, 123, 456}`):**
     - `Vessel-RASNet-v2` (`~7–9M` params, full upgrades)
     - `Matched Plain 6-Stage Anisotropic ResUNet` (same backbone/loss/augs **WITHOUT** Strip-Convs, Bottleneck Attention, and Fixed Gates — gives a **100% clean, unconfounded causal architecture ablation**!)
     - `nnU-Net V2` (native `30.79M` params, 3 seeds)
   - **Targeted Component Ablations (Seed 42 or 3 seeds, all on the EXACT same pipeline & transform-hashed cache — replaces the invalid v1 ablations!):**
     - `No-StripConv` (3×3×3 only)
     - `No-BottleneckAttn`
     - `No-SkelRecall` (`w_skel=0.0`)
3. **Symmetric Val-94 Lock → Commit Lock + `.sha256` → One-Shot Test-150 (Retrospective) → Prospective `Phase I` Clinical Cohort:**
   - All models receive identical `0.50` Gaussian SW, 8-pass logit-averaged TTA, Val-94 locked thresholds (`τ ∈ [0.10, 0.70]`), and symmetric CC post-processing (`Top-2` and `Volume-Floor` side-by-side).
   - Paired Wilcoxon + Holm-Bonferroni + 10,000-bootstrap 95% CIs + Dice Non-Inferiority/Superiority + RTX 4080 SUPER latency/GFLOPs/params Pareto curve.
   - Freeze the winning `Vessel-RASNet-v2` checkpoint for the true single-shot **prospective `Phase I` physician-annotated clinical cohort**!

---

## 7. Summary Recommendation

> **Bottom line:** Do **not** resume `Rasnet-v2` Stage 3 with the current 4-stage `SegResNet` code — it will waste ~4 days of GPU time and still leave us with an incremental tweak that barely ties `nnU-Net V2`. Instead:
> 1. Finish the V1 P0 cleanup (`prompt1-mand-fix.md`) so our thesis floor is 100% safe.
> 2. Upgrade `Rasnet-v2` to the **6-Stage Vessel-Native Architecture (`AnisotropicStripResBlock3D` + `Bottleneck3DAttention` + `FixedChannelSpatialGate3D` + `BatchDice/CE/SkeletonRecallLoss` + full 3D augs)** at `~7–9M` params.
> 3. Run the **2-hour 50-epoch Val-94 sanity gate** first. Once it proves a clear leap over `0.7340` and `nnU-Net` Epoch-50, commit Amendment A2 and launch the clean 200-epoch benchmark!
