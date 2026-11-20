# RASNet-v2 Decision & Architecture Summary (Plain-English Guide)

A simple overview of [`V2_DECISION_AND_ARCHITECTURE_GUIDE.md`](V2_DECISION_AND_ARCHITECTURE_GUIDE.md) — what the numbers mean, why `nnU-Net V2` still edged out V1, and what to do next.

---

## 1. The Big Choice (3 Options)

Before spending days running the GPU, there are **3 choices**:

- **Option 1 — Stop at V1 and polish it (0 GPU hours):**
  Finish the cleanup tasks, keep V1 as-is, and optionally do small non-training tweaks. Good if the defense deadline is very close, **but** V1 still only ties `nnU-Net V2` (`0.8039` vs `0.8059` Dice)—it won't beat it in every way.
- **Option 2 — Run the current `Rasnet-v2` code as-is (3–5 GPU days) → DO NOT DO THIS:**
  The current `Rasnet-v2` folder still uses the same shallow 4-layer `SegResNet` backbone, weak augmentations, and a noisy loss. If you run it for 4 days right now, it will almost certainly just tie `nnU-Net` again.
- **Option 3 — Upgrade the architecture first, test for 2 hours, then run (RECOMMENDED):**
  Redesign `RASNet-v2` specifically for **tubular blood vessels**, test it in a fast **2-hour pilot** first to prove it actually jumps ahead, and *only then* run the full multi-day training.

---

## 2. Why `nnU-Net` Still Won in V1 (and Why Current V2 Is Still Stuck)

### A. The "Top-2 Cleanup" Trick Helped `nnU-Net` More Than `RASNet`
- Without cleanup (`R2`), **RASNet actually beat nnU-Net** (`0.7863` vs `0.7760`).
- `nnU-Net` finds more tiny, faint artery tips, **but** it also leaves floating noise blobs outside the heart. When Top-2 cleanup deleted those floating blobs, `nnU-Net` got a massive free boost (`0.8059`).
- Meanwhile, on `RASNet`, if a thin artery had a small break in it, Top-2 cleanup thought the broken piece was a "3rd blob" and **accidentally deleted real artery branches** (for example, on Case 930 alone, that cost RASNet `-0.21` Dice).

### B. Current `Rasnet-v2` Still Has 6 Hidden Handicaps
1. **Too nearsighted:** `SegResNet` only has 4 layers (`8×` zoom-out), so it only sees a tiny ~1.5 cm cube at a time. `nnU-Net` has 6 layers and sees the whole heart patch (`~5.6 cm`).
2. **Weak decoder:** Uses 1 block per layer and adds skip features together instead of concatenating them, which blurs thin vessel edges.
3. **Not built for tubes:** Its attention gates are just `1×1×1` dots—nothing in the network actually understands long, curving 3D tubes.
4. **Loss issues:** It uses per-patch Dice (noisy when 2/3 of patches are mostly empty), weak voxel penalty, and a `clDice` (`soft_skel`) loss that **actually hurt** the 50-epoch score (`-0.0245`).
5. **Weak augmentations:** Only flips and 90° turns—missing 3D rotation, zooming, noise, blur, and contrast jitter that `nnU-Net` uses.
6. **Optimizer:** Only tried `AdamW`; `nnU-Net` uses `SGD + PolyLR`, which usually trains sharper 3D medical UNets.

---

## 3. How Option 3 Fixes Everything (The New Design)

Instead of tweaking `SegResNet`, we build a **compact `~7–9M`-parameter vessel-native network** (still **3.5–4× smaller** than `nnU-Net`'s `30.8M`):

1. **6 Layers Deep (Full Heart View):** Matches `nnU-Net`'s receptive field so arteries stay connected from the aorta to the distal tips.
2. **Tube-Shaped "Strip" Convolutions (`7×1×1`, `1×7×1`, `1×1×5`):** `nnU-Net` only uses generic `3×3×3` cubes. Long, thin 1D strip kernels trace curving arteries and bridge across calcified blockages much better.
3. **Tiny Global Self-Attention at the Bottom Layer:** At the deepest layer (`5×5×6`), the feature map is so small that full 3D attention costs almost **zero** speed/memory, yet lets every distal artery see the whole heart and kill distant false positives.
4. **Fixed Attention Gates:** Removes the `GroupNorm(1,1)` bug that prevented v1 gates from closing on background tissue, and adds channel gating.
5. **Better Loss (`BatchDice + Full CE + Skeleton Recall Loss`):**
   - Drops the `soft_skel` loss that hurt Stage 2B.
   - Uses **Skeleton Recall Loss** (invented by the `nnU-Net` team in 2024): precomputes the vessel centerline once before training, then rewards the model for hitting that centerline—boosting thin-vessel recall and `clDice` with zero slowdown.
6. **Full 3D Augmentations + `SGD` Test + Better Cleanup Rule:**
   - Adds 3D rotation, scaling, noise, blur, and contrast/gamma.
   - Replaces hard "keep only Top-2 blobs" with a **physical size floor** (keep any connected vessel piece bigger than `V_min` up to Top-3), so real 3rd branches don't get deleted.

---

## 4. The Safe Step-by-Step Game Plan

1. **Phase 0 (1–2 hours, no GPU):** Finish cleaning up V1 (remove the tracked patient CT zips from Git and apply `prompt1-mand-fix.md`) so the V1 thesis baseline is 100% safe.
2. **Phase 1 (2-hour Go/No-Go GPU Test):** Build the new design and run **just two fast 50-epoch pilots** (`SGD` vs `AdamW`).
   - If it jumps way past the old `0.7340` 50-epoch score (e.g., `0.76+` and beats `nnU-Net` at epoch 50) → **GO** to full 200-epoch training!
   - If it doesn't → **STOP** immediately so you don't waste 4 days of GPU time.
3. **Phase 2 (Full 200-Epoch Study, only if Phase 1 passes):** Update `PREREGISTRATION_V2.md` (Amendment A2), train the new `RASNet-v2` + matched plain backbone + `nnU-Net V2` (3 seeds) + clean ablations, and evaluate on Test-150 and the prospective `Phase I` clinical cohort.
