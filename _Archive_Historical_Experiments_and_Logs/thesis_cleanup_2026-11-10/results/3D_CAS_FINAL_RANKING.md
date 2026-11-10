# 🌐 3D CAS EXTERNAL DATASET RANKING & GENERALIZATION VERDICT

**Target Document**: `results/3D_CAS_FINAL_RANKING.md`  
**Dataset**: Pure Unseen External Cohort ($N=66$ Cases, `dia_0.nii`, Zero Training Exposure)  
**Evaluation Standard**: Strict Symmetry ($\tau=0.50$ & $\tau=0.60$, identical `cc3d` filtering, zero TTA)  

---

## 1. External Cohort Metric Ranking Table

| Metric | #1 Rank | #2 Rank | #3 Rank | #4 Rank | #5 Rank | #6 Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice (DSC) ↑** | **SegResNet** (0.7546) | **RASNet @ 0.50** (0.7397) | **RASNet @ 0.60** (0.7376) | **V-Net** (0.7119) | **3D U-Net** (0.5322) | **nnU-Net** (0.4704) |
| **Precision (PPV) ↑** | **RASNet @ 0.60** (0.8553) | **RASNet @ 0.50** (0.8427) | **SegResNet** (0.8119) | **V-Net** (0.8065) | **nnU-Net** (0.7597) | **3D U-Net** (0.6019) |
| **Recall / Sensitivity ↑**| **SegResNet** (0.7136) | **RASNet @ 0.50** (0.6701) | **RASNet @ 0.60** (0.6595) | **V-Net** (0.6501) | **3D U-Net** (0.4833) | **nnU-Net** (0.3720) |
| **95% Hausdorff (HD95) ↓**| **SegResNet** (12.90 mm) | **3D U-Net** (13.80 mm) | **RASNet @ 0.50** (14.51 mm) | **RASNet @ 0.60** (15.12 mm) | **V-Net** (19.41 mm) | **nnU-Net** (48.70 mm) |
| **Average Surface Dist (ASD) ↓**| **SegResNet** (2.22 mm) | **RASNet @ 0.50** (2.33 mm) | **RASNet @ 0.60** (2.40 mm) | **3D U-Net** (2.40 mm) | **V-Net** (3.53 mm) | **nnU-Net** (12.66 mm) |
| **Centerline Dice (clDice) ↑**| **SegResNet** (0.8305) | **RASNet @ 0.50** (0.8248) | **RASNet @ 0.60** (0.8215) | **V-Net** (0.7876) | **3D U-Net** (0.6717) | **nnU-Net** (0.4844) |
| **Tree Recall ($T_{sens}$) ↑**| **SegResNet** (0.8065) | **RASNet @ 0.50** (0.7699) | **RASNet @ 0.60** (0.7595) | **V-Net** (0.7277) | **3D U-Net** (0.6089) | **nnU-Net** (0.3525) |
| **Intersection over Union (IoU) ↑**| **SegResNet** (0.6117) | **RASNet @ 0.50** (0.5929) | **RASNet @ 0.60** (0.5899) | **V-Net** (0.5610) | **3D U-Net** (0.3639) | **nnU-Net** (0.3225) |

---

## 2. Forensic Answers to the 6 Core External Questions

### Q1: Does RASNet still rank #1 externally?
**NO. SegResNet ranks #1 in overall segmentation performance on 3D CAS.**
- SegResNet achieves the highest Dice (**0.7546**), highest IoU (**0.6117**), lowest HD95 (**12.90 mm**), lowest ASD (**2.22 mm**), highest clDice (**0.8305**), and highest Recall (**0.7136**).
- RASNet ranks **#2** in Dice (0.7397 @ 0.50 / 0.7376 @ 0.60) and HD95 (14.51 mm @ 0.50 / 15.12 mm @ 0.60).
- RASNet retains the **#1 ranking exclusively in Precision** (0.8553 @ 0.60 / 0.8427 @ 0.50).

### Q2: Does RASNet retain its advantage?
**PARTIALLY.**
- **Precision Advantage Retained**: RASNet retains a statistically significant Precision advantage (+4.3% at $\tau=0.60$ and +3.1% at $\tau=0.50$ over SegResNet, $p < 10^{-10}$).
- **Boundary & Overlap Advantage Lost**: The primary dataset HD95 advantage (10.29 mm vs 31.48 mm) evaporates once `cc3d` is applied symmetrically to SegResNet (SegResNet HD95 drops to 12.90 mm, beating RASNet's 14.51 mm).

### Q3: Does performance degrade compared with the primary dataset?
**YES, for RASNet, V-Net, and nnU-Net:**
- **RASNet**: Dice drops from **0.7765** to **0.7376** ($\Delta = -0.0389$, a $-5.0\%$ degradation). HD95 increases from **10.29 mm** to **15.12 mm** ($+4.83\text{ mm}$).
- **V-Net**: Dice drops from **0.7474** to **0.7119** ($\Delta = -0.0355$, a $-4.7\%$ degradation).
- **nnU-Net V2**: Suffers **catastrophic collapse**: Dice plunges from **0.7687** down to **0.4704** ($\Delta = -0.2983$, a $-38.8\%$ collapse) and HD95 explodes from **21.10 mm** to **48.70 mm**.

### Q4: Do any baselines generalize better?
**YES. SegResNet generalizes best.**
- SegResNet's Dice on the unseen external cohort is **0.7546**, which actually exceeds its primary test performance (**0.7469**, $\Delta = +0.0077$).
- SegResNet's HD95 dramatically improves on the unseen cohort (from **31.48 mm** on primary down to **12.90 mm**) because `cc3d` filtering removed the isolated extracardiac noise clusters that previously penalized it.

### Q5: Is the ranking consistent between Primary and External datasets?
**NO. The model ranking changes fundamentally across domains:**
- **Primary ImageCAS Ranking (Dice)**: RASNet (0.7765) > nnU-Net (0.7687) > V-Net (0.7474) > SegResNet (0.7469) > 3D U-Net (0.5561).
- **External 3D CAS Ranking (Dice)**: SegResNet (0.7546) > RASNet (0.7397) > V-Net (0.7119) > 3D U-Net (0.5322) > nnU-Net (0.4704).
- The most dramatic reversals are:
  1. SegResNet jumping from #4 on primary to #1 on external.
  2. nnU-Net collapsing from #2 on primary to #5 on external.

### Q6: Is the difference statistically significant?
**YES, with bidirectional statistical significance:**
- **SegResNet > RASNet in Dice**: Paired Wilcoxon raw $p = 3.69 \times 10^{-06}$, Holm-Bonferroni adjusted **$p = 1.35 \times 10^{-05}$** (Statistically significant in favor of SegResNet).
- **SegResNet > RASNet in HD95**: Paired Wilcoxon raw $p = 2.70 \times 10^{-06}$, adjusted **$p = 1.35 \times 10^{-05}$** (Statistically significant in favor of SegResNet).
- **SegResNet > RASNet in Recall**: Adjusted **$p = 1.02 \times 10^{-09}$** (Statistically significant in favor of SegResNet).
- **RASNet > SegResNet in Precision**: Adjusted **$p = 1.48 \times 10^{-11}$** (Statistically significant in favor of RASNet).
- **RASNet > nnU-Net across all metrics**: Adjusted **$p < 10^{-11}$** (Statistically significant in favor of RASNet).
