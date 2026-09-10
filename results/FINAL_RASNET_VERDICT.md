# VERDICT: RASNet IS GENERALLY BETTER BUT NOT DOMINANT

**Target Document**: `results/FINAL_RASNET_VERDICT.md`  
**Audit Standard**: Rigorous Evidence-Based Research Verification  
**Evaluation Scope**: Primary ImageCAS Test Set ($N=150$) + External 3D CAS Pure Unseen Cohort ($N=66$)  

---

## 1. The Definitive Forensic Verdict

Based strictly on verified empirical evidence, **RASNet is generally better than the baseline models across a majority of clinical and structural dimensions, but it is NOT universally dominant.**

The claim that *"RASNet outperforms all baselines across all datasets and metrics"* is **empirically false and scientifically indefensible**. A peer reviewer evaluating the full benchmark would immediately identify two critical exceptions:
1. On the **Primary ImageCAS Test Set**, **nnU-Net V2 achieves significantly higher vessel recall** (**0.8106 vs 0.7016**, $p = 1.84 \times 10^{-25}$) and higher centerline tree completeness (**0.9111 vs 0.8018**), while the Dice difference between RASNet and nnU-Net (**0.7765 vs 0.7687**, $\Delta = +0.0078$) is **NOT statistically significant** (paired Wilcoxon raw $p = 0.1152$, Holm-Bonferroni adjusted $p = 0.2262$).
2. On the **External 3D CAS Unseen Dataset**, **SegResNet ranks #1 in volumetric overlap and boundary error**, statistically significantly defeating RASNet in Dice (**0.7546 vs 0.7397**, $p = 1.35 \times 10^{-05}$) and HD95 (**12.90 mm vs 14.51 mm**, $p = 1.35 \times 10^{-05}$).

However, **RASNet possesses genuine, statistically verified strengths that make it a superior overall architecture for clinical precision and boundary fidelity**:
- **Dominant Precision Across All Datasets**: RASNet achieves **0.8801 Precision on Primary** (+14.1% over nnU-Net, +14.9% over SegResNet, $p < 10^{-24}$) and **0.8553 / 0.8427 Precision on External 3D CAS** (+3.1% to +4.3% over SegResNet, $p < 10^{-10}$). It almost completely eliminates false-positive arterial hallucinations.
- **Superior Boundary & Topological Quality In-Domain**: RASNet achieves a 2× lower HD95 error than nnU-Net (10.29 mm vs 21.10 mm, $p = 4.60 \times 10^{-10}$) and 3× lower than SegResNet (31.48 mm, $p = 2.05 \times 10^{-20}$), while achieving the highest clDice (0.8592 vs 0.8201, $p = 3.96 \times 10^{-09}$).
- **Severe Domain-Shift Resistance**: While nnU-Net experiences catastrophic collapse on 3D CAS (Dice dropping by 38.8% to 0.4704 and HD95 exploding to 48.70 mm), RASNet maintains high cross-domain transfer (0.7397 Dice), proving robust against out-of-distribution scanner variations.
- **High Computational Efficiency**: Consuming only **4.71M parameters** and **123.39 GFLOPs**, RASNet matches or beats models requiring 3.5× to 9.7× more parameters (nnU-Net at 16.5M, V-Net at 45.6M).

---

## 2. Answers to the 9 Core Forensic Questions

### 1. Is RASNet best on the primary dataset?
**MOSTLY YES, but with an important trade-off.**
- RASNet ranks **#1 in Dice (0.7765)**, **#1 in Precision (0.8801)**, **#1 in clDice (0.8592)**, **#1 in ASD (1.60 mm)**, and **#1 in IoU (0.6396)**.
- However, it ranks **#4 in Recall (0.7016)** and **#4 in Tree Recall (0.8018)**, yielding to nnU-Net V2 (0.8106 and 0.9111).
- The Dice lead over nnU-Net (+0.0078) is not statistically significant ($p = 0.2262$).

### 2. Is RASNet best on 3D CAS?
**NO. SegResNet ranks #1 overall on 3D CAS.**
- SegResNet achieves higher Dice (**0.7546 vs 0.7397**), higher IoU (**0.6117 vs 0.5929**), lower HD95 (**12.90 mm vs 14.51 mm**), and higher Recall (**0.7136 vs 0.6701**).
- RASNet ranks **#2 in Dice and HD95**, but retains **#1 in Precision (0.8553 / 0.8427 vs SegResNet 0.8119)**.

### 3. Is RASNet statistically significantly better?
**YES for Precision, HD95, and clDice in-domain; NO for Primary Dice vs nnU-Net; and NO on External 3D CAS against SegResNet.**
- On Primary: Statistically superior to SegResNet, V-Net, and 3D U-Net across all metrics ($p < 10^{-10}$). Statistically superior to nnU-Net on Precision ($p = 1.84 \times 10^{-25}$), HD95 ($p = 4.60 \times 10^{-10}$), ASD ($p = 6.29 \times 10^{-12}$), and clDice ($p = 3.96 \times 10^{-09}$), but NOT on Dice ($p = 0.2262$), and statistically inferior to nnU-Net on Recall ($p = 1.84 \times 10^{-25}$).
- On 3D CAS: SegResNet is statistically significantly better than RASNet in Dice ($p = 1.35 \times 10^{-05}$) and HD95 ($p = 1.35 \times 10^{-05}$), while RASNet is statistically significantly better than SegResNet in Precision ($p = 1.48 \times 10^{-11}$).

### 4. Is RASNet more robust?
**YES, in preventing catastrophic false-positive spikes and severe boundary divergence.**
- On Primary, RASNet caps maximum HD95 at **52.24 mm** (vs SegResNet 76.85 mm, V-Net 98.56 mm).
- On External 3D CAS, RASNet caps maximum HD95 at **54.32 mm** (vs SegResNet 73.32 mm, nnU-Net 81.53 mm, V-Net 89.61 mm).
- RASNet's minimum precision across both datasets never drops below **0.6724**, whereas SegResNet drops to **0.4399** and V-Net drops to **0.4439**.

### 5. Does RASNet generalize better?
**BETTER THAN nnU-Net and V-Net, but WORSE than SegResNet.**
- SegResNet generalizes best (Dice $+1.03\%$ on external unseen data).
- RASNet generalizes strongly (Dice $-3.7\%$ on raw architecture, preserving 0.74 Dice).
- nnU-Net generalizes very poorly without recalibration (Dice $-38.8\%$ collapse down to 0.4704).

### 6. Is RASNet computationally reasonable?
**YES, EXTREMELY.**
- At **4.71M parameters** and **123.39 GFLOPs**, it is 72% smaller and 72% less computationally demanding than nnU-Net V2, operating at a lightweight 751 MB VRAM footprint.

### 7. Are the architectural ablations supportive?
**YES.**
- Matched 200-epoch retraining from scratch confirms that Attention Gates ($-8.04\text{ mm}$ HD95), Deep Supervision ($+0.0095$ Dice), and StenosisAwareLoss ($+0.0146$ Dice, $-8.16\text{ mm}$ HD95, $+4.27\%$ Precision) drive the majority of performance gains.
- The raw network architecture alone achieves **0.7681 Dice and 13.45 mm HD95**, matching nnU-Net without any TTA or `cc3d`.
- Post-processing (TTA + `cc3d`) accounts for the remaining refinement (+0.0084 Dice, +0.0674 Precision, -3.16 mm HD95).

### 8. Are there any baselines that beat RASNet on important metrics?
**YES, TWO BASELINES BEAT RASNet ON IMPORTANT METRICS:**
1. **nnU-Net V2 beats RASNet in Recall on the Primary Dataset** (0.8106 vs 0.7016, a $+10.9\%$ advantage) and Tree Recall (0.9111 vs 0.8018).
2. **SegResNet beats RASNet in Dice and HD95 on External 3D CAS** (0.7546 vs 0.7397 Dice, and 12.90 mm vs 14.51 mm HD95).

### 9. Is the claim "RASNet outperforms all baselines" actually defensible?
**NO. IT IS SCIENTIFICALLY INDEFENSIBLE.**
- A claim of unqualified superiority will be rejected by rigorous Q1 reviewers.
- **The Defensible Claim**: *"RASNet provides state-of-the-art precision, topological connectivity, and boundary fidelity with a lightweight 4.7M parameter footprint, resisting out-of-distribution domain collapse, while trading off distal vessel recall to nnU-Net on in-domain data and volumetric overlap to SegResNet on external transfer."*
