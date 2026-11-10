# 🔍 PAPER CLAIM AUDIT & DEFENSIVE POSITIONING

**Target Document**: `results/PAPER_CLAIM_AUDIT.md`  
**Audit Scope**: Research manuscripts, thesis chapters, READMEs, and presentation slides  
**Standard**: CLAIM 2024 / Q1 Biomedical Journal Reviewer Defense Protocol  

---

## 1. Master Claim Verification Matrix

| # | Current Manuscript Claim | Audit Classification | Forensic Evidence & Audit Rationale | Recommended Scientifically Defensible Replacement |
| :-: | :--- | :---: | :--- | :--- |
| **1** | *"RASNet outperforms all baselines across all datasets and metrics."* | ❌ **UNSUPPORTED** | Empirically false. nnU-Net V2 beats RASNet in Recall on ImageCAS (0.8106 vs 0.7016, $p < 10^{-24}$). SegResNet beats RASNet in Dice on 3D CAS Unseen (0.7546 vs 0.7397, $p < 10^{-4}$) and in HD95 (12.90 mm vs 14.51 mm). | *"RASNet achieves state-of-the-art precision, topological connectivity, and boundary accuracy, outperforming baselines in false-positive suppression while yielding distal vessel recall to nnU-Net."* |
| **2** | *"RASNet achieves state-of-the-art performance in 3D coronary artery segmentation."* | 🟡 **PARTIALLY SUPPORTED** | Supported on the primary ImageCAS benchmark for Precision (88.01%), HD95 (10.29 mm), and clDice (0.8592), but Dice difference vs nnU-Net (0.7765 vs 0.7687) is not statistically significant ($p = 0.2262$). | *"RASNet achieves competitive state-of-the-art segmentation with superior boundary precision and topological fidelity on the 150-case ImageCAS test benchmark."* |
| **3** | *"RASNet is statistically significantly superior to all competing models."* | ❌ **UNSUPPORTED** | Paired Wilcoxon signed-rank test between RASNet and nnU-Net V2 on Dice yields raw $p = 0.1152$ and Holm-Bonferroni adjusted $p = 0.2262$ ($p > 0.05$). Furthermore, nnU-Net is significantly superior to RASNet in Recall ($p < 10^{-24}$). | *"RASNet demonstrates statistically significant improvements in precision, boundary distance (HD95/ASD), and centerline Dice over all baselines, while matching nnU-Net V2 in volumetric overlap (Dice p = 0.226)."* |
| **4** | *"RASNet generalizes exceptionally to independent external cohorts."* | 🟡 **PARTIALLY SUPPORTED** | RASNet generalizes well (0.7397 Dice on 3D CAS, resisting the collapse seen in nnU-Net), but SegResNet achieves higher external Dice (0.7546) and lower external HD95 (12.90 mm). Furthermore, 3D CAS is an ImageCAS-derived cohort, not an independent international hospital dataset. | *"RASNet demonstrates robust out-of-distribution transfer on 66 unseen CCTA volumes, maintaining ~0.74 Dice and 85.5% precision where self-configuring baselines suffer severe domain collapse."* |
| **5** | *"RASNet is computationally efficient and real-time capable."* | ✅ **SUPPORTED** | With 4.71M parameters and 123.39 GFLOPs, RASNet is 72% smaller and 72% less compute-intensive than nnU-Net V2, requiring only 751 MB peak VRAM. Single-pass latency is 0.45s and TTA latency is 1.85s. | *"With a compact 4.71M parameter footprint and 123.39 GFLOPs complexity, RASNet delivers high-precision segmentation within 1.85 seconds on standard clinical GPU hardware."* |
| **6** | *"RASNet detects coronary artery stenosis."* | ❌ **UNSUPPORTED (If applied to ImageCAS/3D CAS)** / 🟡 **PARTIALLY SUPPORTED (On Local Cohort)** | ImageCAS and 3D CAS provide only binary lumen annotations, not clinical stenosis ground truth. The 3D CAS analysis is an in-silico geometric cross-sectional narrowing sanity check. Only the audited N=21 Ibrahim Cardiac Hospital cohort has radiologist CAD-RADS stenosis badges, showing 76.2% accuracy and 95.2% adjacent-tier agreement. | *"Geometric profiling demonstrates that RASNet preserves focal luminal narrowing sites without artificial bridging; quantitative clinical stenosis grading was validated on an audited N=21 hospital cohort."* |
| **7** | *"Ablation experiments prove that all claimed components improve performance."* | ✅ **SUPPORTED** | Authentic 200-epoch lab retraining confirms that Attention Gates drop HD95 by 8.04 mm, Deep Supervision adds +0.0095 Dice, and StenosisAwareLoss adds +0.0146 Dice and +4.27% Precision. Raw architecture reaches 0.7681 Dice without post-processing. | *"Ablation experiments confirm that Attention Gates, Deep Supervision, and StenosisAwareLoss synergistically enhance boundary delineation and false-positive suppression, with TTA and cc3d providing final refinement."* |
| **8** | *"RASNet is clinically accurate for CAD-RADS diagnosis."* | 🟡 **PARTIALLY SUPPORTED** | On the audited N=21 clinical cohort, the production QCA engine achieved 76.2% exact CAD-RADS accuracy, 95.2% adjacent-tier accuracy, and Spearman $\rho = 0.5410$ ($p = 0.0113$), but sample size is limited ($N=21$) and sensitivity is 78.9% with 50.0% specificity. | *"In a pilot clinical validation cohort (N=21), automated QCA profiling demonstrated moderate correlation with radiologist-adjudicated stenosis grades (Spearman ρ = 0.541, p = 0.011) and 95.2% adjacent CAD-RADS agreement."* |

---

## 2. Defensive Framing Guidelines for Publication

1. **Acknowledge the Sensitivity/Specificity Trade-off Openly**:
   - Rather than hiding nnU-Net's higher recall (0.8106 vs 0.7016), frame it as a deliberate clinical engineering design choice: *"While nnU-Net optimizes for broad voxel sensitivity, resulting in higher distal recall at the expense of lower precision (73.9%) and higher boundary error (21.1 mm), RASNet prioritizes high positive predictive value (88.0%) and tight boundary adherence (10.29 mm HD95), which is essential for accurate lumen cross-sectional quantification."*
2. **Acknowledge SegResNet's External Strength**:
   - *"On the 66-case unseen cohort, baseline SegResNet equipped with cc3d filtering achieved marginally higher volumetric overlap (0.7546 vs 0.7397 Dice), while RASNet maintained superior precision (85.5% vs 81.2%), demonstrating complementary strengths in vascular reconstruction."*
3. **Never Claim Stenosis 'Detection' Without Clinical Provenance**:
   - Always clearly distinguish in-silico geometric area stenosis profiling on binary datasets from clinical QCA grading against radiologist badges.
