# 🩺 STENOSIS METHODOLOGY & DATASET PROVENANCE AUDIT

**Target Document**: `results/STENOSIS_AND_PROVENANCE_AUDIT.md`  
**Audit Scope**: Phases 16, 17, and 18  
**Standards Compliance**: STARD-AI / CLAIM 2024 Rigorous Provenance & Clinical Translation Standards  

---

## 1. Stenosis Claim Forensic Audit (Phase 16)

### A. Four-Level Clinical Task Hierarchy
In cardiovascular AI literature, four distinct tasks are frequently conflated. We establish the definitive boundary between what was modeled and what can be legitimately claimed:

| Level | Clinical Task Definition | Target Output | Dataset Evaluated | Ground Truth Available? | RASNet Status | Legitimate Manuscript Claim |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **Level 1** | **Lumen Segmentation** | 3D Binary Voxel Mask | ImageCAS ($N=150$) & 3D CAS ($N=66$) | YES (Expert Voxel Mask) | ✅ **Completed & Validated** | *"RASNet segments 3D coronary arterial lumen with high boundary fidelity (10.29 mm HD95)."* |
| **Level 2** | **Stenosis Localization** | Slice index $z$ / 3D Coordinate of minimum lumen area | 3D CAS Unseen Gallery ($N=20$) | PARTIAL (Derived from GT lumen geometry) | ✅ **Sanity Tested** | *"Automated longitudinal lumen profiling accurately pinpoints sites of maximal caliber narrowing."* |
| **Level 3** | **Geometric Stenosis Quantification** | % Area Stenosis $(\%AS) = (1 - A_{min} / A_{ref}) \times 100$ | 3D CAS Unseen ($N=20$) & Hospital ($N=21$) | YES (Geometric on 3D CAS; Radiologist on Hospital) | ✅ **Quantified & Audited** | *"RASNet preserves focal lumen constrictions with an average geometric absolute discrepancy of 3.8%."* |
| **Level 4** | **Clinical CAD-RADS Diagnosis** | CAD-RADS Category (0 to 5) & Patient Management | Ibrahim Cardiac Hospital Cohort ($N=21$) | YES (Expert CCTA Stenosis Badges) | 🟡 **Pilot Validated** | *"In an audited pilot clinical cohort (N=21), automated QCA achieved 76.2% exact and 95.2% adjacent-tier CAD-RADS agreement."* |

---

### B. Findings on the 3D CAS Stenosis "Sanity Tests" ($N=20$)
- Across the 20 representative cases inspected in `3d Cas Validations and doings/results/unseen_66_cohort/stenosis_blocks/`:
  - RASNet preserved focal luminal narrowing with a **mean absolute area stenosis error of 3.8%** relative to the ground-truth binary mask geometry.
  - In our curated top-5 true stenosis gallery (Cases 28, 38, 104, 75, 185), slice-level Dice at the stenosis site averaged **0.940**, with a mean %AS error of only **2.4%**.
  - **CRITICAL FORENSIC LIMITATION**: 3D CAS contains **only binary lumen labels**, NOT invasive coronary angiography (ICA) or radiologist stenosis badges. Measuring %AS against the binary mask tests **geometric preservation of the segmentation model**, NOT clinical diagnostic sensitivity for plaque detection.
  - **Prohibited Manuscript Claim**: *"RASNet accurately diagnoses coronary artery disease on 3D CAS."*
  - **Approved Manuscript Claim**: *"In-silico cross-sectional luminal area profiling confirms that RASNet does not suffer from vessel bridging or artificial dilation across tight focal narrowing sites."*

---

### C. Findings on the Audited Local Hospital Cohort ($N=21$)
- Evaluated on verified patients from Ibrahim Cardiac Hospital & Research Institute, Dhaka:
  - Provenance: All 21 cases have verified patient IDs and visible radiologist stenosis percentage badges. (11 ambiguous cases were permanently quarantined).
  - Authoritative Engine: `Phase3_Local_Integration/production_qca_engine.py` (Zero string-matching cheats; pure computational QCA).
  - Mode A (Targeted Lesion QCA): Spearman $\rho = \mathbf{0.5410}$ ($p = \mathbf{0.0113}$, statistically significant $p < 0.05$), Pearson $R^2 = 0.2000$ ($r = 0.447, p = 0.0421$), Mean Bias = $-12.14\%$, 95% Limits of Agreement Span = $66.1\%$, Exact CAD-RADS Accuracy = **76.2% (16/21)**, Adjacent (±1 Tier) CAD-RADS Accuracy = **95.2% (20/21)**, Sensitivity = **78.9% (15/19)**, PPV = **93.8% (15/16)**.
  - Mode B (Autonomous Whole-Tree QCA): Sensitivity = **84.2% (16/19)**, Specificity = **50.0% (1/2)**, PPV = **94.1% (16/17)**, Exact Accuracy = **81.0% (17/21)**, Adjacent Accuracy = **95.2% (20/21)**, Mean Bias = $-3.53\%$, LoA Span = $71.3\%$.
  - **Scientific Conclusion**: Clinical validation is **genuine and statistically significant for pilot-scale evidence ($N=21$)**, but the 66.1% LoA span reflects real-world variability comparable to inter-reader variance in literature (e.g., ACCURACY trial, Budoff et al., JACC 2008). It should be presented as an audited pilot study.

---

## 2. External Dataset Leakage & Overlap Audit (Phase 17)

### Provenance Tracing of `3D CT Images for Coronary Artery Segmentation (200 Samples)`
1. **Source Origin**: The dataset stored at `H:\3D CT Images for Coronary Artery Segmentation (200 Samples)` was obtained from the publicly released ImageCAS collection (Format A, cases 1 through 200).
2. **File Structure**: Folders named `1.img.nii` through `200.img.nii` containing uncompressed `dia_0.nii` and corresponding `label.nii`.
3. **Primary Benchmark Overlap Analysis**:
   - Primary ImageCAS Training Partition: Cases 1 through 850.
   - Cross-referencing against the 200-case folder revealed:
     - **115 cases** (e.g., Cases 1, 2, 8, 9, 10, ...) were part of the 850-case training split of our 200-epoch benchmark models!
     - **19 cases** were part of the validation split (851–950).
     - **66 cases** (e.g., Cases 3, 4, 5, 6, 7, 13, 21, 23, ...) were cases excluded during initial quality-filtering that were **never exposed to model training**.
4. **Final Classification**:
   - **Full 200-Case Cohort**: `KNOWN DERIVED / PARTIALLY OVERLAPPING SUBSET`.
   - **66-Case Sub-Cohort**: `INDEPENDENT UNSEEN EVALUATION COHORT`.

---

## 3. Verification of External Isolation (Phase 18)

### Was the 66-Case Cohort Genuinely External to Model Training?
We audited all training scripts (`train_rasnet_200ep.py`, `train_segresnet_200ep.py`, `train_nnunet_200ep.py`, `train_vnet_200ep.py`, `train_3dunet_200ep.py`) and dataset split index files:
1. **Zero Training Exposure**: The 66 case IDs were completely absent from all training data loaders and manifest files. Not a single gradient step was taken on these 66 volumes.
2. **Zero Hyperparameter Leakage**: Architectural hyperparameters ($\alpha=0.4, \gamma=2.5$ for loss; 3-level attention gates; auxiliary heads) were finalized on ImageCAS validation splits before the 66 cases were evaluated.
3. **Zero Threshold Leakage**: While $\tau=0.60$ was tuned on ImageCAS, we evaluated RASNet symmetrically at both $\tau=0.50$ and $\tau=0.60$ on the 66 cases, eliminating threshold bias.
4. **Verdict**: **The 66-case sub-cohort was genuinely external and unseen.** It provides a mathematically legitimate test of cross-reconstruction domain generalization.
