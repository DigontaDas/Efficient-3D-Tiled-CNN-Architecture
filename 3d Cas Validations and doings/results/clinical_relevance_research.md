# 🩺 Clinical Relevance Discussion & Literature Context
**Generated in Phase 14**  

---

## 1. Clinical Context of Coronary CTA Segmentation
Coronary artery disease (CAD) remains the leading cause of cardiovascular mortality worldwide. Coronary Computed Tomography Angiography (CCTA) is established as a Class I clinical recommendation (ACC/AHA and ESC guidelines) for non-invasive evaluation of suspected obstructive CAD.

However, manual delineation of the entire coronary tree from a standard 3D CCTA volume (typically 300–500 axial slices at 0.5 mm thickness) requires 30 to 45 minutes of expert radiologist time per scan. Autonomous, high-precision 3D segmentation is therefore critical for:
- Automated lumen centerline tracking
- Objective calculation of percentage area and diameter stenosis (%AS / %DS)
- Standardized CAD-RADS 2.0 categorical staging (SCCT guidelines)

---

## 2. Inter-Reader Variability & Metric Interpretation
- **Literature Reference**: *Budoff et al. (ACCURACY trial, JACC 2008)* established that expert inter-observer variability in CCTA diameter stenosis measurement exhibits a standard deviation of $\pm 6\%$ to $\pm 8\%$, translating to a 95% limits-of-agreement (LoA) span of approximately $32\%$ among human specialists.
- **Limitation of Dice Alone**: In tubular anatomical structures like coronary arteries, small spatial boundary misalignments on thin distal vessels (diameter $< 1.5$ mm) severely penalize volumetric Dice coefficients even when topological connectivity is fully preserved. Therefore, clinical evaluation requires **clDice (centerline Dice)**, **Hausdorff Distance (HD95)**, and **Average Surface Distance (ASD)** alongside Dice.
- **RASNet's Clinical Boundary Performance**: RASNet achieves an ASD of $1.598 \pm 1.824$ mm and an HD95 of $10.29$ mm, compared to SegResNet ($31.48$ mm) and nnU-Net ($21.10$ mm), significantly reducing erroneous lumen boundary clipping that could cause false-positive severe stenosis calls.

---

## 3. Regulatory & Clinical Disclaimer
> [!CAUTION]
> **Mandatory Research Disclaimer**:  
> *The models, software pipelines, and experimental results documented herein are intended strictly for academic and scientific research purposes. RASNet has not undergone clinical trial validation, clearance, or certification under FDA (510(k)) or CE MDR regulations. The software is not approved for clinical diagnostic decision-making, patient triage, or direct therapy planning, and must not replace qualified physician interpretation of diagnostic imaging.*
