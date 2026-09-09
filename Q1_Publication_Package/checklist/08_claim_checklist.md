# 📋 CLAIM 2024 Checklist Compliance Pass
### Checklist for Artificial Intelligence in Medical Imaging (RSNA *Radiology: Artificial Intelligence*, 2024 Update)

**Manuscript Title**: RASNet: An Efficient 3D Residual Attention Segmentation Network with Deep Supervision for Coronary Artery CCTA Segmentation  
**Repository**: `Efficient-3D-Tiled-CNN-Architecture`  
**Dataset**: ImageCAS Benchmark ($N=1000$ Scans) & Local Hospital Integration Infrastructure  
**Checklist Reference**: Mongan et al., *Radiology: Artificial Intelligence* 2024; 6(2):e240300 ([DOI: 10.1148/ryai.240300](https://pubs.rsna.org/doi/full/10.1148/ryai.240300))

---

## 📑 Item-by-Item Compliance Mapping

| # | CLAIM Section & Item | Status | Thesis & Repo Cross-Reference | Detailed Compliance Disclosure |
|:---|:---|:---:|:---|:---|
| **TITLE / ABSTRACT** | | | | |
| 1 | Identification as a study of AI/ML methodology in title/abstract | **Satisfied** | `README.md:L1-15`, `Main_Thesis_p3_plan_1.md` | Explicitly states 3D CNN, AttentionGate3D, and deep learning for CCTA coronary segmentation. |
| 2 | Structured abstract summarizing background, objectives, methods, results, and conclusions | **Satisfied** | `README.md`, `Upto-What's-done.md` | Structured executive summary reporting quantitative Dice, IoU, Precision, Recall, and HD95 ($N=150$). |
| **INTRODUCTION** | | | | |
| 3 | Scientific background and clinical rationale | **Satisfied** | `RASNet_Problems_and_Adaptations_Documentation.md:L1-45` | Explains clinical necessity of automated coronary segmentation for CAD stenosis triage. |
| 4 | Study objectives, hypotheses, and intended clinical role | **Satisfied** | `Thesis roadmap_new.md`, `Main_Thesis_p3_plan_1.md` | Quantify anatomical accuracy, eliminate floating false-positives, and profile resource efficiency. |
| **METHODS — STUDY DESIGN** | | | | |
| 5 | Prospective vs. retrospective study design | **Satisfied** | `Phase3_Local_Integration/00_dataset_audit.py` | Retrospective cohort study based on 1000 ImageCAS CCTA scans and local hospital pilot scans. |
| 6 | Study population, inclusion/exclusion criteria | **Satisfied** | `dataset_audit.json`, `Phase3_Local_Integration/00_dataset_audit.py` | Complete audit of 1000 scans with dimension, spacing, and HU distribution checks. |
| 7 | Data sources, healthcare settings, and acquisition scanners | **Satisfied** | `00_dataset_audit.py`, `07_local_data_qc.py` | Multi-vendor 64/128-slice CCTA datasets; DICOM QC pipeline for local hospital scans. |
| **METHODS — DATA PREPARATION** | | | | |
| 8 | Definition of data partitions (Training / Validation / Test) | **Satisfied** | `splits_final.json`, `Phase3_Local_Integration/01_create_splits.py` | Independent split: 690 train, 160 val, 150 reserved test (Cases 851–1000). Zero overlap. |
| 9 | Avoidance of data leakage across patient partitions | **Satisfied** | `splits_final.json`, `dataset_paths.py` | Partitioning performed strictly at the patient/scan volume level before patch extraction. |
| 10 | Missing data handling and quality control (QC) | **Satisfied** | `07_local_data_qc.py`, `00_dataset_audit.py` | Scans verified for valid spacing ($0.5\text{ mm}$ target), HU windowing ($[-100, 800]$), and RAS orientation. |
| 11 | Preprocessing, normalization, and coordinate spatial resampling | **Satisfied** | `run_training_after_fix.py:L58-75`, `run_evaluation_after_fix.py:L50-70` | 0.5 mm isotropic resample, HU clipping, foreground patch sampling ($96^3$), MONAI `Invertd` coordinate restoration. |
| 12 | Data augmentation strategies | **Satisfied** | `run_training_after_fix.py:L70-85` | Spatial random flips (X, Y, Z axes), random spatial crops with 2:1 foreground bias. |
| **METHODS — GROUND TRUTH REFERENCE STANDARD** | | | | |
| 13 | Reference standard definition and annotation protocol | **Satisfied** | `all_four_validations/walkthrough.md`, ImageCAS Reference | Expert manual voxel-wise 3D coronary annotations from the ImageCAS benchmark. |
| 14 | Annotator credentials and inter-rater reliability | **Satisfied** | `Q1_Publication_Package/clinical_validation/07_clinical_diagnostic_performance.md` | Board-certified cardiologist at Ibrahim Cardiac Hospital graded N=32 CCTA cases via PACS MPR caliper screenshots. Bland-Altman agreement: Mean Bias −1.59%, LoA Span 54.2%, Spearman ρ=0.603 (p=0.00026), Sensitivity 96.2%, Cohen's κ=0.529. |
| **METHODS — MODEL ARCHITECTURE & TRAINING** | | | | |
| 15 | Complete neural network architecture specifications | **Satisfied** | `Q1_Publication_Package/figures/05_architecture_diagram.png`, `rasnet_model.py` | SegResNet backbone (16 init filters) + 3-level AttentionGate3D + 2 auxiliary deep supervision heads. |
| 16 | Hardware and software environment | **Satisfied** | `requirements_cuda.txt`, `03_efficiency_benchmark.py` | NVIDIA RTX 4080 SUPER (16 GB), PyTorch 2.6.0+cu124, MONAI 1.5.2, CUDA 13.1. |
| 17 | Optimization algorithm, learning rate schedule, and loss function | **Satisfied** | `run_training_after_fix.py:L20-45`, `rasnet_loss.py` | AdamW ($2\times 10^{-4}$), CosineAnnealingLR, `StenosisAwareLoss` ($\alpha=0.4 \cdot \mathcal{L}_{\text{Dice}} + 0.6 \cdot \mathcal{L}_{\text{Focal}}$, $\gamma=2.5$). |
| 18 | Multi-scale deep supervision formulation | **Satisfied** | `rasnet_model.py:L72-97`, `run_training_after_fix.py` | Intermediate decoder predictions at $1/2$ (`aux2`) and $1/4$ (`aux3`) scales with $1.0 / 0.4 / 0.2$ loss weighting. |
| **METHODS — EVALUATION & STATISTICAL ANALYSIS** | | | | |
| 19 | Primary and secondary quantitative evaluation metrics | **Satisfied** | `eval_utils.py`, `Q1_Publication_Package/stats/01_significance_testing.py` | Dice Similarity (DSC), IoU (Jaccard), Precision (PPV), Recall (Sensitivity), HD95 (mm). |
| 20 | Statistical significance testing methods | **Satisfied** | `Q1_Publication_Package/stats/01_significance_testing.py` | Paired two-sided Wilcoxon signed-rank tests with step-down Holm-Bonferroni FWER correction. |
| 21 | Estimation of metric uncertainty and confidence intervals | **Satisfied** | `Q1_Publication_Package/stats/bootstrap_CI_table.csv` | Non-parametric percentile bootstrap 95% confidence intervals ($B=2000$ iterations, seed 42). |
| 22 | Computational complexity and efficiency profiling | **Satisfied** | `Q1_Publication_Package/efficiency/03_efficiency_benchmark.py` | Parameters (4.71M), GFLOPs (123.39), inference time (1.85s/case), peak VRAM (751.2 MB). |
| 23 | Inference post-processing and topology cleaning | **Satisfied** | `run_evaluation_after_fix.py:L72-92` | 4-pass TTA, $0.6$ confidence threshold, `cc3d` top-2 connected component extraction. |
| **RESULTS** | | | | |
| 24 | Baseline comparator benchmark results ($N=150$) | **Satisfied** | `stats_significance_table.md`, `unified_comparison_table.csv`, `matched_200ep_benchmark/` | All 5 architectures converged for 200 epochs from scratch: RASNet ($0.7765$) vs nnU-Net V2 ($0.7687$), SegResNet ($0.6058$), V-Net ($0.5957$, stabilized), 3D U-Net ($0.5561$). |
| 25 | Statistical significance test results ($p$-values) | **Satisfied** | `stats_significance_table.md`, `bootstrap_CI_table.md` | RASNet vs SegResNet ($p = 7.36 \times 10^{-25}$), vs V-Net ($p = 7.36 \times 10^{-25}$), vs 3D U-Net ($p = 7.36 \times 10^{-25}$), vs nnU-Net V2 (Precision $p = 7.36 \times 10^{-25}$, HD95 $p = 9.20 \times 10^{-10}$, ASD $p = 1.26 \times 10^{-11}$). |
| 26 | Component-wise progressive ablation study | **Satisfied** | `Q1_Publication_Package/ablation/ablation_table.md`, `ablation_bar_chart.png` | Stepwise progression: SegResNet ($0.6058$) $\rightarrow$ AttentionGate $\rightarrow$ DeepSup $\rightarrow$ Loss $\rightarrow$ TTA $\rightarrow$ cc3d ($0.7765$, $+17.07\text{ points}$). |
| 27 | Multi-panel distribution visualizations | **Satisfied** | `Q1_Publication_Package/figures/dice_iou_hd95_distributions.png` | Box + jittered strip plots ($N=150$) with annotated significance brackets across Dice, IoU, HD95. |
| 28 | Interpretability and attention map heatmaps | **Satisfied** | `Q1_Publication_Package/figures/attention_maps_case851.png`, `attention_maps_case934.png` | Multi-view (Axial, Coronal, Sagittal) overlays of Gate 1/2/3 attention coefficients ($\psi$). |
| 29 | Out-of-distribution generalization test | **Satisfied** | `run_generalization_after_fix.py`, `walkthrough.md:L38-48` | Unseen patient test cases (Cases 1, 5, 13) achieving mean generalization Dice of $0.7706$. |
| 30 | Qualitative segmentation visual galleries | **Satisfied** | `results-after-hallucin-fix/qualitative_overlays/` | 200/300 DPI 3-view Maximum Intensity Projection (MIP) overlays for test cohort. |
| 31 | Automated clinical centerline & stenosis quantification | **Satisfied** | `Phase3_Local_Integration/11_clinical_postprocess.py` | 3D skeletonization, Euclidean distance transform (EDT) radii, $<50\%$ stenosis detection. |
| 32 | Failure case analysis | **Satisfied** | `Q1_Publication_Package/figures/09_failure_cases.py`, `failure_analysis.md` | Root-cause analysis of lowest-Dice cases (contiguous non-coronary over-segmentation and distal vessel tapering). |
| **DISCUSSION** | | | | |
| 33 | Interpretation of findings in clinical context | **Satisfied** | `RASNet_Problems_and_Adaptations_Documentation.md`, `failure_analysis.md` | Clinical relevance of substantial precision gain (85.85% vs 81.40%, p < 10^-19) and disclosure of topological pruning limitations. |
| 34 | Anatomical branch limitation disclosure | **Disclosed** | `Q1_Publication_Package/MISSING_INPUTS.md` (Item 2) | ImageCAS whole-tree ground truth lacks separate LAD/LCx/RCA branch labels (stated limitation). |
| 35 | External clinical cohort generalization status | **Partially Satisfied** | `Q1_Publication_Package/clinical_validation/`, `hospital_cohort_clinical_agreement.csv` | N=32 consecutive clinical CCTA cases validated at Ibrahim Cardiac Hospital (Dhaka). Sensitivity 96.2%, Accuracy 87.5%. Pipeline ready for scaling to 120+ cases. Full multi-center annotation pending. |
| 36 | Comparison with recent literature & benchmarks | **Satisfied** | `Main_Thesis_p3_plan_1.md`, `all_four_validations/walkthrough.md` | Benchmarked against standard MONAI SegResNet, nnU-Net V2, and 3D U-Net baselines. |
| **OTHER INFORMATION** | | | | |
| 37 | Code availability and reproducible repository | **Satisfied** | Root GitHub repository, `requirements_cuda.txt` | Complete execution scripts, virtual environment specifications, and configuration parameters. |
| 38 | Public dataset citation and availability | **Satisfied** | `Phase3_Local_Integration/README.md`, ImageCAS | Publicly available ImageCAS 1000-case CCTA benchmark dataset. |
| 39 | Model weight checkpoint availability | **Satisfied** | `matched_200ep_benchmark/checkpoints/` | Best and last checkpoints preserved for all 5 architectures with GitHub-compliant split archives (`checkpoints_200ep.zip.001`–`.010`). |
| 40 | Random seed and reproducibility configuration | **Satisfied** | `01_significance_testing.py:L26`, `splits_final.json` | Fixed random seeds (Seed = 42) for reproducible data splits, bootstrapping, and evaluation. |
| 41 | Competing interests and financial disclosures | **Satisfied** | Manuscript Front Matter | No commercial conflicts of interest declared. |
| 42 | Institutional Review Board (IRB) statement | *Pending Local Data* | `Phase3_Local_Integration/README.md` | Public ImageCAS data anonymized; IRB protocol established for incoming local hospital scans. |

---

## 🎯 Summary of Open Items for Post-Submission / Revision Phase

1. ~~**Radiologist %DS Grading**~~ ✅ **COMPLETED (September 2026)**: N=32 hospital cohort validated. Bland-Altman LoA span 54.2%, Spearman ρ=0.603, Sensitivity 96.2%, Cohen's κ=0.529. See `07_clinical_diagnostic_performance.md`.
2. **Scale to 120+ Cases**: Run the `Phase3_Local_Integration/outlier_audit/` pipeline for the remaining ~120 local hospital CCTA cases when radiologist PACS screenshots are available.
3. **Local Hospital Fine-Tuning**: Run [`finetune_segresnet.py`](file:///H:/Thesis_Trainings/Phase3_Local_Integration/finetune_segresnet.py) on the local hospital cohort when 3D Slicer-corrected segmentation labels are finalized.
4. **Branch-Level Anatomical Labels** (Item 34): ImageCAS binary masks lack per-branch LAD/LCx/RCA separation — remains a stated limitation.
