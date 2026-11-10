# ✅ Final Research Checklist (CLAIM 2024 Compliant)
**Generated in Phase 17**  

- [x] **3D CAS dataset inspected**: All 200 cases audited in `3D_CAS_DATASET_AUDIT.md`.
- [x] **200 cases accounted for**: 200 image NIfTIs and 200 label NIfTIs verified in `3d_cas_dataset_inventory.csv`.
- [x] **Dataset metadata verified**: Formatted in `results/3d_cas_dataset_description.md` and `Table1`.
- [x] **Dataset provenance investigated**: Provenance traced to ImageCAS Cases 1–200.
- [x] **Overlap risk investigated**: Verified 115 train overlap, 19 val overlap, 66 unseen cases.
- [x] **Two-tier reporting established**: Disaggregated reporting prevents train-data leakage claims.
- [x] **External evaluation completed**: Evaluated with AMP FP16 and `sw_batch_size=4` on RTX 3060 Ti.
- [x] **RASNet evaluated**: Recorded in `results/3d_cas_rasnet_case_metrics.csv`.
- [x] **Baseline models evaluated**: SegResNet, nnU-Net, 3D U-Net evaluated and summarized in `Table3`.
- [x] **Primary-set Wilcoxon tests completed**: Paired tests with Holm-Bonferroni correction in `Table4`.
- [x] **External-set Wilcoxon tests completed**: Documented in `Table5`.
- [x] **95% CIs calculated**: Bootstrap percentile ($B=2000$) in `Table6`.
- [x] **Computational cost measured**: Measured latency, FLOPs, parameters in `Table7`.
- [x] **Failure case analysis generated**: Worst cases rendered in `results/failure_cases/`.
- [x] **Stenosis sanity check generated**: 10-case geometric luminal analysis in `results/stenosis_sanity_check/`.
- [x] **Component-wise ablation completed**: Primary dataset ablation compiled in `Table8`.
- [x] **Reproducibility information recorded**: Split seeds, hyperparameters, versions in `reproducibility.md`.
- [x] **Software versions recorded**: Python 3.14, PyTorch 2.14.0+cu126, MONAI 1.5.1, SimpleITK 2.5.6.
- [x] **Related-work research completed**: 2023–2025 literature synthesized in `Table9`.
- [x] **Clinical relevance research completed**: ACCURACY trial and CAD-RADS 2.0 cited in `clinical_relevance_research.md`.
- [x] **Clinical disclaimer included**: Research-only disclaimer prominent in all reports.
- [x] **All figures saved**: High-res vector plots (PNG & SVG) in `results/figures/`.
- [x] **All tables saved**: Tables 1 through 9 in Markdown and CSV in `results/final_tables/`.
- [x] **No fabricated values**: 100% computed from active models, checkpoints, and files.
- [x] **No unsupported overlap claims**: Overlap verified against `splits_final.json`.
