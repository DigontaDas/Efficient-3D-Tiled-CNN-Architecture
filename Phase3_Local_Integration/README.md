# Phase 3 Local Integration: Coronary Artery Segmentation Evaluation

This directory contains the scripts and configurations to run inference, evaluate outputs, and generate comparison assets for the coronary artery segmentation task on the ImageCAS dataset.

---

## 1. Setup and Environment

All scripts are configured to execute in the local virtual environment:
- **Executable**: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe`
- **Dependencies**: Python libraries including PyTorch (with CUDA 12.4), MONAI, SimpleITK, Matplotlib, Pandas, and NumPy (see [requirements_cuda.txt](../requirements_cuda.txt) for full list).

---

## 2. Directory Structure

The files inside `Phase3_Local_Integration/` are organized sequentially:

- **`dataset_paths.py`**: Shared utility containing absolute path mappings for local CT volumes and annotations.
- **`eval_utils.py`**: Core metrics computation script (Dice coefficient, IoU, Precision, Recall, and Hausdorff Distance 95) using SimpleITK.
- **`00_dataset_audit.py`**: Validates paths and verifies that all images and ground-truth labels exist.
- **`01_create_splits.py`**: Generates training, validation, and test splits, saving them to `splits_final.json`.
- **`03_eval_3dunet.py`**: Performs test-set evaluation for the 3D U-Net model.
- **`04_run_nnunet_inference.py`**: Triggers batch inference for the nnU-Net baseline.
- **`05_eval_nnunet.py`**: Evaluates nnU-Net prediction masks against the ground truth.
- **`06_run_segresnet_inference.py`**: Performs SegResNet sliding window inference on the GPU using MONAI's `Invertd` transform to guarantee physical space coordinate alignment.
- **`07_eval_segresnet.py`**: Multi-core parallel metric evaluation script for SegResNet test predictions.
- **`07_local_data_qc.py`**: Audits predictions to verify voxel orientation and bounding boxes.
- **`08_build_comparison_table.py`**: Compiles metrics from all models into `unified_comparison_table.csv`.
- **`08_plot_training_curves.py`**: Extracts loss curves and plots training/validation progress.
- **`09_plot_slice_overlays.py`**: Generates qualitative slice overlay visualizations showing ground-truth vs. predicted contours.
- **`10_plot_comparison_bar.py`**: Renders a grouped bar chart comparing overlap metrics and 95% Hausdorff Distance.

---

## 3. Running the Evaluation Pipeline

To execute the entire post-processing and analysis pipeline, run the scripts in sequence:

```bash
# Step 1: Run SegResNet Inference (requires GPU)
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe Phase3_Local_Integration\06_run_segresnet_inference.py

# Step 2: Evaluate SegResNet metrics (parallel CPU execution)
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe Phase3_Local_Integration\07_eval_segresnet.py

# Step 3: Compile final comparison table
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe Phase3_Local_Integration\08_build_comparison_table.py

# Step 4: Plot training curves
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe Phase3_Local_Integration\08_plot_training_curves.py

# Step 5: Plot qualitative slice overlays
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe Phase3_Local_Integration\09_plot_slice_overlays.py

# Step 6: Plot comparative bar charts
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe Phase3_Local_Integration\10_plot_comparison_bar.py
```

---

## 4. Key Outputs

All generated assets are stored in:
- **Metrics Table**: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\unified_comparison_table.csv`
- **Visual Artifacts**: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\`
  - `training_curves.png`
  - `slice_overlays.png`
  - `grouped_metrics_barchart.png`
- **Decision Log**: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\main_thesis_decision_log.md`

---

## 5. Clinical Validation Pipeline (Phase 8 — N=32 Hospital Cohort)

Scripts for the local hospital CCTA clinical stenosis validation against radiologist PACS measurements:

- **`run_new_cases_gpu.py`**: Runs 4-pass TTA AMP FP16 inference on RTX 3060 Ti for new local CCTA cases (automatic DICOM series selection + NIfTI conversion + GPU segmentation).
- **`refine_clinical_postprocess.py`**: Extracts 3D vessel centerlines via skeletonization, computes per-branch EDT radii, identifies minimum lumen diameter (MLD), and generates a 2-panel vessel overlay PNG + Bland-Altman agreement figure.
- **`hospital_cohort_automated_stenosis.csv`**: Raw AI-only stenosis outputs (backup, before radiologist comparison).

### Running a New Local Case
```bash
# Run GPU inference on a new case (e.g. CT95)
H:\Thesis_Trainings\.venv_cuda\Scripts\python.exe Phase3_Local_Integration\run_new_cases_gpu.py
# Then run postprocessing
H:\Thesis_Trainings\.venv_cuda\Scripts\python.exe Phase3_Local_Integration\refine_clinical_postprocess.py
```

---

## 6. Outlier Audit Pipeline (`outlier_audit/`)

When expanding the clinical cohort (e.g. from N=32 to N=150+), use this ordered pipeline to systematically identify and correct data errors without touching global algorithm parameters:

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `01_pull_top_outliers.py` | Sort CSV by `\|difference\|` descending | First step after any new batch run |
| `02_audit_outliers_detail.py` | Document root cause per outlier case | For each case with `\|diff\|` > 20% |
| `03_apply_case_corrections.py` | Apply CSV corrections + reconvert DICOM if needed | After root cause is confirmed |
| `04_reinfer_ct70.py` | Template to re-run single-case GPU inference | When DICOM series was wrong |
| `05_compute_diagnostic_metrics.py` | 2×2 confusion matrix + Sensitivity/Specificity/κ | After any full cohort correction cycle |

> [!IMPORTANT]
> **Do NOT modify global algorithm parameters** (percentile cutoffs, diameter thresholds) when fixing outliers. All corrections must be case-specific (target branch string, DICOM series, or local mask repair in 3D Slicer). Changing global parameters risks overfitting the postprocessing logic to the validation set.
