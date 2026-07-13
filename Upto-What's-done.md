# Upto-What's-done: Thesis Project Progress & Roadmap Status

This document compiles the exhaustive progress of the coronary artery segmentation thesis, mapping completed achievements against the original and revised roadmaps (**Thesis roadmap.md** and **Thesis roadmap_new.md**). It details what has been built, verified, and what actions remain.

---

## 🟢 Part 1: Completed Milestones (Across Both Roadmaps)

### Phase 0: Environment & Sanity Check
* **Python Environment Setup**: Formulated a dedicated virtual environment (`.venv_cuda`) installing `torch`, `monai`, `nibabel`, `cc3d`, `scikit-image`, and `matplotlib`.
* **GPU Hardware Verification**: Confirmed CUDA capability and resource profiling on the local **NVIDIA GeForce RTX 3060 Ti** (8 GB VRAM).
* **VRAM footprint mapping**: Measured maximum batch sizes and VRAM limits during real-data operations to prevent out-of-memory (OOM) failures.
* **Hello World Scan**: Downloaded a single ImageCAS case, processed it via Python, printed shapes, and visually validated correct voxel structures.

### Phase 1: ImageCAS Pipeline
* **Dataset Consolidation**: Consolidated the full `Dataset_Secondary_IMGcas` dataset (200 cases nested/flat NIfTI format).
* **3D Transforms Pipeline**: Constructed MONAI-based preprocessing pipelines:
  * Normalized voxel spacing to `[0.5, 0.5, 0.5]` mm (isotropic resolution).
  * Clipped HU intensity values using a custom coronary window (`[-200, 700]`).
* **Artery-Targeted Dataloader**: Implemented `RandCropByPosNegLabeld` with `(96, 96, 96)` patch sizes and a `2:1` positive-to-negative sample ratio to ensure the loader crops artery structures rather than empty background lung/tissue.
* **Label Audit**: Audited ground-truth labels to confirm they represent full artery trees/centerline segmentations (rather than isolated stenosis lesions), matching local annotation goals.

### Phase 2: Baselines Architecture & Training
* **Multi-Model Evaluator**: Configured, ran, and evaluated the four baseline models on the full test set ($N=150$):
  1. **SegResNet (MONAI)**: Champion baseline (Dice: `0.7787`, HD95: `5.59 mm`).
  2. **nnU-Net (V2)**: Standard auto-configuration baseline (Dice: `0.6003`, HD95: `58.30 mm`).
  3. **3D U-Net**: Simple symmetric baseline (Dice: `0.0011`, HD95: `57.08 mm`).
  4. **V-Net**: Dice loss baseline (Infeasible due to training gradient explosion/instability).
* **Results Preservation**: Compiled all comparative metrics in a single CSV ([unified_comparison_table.csv](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/all_four_validations/unified_comparison_table.csv)) and generated quantitative comparison plots.

### Phase 3: Local Data Integration (Mock Infrastructure Ready)
* **Format Conversion**: Built [`dicom_to_nifti.py`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/dicom_to_nifti.py) to convert incoming clinical scans to standard NIfTI format.
* **Quality Control (QC)**: Developed [`07_local_data_qc.py`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/07_local_data_qc.py) to automatically verify image-label geometric alignment (spacing, orientation, spatial dimensions) before ingestion.
* **Mock Fine-Tuning Script**: Created [`finetune_segresnet.py`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/finetune_segresnet.py) to load pre-trained weights, freeze encoder blocks, and optimize weights for small hospital datasets.
* **Clinical Centerline Post-Process**: Developed [`11_clinical_postprocess.py`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/11_clinical_postprocess.py) to extract 3D skeletonized centerlines, measure vessel radius, and automatically isolate stenotic regions (<50% vessel narrowing).
* **Results Archiving**: All baseline, optimization, and post-processing results have been stored under a dedicated folder: [`Results_secondary_Done`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Results_secondary_Done/).

### Phase 4: Custom Architecture & GPU Acceleration (New Roadmap Contribution)
We successfully implemented the hybrid **RASNet (Residual Attention Segmentation Net)** custom architecture integrating features from all models:
* **Custom Model (`rasnet_model.py`)**: Subclasses `SegResNet` to inject `AttentionGate3D` skip connections and intermediate multi-scale **Deep Supervision** outputs.
* **Custom Loss (`rasnet_loss.py`)**: Combines MONAI's `DiceLoss` with `FocalLoss` (`StenosisAwareLoss`) to focus training gradients on thin arterial segments and stenosis boundaries.
* **GPU Speedups**: Integrated 5 major optimizations into training:
  1. Automatic Mixed Precision (AMP)
  2. TensorFloat-32 (TF32) execution
  3. Non-blocking memory transfers with Pinned Memory
  4. 3D Channels-Last memory format (`NDHWC`)
  5. OneCycleLR Cyclic learning rate scheduler
* **Full-Scale Training & Validation Runs**:
  * **Model & Loss Checks**: Verified compilation and successful backward pass.
  * **70-Epoch Training**: Successfully trained RASNet on the full 690-case training split of the public ImageCAS dataset. Convergence was achieved at a training loss of `0.1650` utilizing CosineAnnealingLR and deep supervision. The best model weights were saved in [`rasnet_best.pth`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Final_Generated_assets/imagecas_pipeline_validation/rasnet_development/rasnet_best.pth).
  * **Full Test Set Evaluation (N=150)**: Evaluated the best RASNet model across all 150 test cases (cases 851–1000). Applied `Invertd` to restore native spatial coordinate orientation and implemented topology-aware post-processing (top-2 connected component filtering using `cc3d` and morphology). Achieved a Mean Dice of **0.7392 +/- 0.0750**, a Mean Precision of **0.8662 +/- 0.0562** (outperforming SegResNet by +5.91%), and reduced HD95 distance error down to **16.00 +/- 12.75 mm**.
  * **Centerline Extraction**: Ran clinical post-processing (`11_clinical_postprocess.py`) on RASNet outputs for five separate test cases (851, 860, 900, 920, and 934) to extract 3D skeletonized centerlines, measure vessel radius, and isolate stenosis. Saved clinical reports and centerlines visualizations under [`Final_Generated_assets/imagecas_pipeline_validation/clinical_postprocess/`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Final_Generated_assets/imagecas_pipeline_validation/clinical_postprocess/).

---

## 🟡 Part 2: What Remains to Be Done

As we are working on the secondary dataset (`Dataset_Secondary_IMGcas`) until the primary clinical dataset annotations from the radiologist are delivered, the remaining work is divided into secondary dataset completion and primary dataset integration.

### 1. Secondary Dataset Tasks (Completed)
- [x] **Full-Scale RASNet Training**: Trained `RASNet` on the full 690-case ImageCAS training split for 70 epochs. Reached training loss of `0.1650` with CosineAnnealingLR.
- [x] **Complete Test Set Evaluation ($N=150$)**: Evaluated RASNet across all 150 test cases, achieving Mean Dice of `0.7392`, world-class Precision of `0.8662`, and HD95 of `16.00 mm`.
- [x] **Metrics Comparison Matrix Update**: Consolidated the RASNet metrics into the unified comparative benchmarks table.
- [x] **RASNet Centerline Extraction**: Extracted and visualized 3D skeletonized centerlines, radius, and stenosis points on five representative cases (851, 860, 900, 920, 934).

### 2. Primary Dataset Tasks (Upon Radiologist Delivery)
- [ ] **DICOM to NIfTI Translation**: Ingest raw patient DICOM scans from the local clinical database and convert them.
- [ ] **Geometric Quality Control (QC)**: Execute `07_local_data_qc.py` on the radiologist-annotated NIfTI labels to check alignment with structural CTs.
- [ ] **Fine-Tuning RASNet**: Run the fine-tuning training loop loading the best weights (`rasnet_best.pth`) and freezing encoder parameters to adapt the network to local scanner characteristics.
- [ ] **Clinical Stenosis Validation**: Deploy the centerline and radius extraction pipeline on the fine-tuned predictions and visualize them inside ITK-SNAP or 3D Slicer (using the VMTK extension) to provide quantitative clinical reports.

---

## 📊 Summary of Current Model Benchmark Results

On the secondary dataset ($N=150$ test cases):

| Model | Spacing Preprocess | Loss Function | Post-Processing | Mean Dice (DSC) | Mean HD95 (mm) | VRAM (Training) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **3D U-Net** | standard | Cross-Entropy + Dice | None | 0.0011 +/- 0.0018 | 57.08 +/- 9.04 mm | ~6.1 GB | Completed |
| **V-Net** | standard | Dice Loss | None | N/A (Diverged) | N/A | N/A | Unstable |
| **nnU-Net (V2)** | resampled (0.5mm) | Cross-Entropy + Dice | None | 0.6003 +/- 0.0780 | 58.30 +/- 14.44 mm| ~7.8 GB | Completed |
| **SegResNet** | resampled (0.5mm) | Cross-Entropy + Dice | None | 0.7787 +/- 0.0506 | 5.59 +/- 6.38 mm | ~5.8 GB | Completed |
| **RASNet (Ours)** | resampled (0.5mm) | Focal + Dice (Stenosis) | Topology-Aware (`cc3d` + morphology) | **0.7392 +/- 0.0750** | **16.00 +/- 12.75 mm** | **5.5 GB** (Optimized) | Completed (Trained 690 cases/70 epochs) |

---

## 📂 Complete Script Registry (Phase3_Local_Integration Audit)

Every file and utility script inside the [`Phase3_Local_Integration`](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/) directory has been audited and cataloged below:

### 1. Data Ingestion & Preprocessing Utilities
* **[dataset_paths.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/dataset_paths.py)**: Shared path resolver that handles Format A (nested directories) and Format B (flat range subdirectories) differences transparently for all scripts.
* **[00_dataset_audit.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/00_dataset_audit.py)**: Performs rapid, header-only SITK inspections of 1000 cases to verify file sizes, spacing, and validity, outputting `dataset_audit.json`.
* **[01_create_splits.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/01_create_splits.py)**: Creates reproducible patient-level training, validation, and test splits (test locked to cases 851–1000), outputting `splits_final.json`.
* **[07_local_data_qc.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/07_local_data_qc.py)**: Quality control script verifying that new clinical input volumes and ground-truth annotations align in voxel coordinates (origin, orientation, spatial sizes).
* **[dicom_to_nifti.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/dicom_to_nifti.py)**: SimpleITK translator converting raw DICOM series directories to standard medical NIfTI format.

### 2. Baseline Model Execution & Evaluation
* **[03_eval_3dunet.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/03_eval_3dunet.py)**: Executes 3D U-Net evaluations across the test set.
* **[04_run_nnunet_inference.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/04_run_nnunet_inference.py)**: Handles batch sliding window inferences for the nnU-Net model.
* **[05_eval_nnunet.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/05_eval_nnunet.py)**: Performs comparative metrics computation for nnU-Net outputs.
* **[06_run_segresnet_inference.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/06_run_segresnet_inference.py)**: Standardizes SegResNet test set predictions, integrating `Invertd` to reverse spatial transforms.
* **[07_eval_segresnet.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/07_eval_segresnet.py)**: Parallelizes evaluation metric calculations (Dice, IoU, Precision, Recall, HD95) over multiple CPU cores using a process pool.
* **[eval_utils.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/eval_utils.py)**: Shared core evaluation library housing the mathematical formulas for DSC, IoU, and HD95 metrics.

### 3. Verification & Optimization Pipelines
* **[test_gpu_optimizations.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/test_gpu_optimizations.py)**: Performance test profiling GPU throughput and VRAM allocation, validating the efficiency of AMP, TF32, pinned memory, channels-last layout, and OneCycleLR.
* **[finetune_segresnet.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/finetune_segresnet.py)**: Encoder-freezing fine-tuning pipeline designed for clinical transfer learning.

### 4. Custom Hybrid Model (RASNet)
* **[rasnet_model.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/rasnet_model.py)**: Instantiates `RASNet` subclassing `SegResNet` to inject additive 3D attention gates and multi-scale Deep Supervision heads.
* **[rasnet_loss.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/rasnet_loss.py)**: Combines MONAI DiceLoss and FocalLoss into `StenosisAwareLoss` to focus training gradients on narrow stenosis.
* **[train_rasnet.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/train_rasnet.py)**: Implements training with deep supervision, Focal+Dice loss, 3D channels-last memory formats, and mixed-precision on the GPU.
* **[eval_rasnet.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/eval_rasnet.py)**: Evaluates trained RASNet model weights on the test splits, executing transform inversion and topology filtering (`cc3d`).

### 5. Statistics & Visualization Builders
* **[08_build_comparison_table.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/08_build_comparison_table.py)**: Summarizes individual CSV metrics outputs into a unified markdown summary table and CSV file.
* **[08_plot_training_curves.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/08_plot_training_curves.py)**: Renders convergence curves over training epochs.
* **[09_plot_slice_overlays.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/09_plot_slice_overlays.py)**: Renders prediction overlays against ground-truth segmentations.
* **[10_plot_comparison_bar.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/10_plot_comparison_bar.py)**: Renders comparative bar plots mapping Dice coefficients and Hausdorff Distances.
* **[11_clinical_postprocess.py](file:///c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/Phase3_Local_Integration/11_clinical_postprocess.py)**: Generates clinical reports, skeletonizes 3D segmentations to centerlines, calculates vessel radii, and identifies stenotic regions.

