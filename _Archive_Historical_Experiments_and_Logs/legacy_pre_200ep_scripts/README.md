# Legacy Pre-200-Epoch Scripts Archive

This directory contains outdated, pre-200-epoch training and evaluation scripts that were previously stored in `Phase3_Local_Integration/`.

### Reason for Archival:
These scripts represent older development phases (such as 70-epoch runs, curriculum stage experiments, and legacy evaluation runs with $0.8\text{ mm}$ spacing).

### Authoritative Active Scripts:
All authoritative benchmark training, evaluation, and logging for the 200-epoch runs reside strictly inside:
- **`H:\Thesis_Trainings\Q1_Publication_Package\matched_200ep_benchmark\`**

### Active Clinical Production Modules:
The active clinical inference and QCA pipeline scripts remain in `Phase3_Local_Integration/`:
- `production_qca_engine.py` (authoritative clinical QCA engine)
- `run_hospital_cohort_pipeline.py` (authoritative GPU inference pipeline)
- `dicom_to_nifti.py`
- `07_local_data_qc.py`
