You are working on the final experimental/research phase of our coronary artery segmentation project using **RASNet** and baseline models.

## IMPORTANT CONTEXT

Our new external dataset is located at:

`H:\3D CT Images for Coronary Artery Segmentation (200 Samples)`

This dataset contains **200 cases** and is intended to be used as a **completely external test dataset**.

My local GPU is:

- GPU: NVIDIA RTX 3060 Ti
- VRAM: 8 GB

The goal is NOT to redesign the model or casually experiment. The goal is to produce the missing experimental evidence, tables, figures, metadata, sanity checks, and reproducibility information required for the final research report/paper.

### CRITICAL RULES

1. Do NOT fabricate any result, metric, dataset metadata, paper result, hardware specification, or clinical benchmark.
2. Do NOT modify the primary experimental results unless explicitly necessary.
3. Keep the **primary dataset** and **3D CAS Images dataset** completely separate in reporting.
4. Treat 3D CAS Images as an **external dataset**.
5. Determine whether any cases/images in 3D CAS Images overlap with, originate from, or are repackaged from our primary dataset. If this cannot be conclusively established, explicitly report that it could not be verified rather than guessing.
6. Record every experiment in reproducible files/logs.
7. Save all generated predictions, metrics, visualizations, and tables in clearly named directories.
8. Before starting expensive experiments, inspect the existing repository and determine:
   - existing model implementations
   - existing checkpoints
   - existing preprocessing
   - existing evaluation scripts
   - existing metric implementations
   - existing dataset loaders
   - existing training configuration
   - existing baseline results
   - existing ablation results
9. Reuse existing checkpoints whenever possible. Do NOT retrain models unnecessarily.
10. Because the GPU is an RTX 3060 Ti with 8 GB VRAM, use memory-efficient inference/training settings and process cases sequentially where appropriate.
11. Never silently change preprocessing between the primary dataset and external dataset. Document every necessary adaptation.
12. Every headline metric must have a clearly defined calculation method.

---

# PHASE 1 — INSPECT THE PROJECT FIRST

Before running anything, inspect the complete project/repository.

Create a report:

`EXPERIMENT_AUDIT.md`

Include:

- model architecture location
- RASNet implementation
- SegResNet implementation
- nnU-Net implementation
- 3D U-Net implementation
- checkpoints available
- dataset loaders
- preprocessing pipeline
- post-processing
- evaluation scripts
- metric implementations
- existing results
- existing ablation experiments
- training configurations
- dependency/environment files
- CUDA/PyTorch versions if detectable

Also identify whether the existing implementation already supports:

- Dice
- HD95
- IoU
- Precision
- Recall/Sensitivity
- Specificity
- ASSD
- volume metrics
- connected-component postprocessing
- test-time augmentation
- stenosis-aware loss
- attention gates
- deep supervision

Do not start a large experiment until this audit is completed.

---

# PHASE 2 — UNDERSTAND THE NEW 3D CAS DATASET

Inspect:

`H:\3D CT Images for Coronary Artery Segmentation (200 Samples)`

Determine:

- number of cases
- exact file structure
- image files
- segmentation/label files
- file extensions
- NIfTI structure
- image dimensions
- voxel spacing
- orientation
- intensity characteristics
- available metadata
- whether labels are binary or multi-class
- whether coronary artery labels are available
- whether stenosis annotations are available
- whether case IDs are available

Generate:

`3D_CAS_DATASET_AUDIT.md`

Also generate a machine-readable summary such as:

`results/3d_cas_dataset_inventory.csv`

with one row per case if feasible.

At minimum include:

- case ID
- image path
- label path
- image dimensions
- voxel spacing
- label presence
- label voxel count
- preprocessing status

---

# PHASE 3 — DATASET DESCRIPTION TABLE

Create the information required for the final paper's dataset description table.

Required fields:

| Field                                  | Value                   |
| -------------------------------------- | ----------------------- |
| Dataset                                | 3D CAS Images           |
| Number of cases                        | 200                     |
| Format                                 | NIfTI                   |
| Modality                               | Verify from source/data |
| Source/curator                         | Verify                  |
| License                                | Verify                  |
| Official/source link                   | Verify                  |
| Dataset description                    | Verify                  |
| Annotation type                        | Verify                  |
| Any preprocessing                      | Document                |
| Potential overlap with primary dataset | Investigate carefully   |

VERY IMPORTANT:

Determine whether 3D CAS Images is:

- an independent dataset,
- a repackaged version of another dataset,
- a subset of ImageCAS,
- derived from ImageCAS,
- or otherwise overlapping with the primary dataset.

Search the dataset documentation/source and relevant publications if necessary.

If it is a repackaged/subset dataset, explicitly document this.

If overlap cannot be conclusively determined, say:

"Dataset provenance/overlap could not be independently verified from the available documentation."

Never claim "zero overlap" without evidence.

Save the final verified metadata to:

`results/3d_cas_dataset_description.md`

and

`results/3d_cas_dataset_description.csv`

---

# PHASE 4 — EXTERNAL VALIDATION ON ALL 200 CASES

Run RASNet on the entire 3D CAS dataset.

IMPORTANT:

The external dataset must NOT be used for:

- training
- model selection
- hyperparameter tuning
- threshold selection
- checkpoint selection

It is strictly an external evaluation dataset.

If labels are available, calculate the same headline metrics used on the primary dataset.

At minimum investigate:

- Dice
- HD95
- IoU
- Precision
- Recall/Sensitivity
- Specificity
- ASSD if already used by the project

Use exactly the same metric definitions as the primary evaluation wherever possible.

Generate:

`results/3d_cas_rasnet_case_metrics.csv`

with one row per case.

Also generate:

`results/3d_cas_rasnet_summary.csv`

containing:

- mean
- standard deviation
- median
- minimum
- maximum
- 95% confidence interval

for every headline metric.

Use an appropriate confidence interval method and document it.

For example, bootstrap 95% CI is acceptable if appropriate, but explicitly document the method and number of bootstrap samples.

---

# PHASE 5 — BASELINE MODELS ON 3D CAS

Evaluate the available baseline models on the same external dataset:

1. RASNet
2. SegResNet
3. nnU-Net
4. 3D U-Net

Use equivalent preprocessing/evaluation conditions wherever technically possible.

Do not create an unfair comparison.

Generate:

`results/3d_cas_model_comparison.csv`

Columns should include:

- Model
- Dice
- Dice 95% CI
- HD95
- HD95 95% CI
- IoU
- IoU 95% CI
- Precision
- Recall
- any other headline metrics used in the paper

Also produce a formatted Markdown table:

`results/table_3d_cas_model_comparison.md`

---

# PHASE 6 — PRIMARY DATASET + 3D CAS STATISTICAL SIGNIFICANCE

The final paper requires:

## Wilcoxon signed-rank tests

Compare:

- RASNet vs SegResNet
- RASNet vs nnU-Net
- RASNet vs 3D U-Net

Run these separately on:

### Dataset A

Primary test set

### Dataset B

3D CAS external dataset

Use paired per-case metrics where the comparison is valid.

For each comparison and each headline metric, report:

- Wilcoxon statistic
- p-value
- sample size
- effect direction
- corrected p-value if multiple-comparison correction is appropriate

Clearly document the multiple-testing correction.

Create:

`results/statistical_significance_primary.csv`

`results/statistical_significance_3d_cas.csv`

and a final formatted table:

`results/table_statistical_significance.md`

Do NOT perform a statistical test if the pairing assumptions are invalid. Explain the limitation instead.

---

# PHASE 7 — 95% CONFIDENCE INTERVALS

For EVERY headline metric calculate 95% confidence intervals on BOTH:

1. Primary test set
2. 3D CAS external dataset

Metrics should include whichever are actually headline metrics in our paper, especially:

- Dice
- HD95
- IoU
- Precision
- Recall/Sensitivity

Do not invent metrics simply to make the table larger.

Create:

`results/table_confidence_intervals.md`

and machine-readable CSV files.

---

# PHASE 8 — COMPUTATIONAL COST TABLE

Create a computational cost comparison for:

- RASNet
- SegResNet
- nnU-Net
- 3D U-Net

Measure/report:

- Number of parameters
- FLOPs
- training time
- inference time per scan
- GPU
- VRAM usage

Hardware:

NVIDIA RTX 3060 Ti, 8 GB VRAM.

IMPORTANT:

Clearly distinguish:

- measured values
- estimated values
- values taken from official/model documentation

For inference time:

- use a clearly defined timing protocol
- warm up GPU before timing
- measure multiple cases
- report mean ± standard deviation if possible
- exclude/include preprocessing explicitly and consistently

For FLOPs:

Document input shape/resolution and calculation methodology because FLOPs depend on input size.

Create:

`results/table_computational_cost.md`

and

`results/computational_cost.csv`

---

# PHASE 9 — FAILURE CASE ANALYSIS

Find the **2–3 worst RASNet cases specifically from the 3D CAS external dataset**.

Do NOT select these from the primary dataset.

Rank cases using an appropriate failure criterion, preferably considering:

- lowest Dice
- highest HD95
- major anatomical segmentation errors

For each selected case save:

1. Case ID
2. Dice
3. HD95
4. relevant additional metrics
5. ground truth visualization
6. predicted segmentation visualization
7. overlay visualization
8. short explanation of the failure

Look for failure causes such as:

- missed coronary branches
- false-positive regions
- discontinuous vessel segmentation
- small distal vessels
- severe stenosis
- unusual anatomy
- image quality
- artifacts
- poor contrast
- preprocessing mismatch

Do NOT claim a cause unless visually supported.

Save figures under:

`results/failure_cases/`

Create:

`results/failure_cases_3d_cas.md`

These figures are intended for direct use in the final paper.

---

# PHASE 10 — STENOSIS SANITY TEST

This is VERY IMPORTANT.

We want approximately 10 representative test cases from the available data to perform a sanity check specifically for stenosis localization/segmentation.

First determine what stenosis annotation actually exists in our data/model pipeline.

DO NOT invent stenosis ground truth.

Select approximately 10 representative cases where stenosis information/annotation can be meaningfully inspected.

For each case:

- identify the stenosis region/location
- generate the relevant segmentation/prediction visualization
- show the coronary artery
- show the predicted segmentation
- mark/highlight the stenotic region if ground-truth stenosis annotation exists
- if stenosis is inferred from vessel geometry rather than directly annotated, clearly label it as an inferred/sanity-check result
- do not present inferred stenosis as clinically validated stenosis detection

Save the raw outputs and figures.

Directory:

`results/stenosis_sanity_check/`

Create:

`results/stenosis_sanity_check.md`

and:

`results/stenosis_sanity_cases.csv`

The CSV should include:

- case ID
- stenosis annotation available? yes/no
- stenosis location if available
- method used to identify stenosis
- visualization path
- notes

The final report should be scientifically conservative: this is a sanity test, NOT a clinical stenosis diagnosis system.

---

# PHASE 11 — COMPONENT-WISE ABLATION

On the PRIMARY dataset only, run isolated ablations for:

1. Attention gates only
2. Deep supervision only
3. StenosisAwareLoss only
4. TTA only
5. CC3D/post-processing only

Each component should be evaluated independently against the complete RASNet configuration or the project's established baseline.

Make the experimental comparison scientifically clear.

Create:

`results/ablation_results.csv`

and:

`results/table_ablation.md`

Required columns:

- Configuration
- Dice
- HD95
- IoU
- Precision
- Recall
- Δ Dice
- Δ HD95
- notes

Do not rerun ablations on 3D CAS unless computationally practical and useful. Primary-dataset ablation is sufficient.

---

# PHASE 12 — REPRODUCIBILITY BLOCK

Create a reproducibility document:

`results/reproducibility.md`

It must contain:

## Primary dataset

- train/validation/test split
- case IDs if available
- split ratio
- random seed
- preprocessing
- augmentation
- normalization
- patch size
- batch size
- learning rate
- optimizer
- scheduler
- number of epochs
- early stopping/checkpoint strategy
- loss function
- model configuration

## External dataset

Explicitly state:

"3D CAS Images was used as an external test dataset and was not used for model training, validation, hyperparameter tuning, or checkpoint selection."

Only use this exact claim if the experiment history confirms it.

Also include:

- number of external cases
- whether all 200 cases were evaluated
- any cases excluded and why
- preprocessing applied to external cases

## Software environment

Record:

- Python
- PyTorch
- CUDA
- cuDNN if available
- MONAI if used
- nnU-Net version if used
- NumPy
- SciPy
- SimpleITK
- nibabel
- other relevant libraries

Record the actual installed versions rather than guessing.

---

# PHASE 13 — RELATED WORK: 2023–2025

Research 4–6 recent coronary artery segmentation papers published approximately between 2023 and 2025.

Prioritize:

- 3D coronary artery segmentation
- coronary CTA/CT segmentation
- deep-learning-based coronary vessel segmentation
- papers with quantitative Dice and/or HD95
- papers using public coronary datasets

For each paper collect:

- paper title
- year
- model
- dataset
- number of cases if available
- Dice
- HD95 if reported
- other relevant metric
- citation/source
- important methodological distinction

Create:

`results/related_work_research.md`

and:

`results/table_related_work.md`

VERY IMPORTANT:

Do NOT compare results as though they are directly equivalent when:

- datasets differ
- annotation protocols differ
- preprocessing differs
- task definitions differ
- evaluation protocols differ
- train/test splits differ

Include a clear caveat such as:

"Reported performance is not directly comparable across studies because datasets, annotation protocols, preprocessing pipelines, and evaluation procedures differ."

Use primary papers whenever possible rather than relying on random summaries.

---

# PHASE 14 — CLINICAL RELEVANCE

Research this carefully.

We need a scientifically defensible paragraph discussing the clinical relevance of coronary artery segmentation performance.

Investigate literature for:

- clinically useful coronary segmentation accuracy
- Dice/overlap benchmarks
- HD95 or boundary-distance considerations
- coronary CTA segmentation requirements
- limitations of using Dice alone
- relationship between segmentation quality and downstream stenosis analysis

DO NOT invent a universal "clinically acceptable Dice threshold" if the literature does not establish one.

If there is no universally accepted Dice/HD95 clinical threshold, explicitly say so.

Instead, explain that clinical usefulness depends on:

- anatomical accuracy
- small-vessel preservation
- boundary accuracy
- stenosis localization
- downstream quantitative measurements
- image quality
- clinical task

Compare our performance on:

1. Primary test dataset
2. 3D CAS external dataset

against relevant literature benchmarks where genuinely comparable.

The final section must include a clear disclaimer that the model is:

- research-only
- not clinically validated
- not intended for diagnosis
- not a replacement for clinician interpretation

Create:

`results/clinical_relevance_research.md`

---

# PHASE 15 — FINAL FIGURE GENERATION

Generate publication-quality figures for:

1. Primary vs external dataset performance
2. RASNet vs baselines
3. confidence intervals
4. failure cases from 3D CAS
5. stenosis sanity-check cases
6. ablation results if useful

Use consistent:

- font
- labels
- units
- figure dimensions
- naming
- terminology

Save all figures in:

`results/figures/`

Use high-resolution output suitable for a research paper.

---

# PHASE 16 — FINAL MASTER TABLES

Create a directory:

`results/final_tables/`

Generate final versions of:

### Table 1

3D CAS dataset description

### Table 2

RASNet vs baselines on primary dataset

### Table 3

RASNet vs baselines on 3D CAS external dataset

### Table 4

Statistical significance — primary dataset

### Table 5

Statistical significance — 3D CAS

### Table 6

95% confidence intervals

### Table 7

Computational cost

### Table 8

Component-wise ablation

### Table 9

Related work comparison

Make both:

- Markdown
- CSV

where appropriate.

---

# PHASE 17 — FINAL RESEARCH CHECKLIST

Create:

`FINAL_EXPERIMENT_CHECKLIST.md`

Use checkboxes.

The checklist must verify:

- [ ] 3D CAS dataset inspected
- [ ] 200 cases accounted for
- [ ] dataset metadata verified
- [ ] dataset provenance investigated
- [ ] overlap risk investigated
- [ ] external evaluation completed
- [ ] RASNet evaluated
- [ ] SegResNet evaluated
- [ ] nnU-Net evaluated
- [ ] 3D U-Net evaluated
- [ ] primary-set Wilcoxon tests completed
- [ ] external-set Wilcoxon tests completed
- [ ] 95% CIs calculated
- [ ] computational cost measured
- [ ] 2–3 external-dataset failure cases generated
- [ ] \~10-case stenosis sanity check generated
- [ ] ablation experiments completed
- [ ] reproducibility information recorded
- [ ] software versions recorded
- [ ] related-work research completed
- [ ] clinical relevance research completed
- [ ] clinical disclaimer included
- [ ] all figures saved
- [ ] all tables saved
- [ ] no fabricated values
- [ ] no unsupported overlap claims

---

# EXECUTION STRATEGY FOR RTX 3060 Ti

Because this machine has only 8 GB VRAM:

1. First use existing checkpoints.
2. Prefer inference over retraining where possible.
3. Process external cases sequentially.
4. Use mixed precision only if the existing pipeline supports it safely.
5. Avoid unnecessarily large batch sizes.
6. Monitor VRAM.
7. Do not run all four models simultaneously.
8. Run expensive experiments one model at a time.
9. Save intermediate predictions so interrupted experiments can resume.
10. Never lose already-computed results because of an interrupted run.

Before any long experiment, estimate:

- expected runtime
- VRAM usage
- disk usage

and write the estimate to:

`results/experiment_runtime_plan.md`

---

# MOST IMPORTANT FINAL REQUIREMENT

At the end, produce:

`FINAL_RESULTS_SUMMARY.md`

This should contain a concise research-ready summary of:

1. Primary dataset results
2. 3D CAS external results
3. RASNet vs baselines
4. statistical significance
5. confidence intervals
6. computational cost
7. failure cases
8. stenosis sanity check
9. ablation findings
10. reproducibility information
11. related work
12. clinical relevance
13. limitations
14. any unresolved issues

For every unresolved issue, explicitly say:

- what is missing
- why it is missing
- whether it affects the validity of the experiment
- what needs to be done manually

DO NOT start writing the final paper itself yet.

The objective of this task is to produce **all verified experimental evidence, tables, figures, metadata, and research notes needed before we begin writing the final report.**

If something cannot be verified, stop and flag it clearly rather than guessing.
