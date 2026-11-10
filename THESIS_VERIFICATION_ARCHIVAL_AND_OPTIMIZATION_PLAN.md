# Thesis verification, archival, reporting, and later RASNet optimization

Created: 2026-11-10, Asia/Dhaka.

Status: PLAN ONLY. Creating this document does not execute or approve its workflows. No existing code, configuration, results, datasets, checkpoints, or documentation were changed to prepare it.

## 1. Purpose and order of work

Establish a trustworthy comparison of the existing RASNet, SegResNet, nnU-Net V2, 3D U-Net, and V-Net checkpoints on secondary ImageCAS. Preserve unfavorable results. Isolate superseded experiments and claims so the thesis has one identifiable current evidence base. Consider RASNet optimization only after verification, documentation, archival, and reporting are complete.

Required order:

1. Establish the current run's actual state; preserve its inputs and outputs.
2. Verify inference, geometry, metrics, thresholds, completeness, and provenance.
3. Correct evaluation defects where demonstrated; repeat only affected computations.
4. Complete statistics and historical comparisons.
5. Correct active documentation and establish authoritative reports.
6. Archive superseded material after dependency checks.
7. Produce the thesis tables, figures, methods, limitations, and claims.
8. Conduct a separate fixed-weight RASNet optimization experiment.
9. Confirm performance on an untouched portion of the forthcoming primary dataset.

Each phase requires its acceptance checks before dependent work proceeds. No model must win to satisfy this plan.

## 2. Scope and constraints

- Dataset: `C:/Thesis_RASNET/secondary-imagcas-dataset`.
- Split source: `Phase3_Local_Integration/splits_final.json`: 690 train, 94 validation, 150 test IDs (851–1000), subject to fresh verification.
- The 150 cases were previously inspected. Their corrected evaluation is retrospective, not an untouched confirmatory test.
- Do not run the 66-case cohort or hospital evaluation. Preserve their existing records as historical work.
- Do not delete data, overwrite checkpoints, replace BEST with LAST, commit, push, or retrain without separate authorization.
- Preserve historical result bytes. Generate corrections in new, uniquely identified directories.
- Do not modify the current run midway. Do not launch competing GPU work while its state is unresolved.
- Evaluation-infrastructure fixes proposed here require later implementation authorization; architecture, loss behavior, training hyperparameters, and existing weights remain frozen for this comparison.
- Retraining all models for another 200 epochs is not the default remedy for evaluation or reporting problems.
- The approved action for this chat was creating this Markdown file only.

## 3. Observed state and evidence limitations

On 2026-11-10 the repository was on `main` with existing modified/deleted/untracked files. The benchmark implementation and audit documents were untracked. Preserve all existing user changes.

Observed run directories:

- `results/benchmark_runs/smoke_val_all5/`: existing manifest and smoke outputs.
- `results/benchmark_runs/val_full_94/`: existing manifest, five-model Arm A metrics, statistics, threshold search, and threshold JSON.
- `results/benchmark_runs/test_retrospective_150/`: observed RASNet and SegResNet Arm A/B/C CSVs and a RASNet historical-recipe diagnostic CSV, but no completion manifest at inspection.

Partial files do not establish successful completion, failure, or whether a process is currently running. Determine actual process/log state before acting. The current source may differ from the executing source; do not assume otherwise.

Earlier read-only checks found matching hashes for five validation-manifest checkpoints, unique/disjoint split IDs, matching image/label headers on all 244 validation/test pairs, and matching headers for 30 smoke masks. These are useful observations, not a substitute for spatial-content validation or a fresh acceptance audit. No complete synthetic suite was executed by this audit.

## 4. Phase A — Preserve and establish run state

- [ ] Identify the actual command, interpreter, start time, process state, logs, and output destination without interrupting execution.
- [ ] If running normally, allow completion unchanged. If stopped, determine the actual error before proposing a restart.
- [ ] Require the completion manifest and complete case/model/arm inventory; do not infer completion from one model's CSV.
- [ ] Preserve the original run directory and available logs unchanged, including failures and partial outputs.
- [ ] Record branch, commit, Git status, Python executable/version, GPU/driver, CUDA runtime, package versions, configuration bytes, seed settings, and device settings.
- [ ] Preserve a source/configuration snapshot with SHA-256 inventory. Exclude credentials, patient data, and checkpoint duplication.
- [ ] Establish whether source changed during execution. A post-run snapshot alone cannot prove which code was executed.
- [ ] Record SHA-256 for the split manifest, all BEST checkpoints, plans, dataset metadata, and consumed threshold file.

Acceptance: either a completed identifiable run is available, or a specific failure/uncertainty is documented. Preserve incomplete runs; a replacement uses a new run ID, never stale-output reuse.

Proposed future audit outputs, in a new directory under `results/audits/<audit_id>/`:

`run_state.md`, `artifact_inventory.csv`, `environment.json`, `source_inventory.csv`, `checkpoint_verification.csv`, and `issues.csv`. These paths are proposals, not existing artifacts.

## 5. Phase B — Verify inference and experimental identity

### B1. Dataset and splits

- [ ] Verify exact unique train/validation/test IDs and all pairwise disjointness.
- [ ] Resolve each requested image/label pair once; reject missing or duplicate identities.
- [ ] Check size, spacing, origin, direction, and valid finite geometry for every evaluated pair.
- [ ] Record orientation and geometry evidence. Header agreement does not establish patient-level independence or correct content.
- [ ] Report any unsupported patient-independence claim; do not equate absence of exact file duplicates with proof of distinct patients.

### B2. Checkpoints and training provenance

- [ ] Strictly load each exact BEST checkpoint into the source-matching architecture.
- [ ] Verify checkpoint formats and metadata against the files themselves, not filenames alone.
- [ ] Verify the native nnU-Net folder/fold/checkpoint actually loaded matches the recorded checkpoint hash; confirm plans, dataset metadata, configuration, label mapping, reader, normalization, spacing, and mirroring axes.
- [ ] Verify the actual `nnUNetTrainer_200epochs` implementation or demonstrate the registered fallback is equivalent for inference. Do not silently substitute an unverified trainer implementation.
- [ ] Trace model training/selection case inventories. `numTraining=784` alone does not prove exact membership.
- [ ] Reconcile nnU-Net parsed duplicate epochs 51/101/151 and empty epoch-201 record with native logs and checkpoint training metadata. Preserve raw logs.
- [ ] Trace MONAI interruptions/resumes, especially recovered SegResNet logs. Document that source resume paths do not restore optimizer state; determine which runs used them.
- [ ] Record actual validation/checkpoint-selection frequencies. Equal epoch counts do not imply equal compute, data exposure, selection opportunities, or tuning budgets.

Acceptance: each checkpoint has defensible identity and provenance, or a specific BLOCKED/LAB-CHECK item. An epoch-160 BEST checkpoint is valid when the training run continued to 200.

### B3. Coordinate inversion and native nnU-Net execution

- [ ] Add meaningful tests first for identified missing coverage; demonstrate failures before fixing defects.
- [ ] Use asymmetric phantoms with known locations, anisotropic spacing, nonzero origins, and differing direction encodings. Verify recovered physical feature locations, not only output shapes/headers.
- [ ] Cover crop inversion, spacing inversion, 3D U-Net 128-cubed continuous-probability resizing/inversion, and V-Net window padding.
- [ ] Exercise the actual optimized runner path as well as helper adapters.
- [ ] Compare full-run nnU-Net batch preprocessing/export with native single-case inference on the same validation smoke cases, with tolerance declared before inspecting differences.
- [ ] Verify case order/identity through multiprocessing and restored native-grid coordinates. Test missing/failed iterator outputs.
- [ ] Inspect real smoke overlays against images and labels using a predefined case/view selection.
- [ ] If test masks are saved, inspect a predefined sample plus all identified failures. If absent, record the limitation and decide whether targeted regeneration is necessary.

Acceptance: no unexplained spatial, ordering, or identity discrepancy. A copied affine is not proof of correct anatomical alignment.

## 6. Phase C — Verify metrics, arms, safeguards, and outputs

### C1. Metrics and status policy

- [ ] Validate true surface-to-surface distances against brute-force physical surface points, including anisotropic and nested-mask examples.
- [ ] Document erosion connectivity, voxel-center surface convention, percentile convention, and pooled symmetric HD95/ASD definitions. Do not conflate these with every alternative surface-metric convention.
- [ ] Verify skeleton metrics and explicit empty-mask/empty-skeleton policies.
- [ ] Reject invalid/non-finite probabilities before thresholding; never turn inference failures into ordinary masks or fabricated scores.
- [ ] Distinguish a valid empty prediction from an inference exception; expose all undefined values and failure counts.
- [ ] Run the complete synthetic suite after changes. Record real random architectures versus explicit test doubles.

### C2. Frozen arm verification

| Arm | Required policy |
| --- | --- |
| A | One inference pass, threshold 0.50, no TTA or component pruning; preserve model-specific necessary preprocessing. |
| B | Documented RASNet four-pass TTA and native nnU-Net mirroring; other baselines single-pass where no standard TTA was established; threshold 0.50, no pruning. This compares documented pipelines, not equal augmentation budgets. |
| C | Single-pass inference; thresholds from all 94 validation cases; common top-two-component filter with 26-connectivity. Log retained/removed components, voxels, and physical volumes. |

- [ ] Preserve the existing grid 0.30–0.70 in steps of 0.05, mean validation Dice objective after filtering, and tie rule (closest to 0.50, then lower).
- [ ] Verify the consumed thresholds match validation IDs, checkpoint hashes, source/settings provenance, and post-processing.
- [ ] Strengthen threshold validation to reject absent/incompatible provenance and incomplete entries.
- [ ] Describe selected values as best within the declared grid; do not extend the grid because of test outcomes.
- [ ] Keep the RASNet tau-0.60 historical-recipe diagnostic separate from Arms A/B/C and from primary hypothesis testing.
- [ ] Confirm caching is disabled or fully validated against inputs, exact IDs, checkpoints, settings, metric implementation, and executed code identity.
- [ ] Confirm each arm/model contains exactly 150 unique IDs and that expected mask exports are complete where requested.

### C3. Decide what must be repeated

| Finding | Action |
| --- | --- |
| Documentation error only | Correct documentation; retain verified computations. |
| Statistical output incomplete | Recompute statistics from verified case-level metrics. |
| Metric defect | Recompute metrics from valid saved masks, if available; otherwise regenerate only needed predictions. |
| Wrong coordinates, preprocessing, loaded checkpoint, or case mapping | Correct and test; rerun affected inference in a new directory. |
| Demonstrated validation-selection defect | Repeat affected selection on full validation, then affected evaluation; disclose correction after test inspection. |
| Unsupported executed-source identity | Attempt reconstruction; otherwise retain explicit uncertainty or perform a reproducibly captured replacement run. |
| Consequential training defect | Stop affected interpretation and request a separate decision about retraining. Do not retrain all models automatically. |

After repairs, run all-five three-case smoke verification before dependent full-data stages. Preserve original thresholds/results; corrected artifacts receive new paths. Test-outcome-driven model improvement belongs to Phase H, not benchmark repair.

## 7. Phase D — Statistics and historical reconciliation

- [ ] Validate unique identical case sets before any merge/filtering; never pair by row order.
- [ ] Preserve the declared primary family (RASNet vs nnU-Net precision/HD95/ASD) and secondary family (four baselines times eight metrics), with Holm correction as specified within each arm.
- [ ] State that correction within each arm does not automatically control arbitrary claims across all arms. Identify primary versus exploratory interpretations without retroactive selection.
- [ ] Report two-sided paired Wilcoxon tests, rank-biserial effects, paired mean differences, paired bootstrap confidence intervals, and direction-correct wins/ties/losses.
- [ ] State effect-size sign convention: a negative raw difference on a lower-is-better metric favors RASNet unless a benefit-oriented convention is explicitly adopted.
- [ ] Report metric-specific paired denominators, exclusions, ties, and failure patterns. Finite-pair selection must not conceal inferior empty predictions.
- [ ] Retain seed 42 and 2,000 bootstrap resamples where prescribed; do not select a procedure based on favorable significance.
- [ ] Compare new and historical CSVs using exact unique IDs and the declared tolerance 0.0001. Report missing/unmatched cases and metrics explicitly.
- [ ] Separate recipe changes from metric corrections when explaining differences; neither is evidence of improved trained weights.
- [ ] Audit ablation sources separately. Current Table8/ablation_table Dice means agree with their case CSVs, while `3d Cas Validations and doings/results/ablation_results.csv` disagrees.
- [ ] Record that the matched SegResNet baseline uses DiceFocalLoss whereas attention/deep-supervision ablation variants use DiceCELoss: the first comparison is not an attention-only intervention.

Acceptance: neutral tables for every arm and all models, case-level traceability, and explicit limitations. Failure to reject a Dice difference does not establish equivalence.

## 8. Phase E — Correct the active documentation

Preserve original document bytes first. Add dated corrections; never rewrite historical preregistration as if later decisions preceded results.

| Exact path | Required work |
| --- | --- |
| `README.md` | Correct addition-vs-concatenation diagram, spatial-attention description, case-1 unseen claim, guaranteed blob-removal language, retrospective status, and evidence-backed results. |
| `GEMINI.md` | Replace mixed N=32/N=21 clinical/publication-ready claims and stale rankings with active scope and verified status. Agent memory must not keep instructing future work from obsolete claims. |
| `CLAUDE.md` | Correct unavailable-data/checkpoint machine context while retaining valid integrity rules. |
| `CHECKPOINTS.md` | Correct source auxiliary weights (1.0/0.4/0.2), unsupported `ds_weights` constructor claim, excluded-ID examples, Python version discrepancy, and any checkpoint-format/provenance mismatches. |
| `INFERENCE_SPEC.md` | Distinguish verified implementation, source-derived recipes, resolved conflicts, and remaining checks; include actual batch inference path. |
| `LAB_RUN_PLAN.md` | Replace obsolete unimplemented-runner status with exact commands actually used, outputs, checks, failures, and replacement runs. |
| `BENCHMARK_PREREGISTRATION.md` | Preserve original; append dated amendments and truthful timing/status. |
| `AUDIT_FINDINGS.md`, `LAB_CHECKS.md` | Retain original findings; add current resolution and evidence references. Correct outdated assertions without erasing their historical date/context. |
| `Phase3_Local_Integration/README.md` | Correct environment paths and active/shared dependencies. |
| `Phase3_Local_Integration/rasnet_loss.py` documentation only | Describe Dice-plus-focal formulation accurately; no explicit lesion-specific supervision or clinical mechanism is established. Do not change loss behavior. |

Create a proposed new `CURRENT_EVIDENCE_INDEX.md` only during authorized execution: point to verified run IDs, accepted tables, unresolved limitations, current plans, and archive index. Avoid adding another free-standing unsupported verdict.

## 9. Phase F — Dependency-safe archival inventory

Archiving means removing material from the active evidence workflow while preserving its bytes, original paths, and context. Historical does not mean false. Use statuses: `superseded`, `outside_current_scope`, `known_discrepancy`, or `unverified`.

Proposed archive destination: `_Archive_Historical_Experiments_and_Logs/thesis_cleanup_<execution_date>/`. Keep original relative paths underneath it. Do not create or populate it until authorized.

### F1. Keep active/in place

- `benchmark/`, `tests/`, `config.lab.yaml`.
- `secondary-imagcas-dataset/`; all source datasets stay in place.
- `results/benchmark_runs/` and `results/thresholds.json`.
- `Phase3_Local_Integration/rasnet_model.py`, `rasnet_loss.py`, `splits_final.json`.
- `Q1_Publication_Package/matched_200ep_benchmark/checkpoints/`, `nnUNet_results/`, `checkpoints_best_and_last/`, `logs/`, the four `train_*_200ep.py` files, `prepare_and_run_nnunet_200ep.py`, `resume_rasnet_200ep.py`, and `run_all_200ep.py` as frozen provenance. Keeping training launchers is not authorization to run them.
- `Q1_Publication_Package/ablation/train_ablation_models_200ep.py` may remain in place as frozen provenance if dependencies require it; otherwise archive with the ablation package and record its path.
- Active documentation listed in Phase E, this plan, `LICENSE`, environment requirements, and workspace/Git configuration.
- Existing historical archive: leave intact; do not recursively reorganize it.

Do not move the whole `Phase3_Local_Integration/` or `Q1_Publication_Package/`. The active adapter imports the former's RASNet model; tests/configuration reference the latter's sources/artifacts.

### F2. Archive superseded root documents

- `LAB_4080_SUPER_MASTER_PLAN.md`
- `Upto-What's-done.md`
- `IEEE_JBHI_UNSEEN_66_FIGURES_AND_TABLES.md`
- `FINAL_FORENSIC_AUDIT.md` (preserve as a dated historical assessment, not a universally false document)

### F3. Archive exact loose historical files under `results/`

- `FINAL_MASTER_RESULTS.csv`
- `FINAL_MASTER_RESULTS.md`
- `FINAL_MODEL_SCORECARD.md`
- `FINAL_RASNET_VERDICT.md`
- `FINAL_RESEARCH_STATUS.md`
- `PRIMARY_DATASET_RANKING.md`
- `3D_CAS_FINAL_RANKING.md`
- `RECOMMENDED_PAPER_POSITIONING.md`
- `case_level_model_comparison.csv`
- `case_level_analysis.md`
- `FAIRNESS_AUDIT.md`
- `PAPER_CLAIM_AUDIT.md`
- `ACCURACY_COST_FORENSIC_ANALYSIS.md`
- `ABLATION_FORENSIC_ANALYSIS.md`
- `STENOSIS_AND_PROVENANCE_AUDIT.md`

### F4. Archive historical packages after reference checks

- `3d Cas Validations and doings/`: preserve complete 66-case/repackaged-cohort workflows, scripts, metrics, conflicting/corrected summaries, figures, and identity/provenance records together.
- `results-after-hallucin-fix/`.
- `Final_Generated_assets/`.
- `scratch/`.
- `Q1_Publication_Package/clinical_validation/`.
- `Q1_Publication_Package/stats/`.
- `Q1_Publication_Package/efficiency/`.
- `Q1_Publication_Package/figures/`: preserve original figure generators with figures, including original convergence PNG/SVG.
- `Q1_Publication_Package/ablation/`: preserve source training/evaluation scripts and case-level evidence; apply the provenance exception in F1 if required.
- `Q1_Publication_Package/checklist/`.
- `Q1_Publication_Package/MISSING_INPUTS.md`.
- `Q1_Publication_Package/package_handoff_for_digonto.py`.

### F5. Historical matched evaluator and CSVs: defer movement

`Q1_Publication_Package/matched_200ep_benchmark/evaluate_matched_200ep_models.py` and `evaluation_results/` are historical. Preserve them unchanged through Phase D because `config.lab.yaml` directly references five historical case CSVs.

After comparisons, either retain their original paths with explicit historical classification or migrate them and update every active reference in a separately authorized change. Revalidate source-contract tests and comparison loading. Do not strand old result references inside manifests; record old-to-new mapping in the archive index.

### F6. Hospital-specific files in the shared Phase3 folder

Archive individually after import/reference checks:

- `run_hospital_cohort_pipeline.py`
- `production_qca_engine.py`
- `hospital_cohort_automated_stenosis.csv`
- `dicom_to_nifti.py`
- `generate_stenosis_annotation_package.py`
- `apply_ct4_adjudication.py`
- `07_local_data_qc.py`
- `audit_case_provenance.py`
- `audit_3d_mask_completeness.py`
- `outlier_audit/`

All paths above are under `Phase3_Local_Integration/`. Leave `local_data/` and patient/source files where they are.

Hold `dataset_paths.py`, `eval_utils.py`, `dataset_audit.json`, and `curriculum_splits.json` in place until dependencies and historical-training relevance are resolved. Do not classify unknown files by name alone.

### F7. Safe archival procedure and acceptance

- [ ] Freeze an explicit proposed move list after the run, including only existing verified paths.
- [ ] Inventory source hashes, original paths, purpose, status/reason, dependent files, destination, and replacement if any.
- [ ] Search imports, scripts, configuration, tests, packaging rules, and Markdown links. Keep shared dependencies until an authorized migration is complete.
- [ ] Preserve original active-document snapshots before corrections.
- [ ] Resolve and verify absolute source/destination paths stay inside the intended workspace/archive. No deletion or cross-shell deletion/move commands.
- [ ] Use non-overwriting moves after approval; do not merge colliding destinations or copy checkpoints/datasets.
- [ ] Verify destination hashes and file counts; record missing/unmoved items explicitly.
- [ ] Create `ARCHIVE_INDEX.csv` and `README.md` in the dated archive. Explain that archived outputs are excluded from current claims.
- [ ] Check active dependencies and links after movement; run appropriate checks for changed references. No need to rerun valid inference just to test archival.

Acceptance: every moved file is traceable, bytes preserved, active benchmark dependencies resolve, and old claims no longer appear as current evidence.

## 10. Phase G — Thesis analysis, figures, and claims

- [ ] Build one set of tables from accepted new manifests and case CSVs. Include all arms, all models, sample sizes, uncertainty, failures, and trade-offs.
- [ ] Recreate figures from verified sources in a new output directory; preserve old publication assets.
- [ ] Correct convergence figure title: validation N is 94, not 150. Distinguish nnU-Net online pseudo-Dice from MONAI transformed-volume validation Dice; identify actual validation frequency and smoothing.
- [ ] Do not imply four MONAI paths all use sliding windows: 3D U-Net uses a direct resized-volume pass.
- [ ] Do not compare different loss magnitudes as model quality; adding 1 to nnU-Net loss does not make it comparable to Dice-plus-focal with deep supervision.
- [ ] Compare identical checkpoint/inference/grid protocols where possible; epoch-200 training metrics and BEST-checkpoint evaluation are not interchangeable.
- [ ] Do not invent unmeasured RASNet patch Dice (such as 0.83+), claim zero overfitting from similar means, or attribute nnU-Net errors to capacity without evidence.
- [ ] Keep nnU-Net as a relevant strong comparator. Do not remove unfavorable baselines.
- [ ] Report current Arm A validation trade-offs honestly: RASNet precision/boundary/clDice advantages versus nnU-Net recall/centerline recall advantages, with no established Dice equivalence.
- [ ] Include the validation-selection evidence: best grid/postprocessed Dice was approximately nnU-Net 0.8075, SegResNet 0.7901, RASNet 0.7897. These are exploratory selection-set scores, not independent test performance.
- [ ] Do not claim RASNet beats 3D U-Net on HD95/ASD from the cited Arm A validation table.
- [ ] Measure deployment efficiency separately using verified architectures and a declared timing scope (loading/preprocessing/inference/inversion/postprocessing/export, hardware, warmup, synchronization, repeats, and distribution). The old efficiency script hardcodes 1.85 seconds; that is not new timing evidence.
- [ ] Keep hospital/66-case claims out of current conclusions. Do not describe a repackaged ImageCAS subset as external scanner-domain validation.

HeartFlow/Cook discussion is background only. The Cook 2017 review reports FFR-CT diagnostic accuracy, not segmentation accuracy or contemporary head-to-head RASNet performance. Do not compare anatomical grading percentages with FFR diagnostic percentages, claim precision is inherently more clinically important than recall, or attribute gray-zone performance solely to segmentation. Poiseuille's idealized radius sensitivity does not establish clinical benefit from whole-tree HD95/ASD. Do not claim zero operating cost, legal violations, HeartFlow replacement, or superiority without separate evidence.

Reference links already reviewed:

- Cook et al.: https://jamanetwork.com/journals/jamacardiology/article-abstract/2629072
- HeartFlow FFRCT description: https://www.heartflow.com/heartflow-one/ffrct-analysis/
- nnU-Net compound-loss source: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/loss/compound_losses.py

External `F:/Thesis/...` manuscript/plotting paths quoted in earlier advice were not found in this workspace scan; locate and inspect them separately before claiming manuscript corrections are complete.

Acceptance: methods and limitations match execution; every numerical claim is tied to accepted artifacts; no universal-superiority or clinical-deployment claim exceeds evidence.

## 11. Phase H — Separate fixed-weight RASNet optimization, only after A–G

Goal: investigate improved recall while preserving a predefined precision level, with the current architecture and checkpoint. Improvement is not guaranteed.

- [ ] Start a separate experimental protocol and output root. Never overwrite benchmark thresholds or results.
- [ ] Select and freeze one reference pipeline. Arm A at threshold 0.50 is a clean initial reference; distinguish any optimized deployment reference explicitly.
- [ ] Define mean per-case recall as the selection objective subject to a precision constraint. Define whether literal mean precision preservation or a justified non-inferiority margin is required before experiments. Also set/report per-case deterioration criteria.
- [ ] Use only development/validation cases; acknowledge these 94 cases have already informed development and are not independent confirmation.
- [ ] Examine whether missed vessels have usable foreground probabilities. If they are indistinguishable from background, post-processing may not solve the deficit.
- [ ] Freeze a small candidate grid before comparison: lower-threshold baseline; documented four-pass TTA; two-threshold connected recovery using high-confidence seeds and lower-confidence connected extensions, with any physical growth limit/connectivity specified.
- [ ] Compare each candidate against the frozen reference using paired precision/recall changes, added true/false positives, clDice, centerline recall, Dice, boundary metrics, failure counts, and runtime.
- [ ] Inspect predefined validation cases and all major precision failures for real branches versus lumen enlargement, leakage, and false bridges. Disconnected branches without seeds remain a known limitation.
- [ ] Select only candidates satisfying the declared precision constraint; break near-ties by simplicity/cost. Report no qualifying improvement if necessary.
- [ ] Freeze the selected pipeline before independent evaluation. A nonsignificant precision decrease is not proof of preservation.
- [ ] If presenting optimized RASNet versus baselines as a new benchmark, offer comparable development budgets or explicitly label the asymmetric pipeline-development comparison.

No new architectures, losses, retraining, or ensembles of new checkpoints are authorized by this plan. If needed later, design a separate training study with fresh approval and provenance rather than repeatedly tuning against the retrospective test set.

## 12. Phase I — Forthcoming primary dataset and final confirmation

- [ ] Establish dataset identity, consent/access scope where applicable, labeling protocol, patient-level grouping, and overlap checks against development sources.
- [ ] Reserve an untouched patient-level test portion before tuning. If adapting/training on the primary dataset, maintain separate train/validation/test portions.
- [ ] Freeze endpoints, hypothesis families, precision criterion, uncertainty analysis, and all pipelines before test-label analysis.
- [ ] Evaluate all retained comparators with compatible necessary preprocessing and disclose domain adaptations.
- [ ] Report any domain shift, exclusions, failures, and uncertainty. Do not call a development-used cohort independent confirmation.
- [ ] Make clinical stenosis/FFR claims only in a separately designed study with appropriate references and endpoints.

## 13. Operational commands to document during execution

These are future examples, not commands run to produce this plan. Verify interpreter and paths first. Actual executed commands and outcomes belong in `LAB_RUN_PLAN.md` and the audit log.

```powershell
git status --short
git rev-parse HEAD
& 'C:\Thesis_RASNET_env\bench\Scripts\python.exe' --version
nvidia-smi
& 'C:\Thesis_RASNET_env\bench\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests
Get-FileHash -Algorithm SHA256 -LiteralPath 'C:\Thesis_RASNET\results\thresholds.json'
```

Tests write temporary files; run them only during authorized execution, after resource availability is established. Do not start an inference command solely because it appears in an old plan. If new smoke/validation/test runs are necessary, derive exact CLI arguments from the verified runner, assign fresh run IDs and configurations, and document the reason first.

## 14. Final delivery checklist

- [ ] Run-state and provenance inventory, including source uncertainty if unresolved.
- [ ] Checkpoint/nnU-Net metadata, split, geometry, inference-path, and synthetic-test evidence.
- [ ] Accepted manifests, case metrics/statuses, summaries, paired statistical tables, historical diffs, and limitations.
- [ ] Dated issue ledger showing corrected, unresolved, superseded, and outside-scope items.
- [ ] Corrected active documentation and one current evidence index.
- [ ] Archive index with original/destination paths, reasons, hashes, dependencies, and preserved original documents.
- [ ] Reproducible thesis tables/figures with neutral interpretation and explicit retrospective status.
- [ ] Separate optional optimization protocol/results only after earlier phases pass.
- [ ] Independent-primary-dataset protocol before any confirmatory evaluation.
- [ ] No dataset deletion, checkpoint overwriting, silent historical rewriting, unauthorized retraining, commits, or pushes.

Completion means a trustworthy, traceable thesis comparison with acknowledged limits. It does not require RASNet to rank first.
