# Historical Archive: thesis_cleanup_2026-11-10

Archived on: 2026-11-10
Destination: `_Archive_Historical_Experiments_and_Logs\thesis_cleanup_2026-11-10`

## Purpose and Status Notice
This archive contains superseded experimental logs, historical reports, 66-case cohort workflows, legacy assets, and hospital cohort scripts archived per Section 9 of `THESIS_VERIFICATION_ARCHIVAL_AND_OPTIMIZATION_PLAN.md` and approved user authorization.

> **CRITICAL EVIDENCE BOUNDARY**:
> All files in this archive are formally **excluded from current thesis claims and benchmarks**.
> Historical does not mean fraudulent; it documents prior iterations, superseded intermediate states, or components outside the current benchmark scope.

## Status Taxonomy
- `superseded`: Prior iteration replaced by the verified 5-model benchmark pipeline or updated documentation.
- `outside_current_scope`: Experiments conducted on alternative cohorts (e.g. Unseen 66 cohort, hospital clinical cohort) not forming part of the active 5-model 200-epoch ImageCAS benchmark.
- `known_discrepancy`: Files containing demonstrated historical discrepancies (e.g. hardcoded latency/scores, threshold/loss mismatches) preserved for historical transparency and provenance audit.
- `unverified`: Intermediate assets preserved without active claim endorsement.

## Known Limitations of Archived Code
Scripts moved into this archive that rely on relative imports (e.g. importing `dataset_paths` or `eval_utils` from `Phase3_Local_Integration`) are preserved as **frozen provenance** and will not execute out-of-the-box from their new relative paths without path adjustments.

## File Manifest
A complete machine-readable inventory of all 306 files with original paths, sizes, reasons, and verified pre-move SHA-256 hashes is available in `ARCHIVE_INDEX.csv`.
