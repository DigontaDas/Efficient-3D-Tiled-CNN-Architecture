# Lab run plan — BLOCKED until runner completion

Do not evaluate the locked test set yet. The new metrics and model helpers are research work in progress, not a complete five-model benchmark. `benchmark/run_benchmark.py`, the nnU-Net adapter, validation-only threshold selection, run manifests, the fully integrated CPU dry-run, statistics CLI and per-case diff are not implemented. Consequently there are no honest exact smoke, validation, test, statistics or diff commands yet. Do not edit code or tune on the lab PC to work around this; return the implementation to the development machine first. Do not treat prior tables as outputs of this new benchmark.

Once the implementation is completed and independently verified, the required order is: check environment and record each best-checkpoint SHA-256 in `CHECKPOINTS.md`; run the full synthetic test suite; smoke all five; perform the full validation run and freeze `thresholds.json`; run locked test arms A, B and C without selection; compute statistics with the declared family; compare each case with previous CSVs using a stated tolerance; return manifests, CSVs, differences and any failures. Stop at any failure. All reported study numbers must be backed by files from a manifested run. Ask Digonta how the historic 0.60 threshold was selected before interpreting the old numbers; no message has been sent.

Currently available read-only lab checks from the repository root (PowerShell):

```powershell
python --version
python -c "import torch,monai,SimpleITK,scipy,skimage; print(torch.__version__,monai.__version__,SimpleITK.Version_VersionString(),scipy.__version__,skimage.__version__)"
python -B -m pytest -q -p no:cacheprovider tests
Get-FileHash -Algorithm SHA256 -LiteralPath '<verified-best-checkpoint-path>'
```

Pass only if imports and all tests succeed, and every recorded SHA-256 independently matches the correct named best checkpoint. The placeholder must be replaced by the operator with each actual lab path; it is intentionally not a fabricated path. These checks alone do not authorize or constitute a benchmark run. Dataset geometry, split IDs, checkpoint identity, package compatibility and nnU-Net plans remain LAB-CHECK items.
