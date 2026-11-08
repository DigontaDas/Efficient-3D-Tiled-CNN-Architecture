# Persistent thesis benchmark rules

- Datasets, checkpoints and patient data are not available on this machine. Do not invent paths, results or numbers. Run synthetic tests only.
- Put the lab dataset root only in config, never in code.
- Do not change model architectures, losses, hyperparameters, epoch counts or checkpoints.
- Test cases 851–1000 are locked: no threshold, parameter or checkpoint selection may use them. Select on validation only.
- Every reported number must come from a file produced by a run with a manifest.
- Write tests first and show the failing test, then fix.
- Never commit data or checkpoints. Record checkpoint SHA-256 in CHECKPOINTS.md.
- Mark what cannot be verified here as LAB-CHECK. Do not push without explicit authorization.
