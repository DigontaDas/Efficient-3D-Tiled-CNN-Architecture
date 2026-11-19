You are running on the lab PC (`C:\Thesis_RASNET`, RTX 4080 SUPER 16 GB, `$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"`).
Read `GEMINI.md`, `RASNET_MASTER_FIX_AND_RETRAIN_PLAN.md` (PART II), and `Rasnet-v2/results/pilots_val94/pilot_lock_decision.md`.

First verify `git status` is completely clean (`dirty_code = false`) and check `Rasnet-v2/logs/queue_status.json` + `Rasnet-v2/STATE_HANDOFF.md` to confirm the GPU is free (Rule 8).

I want to execute **PART II — [CHOOSE ONE: Strategy A (1-seed fair upgraded 200-epoch retrain: RASNet + SegResNet + V-Net + 2 ablations, reusing existing nnU-Net V2)  OR  Strategy B (Amendment A2 to unpause Rasnet-v2 3-seed x 200-epoch controlled study)]**.

### Strict Two-Step Execution Gate
1. **PHASE A (Plan & Pre-Registration Amendment — STOP FOR MY APPROVAL):**
   - Draft the dated pre-registration amendment (`BENCHMARK_PREREGISTRATION.md` for Strategy A, or `Rasnet-v2/PREREGISTRATION_V2.md` Amendment A2 for Strategy B) covering:
     - Exact architecture/loss/pipeline fixes (`conv_bias` gate, fixed DS weighting `1.0 / 0.5 / 0.25` with downsampled GT, `DiceCELoss(include_background=False, softmax=True, batch=True)`, native `[0.5, 0.349609375, 0.349609375] mm` spacing, `96x160x160` patch, CT z-score normalization, full 3D augmentations, `clip_grad_norm_(1.0)`, full optimizer/scaler/RNG resume checkpoints, transform-hashed `PersistentDataset` caches, matched validation cadence).
     - Exact list of models, seeds, epochs (`200`), fresh checkpoint & `results/` output directories (never overwriting protected v1 directories), and widened Val-94 threshold grid `tau in [0.10, 0.70]`.
   - Write unit tests first (Rule 6) for any new/modified training code and verify they pass.
   - **STOP and ask for my explicit approval** before committing the amendment or launching Epoch 1 on the GPU.

2. **PHASE B (After I reply "Approved" — Full Autonomous Run):**
   - Commit the approved amendment + code so `git.dirty_code = false`.
   - Run 200-epoch training sequentially (one GPU job at a time), logging milestones only when validation epochs or full runs complete (Rule 11), and record every new checkpoint's SHA-256 in `CHECKPOINTS.md`.
   - Run symmetric Val-94 grid evaluation (`0.50` Gaussian SW, 8-pass logit-averaged TTA, Top-2 CC / locked symmetric CC) -> write and commit `locked_recipes.json` + `.sha256` sidecar.
   - Run one-shot Test-150 evaluation (explicitly labeled retrospective), paired Wilcoxon + Holm-Bonferroni + 10,000-bootstrap 95% CIs + Dice non-inferiority (`margin = -0.005`), efficiency timing, and generate final tables/figures in the new `results/` folder.
   - Report final verified metrics from the generated CSVs without pushing to GitHub.