# V1 Epoch-200 Val94 Endpoint Evaluation Report (`20261117_041500Z`)

- **Generated (UTC):** `2026-11-17T06:16:55Z`
- **Checkpoint Role:** `final_epoch200` (`RASNet`: epoch 200 by script/log/archive provenance, `embedded_epoch_metadata_present=False`; `nnU-Net V2`: embedded `current_epoch=200`)
- **Validation Split:** `Val94` (`N = 94`, `splits_sha256: efe3f06cfe9c7dff646224378ad4d0ea5ce50fb4bf920807b17054af4bdf953c`)
- **Statuses (`RASNet` / `nnU-Net`):** `{'ok': 94}` / `{'ok': 94}` (`0` dropped, `0` empty predictions)
- **Mask Geometry Verification:** `188/188` exported masks verified against source image and label headers.
- **Timing (`manifest.json`):** `rasnet = 0.0s`, `nnunet = 1579.378s` (for resumed run, `rasnet = 0.0s` reflects pre-resume completion of 94/94 cases).

# V1 Epoch-200 Endpoint Comparison vs. Existing Best-Checkpoint References (Val94)

> **Evidence Separation Rule**: Panel A (`final_epoch200`, `1-pass, 0.50 Gaussian SW, tau=0.50, no CC`) and Panel B (existing `best` checkpoint runs at differing epochs and differing inference/post-processing recipes) are kept strictly distinct. Do **not** merge rows across panels into a single ranking or infer a training trajectory across unmatched recipes.

## Panel A — Newly Verified Epoch-200 Endpoint Comparison (Val N = 94; Symmetric 1-Pass, 0.50 Gaussian SW, τ = 0.50, No CC)

| Model | Role | Epoch Provenance | N | Dice (Mean ± SD) | Precision | Recall | HD95 (mm) | ASD (mm) | clDice | Centerline Recall |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **rasnet** | `final_epoch200` | epoch 200 (script/log/archive provenance; no embedded ckpt metadata) | 94 | `0.7792 ± 0.0569` | `0.7586` | `0.8065` | `17.32` | `2.619` | `0.8413` | `0.9068` |
| **nnunet** | `final_epoch200` | current_epoch=200 (embedded ckpt metadata) | 94 | `0.7767 ± 0.0562` | `0.7428` | `0.8203` | `18.67` | `2.748` | `0.8347` | `0.9142` |

- **Paired Dice Difference (`RASNet − nnU-Net`, N=94):** `+0.0025` (95% bootstrap CI `[-0.0027, +0.0075]`, seed `42`, resamples `2000`; median `+0.0032`, Wilcoxon `p = 0.2401`, W/T/L = `54/0/40` with tie tolerance `1e-12`).

## Panel B — Existing Best-Checkpoint Val94 References (Strictly Separate Recipes & Selected Epochs)

| Comparison Group | Model | Role | Epoch Record | SW / Preprocessing Recipe | Post-Processing | N | Dice (Mean ± SD) | Precision | Recall | HD95 (mm) | clDice |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Val94 Arm A (best; asymmetric SW 0.25 const vs 0.50 gauss, no CC) | **rasnet** | `best` | epoch 160 (rasnet_200ep_log.csv) | 1-pass, 0.25 constant SW, tau=0.50 | none | 94 | `0.7789 ± 0.0521` | `0.8165` | `0.7483` | `13.07` | `0.8480` |
| Val94 Arm A (best; asymmetric SW 0.25 const vs 0.50 gauss, no CC) | **nnunet** | `best` | current_epoch=199 (best_ema=0.8162; raw mean_fg_dice peak ep 189=0.8376) | 1-pass, 0.50 Gaussian SW, tau=0.50 | none | 94 | `0.7761 ± 0.0567` | `0.7467` | `0.8144` | `18.76` | `0.8339` |
| Option A Val94 (best; 0.50 gauss SW, tau=0.50, WITH Top-2 CC) | **rasnet** | `best` | epoch 160 (rasnet_200ep_log.csv) | 1-pass, 0.50 Gaussian SW, tau=0.50 | 26-connected Top-2 CC | 94 | `0.7953 ± 0.0657` | `—` | `—` | `—` | `—` |
| Option A Val94 (best; 0.50 gauss SW, tau=0.50, WITH Top-2 CC) | **nnunet** | `best` | current_epoch=199 (best_ema=0.8162; raw mean_fg_dice peak ep 189=0.8376) | 1-pass, 0.50 Gaussian SW, tau=0.50 | 26-connected Top-2 CC | 94 | `0.8070 ± 0.0585` | `—` | `—` | `—` | `—` |
