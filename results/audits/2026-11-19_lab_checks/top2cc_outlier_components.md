# Option A Top-2 CC Outlier Component Inspection (Cases 930, 948, 861, 978)

- **Audit UTC:** `2026-11-22T04:51:56Z`
- **Sources:** `results/optionA_v1_symmetric_sw050/test_per_case.csv` and native ImageCAS GT NIfTI labels (`config.lab.yaml`).
- **Exported Option A Prediction Masks on Disk:** `0` (`run_optionA_symmetric_v1_eval.py` evaluates probabilities in memory and records component telemetry in `test_per_case.csv` without exporting NIfTI masks; per-component predicted volume lists therefore require an inference-only re-run if Stage 5C is selected).

## 1. Ground-Truth 26-Connected Component Breakdown (Native Physical Grid)

| Case ID | Spacing (Z, Y, X) mm | Voxel Vol (mm³) | GT CCs (26-conn) | Total GT Vol (mm³) | GT Component Volumes (mm³, descending) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `930` | `(0.50, 0.3398, 0.3398)` | `0.05775` | `2` | `8231.88` | `4174.06, 4057.82` |
| `948` | `(0.50, 0.3828, 0.3828)` | `0.07327` | `3` | `9399.57` | `5350.67, 4048.32, 0.59` |
| `861` | `(0.50, 0.3711, 0.3711)` | `0.06886` | `2` | `4533.78` | `2321.39, 2212.39` |
| `978` | `(0.50, 0.3613, 0.3613)` | `0.06528` | `1` | `6622.23` | `6622.23` |

## 2. Option A `R2_raw_tta8` vs. `R4_tuned_tta8_top2` Component Pruning Telemetry

| Case ID | Model | R2 Dice | R4 Dice | ΔDice (R4−R2) | R2 CCs | R4 Total→Kept CCs | R4 Retained Vol (mm³) | R4 Removed Vol (mm³) | R2→R4 Recall | R2→R4 HD95 (mm) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `930` | `rasnet` | `0.6814` | `0.4703` | `-0.2112` | `14` | `14→2` | `7414.9` | `4023.7` | `0.790→0.447` | `44.30→57.40` |
| `930` | `nnunet` | `0.6483` | `0.8260` | `+0.1776` | `28` | `27→2` | `7985.0` | `5218.8` | `0.806→0.814` | `48.20→4.32` |
| `930` | `segresnet` | `0.6482` | `0.8163` | `+0.1682` | `22` | `22→2` | `7798.7` | `4365.3` | `0.794→0.795` | `47.97→5.89` |
| `930` | `vnet` | `0.6893` | `0.7918` | `+0.1024` | `26` | `27→2` | `6943.3` | `4243.1` | `0.800→0.730` | `43.75→11.23` |
| `948` | `rasnet` | `0.8787` | `0.7709` | `-0.1079` | `7` | `8→2` | `7083.1` | `2388.8` | `0.861→0.676` | `0.74→51.13` |
| `948` | `nnunet` | `0.8729` | `0.7657` | `-0.1073` | `8` | `8→2` | `7366.0` | `2616.1` | `0.873→0.683` | `0.91→51.35` |
| `948` | `segresnet` | `0.8689` | `0.7629` | `-0.1060` | `13` | `13→2` | `6901.3` | `2432.1` | `0.856→0.662` | `0.86→51.40` |
| `948` | `vnet` | `0.8663` | `0.7293` | `-0.1369` | `9` | `9→2` | `6320.9` | `2838.3` | `0.846→0.610` | `0.91→52.24` |
| `861` | `rasnet` | `0.7481` | `0.6530` | `-0.0951` | `17` | `16→2` | `2558.0` | `1232.3` | `0.665→0.511` | `9.54→48.53` |
| `861` | `nnunet` | `0.7559` | `0.6512` | `-0.1047` | `11` | `11→2` | `3027.6` | `1006.2` | `0.681→0.543` | `8.41→32.58` |
| `861` | `segresnet` | `0.7377` | `0.6448` | `-0.0929` | `20` | `21→2` | `2760.0` | `1278.1` | `0.689→0.519` | `27.60→32.87` |
| `861` | `vnet` | `0.7309` | `0.6180` | `-0.1129` | `14` | `14→2` | `2501.3` | `1181.2` | `0.654→0.479` | `8.47→32.59` |
| `978` | `rasnet` | `0.7229` | `0.6445` | `-0.0784` | `20` | `21→2` | `4314.0` | `2868.0` | `0.730→0.532` | `19.33→34.70` |
| `978` | `nnunet` | `0.6897` | `0.7077` | `+0.0180` | `25` | `24→2` | `5439.1` | `2886.6` | `0.735→0.644` | `26.35→25.38` |
| `978` | `segresnet` | `0.7033` | `0.7410` | `+0.0377` | `25` | `24→2` | `5857.4` | `2290.2` | `0.774→0.698` | `24.15→21.75` |
| `978` | `vnet` | `0.6765` | `0.5916` | `-0.0850` | `22` | `25→2` | `4281.8` | `3478.7` | `0.724→0.487` | `23.08→34.64` |

## 3. Read-Only Stage 5C Prep Summary

- **Case `930`:** Native GT has `2` large balanced trees (`4,174.06 mm³` and `4,057.82 mm³`). Under `R4` Top-2 CC, `RASNet` had `4,023.7 mm³` removed (`recall 0.790→0.447`, `Dice 0.6814→0.4703`, `-0.2112`, `HD95 44.30→57.40 mm`), indicating an entire major coronary tree was dropped (either fragmented into ≥2 pieces or outranked by a larger false-positive component), whereas `nnU-Net` gained `+0.1776` (`0.6483→0.8260`, `HD95 48.20→4.32 mm`) as Top-2 CC removed `5,218.8 mm³` of false positives (`recall 0.806→0.814`).
- **Case `948`:** Native GT has `2` large trees (`5,350.67 mm³`, `4,048.32 mm³`) plus an `8`-voxel (`0.59 mm³`) fragment. All 4 models dropped `-0.1060` to `-0.1369` Dice under Top-2 CC with `2,388.8–2,838.3 mm³` removed (`HD95 0.74–0.91 mm → 51.13–52.24 mm`) and sharp recall drops, confirming that all 4 models fragmented one major arterial tree into ≥2 disconnected segments, causing hard `keep_top_k=2` to amputate a large true branch.
- **Case `861`:** Native GT has `2` trees (`2,321.39 mm³`, `2,212.39 mm³`). All 4 models dropped `-0.093` to `-0.113` Dice under Top-2 CC with `1,006–1,278 mm³` removed (`HD95` jumping from `8–28 mm` to `33–49 mm`), again showing symmetric third-branch amputation.
- **Case `978`:** Native GT is a single connected tree (`6,622.23 mm³`). `RASNet` (`-0.0784`, `2,868.0 mm³` removed) and `V-Net` (`-0.0850`, `3,478.7 mm³` removed) suffered major branch pruning under Top-2 CC, while `nnU-Net` (`+0.0180`) and `SegResNet` (`+0.0377`) gained slightly.
- **Next step if Stage 5C is requested:** Pre-register an inference-only run in `BENCHMARK_PREREGISTRATION.md`, export or retain 8-pass probabilities on Val-94 for all 4 models, and evaluate symmetric physical/relative CC volume-floor rules (e.g., retain CCs `>= V_min` in `{200, 350, 500} mm³` or `>= 10%` of largest CC, capped at top-3) on Val-94 only before any Test-150 evaluation.
