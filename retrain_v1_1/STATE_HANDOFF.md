# Strategy A (`retrain_v1_1/`) Execution State & Resume Handoff

- **Saved At:** `2026-11-24T17:41+06:00` (`2026-11-24T11:41Z`)
- **Status:** **Cleanly Paused (Mid-Stage 5 Retrospective Test-150 Evaluation — `158 / 900` model-cases saved)** — background job (`task-565`) and 1-hour watchdog timer (`task-595`) stopped cleanly; zero `python` processes running; zero `.tmp` files; GPU idle (`41 °C`, `718 / 16,376 MiB` VRAM).
- **Environment:** `C:\Thesis_RASNET_env\bench\Scripts\python.exe` | `NVIDIA GeForce RTX 4080 SUPER` (`16,376 MiB`) | Disk `C:\` Free: **`256.18 GB`** | `HEAD`: `136c66d` (`4` local commits ahead of `origin/main`, `git.dirty_code = false`, zero pushes).
- **Pre-Registration:** `BENCHMARK_PREREGISTRATION.md` Section 7 (approved & committed at `9d108e2`; Val-94 lock committed at `136c66d`).

---

## 1. Completed & Verified Stages

1. **Stage 1 — Transform-Hashed Cache (`--stage cache`) — `784 / 784` COMPLETE:**
   - Directory: `retrain_v1_1/cache/tf_ac5dfd145db81bfa/` (`51.24 GB`, lossless `blosc2` ZSTD-1 `float16` img + `uint8` lbl + `fg/bg` coords)
   - `transform_sha256`: `ac5dfd145db81bfa701d78517bcd46251097c37298ee7ac0a275a44306d8307a`

2. **Stage 2 — 200-Epoch Sequential Training (`--stage train`) — `5 / 5` COMPLETE (`DONE.flag`, all peaked @ ep `200`):**

| # | Model | Params | Raw `Val-94` Dice (`τ=0.50`, no CC) | Verified `best_model.pth` / `final_epoch200.pth` SHA-256 |
|---:|---|---:|---:|---|
| 1 | `rasnet_v1_1` | `4,706,657` | **`0.7940`** | `efa090871fa0ec6bd4b6f57cddb1d343f374488c8e0c14a9481b5efc6e092ab8` |
| 2 | `ablation_no_attn_v1_1` | `4,701,110` | **`0.7917`** | `a146dda1ed79b97e798d58969b8ee26e3e331df61526fb8b7bf493a90b0b84d5` |
| 3 | `ablation_no_ds_v1_1` | `4,706,461` | **`0.7908`** | `819f4035f3c97d79b76d7c89526b0f6a2a4bc1d66e1ca9bd2371eef8dac80ac3` |
| 4 | `segresnet_v1_1` | `4,700,914` | **`0.7872`** | `a512e2f72567f4f3a18a1b3072fdb5e3a48023081200873633a26e5b88d39f48` |
| 5 | `vnet_v1_1` | `45,599,452` | **`0.7783`** | `f7f8577274d4deb74e989f6352bfc729b9f235e0bf0cc002d60d27520592ceea` |
| 6 | `nnunet_v2` (frozen reuse) | `30,788,004` | — | `9accddf283b1c480e386f30f426b9200ca8d4aa1ecd70d5b5af0b8aad6058711` |

   - Recorded in `retrain_v1_1/results/checkpoint_hashes.json` and `CHECKPOINTS.md` §1B (committed at `136c66d`).

3. **Stage 3 & 4 — Symmetric Val-94 Grid (`[0.10..0.70]`) & Threshold Lock (`--stage val`, `--stage lock`) — COMPLETE & COMMITTED (`136c66d`):**
   - `retrain_v1_1/results/val_grid_scores.csv`: `14,664 / 14,664` rows (`6` models × `94` Val cases × `26` mode-thresholds).
   - `retrain_v1_1/results/locked_recipes.json` (`SHA-256: f94866305a202a13243814de6555632a05afed30058acd47ed7836b33acd0bbb`) + `.sha256` sidecar.
   - All `6` models peaked strictly interior (zero lower-edge clamp):
     - `rasnet_v1_1`: 1-pass `τ*=0.50` (`0.8040`) | TTA-8 `τ*=0.50` (`0.8057`)
     - `ablation_no_ds_v1_1`: 1-pass `τ*=0.50` (`0.8008`) | TTA-8 `τ*=0.40` (`0.8051`)
     - `ablation_no_attn_v1_1`: 1-pass `τ*=0.50` (`0.7999`) | TTA-8 `τ*=0.50` (`0.8038`)
     - `segresnet_v1_1`: 1-pass `τ*=0.50` (`0.7983`) | TTA-8 `τ*=0.50` (`0.8006`)
     - `vnet_v1_1`: 1-pass `τ*=0.40` (`0.7800`) | TTA-8 `τ*=0.35` (`0.7816`)
     - `nnunet_v2`: 1-pass `τ*=0.35` (`0.8075`) | TTA-8 `τ*=0.20` (`0.8098`)

---

## 2. Paused Stage — Retrospective Test-150 (`--stage test`)

- **File:** `retrain_v1_1/results/test_per_case.csv` (`147,978` bytes, tail verified intact via `repair_csv_tail`, `632 / 3,600` unique rows, `158 / 900` complete `4`-recipe model-cases, `0` partial cases, `0` empty predictions, `0` NaN Dice).
- **Per-model progress:**
  - `rasnet_v1_1`: **`150 / 150` DONE** (`600` rows — automatically skipped on resume)
  - `segresnet_v1_1`: **`8 / 150` DONE** (`32` rows, up to `case_id=858` — automatically resumes at **`case_id=859`**)
  - `vnet_v1_1`: `0 / 150` PENDING
  - `ablation_no_attn_v1_1`: `0 / 150` PENDING
  - `ablation_no_ds_v1_1`: `0 / 150` PENDING
  - `nnunet_v2`: `0 / 150` PENDING

---

## 3. Exact Commands to Resume Later (PowerShell, `C:\Thesis_RASNET`)

```powershell
$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"
Set-Location C:\Thesis_RASNET

# 1. Resume one-shot retrospective Test-150 evaluation (skips completed 158/900 model-cases automatically)
& $PY -u -m retrain_v1_1.src.run_strategy_a --stage test

# 2. Run Val-only timing/VRAM efficiency benchmark (n=10 Val cases + 1 warm-up per model)
& $PY -u -m retrain_v1_1.src.run_strategy_a --stage timing

# 3. Generate final summary CSVs, 10k-bootstrap paired Wilcoxon + Holm-Bonferroni stats (FA-FE),
#    non-inferiority (-0.005), STRATEGY_A_RESULTS.md, and publication figures
& $PY -u -m retrain_v1_1.src.run_strategy_a --stage stats
```

- **Reminder after `--stage timing`:** also run a quick read-only single-patch GFLOPs/VRAM table (`(1, 1, 160, 160, 96)` MONAI / `(1, 1, 96, 160, 160)` nnU-Net) into `retrain_v1_1/results/patch_efficiency_rtx4080super.csv` to satisfy Part II §3.5.
- **Git Guardrail:** Do **not** `git push` without explicit user authorization; label all Test-150 results **retrospective**.
