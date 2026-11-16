# RASNet-v2 Execution State & Resume Handoff

- **Saved At**: `2026-11-15T16:23+06:00`
- **Status**: **Cleanly Paused (Mid-Stage 2A, Job 15/18)** — all background processes (`task-428`) and monitoring timers (`task-430`) stopped cleanly; zero `python` processes running; no `.tmp` checkpoint files left; GPU idle (`1412 / 16,376 MiB` desktop baseline).
- **Environment**: `C:\Thesis_RASNET_env\bench\Scripts\python.exe` | `NVIDIA GeForce RTX 4080 SUPER` (`16,376 MiB` VRAM) | Disk `C:\` Free: **`107.68 GB`** | `HEAD`: `22d4581` (Option A committed)
- **Git Discipline**: Strict `Rasnet-v2/` isolation (`git status --short` confirms zero changes to verified `v1` files). Caches (`Rasnet-v2/cache_fp16/`, `Rasnet-v2/nnUNet_work/nnUNet_raw/`, `Rasnet-v2/nnUNet_work/nnUNet_preprocessed/`, `Rasnet-v2/results/val94_final/_tmp_probs/`, `.pytest_cache/`) are in `.gitignore`; `Rasnet-v2/nnUNet_work/nnUNet_results/` and `Rasnet-v2/checkpoints/` remain **un-ignored**.

---

## 1. Completed & Verified Milestones

1. **Pre-Stage Testing**:
   - Combined `v1` + `v2` suite verified (`63 passed` before runtime BF16 update); after the BF16 + cache-lifecycle update, `C:\Thesis_RASNET\Rasnet-v2\tests` verified (`32 passed`).
2. **Stage 1 Hardware & Determinism Gates (`src/run_stage1_gates.py`) — ALL PASSED (`2026-11-15T09:47:31Z`)**:
   - Report: `Rasnet-v2/logs/STAGE1_GATES_REPORT.json`
   - Gate 2 (`Val20` deterministic manifest): `PASS` (`val20_sha256: cd29dd64242cc26ee6e2c44f0c930bf96b3a6c2d6675f3faea60ab38b0f084ac`)
   - Gate 3 (`f16` 9322.2 MB / `f32` 8131.2 MB peak VRAM, CPU epoch & mid-epoch resume parity `0.0`, `nnU-Net` batch-vs-single parity `0.0`): `PASS`
   - Gate 4 (`surface-v1` parity `0.0` diff, CSV resume round-trip max diff `7.11e-15`): `PASS`
3. **Training & Validation FP16 Cache (`cache_fp16/`) — `784 / 784` Complete**:
   - Manifest: `Rasnet-v2/cache_fp16/CACHE_MANIFEST.json` (`153.57 GB`, `690 train` + `94 val` foreground-cropped volumes).
   - `nnU-Net` raw NTFS hardlinks (`nlink = 2`, sharing storage with `secondary-imagcas-dataset`) + `.b2nd`/`.pkl` preprocessed blobs ready on disk (`nnUNet_preprocessed = 118.76 GB`).
4. **Stability & Lifecycle Engineering Fixes Applied**:
   - **Hardware `bfloat16` AMP** (`src/train_fair_v2.py`, `src/eval_fair_v2.py`, `src/stage5_stats_and_efficiency.py`): replaced IEEE `float16` (which overflowed `±65504` in `SegResNet`'s `up_layers.2` decoder sum) with hardware `bfloat16` (`±3.4e38`, no `GradScaler`), with a 25-step non-finite fail-fast guard.
   - **VRAM & Disk Lifecycle Management** (`src/train_fair_v2.py`, `src/run_queue.py`): `torch.cuda.empty_cache()` around `val20`, auto-deletion of `mid_epoch_state.pt` once `DONE.flag` is written, disk headroom checks (`min_free_gb=25.0`), and automatic cleanup in `--stage stage3` of `cache_fp16/` (`~153.57 GB`) after the 19 MONAI runs and `nnUNet_preprocessed/` binary blobs (`~118.76 GB`) + `nnUNet_raw/` hardlinks after the 3 nnU-Net seeds.

---

## 2. Stage 2A Status (`14 / 18` Complete, `1` Mid-Epoch Saved, `3` Pending)

Queue state in `Rasnet-v2/logs/queue_status.json`: `current_index = 15 / 18` (`pilot2a_rasnet_v2_f32_lr3e-4_seed42`).

| # | Job ID | Status | Last Ep | Final `train_loss` | Last `val20_dice` (Best) |
|---:|---|---|---:|---:|---:|
| 1 | `pilot2a_segresnet_f16_lr1e-4_seed42` | `DONE` | 50 | `0.3109` | `0.6556` (`0.6556`) |
| 2 | `pilot2a_segresnet_f16_lr3e-4_seed42` | `DONE` | 50 | `0.3219` | `0.6267` (`0.6267`) |
| 3 | `pilot2a_segresnet_f16_lr1e-3_seed42` | `DONE` | 50 | `0.4793` | `0.4438` (`0.4438`) |
| 4 | `pilot2a_rasnet_v2_f16_lr1e-4_seed42` | `DONE` | 50 | `0.5276` | `0.7049` (`0.7049`) |
| 5 | `pilot2a_rasnet_v2_f16_lr3e-4_seed42` | `DONE` | 50 | `0.5365` | `0.6877` (`0.6877`) |
| 6 | `pilot2a_rasnet_v2_f16_lr1e-3_seed42` | `DONE` | 50 | `0.4934` | `0.6973` (`0.6973`) |
| 7 | `pilot2a_vnet_f16_lr1e-4_seed42` | `DONE` | 50 | `0.4853` | `0.1483` (`0.4594`) |
| 8 | `pilot2a_vnet_f16_lr3e-4_seed42` | `DONE` | 50 | `0.3346` | `0.7058` (`0.7058`) |
| 9 | `pilot2a_vnet_f16_lr1e-3_seed42` | `DONE` | 50 | `0.3435` | `0.6946` (`0.6946`) |
| 10 | `pilot2a_unet3d_f16_lr1e-4_seed42` | `DONE` | 50 | `0.3391` | `0.6895` (`0.6909`) |
| 11 | `pilot2a_unet3d_f16_lr3e-4_seed42` | `DONE` | 50 | `0.3334` | `0.6886` (`0.6886`) |
| 12 | `pilot2a_unet3d_f16_lr1e-3_seed42` | `DONE` | 50 | `0.3474` | `0.6378` (`0.6449`) |
| 13 | `pilot2a_segresnet_f32_lr3e-4_seed42` | `DONE` | 50 | `0.3178` | `0.6348` (`0.6348`) |
| 14 | `pilot2a_segresnet_f32_lr1e-3_seed42` | `DONE` | 50 | `0.2819` | `0.6763` (`0.6763`) |
| 15 | `pilot2a_rasnet_v2_f32_lr3e-4_seed42` | **PAUSED (Resumable)** | 35 (`step 8900`, Ep 36 `150/250`) | `0.4920` | `0.6165` at Ep 20 |
| 16 | `pilot2a_rasnet_v2_f32_lr1e-3_seed42` | `PENDING` | — | — | — |
| 17 | `pilot2a_segresnet_f32_lr1e-4_seed42` | `PENDING` | — | — | — |
| 18 | `pilot2a_rasnet_v2_f32_lr1e-4_seed42` | `PENDING` | — | — | — |

### Verified Resume Integrity for Job 15
- `checkpoints/pilot2a_rasnet_v2_f32_lr3e-4_seed42/mid_epoch_state.pt`: loaded cleanly on CPU (`next_epoch=36`, `next_step_in_epoch=150`, `global_opt_step=8900`, `best_val_dice=0.6165`, `elapsed_gpu_sec=4488.52`, `config_sha256=ab49252bf8a11047b642f76371163d9a27a94c654ba6ccfd8229e9f6f4ce97c1`, `sha256=a757487b799d8d5598049488f6702352f8631d7f78ae2876f5aacc500536ad69`).
- `checkpoints/pilot2a_rasnet_v2_f32_lr3e-4_seed42/latest_state.pt`: loaded cleanly on CPU (`next_epoch=36`, `next_step_in_epoch=0`, `global_opt_step=8750`, `best_val_dice=0.6165`, `elapsed_gpu_sec=4417.52`, `sha256=44833da788921dec5f04d0d15a8668c55f0d0ae3390c2fccabba23866b5c244f`).
- Zero `.tmp` files present under `Rasnet-v2/checkpoints/`.

---

## 3. Exact Commands to Resume Later

Run from `C:\Thesis_RASNET` using `C:\Thesis_RASNET_env\bench\Scripts\python.exe`:

```powershell
$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"
$Q  = "C:\Thesis_RASNET\Rasnet-v2\src\run_queue.py"

# 1. Resume & finish Stage 2A (~0.5 h left on Job 15 + ~5.0 h on Jobs 16–18 ≈ ~5.5 h)
& $PY -u $Q --stage stage2a

# 2. Evaluate Stage 2A on full Val (N=94) and lock LRs (results/pilots_val94/locked_lrs.json)
& $PY -u $Q --stage stage2a_eval

# 3. Run & evaluate Stage 2B architectural/loss pilots (~3 new runs -> pilot_lock_decision.{json,md})
#    STOP if all_gates_passed is false.
& $PY -u $Q --stage stage2b
& $PY -u $Q --stage stage2b_eval

# 4. Run full Stage 3 (19 x 200-ep MONAI -> auto-cleans cache_fp16/ [+153.57 GB] -> 3 x nnU-Net V2 seeds -> auto-cleans nnUNet_preprocessed/ blobs [+118.76 GB] & nnUNet_raw/)
& $PY -u $Q --stage stage3

# 5. Run Stages 4A -> 4B (commit thresholds_v2.json before 4C) -> 4C -> 5
& $PY -u $Q --stage stage4a
& $PY -u $Q --stage stage4b
& $PY -u $Q --stage stage4c
& $PY -u $Q --stage stage5
```

- **Stage 6 (Phase I prospective clinical CT cohort)**: Remains deferred until the doctor-annotated prospective CT cases arrive (`src/stage6_phaseI_clinical.py`).

