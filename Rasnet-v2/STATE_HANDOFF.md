# RASNet-v2 Execution State & Resume Handoff

- **Saved At**: `2026-11-12T12:19+06:00`
- **Status**: **Cleanly Paused (Mid-Stage 2A, Job 15/16)** — all background processes and monitoring timers stopped cleanly; no `.tmp` checkpoint files left; GPU released (`601 / 16,376 MiB`).
- **Environment**: `C:\Thesis_RASNET_env\bench\Scripts\python.exe` | `NVIDIA GeForce RTX 4080 SUPER` (`16,376 MiB` VRAM) | Disk `C:\` Free: **`135.88 GB`** (`794.67 GB` used)
- **Git Discipline**: Strict `Rasnet-v2/` isolation (`git status --short` confirms zero changes to verified `v1` files). Caches (`Rasnet-v2/cache_fp16/`, `Rasnet-v2/nnUNet_work/nnUNet_raw/`, `Rasnet-v2/nnUNet_work/nnUNet_preprocessed/`, `Rasnet-v2/results/val94_final/_tmp_probs/`, `.pytest_cache/`) are in `.gitignore`; `Rasnet-v2/nnUNet_work/nnUNet_results/` and `Rasnet-v2/checkpoints/` remain **un-ignored**.

---

## 1. Completed & Verified Milestones

1. **Pre-Stage Testing**:
   - Combined `v1` + `v2` suite verified (`63 passed` before runtime BF16 update); after the BF16 + cache-lifecycle update, `C:\Thesis_RASNET\Rasnet-v2\tests` verified (`32 passed`).
2. **Stage 1 Hardware & Determinism Gates (`src/run_stage1_gates.py`) — ALL PASSED**:
   - Report: `Rasnet-v2/logs/STAGE1_GATES_REPORT.json`
   - Gate 2 (`Val20` deterministic manifest): `PASS` (`val20_sha256: cd29dd64242cc26ee6e2c44f0c930bf96b3a6c2d6675f3faea60ab38b0f084ac`)
   - Gate 3 (`f16` & `f32` peak VRAM, CPU epoch & mid-epoch resume parity, `nnU-Net` batch-vs-single parity): `PASS`
   - Gate 4 (`surface-v1` parity `0.0` diff, CSV resume round-trip max diff `3.55e-15`): `PASS`
3. **Training & Validation FP16 Cache (`cache_fp16/`) — `784 / 784` Complete**:
   - Manifest: `Rasnet-v2/cache_fp16/CACHE_MANIFEST.json` (`153.57 GB`, `690 train` + `94 val` foreground-cropped volumes).
   - `nnU-Net` raw hardlinks + preprocessed blobs also ready on disk (`nnUNet_preprocessed = 118.76 GB`).
4. **Stability & Lifecycle Engineering Fixes Applied**:
   - **Hardware `bfloat16` AMP** (`src/train_fair_v2.py`, `src/eval_fair_v2.py`, `src/stage5_stats_and_efficiency.py`): replaced IEEE `float16` (which overflowed `±65504` in `SegResNet`'s `up_layers.2` decoder sum) with hardware `bfloat16` (`±3.4e38`, no `GradScaler`), with a 25-step non-finite fail-fast guard.
   - **VRAM & Disk Lifecycle Management** (`src/train_fair_v2.py`, `src/run_queue.py`): `torch.cuda.empty_cache()` around `val20`, auto-deletion of `mid_epoch_state.pt` once `DONE.flag` is written, disk headroom checks (`min_free_gb=25.0`), and automatic cleanup in `--stage stage3` of `cache_fp16/` (`~153.57 GB`) after the 17 MONAI runs and `nnUNet_preprocessed/` binary blobs (`~118.76 GB`) + `nnUNet_raw/` hardlinks after the 3 nnU-Net seeds.

---

## 2. Stage 2A Status (`14 / 16` Complete, `1` Mid-Epoch Saved, `1` Pending)

Queue state in `Rasnet-v2/logs/queue_status.json`: `current_index = 15 / 16` (`pilot2a_rasnet_v2_f32_lr3e-4_seed42`).

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
| 15 | `pilot2a_rasnet_v2_f32_lr3e-4_seed42` | **PAUSED (Resumable)** | 29 (`step 7450`, Ep 30 `200/250`) | `0.5439` | `0.6165` at Ep 20 |
| 16 | `pilot2a_rasnet_v2_f32_lr1e-3_seed42` | `PENDING` | — | — | — |

### Verified Resume Integrity for Job 15
- `checkpoints/pilot2a_rasnet_v2_f32_lr3e-4_seed42/mid_epoch_state.pt`: loaded cleanly on CPU (`next_epoch=30`, `next_step_in_epoch=200`, `global_opt_step=7450`, `best_val_dice=0.6165`, `elapsed_gpu_sec=3821.47`, `config_sha256=ab49252bf8a11047b642f76371163d9a27a94c654ba6ccfd8229e9f6f4ce97c1`).
- `checkpoints/pilot2a_rasnet_v2_f32_lr3e-4_seed42/latest_state.pt`: loaded cleanly on CPU (`next_epoch=30`, `next_step_in_epoch=0`, `global_opt_step=7250`, `best_val_dice=0.6165`).
- Zero `.tmp` files present under `Rasnet-v2/checkpoints/`.

---

## 3. Exact Commands to Resume Later

Run from `C:\Thesis_RASNET` using `C:\Thesis_RASNET_env\bench\Scripts\python.exe`:

```powershell
$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"
$Q  = "C:\Thesis_RASNET\Rasnet-v2\src\run_queue.py"

# 1. Resume & finish Stage 2A (~49 min left on Job 15 + ~113 min on Job 16 ≈ ~2.7 h)
& $PY -u $Q --stage stage2a

# 2. Evaluate Stage 2A on full Val (N=94) and lock LRs (results/pilots_val94/locked_lrs.json)
& $PY -u $Q --stage stage2a_eval

# 3. Run & evaluate Stage 2B architectural/loss pilots (~3 deduplicated runs -> pilot_lock_decision.{json,md})
& $PY -u $Q --stage stage2b
& $PY -u $Q --stage stage2b_eval

# 4. Run full Stage 3 (17 x 200-ep MONAI -> auto-cleans cache_fp16/ [+153.57 GB] -> 3 x nnU-Net V2 seeds -> auto-cleans nnUNet_preprocessed/ blobs [+118.76 GB] & nnUNet_raw/)
& $PY -u $Q --stage stage3

# 5. Run Stages 4A -> 4B -> 4C -> 5
& $PY -u $Q --stage stage4a
& $PY -u $Q --stage stage4b
& $PY -u $Q --stage stage4c
& $PY -u $Q --stage stage5
```

- **Stage 6 (Phase I prospective clinical CT cohort)**: Remains deferred until the doctor-annotated prospective CT cases arrive (`src/stage6_phaseI_clinical.py`).

