# `Rasnet-v2` Stage 1 Code-Review Audit Log (`/teamwork` Gate)

- **Reviewer**: Senior Code Reviewer subagent (`7f520571-ca45-4469-a692-691c0198224b`) + live GPU telemetry audit
- **Scope**: All modules under [`C:\Thesis_RASNET\Rasnet-v2\`](file:///C:/Thesis_RASNET/Rasnet-v2)

---

## Findings & Verified Remediations

1. **CRITICAL (`C1`) — In-Loop Validation VRAM Spike (`10,443 MB` / PCIe Paging at `sw_batch_size=8` + Full-Volume GPU Loss)**:
   - **Live Discovery**: During Stage 1 Gate 3 run `_gate3_f16_unint`, Epoch 2 validation spiked PyTorch VRAM to `10,442.9 MB` (`15,942 / 16,376 MiB` total GPU, causing WDDM PCIe paging and `546.26 s` for 3 cases) due to holding full-volume `(1, 2, 480, 480, 220)` logits and computing `SoftclDiceLoss` on GPU over ~50M voxels.
   - **Remediation Applied**:
     - Reduced `inference_eval.sw_batch_size` to `4` in `configs/fair_v2.yaml` (`min(4, 2)` for `F >= 32`).
     - Switched `sliding_window_inference` in `train_fair_v2.py` (`evaluate_cached_val20`) and `eval_fair_v2.py` (`infer_monai_native_zyx`) to `sw_device=device, device=torch.device("cpu")`, computing stitched outputs and `val_dice`/`soft_dice_loss` on CPU (`2,910.2 MB` peak VRAM, `3.24 s` full volume in microbenchmark — `~56×` speedup and zero WDDM paging).

2. **MINOR (`M1`) — Mid-Epoch Resume `train_loss` Divisor (`train_fair_v2.py`)**:
   - **Remediation Applied**: Tracked `steps_executed = max(steps_done_ep - resume_step_in_epoch, 1)` prior to resetting `resume_step_in_epoch` so resumed epochs normalize `running_loss` cleanly.

3. **MINOR (`M2`, `M3`, `M5`) — Evaluation Completeness (`eval_fair_v2.py`)**:
   - **Remediation Applied**: Wired `postprocess_mode="top_k" | "seeded"` (`seeded_component_filter_3d`) into both MONAI and native `nnU-Net V2` evaluators, implemented `evaluate_nnunet_checkpoint_to_csv`, and added `argparse` CLI `main()`.

4. **MINOR (`M4`) — Full Stage Queue Orchestration (`run_queue.py`)**:
   - **Remediation Applied**: Added `build_stage2b_jobs` (`5` design sweeps) and `build_stage3_jobs` (`20` runs: `1+3+3+1+1+3+3+1+1+3`) alongside `build_stage2a_jobs`.

5. **MINOR (`M6`) — Axis Naming Clarity (`cache_builder.py`)**:
   - **Remediation Applied**: Renamed `fg_zyx` -> `fg_xyz` matching MONAI `(1, X, Y, Z)` storage order.

---

## Visual Overlay & Parity Sign-Off
- **Visual Patch Overlays (`logs/patch_overlays/patch_overlay_case_{0014,0045,0047}.png`)**: Inspected visually; Axial (`XY`), Coronal (`XZ`), and Sagittal (`YZ`) green contours align pixel-for-pixel with contrast-enhanced coronary arteries (`0` axis flip or affine mismatch).
- **Immutability**: `git diff` confirms zero modifications to verified `v1` files.

---

## `/boost` Verification & Stage 1 Hardware Gate Sign-Off (`2026-11-11`)
- **Automated `/boost` Fixes**:
  - Replaced silent zero-gradient MONAI `FocalLoss(include_background=False, use_softmax=True)` with fused FP32 2-class softmax `FocalLoss(gamma=1.5)` + `DiceLoss` in `src/rasnet_v2_loss.py` (verified non-zero finite gradients matching MONAI 2-class `DiceFocal` parity).
  - Corrected deep-supervision `ds_weights` to `(0.25, 0.5)` aligned with `[1/4, 1/2]` decoder outputs (`rasnet_v2_loss.py` & `configs/fair_v2.yaml`).
  - Fixed `uint8(255)` wraparound risk via per-key `SpatialPadd` (`cache_builder.py` / `augmentations.py`), fixed `unet3d` `sw_batch_size` throttling (`train_fair_v2.py` & `eval_fair_v2.py`), verified all `784` `nnU-Net` `.b2nd`/`.pkl` preprocessed scans (`train_nnunet_v2_ref.py`), restored `prev_num_threads` on `nnU-Net` exit, and implemented Stage 2–6 modules (`study_registry.py`, `stage2_pilot_eval.py`, `stage4a_val20_curves.py`, `stage4b_val94_lock.py`, `stage4c_test150_eval.py`, `stage5_stats_and_efficiency.py`, `stage6_phaseI_clinical.py`).
- **Automated Test Suite (`task-1826`)**: `pytest Rasnet-v2/tests tests -v` passed **`63/63`** tests on CUDA + CPU (`122.72s`).
- **Live Hardware Gates (`logs/STAGE1_GATES_REPORT.json`)**:
  - `F=16 (mb=2x1)` Peak VRAM `9,322.2 MB` and `F=32 (mb=1x2)` Peak VRAM `8,131.2 MB` (both `< 10.5 GB`).
  - Deterministic CPU resume parity: `0.00e+00` weight, optimizer `exp_avg`, and `train_loss` difference across both epoch-boundary and `mid_epoch_state.pt` step resumes.
  - `FairV2nnUNetTrainer` (`seed=999`) Ep 2->4 resume passed with `checkpoint_epoch_{2,4}.pth`, `checkpoint_best.pth`, and **`0.00e+00`** 2-case (`0014`, `0045`) batch-vs-single probability parity (`1pass` and `8-pass` mirroring TTA).
  - Gate 4 Case `0014` native `(275, 512, 512)` `Invertd` + `surface-v1` parity: `0.00e+00` vs Round 1 (`metrics.py`), `3.55e-15` CSV decimal roundtrip parity.

---

## Stage 2A Live Runtime Remediations (`2026-11-11` Check Cycle)
1. **WDDM Val20 Spike & Transient State Cleanup (`train_fair_v2.py`, `run_queue.py`)**:
   - Added `torch.cuda.empty_cache()` before and after `evaluate_cached_val20()`, reducing validation reserved VRAM by `2,884 MB` on `rasnet_v2_f16`.
   - Added automatic `mid_epoch_state.pt` removal on `DONE.flag` creation, live disk inventory logging in `logs/queue_status.json`, and stage-gated cleanup (`cache_fp16/` deleted only after all 17 Stage 3 MONAI `DONE.flag`s; `nnUNet_preprocessed/` binary blobs deleted only after all 3 `nnU-Net V2` `DONE.flag`s).
2. **`float16` Overflow in `SegResNet` Decoder (`up_layers.2`) at `lr=3e-4` (`train_fair_v2.py`, `eval_fair_v2.py`, `stage5_stats_and_efficiency.py`)**:
   - **Live Finding**: `pilot2a_segresnet_f16_lr3e-4_seed42` produced `loss=nan` from Epoch 7 onward (`val20_dice=0.0000`) despite finite weights (`nan_keys=0`, `max_finite_w=1.253`); forward hooks confirmed `up_layers.2` overflowed IEEE `float16` (`±65504`), driving `GradScaler` scale to `0.0`. Same checkpoint evaluated cleanly in `bfloat16` (`[-0.2109, 0.9453]`) matching FP32 (`[-0.2075, 0.9472]`).
   - **Remediation Applied**: Switched MONAI AMP across training and evaluation to hardware `bfloat16` (`resolve_amp_dtype`), bound `amp_dtype` into `cfg_hash`, added a 25-step consecutive non-finite loss fail-fast guard, and reset the two initial `float16` pilots (`lr=1e-4` & `lr=3e-4`) so all 16 Stage 2A pilots run identically in `bfloat16`.

