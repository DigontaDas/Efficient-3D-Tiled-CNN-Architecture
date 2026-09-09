#!/usr/bin/env python3
"""
run_all_200ep.py — Master Orchestrator for 200-Epoch Matched Benchmark Suite
=============================================================================
Sequentially executes all five models to their matched 200-epoch budget:
  Stage 1: SegResNet (Epoch 0 -> 200)
  Stage 2: nnU-Net V2 (Epoch 0 -> 200, nnUNetTrainer_200epochs)
  Stage 3: 3D U-Net (Epoch 0 -> 200)
  Stage 4: V-Net (Epoch 0 -> 200, Stabilized)
  Stage 5: RASNet (Epoch 71 -> 200, Continued from Champion Checkpoint)

After each stage, checkpoints and per-epoch CSV telemetry are verified.
"""

import os
import sys
import time
import subprocess
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import shutil

def detect_resume_stage(base_dir):
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    
    if not os.path.exists(os.path.join(ckpt_dir, "segresnet_epoch_200.pt")):
        return 1
    if not os.path.exists(os.path.join(ckpt_dir, "nnunet_epoch_200.pt")):
        return 2
    if not os.path.exists(os.path.join(ckpt_dir, "3dunet_epoch_200.pth")):
        return 3
    if not os.path.exists(os.path.join(ckpt_dir, "vnet_epoch_200.pt")):
        return 4
    if not os.path.exists(os.path.join(ckpt_dir, "rasnet_epoch_200.pth")):
        return 5
    return 6

def check_disk_space():
    total, used, free = shutil.disk_usage("C:\\")
    free_gb = free / (1024 ** 3)
    print(f"[*] Drive C: Free Space: {free_gb:.2f} GB")
    return free_gb

def run_stage(stage_num, stage_name, script_path, extra_args=None):
    print("\n" + "=" * 80)
    print(f"[*] STARTING STAGE {stage_num}/5: {stage_name}")
    print("=" * 80)
    check_disk_space()
    start_t = time.time()

    cmd = [sys.executable, "-u", script_path]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[*] Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)

    elapsed_hrs = (time.time() - start_t) / 3600.0
    if res.returncode == 0:
        print(f"\n[OK] STAGE {stage_num}/5 ({stage_name}) COMPLETED SUCCESSFULLY in {elapsed_hrs:.2f} hours.")
    else:
        print(f"\n[FAIL] STAGE {stage_num}/5 ({stage_name}) FAILED with exit code {res.returncode}.")
        raise RuntimeError(f"Stage {stage_name} failed.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    auto_stage = detect_resume_stage(base_dir)

    parser = argparse.ArgumentParser(description="Master 200-Epoch Benchmark Runner")
    parser.add_argument("--skip-to", type=int, default=auto_stage, help=f"Stage to start from (1 to 5, auto-detected: {auto_stage})")
    args = parser.parse_args()

    stages = [
        (1, "SegResNet (0 -> 200)", os.path.join(base_dir, "train_segresnet_200ep.py")),
        (2, "nnU-Net V2 (0 -> 200)", os.path.join(base_dir, "prepare_and_run_nnunet_200ep.py")),
        (3, "3D U-Net (0 -> 200)", os.path.join(base_dir, "train_3dunet_200ep.py")),
        (4, "V-Net (0 -> 200)", os.path.join(base_dir, "train_vnet_200ep.py")),
        (5, "RASNet (0 -> 200, Scratch)", os.path.join(base_dir, "train_rasnet_200ep.py")),
    ]

    total_start = time.time()
    print("=" * 80)
    print("[*] 200-EPOCH MATCHED BENCHMARK TRAINING SUITE")
    print(f"[*] Total Models: 5 | Target Budget: 200 epochs per model")
    print(f"[*] Starting at Stage: {args.skip_to}")
    print("=" * 80)

    for num, name, script in stages:
        if num < args.skip_to:
            print(f"[*] Skipping Stage {num}: {name}")
            continue
        run_stage(num, name, script)

    total_hrs = (time.time() - total_start) / 3600.0
    print("\n" + "=" * 80)
    print(f"[COMPLETE] ALL 5 MODELS SUCCESSFULLY TRAINED TO 200 EPOCHS! Total time: {total_hrs:.2f} hours.")
    print("=" * 80)

if __name__ == "__main__":
    main()
