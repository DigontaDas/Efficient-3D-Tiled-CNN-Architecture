#!/usr/bin/env python3
"""
prepare_and_run_nnunet_200ep.py — Setup & Train nnU-Net V2 to 200 Genuine Epochs
=================================================================================
- Dataset: Dataset501_CoronarySeg (linked via NTFS zero-copy hardlinks from archive/)
- Splits: Injects splits_final.json into preprocessed folder (fold 0: 690 train / 94 val)
- Trainer: nnUNetTrainer_200epochs (num_epochs=200 with rescaled polynomial decay)
- Telemetry: Parses training log to output logs/nnunet_200ep_log.csv in unified format
"""

import os
import sys
import re
import json
import time
import shutil
import subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Phase3_Local_Integration")))
import dataset_paths

def setup_environment(base_dir):
    nnunet_raw = os.path.join(base_dir, "nnUNet_raw")
    nnunet_preprocessed = os.path.join(base_dir, "nnUNet_preprocessed")
    nnunet_results = os.path.join(base_dir, "nnUNet_results")

    os.environ["nnUNet_raw"] = nnunet_raw
    os.environ["nnUNet_preprocessed"] = nnunet_preprocessed
    os.environ["nnUNet_results"] = nnunet_results

    os.makedirs(nnunet_raw, exist_ok=True)
    os.makedirs(nnunet_preprocessed, exist_ok=True)
    os.makedirs(nnunet_results, exist_ok=True)

    return nnunet_raw, nnunet_preprocessed, nnunet_results

def link_dataset(nnunet_raw, splits_file):
    dset_dir = os.path.join(nnunet_raw, "Dataset501_CoronarySeg")
    img_tr_dir = os.path.join(dset_dir, "imagesTr")
    lbl_tr_dir = os.path.join(dset_dir, "labelsTr")
    os.makedirs(img_tr_dir, exist_ok=True)
    os.makedirs(lbl_tr_dir, exist_ok=True)

    with open(splits_file) as f:
        splits = json.load(f)
    train_ids = splits["train"]
    val_ids = splits["val"]
    all_case_ids = sorted(train_ids + val_ids)

    print(f"[*] Linking {len(all_case_ids)} cases into Dataset501_CoronarySeg...")
    linked = 0
    for cid in all_case_ids:
        src_img = dataset_paths.find_image(cid)
        src_lbl = dataset_paths.find_label(cid)
        if not src_img or not src_lbl:
            continue

        dst_img = os.path.join(img_tr_dir, f"case_{cid:04d}_0000.nii.gz")
        dst_lbl = os.path.join(lbl_tr_dir, f"case_{cid:04d}.nii.gz")

        if not os.path.exists(dst_img):
            try:
                os.link(src_img, dst_img)
            except Exception:
                import shutil
                shutil.copy2(src_img, dst_img)

        if not os.path.exists(dst_lbl):
            try:
                os.link(src_lbl, dst_lbl)
            except Exception:
                import shutil
                shutil.copy2(src_lbl, dst_lbl)

        linked += 1

    print(f"[*] Successfully linked {linked} training/validation case pairs.")

    # Create dataset.json
    dset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": linked,
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(dset_dir, "dataset.json"), "w") as f:
        json.dump(dset_json, f, indent=4)

    return all_case_ids, train_ids, val_ids

def inject_splits(nnunet_preprocessed, train_ids, val_ids):
    prep_dset_dir = os.path.join(nnunet_preprocessed, "Dataset501_CoronarySeg")
    os.makedirs(prep_dset_dir, exist_ok=True)

    # Format IDs matching case names
    train_cases = [f"case_{cid:04d}" for cid in train_ids]
    val_cases = [f"case_{cid:04d}" for cid in val_ids]

    splits = [{"train": train_cases, "val": val_cases}]
    splits_dst = os.path.join(prep_dset_dir, "splits_final.json")
    with open(splits_dst, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"[*] Injected thesis splits into {splits_dst} (fold 0: {len(train_cases)} train, {len(val_cases)} val).")

def parse_nnunet_log_to_unified_csv(log_path, output_csv):
    if not os.path.exists(log_path):
        return

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    history = []
    current_ep = None
    ep_loss = np.nan
    val_loss = np.nan
    val_dice = np.nan
    lr = np.nan
    ep_time = np.nan
    cum_time = 0.0

    for line in lines:
        m_ep = re.search(r"\bEpoch\s+(\d+)\s*$", line)
        if m_ep:
            if current_ep is not None:
                history.append({
                    "epoch": current_ep + 1,
                    "train_loss": ep_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "learning_rate": lr,
                    "epoch_time_seconds": ep_time,
                    "cumulative_time_hours": cum_time / 3600.0
                })
            current_ep = int(m_ep.group(1))
            ep_loss = np.nan
            val_loss = np.nan
            val_dice = np.nan
            lr = np.nan
            ep_time = np.nan
            continue

        m_lr = re.search(r"Current learning rate:\s*([0-9eE\.\-]+)", line)
        if m_lr:
            lr = float(m_lr.group(1))

        m_tr = re.search(r"train_loss\s*([0-9eE\.\-]+)", line)
        if m_tr:
            ep_loss = float(m_tr.group(1))

        m_va = re.search(r"val_loss\s*([0-9eE\.\-]+)", line)
        if m_va:
            val_loss = float(m_va.group(1))

        m_dice = re.search(r"Pseudo dice\s*\[np\.float32\(([0-9\.]+)\)\]", line)
        if m_dice:
            val_dice = float(m_dice.group(1))

        m_time = re.search(r"Epoch time:\s*([0-9\.]+)\s*s", line)
        if m_time:
            ep_time = float(m_time.group(1))
            cum_time += ep_time

    if current_ep is not None:
        history.append({
            "epoch": current_ep + 1,
            "train_loss": ep_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "learning_rate": lr,
            "epoch_time_seconds": ep_time,
            "cumulative_time_hours": cum_time / 3600.0
        })

    if history:
        pd.DataFrame(history).to_csv(output_csv, index=False)
        print(f"[*] Parsed {len(history)} nnU-Net epochs to {output_csv}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    splits_file = os.path.abspath(os.path.join(base_dir, "..", "..", "Phase3_Local_Integration", "splits_final.json"))

    nnunet_raw, nnunet_prep, nnunet_res = setup_environment(base_dir)
    all_ids, train_ids, val_ids = link_dataset(nnunet_raw, splits_file)
    inject_splits(nnunet_prep, train_ids, val_ids)

    python_exe = sys.executable

    # 1. Plan and Preprocess
    prep_dset_dir = os.path.join(nnunet_prep, "Dataset501_CoronarySeg")
    raw_dset_dir = os.path.join(nnunet_raw, "Dataset501_CoronarySeg")
    plans_file = os.path.join(prep_dset_dir, "nnUNetPlans.json")
    prep_data_dir = os.path.join(prep_dset_dir, "nnUNetPlans_3d_fullres")
    prep_dataset_json = os.path.join(prep_dset_dir, "dataset.json")
    raw_dataset_json = os.path.join(raw_dset_dir, "dataset.json")

    # Ensure dataset.json exists in preprocessed directory
    if os.path.exists(raw_dataset_json):
        shutil.copy2(raw_dataset_json, prep_dataset_json)

    # Verify if preprocessed files actually exist
    needs_prep = not os.path.exists(prep_data_dir) or not os.path.exists(plans_file)
    if not needs_prep:
        npy_files = [f for f in os.listdir(prep_data_dir) if f.endswith(".npy") or f.endswith(".npz")]
        if len(npy_files) == 0:
            needs_prep = True

    if needs_prep:
        print("[*] Running nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres -np 8 --clean...")
        cmd_prep = [
            python_exe, "-m", "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
            "-d", "501", "-c", "3d_fullres", "-np", "8", "--clean"
        ]
        subprocess.run(cmd_prep, check=True)
        # Ensure dataset.json is in preprocessed folder
        if os.path.exists(raw_dataset_json):
            shutil.copy2(raw_dataset_json, prep_dataset_json)
        # Re-inject splits after preprocessing
        inject_splits(nnunet_prep, train_ids, val_ids)

    # 2. Run nnUNetv2_train targeting 200 epochs
    log_dir = os.path.join(nnunet_res, "Dataset501_CoronarySeg", "nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres", "fold_0")
    cmd_train = [
        python_exe, "-m", "nnunetv2.run.run_training",
        "501", "3d_fullres", "0",
        "-tr", "nnUNetTrainer_200epochs"
    ]
    if os.path.exists(os.path.join(log_dir, "checkpoint_latest.pth")):
        print("[*] Detected existing nnU-Net checkpoint. Resuming training with --c...")
        cmd_train.append("--c")

    print(f"[*] Launching nnU-Net training: {' '.join(cmd_train)}...")
    subprocess.run(cmd_train, check=True)

    # 3. Locate training log, parse, and collect checkpoints
    log_dir = os.path.join(nnunet_res, "Dataset501_CoronarySeg", "nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres", "fold_0")
    if os.path.exists(log_dir):
        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("training_log") and f.endswith(".txt")]
        if log_files:
            log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            unified_csv = os.path.join(base_dir, "logs", "nnunet_200ep_log.csv")
            parse_nnunet_log_to_unified_csv(log_files[0], unified_csv)

        # Preserve checkpoints in main checkpoints/ folder
        ckpt_target_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(ckpt_target_dir, exist_ok=True)
        name_map = {
            "checkpoint_best.pth": "nnunet_best.pt",
            "checkpoint_final.pth": "nnunet_last.pt",
            "checkpoint_epoch_50.pth": "nnunet_epoch_50.pt",
            "checkpoint_epoch_100.pth": "nnunet_epoch_100.pt",
            "checkpoint_epoch_150.pth": "nnunet_epoch_150.pt",
            "checkpoint_epoch_200.pth": "nnunet_epoch_200.pt",
        }
        for f in os.listdir(log_dir):
            if f.startswith("checkpoint_") and f.endswith(".pth"):
                src_ckpt = os.path.join(log_dir, f)
                target_name = name_map.get(f, f"nnunet_{f}")
                dst_ckpt = os.path.join(ckpt_target_dir, target_name)
                shutil.copy2(src_ckpt, dst_ckpt)
                print(f"[*] Preserved checkpoint: {dst_ckpt}")

        # 4. Automatic Cache Management & Space Recovery
        # Training is complete, and all checkpoints/logs are safely preserved in checkpoints/ and logs/.
        # Remove the temporary preprocessed 3D arrays to immediately recover ~125GB of disk space.
        if os.path.exists(prep_data_dir):
            print(f"[*] Reclaiming temporary preprocessed cache from {prep_data_dir}...")
            shutil.rmtree(prep_data_dir, ignore_errors=True)
            print("[OK] Temporary nnU-Net preprocessed cache successfully purged. Headroom restored for Stages 3-5.")

if __name__ == "__main__":
    main()
