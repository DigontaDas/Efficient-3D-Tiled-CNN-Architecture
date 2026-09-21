#!/usr/bin/env python3
r"""
train_rasnet_recall_tuned.py — Recall-Boosted RASNet Training on RTX 3060 Ti (8 GB VRAM)
========================================================================================
Key Enhancements:
1. Compound Loss: StenosisAwareLoss (Dice + Focal) + Soft-Tversky Loss (alpha=0.3, beta=0.7).
   Beta=0.7 heavily penalizes false negatives (missed distal vessels), boosting recall & Dice.
2. Max GPU Throughput on RTX 3060 Ti:
   - batch_size = 2 with num_samples = 4 (total 8 patches of 96x96x96 per forward pass).
   - Mixed Precision fp16 (torch.amp.autocast + GradScaler).
   - DataLoader: num_workers = 4, pin_memory = True, persistent_workers = True.
   - torch.backends.cudnn.benchmark = True.
   - Validation sw_batch_size = 8.
3. Multi-Scale Deep Supervision: [0.5, 0.25, 0.15, 0.10].
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import monai
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.losses import TverskyLoss
from monai.data import Dataset, PersistentDataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Spacingd, RandCropByPosNegLabeld, ToTensord, CropForegroundd,
    RandFlipd, Activations, AsDiscrete
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_ROOT = r"H:\3D CT Images for Coronary Artery Segmentation (200 Samples)"
INVENTORY_CSV = os.path.join(WORK_DIR, "results", "3d_cas_dataset_inventory.csv")

sys.path.insert(0, os.path.join(REPO_ROOT, "Phase3_Local_Integration"))
from rasnet_model import RASNet
from rasnet_loss import StenosisAwareLoss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class ConvertToPlainTensor:
    def __call__(self, data):
        if isinstance(data, dict):
            cleaned = {"image": data["image"], "label": data["label"]}
            for k in ["image", "label"]:
                if hasattr(cleaned[k], "as_tensor"):
                    cleaned[k] = cleaned[k].as_tensor()
                else:
                    cleaned[k] = torch.as_tensor(cleaned[k])
            return cleaned
        return data


class CompoundRecallLoss(nn.Module):
    """
    Combines StenosisAwareLoss (Dice + Focal) with TverskyLoss (alpha=0.3, beta=0.7).
    Tversky with beta=0.7 penalizes false negatives (missed thin vessels) more heavily,
    elevating Recall and Dice without causing boundary dilation artifacts.
    """
    def __init__(self, tversky_weight: float = 0.5, alpha: float = 0.3, beta: float = 0.7):
        super().__init__()
        self.stenosis_loss = StenosisAwareLoss(alpha=0.4, gamma=2.5)
        self.tversky_loss = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            alpha=alpha,
            beta=beta,
            include_background=False,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
        self.tversky_weight = tversky_weight

    def forward(self, pred, target):
        loss_st = self.stenosis_loss(pred, target)
        loss_tv = self.tversky_loss(pred, target)
        return loss_st + self.tversky_weight * loss_tv


def get_train_val_files():
    df = pd.read_csv(INVENTORY_CSV)
    train_df = df[df["primary_benchmark_split"].str.contains("Primary_Train", na=False)]
    val_df = df[df["primary_benchmark_split"].str.contains("Primary_Val", na=False)]

    train_files = []
    for _, row in train_df.iterrows():
        cid = int(row["case_id"])
        img_p = os.path.join(DATASET_ROOT, f"{cid}.img.nii", "dia_0.nii")
        if not os.path.exists(img_p):
            img_p = os.path.join(DATASET_ROOT, f"{cid}.img.nii", "diao_0.nii")
        lbl_p = os.path.join(DATASET_ROOT, f"{cid}.label.nii", "label.nii")
        if os.path.exists(img_p) and os.path.exists(lbl_p):
            train_files.append({"image": img_p, "label": lbl_p, "case_id": cid})

    val_files = []
    for _, row in val_df.iterrows():
        cid = int(row["case_id"])
        img_p = os.path.join(DATASET_ROOT, f"{cid}.img.nii", "dia_0.nii")
        if not os.path.exists(img_p):
            img_p = os.path.join(DATASET_ROOT, f"{cid}.img.nii", "diao_0.nii")
        lbl_p = os.path.join(DATASET_ROOT, f"{cid}.label.nii", "label.nii")
        if os.path.exists(img_p) and os.path.exists(lbl_p):
            val_files.append({"image": img_p, "label": lbl_p, "case_id": cid})

    return train_files, val_files


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(epochs: int = 30, batch_size: int = 2, num_samples: int = 4, lr: float = 1e-4, resume: bool = True, max_train_cases: int = None, max_val_cases: int = None):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.join(WORK_DIR, "results", "unseen_66_cohort", "checkpoints")
    log_dir = os.path.join(WORK_DIR, "results", "unseen_66_cohort", "logs")
    cache_dir = os.path.join(WORK_DIR, "cache", "rasnet_recall_tuned")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    best_ckpt_path = os.path.join(ckpt_dir, "rasnet_recall_best.pth")
    last_ckpt_path = os.path.join(ckpt_dir, "rasnet_recall_last.pth")
    log_csv = os.path.join(log_dir, "rasnet_recall_training_log.csv")

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    train_files, val_files = get_train_val_files()
    if max_train_cases is not None:
        train_files = train_files[:max_train_cases]
    if max_val_cases is not None:
        val_files = val_files[:max_val_cases]

    print(f"[*] Dataset ready: {len(train_files)} Train scans, {len(val_files)} Val scans.", flush=True)
    print(f"[*] GPU: {torch.cuda.get_device_name(0)} | Target VRAM allocation: ~6.0-7.0 GB.", flush=True)
    print(f"[*] Effective batch size: {batch_size} scans x {num_samples} patches = {batch_size * num_samples} patches/step.", flush=True)

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(96, 96, 96), pos=2, neg=1, num_samples=num_samples
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ToTensord(keys=["image", "label"]),
        ConvertToPlainTensor()
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image", "label"]),
        ConvertToPlainTensor()
    ])

    if max_train_cases is not None:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
    else:
        train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=cache_dir)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = RASNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        init_filters=16, dropout_prob=0.1
    ).to(device)

    # Load baseline champion checkpoint if available for fine-tuning
    champion_ckpt = os.path.join(REPO_ROOT, "Q1_Publication_Package", "matched_200ep_benchmark", "checkpoints", "rasnet_best.pth")
    if resume and os.path.exists(champion_ckpt):
        print(f"[+] Initializing from champion checkpoint: {champion_ckpt}")
        ckpt_data = torch.load(champion_ckpt, map_location=device, weights_only=False)
        sd = ckpt_data.get("model_state_dict", ckpt_data)
        model.load_state_dict(sd)

    criterion = CompoundRecallLoss(tversky_weight=0.5, alpha=0.3, beta=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = AsDiscrete(to_onehot=2)

    best_val_dice = -1.0
    history = []
    print(f"\n[*] Starting Recall-Tuned Training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        step_count = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                model_out = model(images)
                if isinstance(model_out, tuple):
                    main_out, ds_outputs = model_out
                    loss_main = criterion(main_out, labels)
                    aux2_pred = torch.nn.functional.interpolate(ds_outputs[0], size=labels.shape[2:], mode="trilinear", align_corners=True)
                    aux3_pred = torch.nn.functional.interpolate(ds_outputs[1], size=labels.shape[2:], mode="trilinear", align_corners=True)
                    aux2_loss = criterion(aux2_pred, labels)
                    aux3_loss = criterion(aux3_pred, labels)
                    loss = loss_main + 0.4 * aux2_loss + 0.2 * aux3_loss
                else:
                    loss = criterion(model_out, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            step_count += 1

        scheduler.step()
        avg_train_loss = train_loss / max(step_count, 1)

        # Validation every 5 epochs or final
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for val_batch in val_loader:
                    v_img = val_batch["image"].to(device, non_blocking=True)
                    v_lbl = val_batch["label"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        val_preds = sliding_window_inference(
                            v_img, (96, 96, 96), sw_batch_size=8,
                            predictor=model, overlap=0.7
                        )
                    val_preds = [post_pred(i) for i in decollate_batch(val_preds)]
                    val_lbls = [post_label(i) for i in decollate_batch(v_lbl)]
                    dice_metric(y_pred=val_preds, y=val_lbls)

            val_dice = float(dice_metric.aggregate().item())
            dice_metric.reset()

            is_best = val_dice > best_val_dice
            if is_best:
                best_val_dice = val_dice
                torch.save(model.state_dict(), best_ckpt_path)

            torch.save(model.state_dict(), last_ckpt_path)

            ep_time = round(time.time() - epoch_start, 1)
            print(f"[Epoch {epoch:03d}/{epochs:03d}] Train Loss: {avg_train_loss:.4f} | Val Dice: {val_dice:.4f} {'(*BEST*)' if is_best else ''} | Time: {ep_time}s")
            history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_dice": val_dice, "epoch_time_s": ep_time})
            pd.DataFrame(history).to_csv(log_csv, index=False)
        else:
            ep_time = round(time.time() - epoch_start, 1)
            print(f"[Epoch {epoch:03d}/{epochs:03d}] Train Loss: {avg_train_loss:.4f} | Time: {ep_time}s")
            history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_dice": np.nan, "epoch_time_s": ep_time})
            pd.DataFrame(history).to_csv(log_csv, index=False)

    print(f"\n[+] Training complete! Best Val Dice: {best_val_dice:.4f}")
    print(f"[+] Saved best checkpoint: {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=2, help="Number of CT scans per batch (default: 2)")
    parser.add_argument("--num_samples", type=int, default=4, help="Patches per CT scan (default: 4 -> total 8 patches/step)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max_train_cases", type=int, default=None, help="Limit training cases for quick dry-runs")
    parser.add_argument("--max_val_cases", type=int, default=None, help="Limit validation cases for quick dry-runs")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        lr=args.lr,
        max_train_cases=args.max_train_cases,
        max_val_cases=args.max_val_cases
    )
