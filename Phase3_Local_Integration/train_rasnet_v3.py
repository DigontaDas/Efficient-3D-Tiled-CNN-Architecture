# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\train_rasnet_v3.py
"""
Retraining Pipeline for RASNet v3.
1. Implements nnU-Net adaptive preprocessing.
2. Injects BoundaryLoss to penalize spillover.
3. Applies RandErodeLabelD annotation-aware GT erosion.
4. Trains for 10 epochs on 100 training cases starting from rasnet_best.pth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from monai.data import PersistentDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Spacingd, RandCropByPosNegLabeld, ToTensord, CropForegroundd, RandFlipd
)
from scipy.ndimage import binary_erosion

# Add current dir to path to import rasnet modules & dataset_paths
sys.path.insert(0, os.path.dirname(__file__))
from rasnet_model import RASNet
from rasnet_loss import StenosisAwareLoss
import dataset_paths

# Enable TensorFloat-32 (TF32) execution
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 4
LR = 1e-4  # Lower learning rate for fine-tuning
EPOCHS = 10

# Output Paths
OUTPUT_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\rasnet_development"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "rasnet_v3_best_10ep.pth")
PLOT_PATH = os.path.join(OUTPUT_DIR, "loss_curves_v3_10ep.png")

# Custom transform for random label erosion
class RandErodeLabelD:
    """Randomly erode GT label by 1 voxel with probability p."""
    def __init__(self, keys=["label"], prob=0.3):
        self.keys = keys
        self.prob = prob

    def __call__(self, data):
        if np.random.random() < self.prob:
            d = dict(data)
            for key in self.keys:
                arr = d[key].numpy() if hasattr(d[key], 'numpy') else np.array(d[key])
                # arr shape: (1, D, H, W). Erode foreground mask.
                eroded = binary_erosion(arr[0] > 0.5).astype(np.float32)
                if torch.is_tensor(d[key]):
                    d[key] = torch.from_numpy(eroded[np.newaxis]).to(d[key].device)
                else:
                    d[key] = eroded[np.newaxis]
            return d
        return data

# Custom transform to prevent PyTorch/MONAI collate crashes due to MetaTensor metadata
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

# Preprocessing transforms (with label erosion)
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=PATCH_SIZE, pos=2, neg=1, num_samples=2
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandErodeLabelD(keys=["label"], prob=0.3),  # 1-voxel random erosion
    ToTensord(keys=["image", "label"]),
    ConvertToPlainTensor(),
])

def train_model():
    print("==================================================")
    print("      RASNET V3 10-EPOCH RETRAINING RUN           ")
    print("==================================================")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. This script must run on a GPU.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Resolve dataset files (take first 100 cases from splits_final.json training split)
    splits_file = os.path.join(os.path.dirname(__file__), "splits_final.json")
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
        
    with open(splits_file) as f:
        splits = json.load(f)
    train_ids = splits.get("train", [])[:100]
    
    print(f"Loading {len(train_ids)} ImageCAS scans for training: {train_ids}")
    
    data_files = []
    for cid in train_ids:
        data_files.append({
            "image": dataset_paths.find_image(cid),
            "label": dataset_paths.find_label(cid)
        })

    # Caching preprocessed volumes on disk
    cache_dir = os.path.join(OUTPUT_DIR, "persistent_cache_v3")
    os.makedirs(cache_dir, exist_ok=True)
    dataset = PersistentDataset(data=data_files, transform=train_transforms, cache_dir=cache_dir)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # 2. Instantiate RASNet and load current best model weights
    model = RASNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    )
    
    pretrained_ckpt = os.path.join(OUTPUT_DIR, "rasnet_best.pth")
    if os.path.exists(pretrained_ckpt):
        print(f"Loading best weights from {pretrained_ckpt} (strict=True)...")
        model.load_state_dict(torch.load(pretrained_ckpt, map_location=DEVICE), strict=True)
    else:
        raise FileNotFoundError(f"Could not find checkpoint to initialize: {pretrained_ckpt}")

    model = model.to(DEVICE, memory_format=torch.channels_last_3d)

    # 3. Setup optimizer, loss function, and scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = StenosisAwareLoss(alpha=0.5, beta=0.3, delta=0.2, gamma=2.5)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(loader), eta_min=1e-6
    )

    losses = []
    best_loss = float("inf")

    print("\nStarting training loop...")
    print("Epoch | Time/Epoch | Mean Loss | Learning Rate | Max VRAM")
    print("--------------------------------------------------")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        model.train()
        for batch in loader:
            imgs = batch["image"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last_3d)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Forward pass returns final output and auxiliary outputs for Deep Supervision
                preds, ds_preds = model(imgs)
                
                # Compute main loss
                main_loss = loss_fn(preds, labels)
                
                # Upsample auxiliary predictions to full resolution and compute loss
                aux2_pred = F.interpolate(ds_preds[0], size=labels.shape[2:], mode='trilinear', align_corners=True)
                aux3_pred = F.interpolate(ds_preds[1], size=labels.shape[2:], mode='trilinear', align_corners=True)
                
                aux2_loss = loss_fn(aux2_pred, labels)
                aux3_loss = loss_fn(aux3_pred, labels)
                
                # Weighted total loss: 1.0 * main_loss + 0.4 * aux2_loss + 0.2 * aux3_loss
                loss = main_loss + 0.4 * aux2_loss + 0.2 * aux3_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        lr_curr = scheduler.get_last_lr()[0]
        max_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

        print(f" {epoch:02d}   |   {epoch_time:.3f}s   |  {avg_loss:.4f}   |   {lr_curr:.6f}   |  {max_mem:.1f} MB")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print("--------------------------------------------------")
    print(f"Training completed. Best loss: {best_loss:.4f}")
    
    # Save Loss curves
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(range(1, EPOCHS + 1), losses, color='#3b82f6', marker='o', linewidth=2, label='RASNet v3 Training Loss')
    plt.title('RASNet v3 10-Epoch Training Curve (BoundaryLoss)', fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontweight='semibold')
    plt.ylabel('Loss (Dice + Focal + Boundary + DS)', fontweight='semibold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved Model: {BEST_MODEL_PATH}")
    print(f"[OK] Saved Plot: {PLOT_PATH}")
    print("==================================================")

if __name__ == "__main__":
    train_model()
