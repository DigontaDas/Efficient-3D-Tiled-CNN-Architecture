# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\run_training_after_fix.py
r"""
RASNet Training Runner (Post-Hallucination Fix).
Outputs all checkpoints, persistent cache, and loss plots directly into:
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\results-after-hallucin-fix\
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import json
import time
import matplotlib.pyplot as plt
from monai.data import PersistentDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Spacingd, RandCropByPosNegLabeld, ToTensord, CropForegroundd, RandFlipd
)

# Add current dir to path to import rasnet modules & dataset_paths
sys.path.insert(0, os.path.dirname(__file__))
from rasnet_model import RASNet
from rasnet_loss import StenosisAwareLoss
import dataset_paths

# Enable TensorFloat-32 (TF32) execution for Ampere GPU Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 4
LR = 2e-4
EPOCHS = 70  # 70 epochs for full-scale training

# Target Output Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "results-after-hallucin-fix")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "rasnet_best.pth")
BEST_MODEL_CKPT_PATH = os.path.join(CKPT_DIR, "rasnet_best.pth")
PLOT_PATH = os.path.join(OUTPUT_DIR, "loss_curves.png")

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

# Preprocessing transforms (nnU-Net style isotropic resampling and intensity clipping)
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
    ToTensord(keys=["image", "label"]),
    ConvertToPlainTensor(),
])

def train_model():
    print("==================================================")
    print("  RASNET TRAINING (RESULTS-AFTER-HALLUCIN-FIX)   ")
    print("==================================================")
    print(f"Device: {DEVICE}")
    print(f"Target Output Directory: {OUTPUT_DIR}")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. This script must run on a GPU.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    # 1. Resolve dataset files from splits_final.json
    splits_file = os.path.join(os.path.dirname(__file__), "splits_final.json")
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
        
    with open(splits_file) as f:
        splits = json.load(f)
    train_ids = splits.get("train", [])
    
    if "--dry-run" in sys.argv:
        train_ids = train_ids[:5]
        epochs = 1
        print(f"Running Dry-Run mode with {len(train_ids)} cases, {epochs} epoch.")
    else:
        epochs = EPOCHS
        print(f"Loading {len(train_ids)} ImageCAS scans for full training ({epochs} epochs)...")
    
    data_files = []
    for cid in train_ids:
        img_p = dataset_paths.find_image(cid)
        lbl_p = dataset_paths.find_label(cid)
        if img_p and lbl_p:
            data_files.append({"image": img_p, "label": lbl_p})

    print(f"Valid dataset files found: {len(data_files)}")

    # Caching preprocessed volumes on disk
    cache_dir = os.path.join(OUTPUT_DIR, "persistent_cache")
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
    
    # 2. Instantiate RASNet and transfer pre-trained weights
    model = RASNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    )
    
    pretrained_candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Final_Generated_assets", "imagecas_pipeline_validation", "rasnet_development", "rasnet_best.pth")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Final_Generated_assets", "imagecas_pipeline_validation", "finetune_dry_run", "finetuned_segresnet_best.pth")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "all_four_validations", "mandatory_artifacts_segresnet", "best_resumed.pt"))
    ]
    pretrained_ckpt = None
    for p in pretrained_candidates:
        if os.path.exists(p):
            pretrained_ckpt = p
            break

    if pretrained_ckpt:
        print(f"Transferring pre-trained weights from {pretrained_ckpt} (strict=False)...")
        state = torch.load(pretrained_ckpt, map_location=DEVICE)
        state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(state_dict, strict=False)
    else:
        print("[WARNING] Pre-trained checkpoint not found. Training from scratch.")

    model = model.to(DEVICE, memory_format=torch.channels_last_3d)

    # 3. Setup optimizer, loss function, and scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = StenosisAwareLoss()  # α=0.4, γ=2.5
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader), eta_min=1e-6
    )

    losses = []
    best_loss = float("inf")

    print("\nStarting training loop...")
    print("Epoch | Time/Epoch | Mean Loss | Learning Rate | Max VRAM")
    print("--------------------------------------------------")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        model.train()
        for batch in loader:
            imgs = batch["image"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last_3d)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                preds, ds_preds = model(imgs)
                
                main_loss = loss_fn(preds, labels)
                aux2_pred = F.interpolate(ds_preds[0], size=labels.shape[2:], mode='trilinear', align_corners=True)
                aux3_pred = F.interpolate(ds_preds[1], size=labels.shape[2:], mode='trilinear', align_corners=True)
                
                aux2_loss = loss_fn(aux2_pred, labels)
                aux3_loss = loss_fn(aux3_pred, labels)
                
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
            torch.save(model.state_dict(), BEST_MODEL_CKPT_PATH)
            
        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = os.path.join(CKPT_DIR, f"rasnet_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f" -> Saved Checkpoint: {ckpt_path}")

    print("--------------------------------------------------")
    print(f"Training completed. Best loss: {best_loss:.4f}")
    
    # Save Loss curves
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(range(1, epochs + 1), losses, color='#10b981', marker='o', linewidth=2, label='RASNet Training Loss')
    plt.title('RASNet Training Curve (Post-Hallucination Fix)', fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontweight='semibold')
    plt.ylabel('Loss (Dice + Focal + Deep Supervision)', fontweight='semibold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved Best Model: {BEST_MODEL_PATH}")
    print(f"[OK] Saved Plot: {PLOT_PATH}")
    print("==================================================")

if __name__ == "__main__":
    train_model()
