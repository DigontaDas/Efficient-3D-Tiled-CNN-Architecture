# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\finetune_segresnet.py
"""
Loads the pre-trained SegResNet checkpoint, freezes the encoder,
and fine-tunes the decoder on a small local dataset (e.g., 20-30 NIfTI cases).
Includes 5 GPU and convergence speedups: AMP, TF32, Pinned Memory, Channels-Last 3D, and OneCycleLR.

Supports a --dry-run option to simulate transfer-learning using 10 ImageCAS cases.
Saves checkpoints and loss plots to c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\finetune_dry_run\
"""
import torch
import sys
import os
import matplotlib.pyplot as plt
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandCropByPosNegLabeld, ToTensord
)

# Enable TensorFloat-32 (TF32) execution for Ampere GPU Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add current dir to path to import dataset_paths
sys.path.insert(0, os.path.dirname(__file__))
try:
    import dataset_paths
except ImportError:
    dataset_paths = None

# ── CONFIG ───────────────────────────────────────────────────────────────────
PRETRAINED_CKPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "all_four_validations", "mandatory_artifacts_segresnet", "best_resumed.pt"))
LOCAL_DATA_DIR  = os.path.join(os.path.dirname(__file__), "local_data")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE      = (96, 96, 96)
LR              = 1e-4

# Dry-run Output config
OUTPUT_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\finetune_dry_run"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "finetuned_segresnet_best.pth")
LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "loss_curves.png")

# ── TRANSFORMS ───────────────────────────────────────────────────────────────
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=PATCH_SIZE, pos=2, neg=1, num_samples=4
    ),
    ToTensord(keys=["image", "label"]),
])

# ── MODEL LOAD + FREEZE ENCODER ──────────────────────────────────────────────
def load_pretrained_segresnet(ckpt_path: str) -> SegResNet:
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1,
    )
    
    print(f"Loading pre-trained SegResNet weights from {ckpt_path}...")
    state = torch.load(ckpt_path, map_location=DEVICE)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    # Freeze encoder layers (everything before "up_layers")
    for name, param in model.named_parameters():
        if "up_layers" not in name and "conv_final" not in name:
            param.requires_grad = False

    # Apply 3D Channels-Last layout memory optimization
    model = model.to(DEVICE, memory_format=torch.channels_last_3d)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model

# ── TRAINING LOOP ────────────────────────────────────────────────────────────
def finetune(model, data_files, num_epochs=50, is_dry_run=False):
    dataset   = CacheDataset(data=data_files, transform=train_transforms, cache_rate=1.0)
    loader    = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn   = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler    = torch.amp.GradScaler('cuda')

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(loader), epochs=num_epochs
    )

    losses = []
    best_loss = float("inf")
    
    print(f"\nStarting fine-tuning training loop ({num_epochs} epochs)...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in loader:
            imgs   = batch["image"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last_3d)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss  = loss_fn(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        losses.append(avg)
        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model checkpoint
        if avg < best_loss:
            best_loss = avg
            save_path = BEST_MODEL_PATH if is_dry_run else r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\finetuned_segresnet_best.pth"
            torch.save(model.state_dict(), save_path)
            
    print("--------------------------------------------------")
    print(f"Training completed. Best loss: {best_loss:.4f}")

    if is_dry_run:
        # Plot and save loss curves
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.figure(figsize=(7, 4.5), dpi=300)
        plt.plot(range(1, num_epochs + 1), losses, color='#2563eb', marker='o', linewidth=2, label='Fine-Tuning Loss')
        plt.title('SegResNet Transfer Learning Convergence (Dry-Run on ImageCAS)', fontsize=11, fontweight='bold', pad=12)
        plt.xlabel('Epoch', fontweight='semibold')
        plt.ylabel('Dice + Cross-Entropy Loss', fontweight='semibold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(LOSS_PLOT_PATH, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved Checkpoint: {BEST_MODEL_PATH}")
        print(f"[OK] Saved Loss Plot: {LOSS_PLOT_PATH}")

if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data_files = []
    
    if is_dry_run:
        print("==================================================")
        print("   MOCK TRANSFER-LEARNING DRY-RUN (IMGCAS)        ")
        print("==================================================")
        if dataset_paths is None:
            print("[ERROR] dataset_paths.py not found. Cannot resolve ImageCAS scans.")
            sys.exit(1)
        valid_ids = dataset_paths.all_valid_ids()
        # Load the first 10 cases to simulate a local clinical dataset
        dry_run_ids = valid_ids[:10]
        print(f"Simulating clinical dataset using ImageCAS cases: {dry_run_ids}")
        for cid in dry_run_ids:
            img_p = dataset_paths.find_image(cid)
            lbl_p = dataset_paths.find_label(cid)
            data_files.append({"image": img_p, "label": lbl_p})
        
        epochs = 10
    else:
        # Normal fine-tuning path using local scans
        print("==================================================")
        print("        LOCAL FINE-TUNING EXECUTION MODE          ")
        print("==================================================")
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        for case in sorted(os.listdir(LOCAL_DATA_DIR)):
            case_path = os.path.join(LOCAL_DATA_DIR, case)
            if os.path.isdir(case_path):
                img = os.path.join(case_path, "image.nii.gz")
                lbl = os.path.join(case_path, "label.nii.gz")
                if os.path.exists(img) and os.path.exists(lbl):
                    data_files.append({"image": img, "label": lbl})
        epochs = 50

    print(f"Found {len(data_files)} dataset cases.")
    if len(data_files) < 5:
        if is_dry_run:
            print("[ERROR] Not enough ImageCAS cases. Verify Dataset_Secondary_IMGcas folder.")
        else:
            print("[WARNING] Need at least 5 local cases to run fine-tuning loop.")
            print(f"Please place local case subfolders (each containing image.nii.gz and label.nii.gz) in: {LOCAL_DATA_DIR}")
        sys.exit(0)

    model = load_pretrained_segresnet(PRETRAINED_CKPT)
    finetune(model, data_files, num_epochs=epochs, is_dry_run=is_dry_run)
