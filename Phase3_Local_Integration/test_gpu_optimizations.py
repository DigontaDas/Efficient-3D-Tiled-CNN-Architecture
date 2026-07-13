# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\test_gpu_optimizations.py
"""
GPU Optimization Benchmark & Verification Script.
Generates synthetic 3D spatial volumes or loads actual NIfTI scans to benchmark:
  1. TensorFloat-32 (TF32) Execution
  2. Page-Locked Pinned Memory Transfers
  3. Automatic Mixed Precision (AMP) Autocast
  4. Channels-Last 3D Memory Tensors
  5. OneCycleLR Learning Rate Scheduling

Saves the metrics to a CSV, a Markdown report, and a performance plot inside:
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\gpu_benchmark\
"""
import torch
import time
import os
import csv
import sys
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 2
LR = 1e-4
TEST_EPOCHS = 5

# Output Paths
OUTPUT_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\gpu_benchmark"
CSV_PATH = os.path.join(OUTPUT_DIR, "gpu_benchmark_results.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "gpu_benchmark_report.md")
PLOT_PATH = os.path.join(OUTPUT_DIR, "gpu_speed_and_vram_plot.png")

# ── TRANSFORMS FOR REAL DATA ──────────────────────────────────────────────────
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

def generate_mock_batch():
    """Generates synthetic 3D spatial data representing CT patch and vessel label."""
    img = torch.randn(BATCH_SIZE, 1, *PATCH_SIZE, dtype=torch.float32)
    lbl = torch.randint(0, 2, (BATCH_SIZE, 1, *PATCH_SIZE), dtype=torch.float32)
    return img.pin_memory(), lbl.pin_memory()

def run_benchmark():
    use_real = "--real-data" in sys.argv
    
    print("==================================================")
    print("   GPU OPTIMIZATION BENCHMARK & VERIFICATION      ")
    print("==================================================")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. This script must run on a GPU.")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"TF32 matmul allowed: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 cudnn allowed: {torch.backends.cudnn.allow_tf32}")
    print(f"Batch size: {BATCH_SIZE} | Patch size: {PATCH_SIZE}")
    print(f"Mode: {'Real ImageCAS Data' if use_real else 'Synthetic Mock Data'}")
    print("--------------------------------------------------")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    
    # ── Loader Setup ─────────────────────────────────────────────────────────
    if use_real:
        if dataset_paths is None:
            print("[ERROR] dataset_paths.py not found. Cannot load ImageCAS scans.")
            return
        
        valid_ids = dataset_paths.all_valid_ids()
        if not valid_ids:
            print("[ERROR] No valid ImageCAS scans found. Verify Dataset_Secondary_IMGcas folder.")
            return
        
        # Take the first 5 valid cases for benchmarking
        test_ids = valid_ids[:5]
        print(f"Loading {len(test_ids)} real ImageCAS volumes: {test_ids}")
        
        data_files = []
        for cid in test_ids:
            img_p = dataset_paths.find_image(cid)
            lbl_p = dataset_paths.find_label(cid)
            data_files.append({"image": img_p, "label": lbl_p})
            
        dataset = CacheDataset(data=data_files, transform=train_transforms, cache_rate=1.0)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        steps_per_epoch = len(loader)
    else:
        # Mock loader config
        steps_per_epoch = 4
        loader = None
        print("Using synthetic page-locked memory tensors.")

    # 3. Initialize SegResNet and convert to Channels-Last 3D memory format
    print("Initializing SegResNet model...")
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1,
    )
    model = model.to(DEVICE, memory_format=torch.channels_last_3d)

    # 4. Setup optimizer, loss function, and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=steps_per_epoch, epochs=TEST_EPOCHS
    )

    print("\nStarting training epochs...")
    print("Epoch | Time/Epoch | Mean Loss | Learning Rate | Max VRAM")
    print("--------------------------------------------------")

    epoch_metrics = []

    for epoch in range(1, TEST_EPOCHS + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        model.train()
        
        if use_real:
            # Real dataloader training loop
            for batch in loader:
                imgs = batch["image"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last_3d)
                labels = batch["label"].to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    loss = loss_fn(preds, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                epoch_loss += loss.item()
        else:
            # Synthetic mock training loop
            for step in range(steps_per_epoch):
                img_host, lbl_host = generate_mock_batch()
                imgs = img_host.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last_3d)
                labels = lbl_host.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    loss = loss_fn(preds, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / steps_per_epoch
        lr_curr = scheduler.get_last_lr()[0]
        max_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2) # MB

        epoch_metrics.append({
            "Epoch": epoch,
            "Time (s)": round(epoch_time, 3),
            "Loss": round(mean_loss, 4),
            "LR": round(lr_curr, 6),
            "Max VRAM (MB)": round(max_mem, 1)
        })

        print(f" {epoch:02d}   |   {epoch_time:.3f}s   |  {mean_loss:.4f}   |   {lr_curr:.6f}   |  {max_mem:.1f} MB")

    print("--------------------------------------------------")
    print("Epochs finished. Saving assets...")

    # A. Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Epoch", "Time (s)", "Loss", "LR", "Max VRAM (MB)"])
        writer.writeheader()
        writer.writerows(epoch_metrics)
    print(f"[OK] Saved CSV: {CSV_PATH}")

    # B. Write Markdown Report
    with open(REPORT_PATH, "w") as f:
        f.write("# GPU Optimization Benchmark Report\n\n")
        f.write(f"- **Device**: {device_name}\n")
        f.write(f"- **Mode**: {'Real ImageCAS NIfTI Data' if use_real else 'Synthetic Mock Data'}\n")
        f.write(f"- **Patch Size**: {PATCH_SIZE}\n")
        f.write(f"- **Batch Size**: {BATCH_SIZE}\n")
        f.write(f"- **TF32 Matmul**: {torch.backends.cuda.matmul.allow_tf32}\n")
        f.write(f"- **TF32 CuDNN**: {torch.backends.cudnn.allow_tf32}\n\n")
        f.write("## Epoch Metrics\n\n")
        f.write("| Epoch | Time (s) | Mean Loss | Learning Rate | Max VRAM (MB) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for m in epoch_metrics:
            f.write(f"| {m['Epoch']:02d} | {m['Time (s)']:.3f}s | {m['Loss']:.4f} | {m['LR']:.6f} | {m['Max VRAM (MB)']:.1f} MB |\n")
        f.write("\n---\n")
        f.write("*Benchmark completed successfully with all 5 active GPU optimizations (AMP, TF32, Pinned Memory, Channels-Last 3D, and OneCycleLR).* \n")
    print(f"[OK] Saved Markdown Report: {REPORT_PATH}")

    # C. Generate and Save Plot
    epochs = [m["Epoch"] for m in epoch_metrics]
    times = [m["Time (s)"] for m in epoch_metrics]
    vrams = [m["Max VRAM (MB)"] for m in epoch_metrics]

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot Epoch Times (Left axis)
    color = '#2563eb' # Blue
    ax1.set_xlabel('Epoch', fontweight='semibold')
    ax1.set_ylabel('Execution Time (seconds)', color=color, fontweight='semibold')
    ax1.plot(epochs, times, color=color, marker='o', linewidth=2, label='Time (s)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(epochs)

    # Plot Max VRAM (Right axis)
    ax2 = ax1.twinx()
    color = '#dc2626' # Red
    ax2.set_ylabel('Peak VRAM Allocation (MB)', color=color, fontweight='semibold')
    ax2.plot(epochs, vrams, color=color, marker='s', linestyle='--', linewidth=2, label='VRAM (MB)')
    ax2.tick_params(axis='y', labelcolor=color)

    title_suffix = "(Real ImageCAS)" if use_real else "(Mock Data)"
    plt.title(f'GPU Performance Benchmark {title_suffix}\n({device_name})', fontsize=12, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(PLOT_PATH, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved Plot: {PLOT_PATH}")

    print("==================================================")
    print("[OK] Benchmark process completed successfully.")
    print("==================================================")

if __name__ == "__main__":
    run_benchmark()
