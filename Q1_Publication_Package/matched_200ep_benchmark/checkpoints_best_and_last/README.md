# Matched 200-Epoch Authoritative Model Checkpoints

This directory contains the authoritative 200-epoch trained model checkpoints (`best` and `last`) for all 5 benchmarked architectures on the ImageCAS dataset:

1. **RASNet (Ours, Champion)**: `rasnet_best.pth` & `rasnet_last.pth`
2. **nnU-Net V2 (Self-Configuring 3D Fullres)**: `nnunet_best.pt` & `nnunet_last.pt`
3. **SegResNet (MONAI Baseline)**: `segresnet_best.pt` & `segresnet_last.pt`
4. **V-Net (Additive Residuals)**: `vnet_best.pt` & `vnet_last.pt`
5. **3D U-Net (Classic Baseline)**: `3dunet_best.pth` & `3dunet_last.pth`

---

## Storage Architecture & GitHub Compliance

Because GitHub strictly enforces a **100.0 MB per-file limit** for Git repositories, the 859 MB collection of models has been packaged into a multi-part split archive (`checkpoints_200ep.zip.001` through `.010`), where each individual file is bounded to **95.0 MB**.

This guarantees:
- **No Git LFS bandwidth/storage quota limitations**.
- **100% bit-for-bit reconstruction** verified against the original training outputs.
- Seamless, reproducible extraction across Windows, Linux, and macOS.

---

## How to Recombine and Extract the Checkpoints

### Option 1: Automated Python Script (Cross-Platform, Recommended)
Run the bundled script from this folder:
```bash
python recombine_and_unzip.py
```
This script will concatenate the 10 split parts into a temporary zip, verify archive integrity, extract all 10 `.pt`/`.pth` checkpoint files into this folder, and remove the temporary zip.

---

### Option 2: Command-Line Recombination

#### On Windows (PowerShell / CMD)
```cmd
copy /b checkpoints_200ep.zip.* checkpoints_matched_200ep.zip
tar -xf checkpoints_matched_200ep.zip
del checkpoints_matched_200ep.zip
```

#### On Linux / macOS (Bash)
```bash
cat checkpoints_200ep.zip.* > checkpoints_matched_200ep.zip
unzip checkpoints_matched_200ep.zip
rm checkpoints_matched_200ep.zip
```

---

## Model Inventory & Sizes

| Model Architecture | Best Checkpoint | Size (MB) | Last Checkpoint | Size (MB) |
| :--- | :--- | :--- | :--- | :--- |
| **RASNet (Champion)** | `rasnet_best.pth` | 18.00 MB | `rasnet_last.pth` | 18.00 MB |
| **nnU-Net V2** | `nnunet_best.pt` | 235.05 MB | `nnunet_last.pt` | 235.05 MB |
| **SegResNet** | `segresnet_best.pt` | 17.96 MB | `segresnet_last.pt` | 17.96 MB |
| **V-Net** | `vnet_best.pt` | 174.03 MB | `vnet_last.pt` | 174.03 MB |
| **3D U-Net** | `3dunet_best.pth` | 18.36 MB | `3dunet_last.pth` | 18.36 MB |

All weights are ready for direct inference, held-out evaluation, and transfer learning fine-tuning.
