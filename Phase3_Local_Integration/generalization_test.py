# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\generalization_test.py
"""
Generalization Test for RASNet — Unseen Cases 1, 2, 3, 4.
These cases were NOT in the 150-case test split (851-1000) and were NOT
used during training (690 training cases). This script proves RASNet
generalizes to fully unseen data.

For each case this script:
  1. Runs 4-pass TTA sliding-window GPU inference.
  2. Applies topological CC3D post-processing.
  3. Saves the predicted NIfTI mask to generalization_outputs/predictions/.
  4. Computes Dice and HD95 metrics vs ground truth.
  5. Generates a 3-panel PNG overlay (axial/coronal/sagittal MIP view)
     showing ground truth vs prediction side by side, saved to
     generalization_outputs/visualizations/.
"""
import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
import monai.transforms as mt
from monai.inferers import sliding_window_inference
import cc3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(__file__))
from rasnet_model import RASNet
from eval_utils import compute_metrics
import dataset_paths

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE  = (96, 96, 96)
CASE_IDS    = [1, 5, 13]  # Fully unseen cases for FP reduction testing

BASE_DIR    = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\rasnet_development"
CKPT_PATH   = os.path.join(BASE_DIR, "rasnet_best.pth")
OUT_DIR     = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\generalization_outputs"
PRED_DIR    = os.path.join(OUT_DIR, "predictions")
VIS_DIR     = os.path.join(OUT_DIR, "visualizations")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
pre_trans = mt.Compose([
    mt.LoadImaged(keys=["image"]),
    mt.EnsureChannelFirstd(keys=["image"]),
    mt.Orientationd(keys=["image"], axcodes="RAS"),
    mt.Spacingd(keys=["image"], pixdim=(0.5, 0.5, 0.5), mode="bilinear"),
    mt.ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
    mt.EnsureTyped(keys=["image"])
])

post_trans = mt.Compose([
    mt.Invertd(
        keys=["pred"],
        transform=pre_trans,
        orig_keys=["image"],
        meta_keys=["pred_meta_dict"],
        orig_meta_keys=["image_meta_dict"],
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True
    )
])

# ── TOPOLOGICAL POST-PROCESSING ───────────────────────────────────────────────
def topological_postprocess(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Keep only the 2 largest connected components (LCA + RCA)."""
    labels = cc3d.connected_components(mask.astype(np.uint8))
    unique, counts = np.unique(labels, return_counts=True)
    # exclude background (label 0)
    vessel_labels = [(lbl, cnt) for lbl, cnt in zip(unique, counts) if lbl != 0 and cnt >= min_size]
    vessel_labels.sort(key=lambda x: x[1], reverse=True)
    top2 = {lbl for lbl, _ in vessel_labels[:2]}
    clean = np.isin(labels, list(top2)).astype(np.uint8)
    return clean

# ── 3D VISUALIZATION ──────────────────────────────────────────────────────────
def generate_overlay_png(case_id: int, img_arr: np.ndarray, gt_arr: np.ndarray,
                          pred_arr: np.ndarray, dice: float, hd95: float):
    """
    Generate a 3-panel figure showing the maximum intensity projection (MIP)
    of CT image with GT (green) and Prediction (red) overlays in
    axial, coronal, and sagittal planes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d0d0d")

    views = [
        ("Axial (Z)", np.max(img_arr, axis=2), np.max(gt_arr, axis=2), np.max(pred_arr, axis=2)),
        ("Coronal (Y)", np.max(img_arr, axis=1), np.max(gt_arr, axis=1), np.max(pred_arr, axis=1)),
        ("Sagittal (X)", np.max(img_arr, axis=0), np.max(gt_arr, axis=0), np.max(pred_arr, axis=0)),
    ]

    for ax, (title, img_mip, gt_mip, pred_mip) in zip(axes, views):
        ax.set_facecolor("#0d0d0d")
        # CT MIP in grayscale
        ax.imshow(img_mip.T, cmap="gray", origin="lower", aspect="equal")

        # Ground truth overlay (green)
        gt_rgba = np.zeros((*gt_mip.T.shape, 4), dtype=np.float32)
        gt_rgba[..., 1] = gt_mip.T.astype(np.float32)  # green channel
        gt_rgba[..., 3] = (gt_mip.T > 0).astype(np.float32) * 0.55
        ax.imshow(gt_rgba, origin="lower", aspect="equal")

        # Prediction overlay (red/orange)
        pred_rgba = np.zeros((*pred_mip.T.shape, 4), dtype=np.float32)
        pred_rgba[..., 0] = pred_mip.T.astype(np.float32)   # red channel
        pred_rgba[..., 1] = pred_mip.T.astype(np.float32) * 0.4  # slight orange tint
        pred_rgba[..., 3] = (pred_mip.T > 0).astype(np.float32) * 0.55
        ax.imshow(pred_rgba, origin="lower", aspect="equal")

        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    # Legend
    gt_patch   = mpatches.Patch(color=(0, 1, 0, 0.8), label="Ground Truth")
    pred_patch = mpatches.Patch(color=(1, 0.4, 0, 0.8), label="RASNet Prediction")
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center", ncol=2,
               fontsize=12, facecolor="#1a1a1a", edgecolor="#555",
               labelcolor="white", framealpha=0.9)

    fig.suptitle(
        f"Case {case_id}  —  RASNet Generalization Test\n"
        f"Dice = {dice:.4f}   |   HD95 = {hd95:.2f} mm",
        color="white", fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    out_path = os.path.join(VIS_DIR, f"case_{case_id:04d}_overlay.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   [Saved] Visualization -> {out_path}")
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RASNET GENERALIZATION TEST -- Unseen Cases 1, 2, 3, 4")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}\n")

    # Load model
    model = RASNet(spatial_dims=3, in_channels=1, out_channels=2,
                   init_filters=16, dropout_prob=0.1)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE), strict=True)
    model = model.to(DEVICE, memory_format=torch.channels_last_3d)
    model.eval()
    print("Model loaded successfully.\n")

    results = []

    for case_id in CASE_IDS:
        print(f"--- Case {case_id} " + "-" * 40)
        img_path = dataset_paths.find_image(case_id)
        gt_path  = dataset_paths.find_label(case_id)

        if not img_path or not gt_path:
            print(f"  [SKIP] Missing raw files for Case {case_id}.")
            continue

        print(f"  Image : {img_path}")
        print(f"  Label : {gt_path}")

        # Preprocessing
        batch = pre_trans({"image": img_path})
        input_tensor = batch["image"].unsqueeze(0).to(DEVICE, memory_format=torch.channels_last_3d)

        # 4-pass TTA inference
        print(f"  Running 4-pass TTA inference...")
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                logits = sliding_window_inference(input_tensor, PATCH_SIZE, 8, model, overlap=0.5)
                prob_avg = torch.softmax(logits, dim=1)

                for flip_dim in [2, 3, 4]:
                    flipped = torch.flip(input_tensor, dims=[flip_dim])
                    logits_f = sliding_window_inference(flipped, PATCH_SIZE, 8, model, overlap=0.5)
                    logits_fr = torch.flip(logits_f, dims=[flip_dim])
                    prob_avg += torch.softmax(logits_fr, dim=1)

                prob_avg /= 4.0

        # Invert transforms
        batch["pred"] = prob_avg.squeeze(0)
        batch = post_trans(batch)

        # D. Threshold at 0.6 confidence for tighter boundary matching
        # pred_probs has shape [2, H, W, D] from post_trans
        pred_probs = batch["pred"]
        fg_prob = pred_probs[1]
        pred_mask = (fg_prob > 0.6).cpu().numpy().astype(np.uint8)
        pred_mask_clean = topological_postprocess(pred_mask, min_size=50)

        # Save predicted NIfTI
        out_nifti = os.path.join(PRED_DIR, f"{case_id}.nii.gz")
        original_img = sitk.ReadImage(img_path)
        pred_sitk_arr = np.transpose(pred_mask_clean, (2, 1, 0))
        pred_sitk = sitk.GetImageFromArray(pred_sitk_arr)
        pred_sitk.CopyInformation(original_img)
        sitk.WriteImage(pred_sitk, out_nifti)
        print(f"  [Saved] Prediction NIfTI -> {out_nifti}")

        # Compute metrics vs ground truth
        metrics = compute_metrics(out_nifti, gt_path)
        dice = metrics.get("dice", 0.0)
        hd95 = metrics.get("hd95", 999.0)
        print(f"  Dice = {dice:.4f}  |  HD95 = {hd95:.2f} mm")

        # Generate 3D overlay visualization
        gt_sitk   = sitk.ReadImage(gt_path)
        img_arr   = sitk.GetArrayFromImage(original_img).astype(np.float32)
        gt_arr    = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)

        # Normalize CT for visualization
        img_arr = np.clip(img_arr, -100, 800)
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)

        generate_overlay_png(case_id, img_arr, gt_arr, pred_mask_clean, dice, hd95)
        results.append({"case_id": case_id, "dice": dice, "hd95": hd95})
        print()

    # Summary
    print("=" * 60)
    print("  GENERALIZATION TEST SUMMARY")
    print("=" * 60)
    print(f"{'Case':>6} | {'Dice':>8} | {'HD95 (mm)':>10}")
    print("-" * 32)
    for r in results:
        print(f"  {r['case_id']:>4} | {r['dice']:>8.4f} | {r['hd95']:>10.2f}")
    if results:
        mean_dice = np.mean([r["dice"] for r in results])
        mean_hd95 = np.mean([r["hd95"] for r in results])
        print("-" * 32)
        print(f"  {'Mean':>4} | {mean_dice:>8.4f} | {mean_hd95:>10.2f}")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
