#!/usr/bin/env python3
"""
evaluate_matched_200ep_models.py — Automated 5-Model 150-Case Benchmark Evaluator
=================================================================================
Evaluates all 5 matched 200-epoch models on the 150 held-out ImageCAS test cases (851-1000):
  1. SegResNet (200 ep)
  2. nnU-Net V2 (200 ep)
  3. 3D U-Net (200 ep)
  4. V-Net (200 ep, stabilized)
  5. RASNet (Ours, 200 ep from scratch)

Computes all 8 publication metrics per case:
  1. Dice Similarity Coefficient (DSC)
  2. Intersection over Union (IoU / Jaccard)
  3. Precision (Positive Predictive Value)
  4. Recall / Sensitivity (True Positive Rate)
  5. 95th Percentile Hausdorff Distance (HD95, mm)
  6. Average Surface Distance (ASD / ASSD, mm)
  7. Centerline Dice (clDice)
  8. Centerline Recall / Tree Completeness (T_sens)

Also outputs:
  - metrics_{model}_200ep.csv for each model
  - benchmark_200ep_summary_table.csv (Mean ± Std, Median [IQR])
  - benchmark_200ep_significance_report.md (Paired Wilcoxon signed-rank tests with Holm-Bonferroni correction)
"""

import os
import sys
import json
import time
import shutil
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import binary_erosion, distance_transform_edt
import SimpleITK as sitk
import skimage.morphology as morph
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ensure paths
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BENCHMARK_DIR, "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "Phase3_Local_Integration")
sys.path.insert(0, PHASE3_DIR)
import dataset_paths

CHECKPOINT_DIR = os.path.join(BENCHMARK_DIR, "checkpoints")
SPLITS_FILE = os.path.join(PHASE3_DIR, "splits_final.json")
OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "evaluation_results")
PRED_DIR = os.path.join(BENCHMARK_DIR, "predictions")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Metric Computation Engine (8 Metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray,
                       spacing: tuple) -> np.ndarray:
    """Distances from every surface voxel of mask_a to the nearest surface voxel of mask_b (mm)."""
    eroded_a = binary_erosion(mask_a)
    surface_a = mask_a & ~eroded_a
    if not surface_a.any():
        return np.array([], dtype=np.float32)

    dt_b = distance_transform_edt(~mask_b, sampling=spacing)
    return dt_b[surface_a]


def compute_8_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple) -> dict:
    """
    Computes all 8 metrics between binary 3D prediction and ground-truth arrays.
    spacing is (sz, sy, sx) in mm.
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    # Overlap metrics
    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    pred_sum = int(pred_bool.sum())
    gt_sum = int(gt_bool.sum())
    union = pred_sum + gt_sum - intersection

    dice = float((2.0 * intersection) / (pred_sum + gt_sum + 1e-8)) if (pred_sum + gt_sum) > 0 else 1.0
    iou = float(intersection / (union + 1e-8)) if union > 0 else 1.0
    precision = float(intersection / (pred_sum + 1e-8)) if pred_sum > 0 else 0.0
    recall = float(intersection / (gt_sum + 1e-8)) if gt_sum > 0 else 0.0

    # Boundary metrics: HD95 and ASD
    if not pred_bool.any() or not gt_bool.any():
        hd95 = 100.0 if not pred_bool.any() else 0.0
        asd = 100.0 if not pred_bool.any() else 0.0
    else:
        coords = np.argwhere(pred_bool | gt_bool)
        min_z, min_y, min_x = coords.min(axis=0)
        max_z, max_y, max_x = coords.max(axis=0)

        margin = 15
        z_start = max(0, min_z - margin)
        z_end = min(pred_bool.shape[0], max_z + margin + 1)
        y_start = max(0, min_y - margin)
        y_end = min(pred_bool.shape[1], max_y + margin + 1)
        x_start = max(0, min_x - margin)
        x_end = min(pred_bool.shape[2], max_x + margin + 1)

        pred_crop = pred_bool[z_start:z_end, y_start:y_end, x_start:x_end]
        gt_crop = gt_bool[z_start:z_end, y_start:y_end, x_start:x_end]

        d_pred_to_gt = _surface_distances(pred_crop, gt_crop, spacing)
        d_gt_to_pred = _surface_distances(gt_crop, pred_crop, spacing)

        all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
        if all_distances.size == 0:
            hd95 = 0.0
            asd = 0.0
        else:
            hd95 = float(np.percentile(all_distances, 95))
            asd = float(np.mean(all_distances))

    # Centerline metrics: clDice and Centerline Recall (T_sens)
    if not gt_bool.any():
        cldice = 1.0 if not pred_bool.any() else 0.0
        cl_recall = 1.0 if not pred_bool.any() else 0.0
    elif not pred_bool.any():
        cldice = 0.0
        cl_recall = 0.0
    else:
        skel_gt = morph.skeletonize(gt_bool)
        skel_pred = morph.skeletonize(pred_bool)

        skel_gt_sum = int(skel_gt.sum())
        skel_pred_sum = int(skel_pred.sum())

        t_prec = float((skel_pred & gt_bool).sum() / (skel_pred_sum + 1e-8)) if skel_pred_sum > 0 else 0.0
        t_sens = float((skel_gt & pred_bool).sum() / (skel_gt_sum + 1e-8)) if skel_gt_sum > 0 else 0.0

        if (t_prec + t_sens) > 0:
            cldice = float(2.0 * t_prec * t_sens / (t_prec + t_sens))
        else:
            cldice = 0.0
        cl_recall = t_sens

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "hd95": hd95,
        "asd": asd,
        "cldice": cldice,
        "centerline_recall": cl_recall
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model Inference Loaders & Runners
# ─────────────────────────────────────────────────────────────────────────────

def get_test_case_ids():
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    test_ids = splits.get("test", [])
    if not test_ids:
        test_ids = list(range(851, 1001))
    return sorted(test_ids)


def run_segresnet_inference(test_ids, pred_model_dir):
    from monai.networks.nets import SegResNet
    from monai.inferers import sliding_window_inference
    import monai.transforms as mt

    ckpt_path = os.path.join(CHECKPOINT_DIR, "segresnet_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, "segresnet_epoch_200.pt")

    print(f"[*] Loading SegResNet checkpoint: {ckpt_path}")
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

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
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
    ])

    for idx, cid in enumerate(test_ids):
        out_file = os.path.join(pred_model_dir, f"{cid}.nii.gz")
        if os.path.exists(out_file):
            continue
        img_p = dataset_paths.find_image(cid)
        if not img_p:
            continue
        batch = pre_trans({"image": img_p})
        inp = batch["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = sliding_window_inference(inp, (96, 96, 96), 4, model, overlap=0.5)
        batch["pred"] = logits.squeeze(0)
        batch = post_trans(batch)
        pred_probs = batch["pred"]
        pred_mask = torch.argmax(pred_probs, dim=0).cpu().numpy().astype(np.uint8)
        pred_mask_sitk = np.transpose(pred_mask, (2, 1, 0))

        orig_img = sitk.ReadImage(img_p)
        sitk_p = sitk.GetImageFromArray(pred_mask_sitk)
        sitk_p.CopyInformation(orig_img)
        sitk.WriteImage(sitk_p, out_file)
        if (idx + 1) % 25 == 0 or idx == len(test_ids) - 1:
            print(f"  [SegResNet] Processed {idx + 1}/{len(test_ids)} cases...")


def run_3dunet_inference(test_ids, pred_model_dir):
    from monai.networks.nets import UNet
    import monai.transforms as mt

    ckpt_path = os.path.join(CHECKPOINT_DIR, "3dunet_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, "3dunet_epoch_200.pth")

    print(f"[*] Loading 3D U-Net checkpoint: {ckpt_path}")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
        dropout=0.15
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    pre_trans = mt.Compose([
        mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.Orientationd(keys=["image"], axcodes="RAS"),
        mt.ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
        mt.Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
        mt.EnsureTyped(keys=["image"])
    ])
    post_trans = mt.Compose([
        mt.Invertd(
            keys=["pred"],
            transform=pre_trans,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
    ])

    for idx, cid in enumerate(test_ids):
        out_file = os.path.join(pred_model_dir, f"{cid}.nii.gz")
        if os.path.exists(out_file):
            continue
        img_p = dataset_paths.find_image(cid)
        if not img_p:
            continue
        batch = pre_trans({"image": img_p})
        inp = batch["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
        probs = torch.sigmoid(logits.squeeze(0))
        batch["pred"] = (probs > 0.5).float()
        batch = post_trans(batch)
        pred_mask = (batch["pred"].squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
        pred_mask_sitk = np.transpose(pred_mask, (2, 1, 0))

        orig_img = sitk.ReadImage(img_p)
        sitk_p = sitk.GetImageFromArray(pred_mask_sitk)
        sitk_p.CopyInformation(orig_img)
        sitk.WriteImage(sitk_p, out_file)
        if (idx + 1) % 25 == 0 or idx == len(test_ids) - 1:
            print(f"  [3D U-Net] Processed {idx + 1}/{len(test_ids)} cases...")


def run_vnet_inference(test_ids, pred_model_dir):
    from monai.networks.nets import VNet
    from monai.inferers import sliding_window_inference
    import monai.transforms as mt

    ckpt_path = os.path.join(CHECKPOINT_DIR, "vnet_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, "vnet_epoch_200.pt")

    print(f"[*] Loading V-Net checkpoint: {ckpt_path}")
    model = VNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        act="elu",
        dropout_prob_down=0.1,
        dropout_prob_up=(0.1, 0.1),
        bias=True
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

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
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
    ])

    for idx, cid in enumerate(test_ids):
        out_file = os.path.join(pred_model_dir, f"{cid}.nii.gz")
        if os.path.exists(out_file):
            continue
        img_p = dataset_paths.find_image(cid)
        if not img_p:
            continue
        batch = pre_trans({"image": img_p})
        inp = batch["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = sliding_window_inference(inp, (96, 96, 96), 4, model, overlap=0.5)
        batch["pred"] = logits.squeeze(0)
        batch = post_trans(batch)
        pred_probs = batch["pred"]
        pred_mask = torch.argmax(pred_probs, dim=0).cpu().numpy().astype(np.uint8)
        pred_mask_sitk = np.transpose(pred_mask, (2, 1, 0))

        orig_img = sitk.ReadImage(img_p)
        sitk_p = sitk.GetImageFromArray(pred_mask_sitk)
        sitk_p.CopyInformation(orig_img)
        sitk.WriteImage(sitk_p, out_file)
        if (idx + 1) % 25 == 0 or idx == len(test_ids) - 1:
            print(f"  [V-Net] Processed {idx + 1}/{len(test_ids)} cases...")


def run_rasnet_inference(test_ids, pred_model_dir):
    from rasnet_model import RASNet
    from monai.inferers import sliding_window_inference
    import monai.transforms as mt
    import cc3d

    ckpt_path = os.path.join(CHECKPOINT_DIR, "rasnet_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, "rasnet_epoch_200.pth")

    print(f"[*] Loading RASNet checkpoint: {ckpt_path}")
    model = RASNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

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
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
    ])

    for idx, cid in enumerate(test_ids):
        out_file = os.path.join(pred_model_dir, f"{cid}.nii.gz")
        if os.path.exists(out_file):
            continue
        img_p = dataset_paths.find_image(cid)
        if not img_p:
            continue
        batch = pre_trans({"image": img_p})
        inp = batch["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Standard 4-pass TTA
            logits = sliding_window_inference(inp, (96, 96, 96), 4, model, overlap=0.5)
            for d in [(2,), (3,), (4,)]:
                logits_f = sliding_window_inference(torch.flip(inp, d), (96, 96, 96), 4, model, overlap=0.5)
                logits += torch.flip(logits_f, d)
            logits /= 4.0

        probs = torch.softmax(logits, dim=1)
        batch["pred"] = probs.squeeze(0)
        batch = post_trans(batch)

        fg_prob = batch["pred"][1].cpu().numpy()
        pred_bin = (fg_prob > 0.6).astype(np.uint8)

        # Topology cleaning (keep top 2 components)
        labeled, N = cc3d.connected_components(pred_bin, return_N=True)
        if N > 0:
            sizes = np.bincount(labeled.flat)
            comps = [(i, sizes[i]) for i in range(1, len(sizes)) if sizes[i] >= 50]
            comps.sort(key=lambda x: x[1], reverse=True)
            cleaned = np.zeros_like(pred_bin, dtype=np.uint8)
            for cid_comp, _ in comps[:2]:
                cleaned[labeled == cid_comp] = 1
            pred_bin = cleaned

        pred_mask_sitk = np.transpose(pred_bin, (2, 1, 0))
        orig_img = sitk.ReadImage(img_p)
        sitk_p = sitk.GetImageFromArray(pred_mask_sitk)
        sitk_p.CopyInformation(orig_img)
        sitk.WriteImage(sitk_p, out_file)
        if (idx + 1) % 25 == 0 or idx == len(test_ids) - 1:
            print(f"  [RASNet] Processed {idx + 1}/{len(test_ids)} cases...")


def run_nnunet_inference(test_ids, pred_model_dir):
    """Executes nnU-Net inference using nnUNetPredictor directly from trained model folder."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    nnunet_res = os.path.join(BENCHMARK_DIR, "nnUNet_results")
    model_folder = os.path.join(nnunet_res, "Dataset501_CoronarySeg", "nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres")

    print(f"[*] Initializing nnUNetPredictor from: {model_folder}")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=DEVICE
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth"
    )

    in_files = []
    out_files = []
    for cid in test_ids:
        out_f = os.path.join(pred_model_dir, f"{cid}.nii.gz")
        if os.path.exists(out_f):
            continue
        src_f = dataset_paths.find_image(cid)
        if src_f and os.path.exists(src_f):
            in_files.append([src_f])
            out_files.append(out_f)

    if in_files:
        print(f"[*] Running nnUNet prediction on {len(in_files)} cases...")
        predictor.predict_from_files(
            in_files,
            out_files,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metric Evaluation Pipeline Across All 5 Models
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions_for_model(model_name: str, pred_model_dir: str, test_ids: list) -> pd.DataFrame:
    csv_out = os.path.join(OUTPUT_DIR, f"metrics_{model_name.lower().replace(' ', '_').replace('-', '_')}_200ep.csv")
    if os.path.exists(csv_out):
        try:
            df_exist = pd.read_csv(csv_out)
            if len(df_exist) == len(test_ids):
                print(f"[OK] Found existing complete metrics for {model_name} ({len(df_exist)} cases). Skipping recomputation.")
                return df_exist
        except Exception:
            pass

    print(f"\n[*] Calculating 8 metrics for {model_name} across {len(test_ids)} cases...")
    rows = []
    t0 = time.time()

    for idx, cid in enumerate(test_ids):
        pred_path = os.path.join(pred_model_dir, f"{cid}.nii.gz")
        gt_path = dataset_paths.find_label(cid)

        if not os.path.exists(pred_path) or not gt_path or not os.path.exists(gt_path):
            continue

        pred_img = sitk.ReadImage(pred_path)
        gt_img = sitk.ReadImage(gt_path)

        pred_arr = sitk.GetArrayFromImage(pred_img).astype(bool)
        gt_arr = sitk.GetArrayFromImage(gt_img).astype(bool)

        if pred_arr.shape != gt_arr.shape:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(gt_img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            pred_img = resampler.Execute(pred_img)
            pred_arr = sitk.GetArrayFromImage(pred_img).astype(bool)

        sp = pred_img.GetSpacing()
        spacing = (sp[2], sp[1], sp[0])

        m = compute_8_metrics(pred_arr, gt_arr, spacing)
        m["case_id"] = cid
        rows.append(m)

        if (idx + 1) % 50 == 0 or idx == len(test_ids) - 1:
            print(f"  [{model_name}] Evaluated {idx + 1}/{len(test_ids)} cases (Dice: {m['dice']:.4f}, clDice: {m['cldice']:.4f})...")

    df = pd.DataFrame(rows)
    cols = ["case_id", "dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall"]
    df = df[cols]

    csv_out = os.path.join(OUTPUT_DIR, f"metrics_{model_name.lower().replace(' ', '_').replace('-', '_')}_200ep.csv")
    df.to_csv(csv_out, index=False)
    print(f"[OK] Saved {model_name} metrics to: {csv_out} (Elapsed: {time.time() - t0:.1f}s)")
    return df


def holm_bonferroni_correction(p_values: list) -> list:
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        k = m - rank
        adj_p = min(1.0, p * k)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max
    return adjusted


def compile_summary_and_significance(all_dfs: dict):
    print("\n" + "=" * 80)
    print("[*] COMPILING 8-METRIC BENCHMARK SUMMARY AND WILCOXON SIGNIFICANCE TESTS")
    print("=" * 80)

    metrics = ["dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall"]
    metric_labels = {
        "dice": "Dice Similarity Coefficient (DSC)",
        "iou": "Intersection-over-Union (IoU)",
        "precision": "Precision (PPV)",
        "recall": "Recall / Sensitivity",
        "hd95": "95% Hausdorff Distance (HD95, mm)",
        "asd": "Average Surface Distance (ASD, mm)",
        "cldice": "Centerline Dice (clDice)",
        "centerline_recall": "Centerline Recall / Tree Completeness (T_sens)"
    }

    summary_rows = []
    for m_name, df in all_dfs.items():
        row = {"Model": m_name, "N": len(df)}
        for m in metrics:
            mean = df[m].mean()
            std = df[m].std()
            med = df[m].median()
            q25 = df[m].quantile(0.25)
            q75 = df[m].quantile(0.75)
            row[f"{m}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
            row[f"{m}_median_iqr"] = f"{med:.4f} [{q25:.4f}, {q75:.4f}]"
        summary_rows.append(row)

    sum_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(OUTPUT_DIR, "benchmark_200ep_summary_table.csv")
    sum_df.to_csv(sum_csv, index=False)
    print(f"[OK] Summary table saved to: {sum_csv}")

    # Significance testing: RASNet vs. each baseline
    rasnet_key = [k for k in all_dfs.keys() if "RASNet" in k][0]
    rasnet_df = all_dfs[rasnet_key].sort_values("case_id").reset_index(drop=True)

    sig_records = []
    for b_name, b_df in all_dfs.items():
        if b_name == rasnet_key:
            continue
        b_df_sorted = b_df.sort_values("case_id").reset_index(drop=True)
        raw_p_vals = []
        for m in metrics:
            x = rasnet_df[m].values
            y = b_df_sorted[m].values
            try:
                res = stats.wilcoxon(x, y, alternative="two-sided")
                raw_p_vals.append(res.pvalue)
            except Exception:
                raw_p_vals.append(1.0)

        adj_p_vals = holm_bonferroni_correction(raw_p_vals)
        for idx, m in enumerate(metrics):
            p_raw = raw_p_vals[idx]
            p_adj = adj_p_vals[idx]
            sig_records.append({
                "Comparison": f"{rasnet_key} vs. {b_name}",
                "Metric": metric_labels[m],
                "RASNet Mean": f"{rasnet_df[m].mean():.4f}",
                "Baseline Mean": f"{b_df_sorted[m].mean():.4f}",
                "Wilcoxon Raw p": f"{p_raw:.2e}" if p_raw < 0.001 else f"{p_raw:.4f}",
                "Holm-Bonferroni p_adj": f"{p_adj:.2e}" if p_adj < 0.001 else f"{p_adj:.4f}",
                "Significant (p < 0.05)": "YES (***)" if p_adj < 0.001 else ("YES (*)" if p_adj < 0.05 else "NO")
            })

    sig_df = pd.DataFrame(sig_records)
    sig_csv = os.path.join(OUTPUT_DIR, "benchmark_200ep_significance_tests.csv")
    sig_df.to_csv(sig_csv, index=False)

    # Markdown Report
    md_path = os.path.join(OUTPUT_DIR, "benchmark_200ep_final_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🏆 200-Epoch Matched Benchmark Suite — Final Publication Report\n\n")
        f.write(f"- **Test Cohort**: N={len(rasnet_df)} Held-Out Cases (ImageCAS 851–1000)\n")
        f.write("- **Models**: SegResNet, nnU-Net V2, 3D U-Net, V-Net, RASNet (Ours)\n")
        f.write("- **Budget**: Exactly 200 Epochs per model from random scratch\n\n")
        f.write("## 1. Primary Metrics (Mean ± Std)\n\n")
        f.write("| Model | Dice (DSC) | IoU | Precision | Recall | HD95 (mm) | ASD (mm) | clDice | Centerline Recall |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for row in summary_rows:
            f.write(f"| **{row['Model']}** | {row['dice_mean_std']} | {row['iou_mean_std']} | {row['precision_mean_std']} | {row['recall_mean_std']} | {row['hd95_mean_std']} | {row['asd_mean_std']} | {row['cldice_mean_std']} | {row['centerline_recall_mean_std']} |\n")

        f.write("\n## 2. Statistical Significance (Paired Two-Sided Wilcoxon Signed-Rank Test with Holm-Bonferroni Correction)\n\n")
        f.write("| Comparison | Metric | RASNet | Baseline | p-value (Holm-Bonferroni) | Sig |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in sig_records:
            f.write(f"| {r['Comparison']} | {r['Metric']} | {r['RASNet Mean']} | {r['Baseline Mean']} | {r['Holm-Bonferroni p_adj']} | {r['Significant (p < 0.05)']} |\n")

    print(f"[OK] Full report written to: {md_path}")


import argparse

def main():
    parser = argparse.ArgumentParser(description="5-Model 150-Case Matched 200-Epoch Benchmark Evaluator")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases (for testing)")
    parser.add_argument("--model", type=str, default=None, help="Run specific model only (SegResNet, 3D U-Net, V-Net, RASNet, nnU-Net V2)")
    parser.add_argument("--metrics-only", action="store_true", help="Skip inference and compute metrics on existing predictions")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    test_ids = get_test_case_ids()
    if args.limit:
        test_ids = test_ids[:args.limit]

    print("=" * 80)
    print(f"[*] 200-EPOCH MATCHED BENCHMARK 5-MODEL EVALUATOR")
    print(f"[*] Target Cases: {len(test_ids)} (IDs {min(test_ids)} - {max(test_ids)})")
    print(f"[*] Device: {DEVICE}")
    print("=" * 80)

    model_runners = [
        ("SegResNet", run_segresnet_inference),
        ("3D U-Net", run_3dunet_inference),
        ("V-Net", run_vnet_inference),
        ("RASNet", run_rasnet_inference),
        ("nnU-Net V2", run_nnunet_inference),
    ]

    if args.model:
        model_runners = [(name, runner) for name, runner in model_runners if args.model.lower() in name.lower()]

    all_dfs = {}
    for name, runner in model_runners:
        pred_model_dir = os.path.join(PRED_DIR, name.lower().replace(" ", "_").replace("-", "_"))
        os.makedirs(pred_model_dir, exist_ok=True)

        if not args.metrics_only:
            print(f"\n[>>>] STAGE: Running Inference for {name}...")
            try:
                runner(test_ids, pred_model_dir)
            except Exception as exc:
                print(f"[WARNING] Inference for {name} encountered an error: {exc}")

        # Compute 8 metrics
        try:
            df = evaluate_predictions_for_model(name, pred_model_dir, test_ids)
            all_dfs[name] = df
        except Exception as exc:
            print(f"[ERROR] Metric calculation for {name} failed: {exc}")

    if len(all_dfs) >= 2:
        compile_summary_and_significance(all_dfs)
    else:
        print("[*] Completed model evaluations. Awaiting remaining models.")


if __name__ == "__main__":
    main()

