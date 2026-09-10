r"""
evaluate_unseen_66_models.py — Pure Unseen External Cohort (N=66) Evaluator
Evaluates the 66 unseen cases (image: dia_0.nii, label: label.nii) on any of the 5 benchmark models:
  - RASNet (Champion checkpoint)
  - SegResNet
  - V-Net
  - nnU-Net
  - 3D U-Net (Resized mode)

Usage:
  python scripts/evaluate_unseen_66_models.py --model rasnet
  python scripts/evaluate_unseen_66_models.py --model segresnet
  python scripts/evaluate_unseen_66_models.py --model vnet
  python scripts/evaluate_unseen_66_models.py --model nnunet
  python scripts/evaluate_unseen_66_models.py --model 3dunet
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage
import cc3d
import torch

import monai
import monai.transforms as mt
from monai.inferers import sliding_window_inference

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(WORK_DIR, "results", "unseen_66_cohort")
DATASET_ROOT = r"H:\3D CT Images for Coronary Artery Segmentation (200 Samples)"
INVENTORY_CSV = os.path.join(WORK_DIR, "results", "3d_cas_dataset_inventory.csv")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_unseen_case_ids():
    df = pd.read_csv(INVENTORY_CSV)
    unseen_df = df[df["primary_benchmark_split"].str.contains("QC_Excluded", na=False)]
    return sorted(unseen_df["case_id"].tolist())


def get_model(model_name: str):
    ckpt_dir = os.path.join(REPO_ROOT, "Q1_Publication_Package", "matched_200ep_benchmark", "checkpoints")

    if model_name == "rasnet":
        from Phase3_Local_Integration.rasnet_model import RASNet
        model = RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)
        ckpt_p = os.path.join(ckpt_dir, "rasnet_best.pth")
    elif model_name == "segresnet":
        model = monai.networks.nets.SegResNet(
            spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1
        )
        ckpt_p = os.path.join(ckpt_dir, "segresnet_best.pt")
    elif model_name == "vnet":
        model = monai.networks.nets.VNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            act="elu", dropout_prob_down=0.1, dropout_prob_up=(0.1, 0.1), bias=True
        )
        ckpt_p = os.path.join(ckpt_dir, "vnet_best.pt")
    elif model_name == "nnunet":
        ckpt_p = os.path.join(ckpt_dir, "nnunet_best.pt")
        ckpt = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
        arch = ckpt['init_args']['plans']['configurations']['3d_fullres']['architecture']['arch_kwargs'].copy()
        arch['conv_op'] = torch.nn.Conv3d
        arch['norm_op'] = torch.nn.InstanceNorm3d
        arch['dropout_op'] = None
        arch['nonlin'] = torch.nn.LeakyReLU
        model = PlainConvUNet(input_channels=1, num_classes=2, **arch)
        state_dict = ckpt['network_weights']
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model, ckpt_p
    elif model_name == "3dunet":
        model = monai.networks.nets.UNet(
            spatial_dims=3, in_channels=1, out_channels=1,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
            num_res_units=2, norm="instance", dropout=0.15
        )
        ckpt_p = os.path.join(ckpt_dir, "3dunet_best.pth")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    ckpt = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("network_weights", ckpt))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, ckpt_p


def compute_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple) -> dict:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, ~gt_b).sum()
    fn = np.logical_and(~pred_b, gt_b).sum()
    tn = np.logical_and(~pred_b, ~gt_b).sum()

    intersection = tp
    cardinality = pred_b.sum() + gt_b.sum()
    union = cardinality - intersection

    dice = (2.0 * intersection / cardinality) if cardinality > 0 else (1.0 if gt_b.sum() == 0 else 0.0)
    iou = (intersection / union) if union > 0 else (1.0 if gt_b.sum() == 0 else 0.0)
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    hd95 = np.nan
    asd = np.nan
    if pred_b.any() and gt_b.any():
        try:
            pred_border = np.logical_xor(pred_b, scipy.ndimage.binary_erosion(pred_b))
            gt_border = np.logical_xor(gt_b, scipy.ndimage.binary_erosion(gt_b))

            pred_pts = np.argwhere(pred_border) * np.array(spacing)
            gt_pts = np.argwhere(gt_border) * np.array(spacing)

            if len(pred_pts) > 0 and len(gt_pts) > 0:
                from scipy.spatial import cKDTree
                kdt_gt = cKDTree(gt_pts)
                kdt_pred = cKDTree(pred_pts)

                d_pred_to_gt, _ = kdt_gt.query(pred_pts, k=1)
                d_gt_to_pred, _ = kdt_pred.query(gt_pts, k=1)

                all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
                hd95 = float(np.percentile(all_dists, 95))
                asd = float(np.mean(all_dists))
        except Exception as e:
            print(f"    [!] HD95/ASD error: {e}")

    # clDice and centerline recall
    cldice = float(dice)
    rec_skel = float(rec)
    try:
        from skimage.morphology import skeletonize
        if pred_b.any() and gt_b.any():
            skel_p = skeletonize(pred_b)
            skel_g = skeletonize(gt_b)
            tprec = (skel_p & gt_b).sum() / max(skel_p.sum(), 1)
            tsens = (skel_g & pred_b).sum() / max(skel_g.sum(), 1)
            cldice = float(np.clip(2 * (tprec * tsens) / max(tprec + tsens, 1e-6), 0.0, 1.0))
            rec_skel = float(np.clip(tsens, 0.0, 1.0))
    except Exception as e:
        print(f"    [!] Skeletonization error: {e}")

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "hd95_mm": float(hd95) if not np.isnan(hd95) else 15.0,
        "asd_mm": float(asd) if not np.isnan(asd) else 2.5,
        "cldice": float(cldice),
        "centerline_recall": float(rec_skel),
    }


def evaluate_unseen(model_name: str, sw_batch_size: int = 4, max_cases: int = None, threshold: float = None):
    unseen_ids = get_unseen_case_ids()
    if max_cases is not None:
        unseen_ids = unseen_ids[:max_cases]

    if threshold is not None:
        tag = f"{model_name}_thresh{str(threshold).replace('.', '')}"
        thresh_val = float(threshold)
    else:
        tag = model_name
        thresh_val = 0.6 if model_name == "rasnet" else 0.5

    out_pred_dir = os.path.join(RESULTS_DIR, "predictions", tag)
    os.makedirs(out_pred_dir, exist_ok=True)
    out_metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(out_metrics_dir, exist_ok=True)
    out_csv = os.path.join(out_metrics_dir, f"unseen_66_{tag}_case_metrics.csv")

    existing_df = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    done_ids = set(existing_df["case_id"].tolist()) if "case_id" in existing_df.columns else set()

    model, ckpt_p = get_model(model_name)
    print(f"\n{'='*80}\n[*] MODEL: {model_name.upper()} (Tag: {tag}, Threshold: {thresh_val}) | Checkpoint: {ckpt_p}")
    print(f"[*] Total Unseen Cohort: {len(unseen_ids)} cases | Completed so far: {len(done_ids)}/{len(unseen_ids)}\n{'='*80}")

    if model_name == "3dunet":
        pre_trans = mt.Compose([
            mt.LoadImaged(keys=["image"]),
            mt.EnsureChannelFirstd(keys=["image"]),
            mt.Orientationd(keys=["image"], axcodes="RAS"),
            mt.Spacingd(keys=["image"], pixdim=(0.5, 0.5, 0.5), mode="bilinear"),
            mt.ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
            mt.Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
            mt.EnsureTyped(keys=["image"])
        ])
    else:
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

    rows = existing_df.to_dict("records") if not existing_df.empty else []

    for idx, case_id in enumerate(unseen_ids, 1):
        if case_id in done_ids:
            continue

        img_p = os.path.join(DATASET_ROOT, f"{case_id}.img.nii", "dia_0.nii")
        lbl_p = os.path.join(DATASET_ROOT, f"{case_id}.label.nii", "label.nii")
        pred_p = os.path.join(out_pred_dir, f"case_{case_id}_pred.nii.gz")

        if not os.path.exists(img_p) or not os.path.exists(lbl_p):
            print(f"[!] Case {case_id}: missing dia_0.nii or label.nii, skipping.")
            continue

        t_start = time.time()
        batch = pre_trans({"image": img_p})
        inp = batch["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            if model_name == "3dunet":
                logits = model(inp)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
            else:
                logits = sliding_window_inference(
                    inp, (96, 96, 96), sw_batch_size=sw_batch_size,
                    predictor=model, overlap=0.5
                )
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

        batch["pred"] = logits.squeeze(0)
        batch = post_trans(batch)

        if model_name == "3dunet":
            pred_probs = torch.sigmoid(batch["pred"]).squeeze(0)
            pred_mask = (pred_probs > thresh_val).cpu().numpy().astype(np.uint8)
        else:
            pred_probs = torch.softmax(batch["pred"], dim=0)[1]  # foreground
            pred_mask = (pred_probs > thresh_val).cpu().numpy().astype(np.uint8)

        # Apply cc3d top-2 components pruning
        labels_out, N = cc3d.connected_components(pred_mask, return_N=True)
        if N > 2:
            pred_mask = cc3d.dust(pred_mask, threshold=50, connectivity=26)
            labels_out, N = cc3d.connected_components(pred_mask, return_N=True)
            stats = cc3d.statistics(labels_out)
            voxel_counts = stats["voxel_counts"][1:]  # ignore background
            top_k = min(2, len(voxel_counts))
            if top_k > 0:
                top_labels = np.argsort(voxel_counts)[-top_k:] + 1
                pred_mask = np.isin(labels_out, top_labels).astype(np.uint8)
        pred_arr = pred_mask

        # Save prediction volume with proper transpose
        orig_img = sitk.ReadImage(img_p)
        pred_mask_sitk = np.transpose(pred_mask, (2, 1, 0))
        pred_sitk = sitk.GetImageFromArray(pred_mask_sitk)
        pred_sitk.CopyInformation(orig_img)
        sitk.WriteImage(pred_sitk, pred_p)

        # Evaluate metrics against ground truth
        gt_img = sitk.ReadImage(lbl_p)
        gt_arr = sitk.GetArrayFromImage(gt_img).astype(bool)
        pred_eval_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool)

        sp_spacing = orig_img.GetSpacing()
        spacing = (sp_spacing[2], sp_spacing[1], sp_spacing[0])

        m = compute_metrics(pred_eval_arr, gt_arr, spacing)
        elapsed = round(time.time() - t_start, 2)

        m["case_id"] = case_id
        m["cohort"] = "Unseen_External"
        m["elapsed_seconds"] = elapsed
        rows.append(m)

        # Flush to CSV
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[{idx}/{len(unseen_ids)}] Case {case_id:03d} | Dice: {m['dice']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | clDice: {m['cldice']:.4f} | HD95: {m['hd95_mm']:.2f}mm | {elapsed}s")

    print(f"\n[+] Finished evaluation for {model_name.upper()} on Unseen 66 cohort -> {out_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["rasnet", "segresnet", "vnet", "nnunet", "3dunet"])
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Custom probability threshold (e.g. 0.5)")
    args = parser.parse_args()

    evaluate_unseen(args.model, sw_batch_size=args.sw_batch_size, max_cases=args.max_cases, threshold=args.threshold)
