#!/usr/bin/env python3
"""
evaluate_ablation_suite.py — Evaluates All Ablation Configurations on N=150 Cases
=================================================================================
Runs 3D sliding-window inference at 0.5mm isotropic spacing and computes all 8 metrics:
  - Step 2: ablation_attngate_best.pth (AttentionGate3D, plain loss)
  - Step 3: ablation_deepsup_best.pth (AttentionGate3D + Deep Sup, plain loss)
  - Step 4: rasnet_best.pth (Raw single pass, no TTA, no cc3d)
  - Step 5: rasnet_best.pth (4-pass TTA, no cc3d)

Outputs:
  - Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step2_attngate_200ep.csv
  - Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step3_deepsup_200ep.csv
  - Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step4_raw_model_200ep.csv
  - Q1_Publication_Package/ablation/evaluation_results/metrics_ablation_step5_tta_200ep.csv
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from monai.inferers import sliding_window_inference
import monai.transforms as mt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "Phase3_Local_Integration"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Q1_Publication_Package", "matched_200ep_benchmark"))

from rasnet_model import RASNet
import dataset_paths
from evaluate_matched_200ep_models import compute_8_metrics, get_test_case_ids

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(REPO_ROOT, "Q1_Publication_Package", "ablation", "evaluation_results")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run_single_model_inference(model_path: str, out_pred_dir: str, test_ids: list, use_tta: bool = False, overwrite: bool = False, sw_batch_size: int = 8):
    os.makedirs(out_pred_dir, exist_ok=True)
    print(f"[*] Loading model from: {model_path} (TTA={use_tta}, SW Batch Size={sw_batch_size})")

    model = RASNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1
    ).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
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
        out_f = os.path.join(out_pred_dir, f"{cid}.nii.gz")
        if os.path.exists(out_f) and not overwrite:
            continue
        img_p = dataset_paths.find_image(cid)
        if not img_p:
            continue

        batch = pre_trans({"image": img_p})
        inp = batch["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if use_tta:
                # 4-Pass TTA: original + 3 flips
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    p1 = sliding_window_inference(inp, (96, 96, 96), sw_batch_size, model, overlap=0.5)
                    p2 = torch.flip(sliding_window_inference(torch.flip(inp, [2]), (96, 96, 96), sw_batch_size, model, overlap=0.5), [2])
                    p3 = torch.flip(sliding_window_inference(torch.flip(inp, [3]), (96, 96, 96), sw_batch_size, model, overlap=0.5), [3])
                    p4 = torch.flip(sliding_window_inference(torch.flip(inp, [4]), (96, 96, 96), sw_batch_size, model, overlap=0.5), [4])
                    logits = (p1 + p2 + p3 + p4) / 4.0
            else:
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = sliding_window_inference(inp, (96, 96, 96), sw_batch_size, model, overlap=0.5)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

        batch["pred"] = logits.squeeze(0)
        batch = post_trans(batch)
        pred_probs = torch.softmax(batch["pred"], dim=0)[1]  # Foreground channel
        pred_mask = (pred_probs > 0.6).cpu().numpy().astype(np.uint8)

        orig_img = sitk.ReadImage(img_p)
        pred_mask_sitk = np.transpose(pred_mask, (2, 1, 0))
        sitk_p = sitk.GetImageFromArray(pred_mask_sitk)
        sitk_p.CopyInformation(orig_img)
        sitk.WriteImage(sitk_p, out_f)

        if (idx + 1) % 50 == 0 or idx == len(test_ids) - 1:
            print(f"  Processed {idx + 1}/{len(test_ids)} cases...")


def evaluate_predictions(pred_dir: str, out_csv: str, test_ids: list, overwrite: bool = False) -> pd.DataFrame:
    if os.path.exists(out_csv) and not overwrite:
        print(f"[OK] Found existing complete metrics: {out_csv}")
        return pd.read_csv(out_csv)

    print(f"[*] Calculating 8 metrics across {len(test_ids)} cases...")
    rows = []
    t0 = time.time()

    for idx, cid in enumerate(test_ids):
        pred_p = os.path.join(pred_dir, f"{cid}.nii.gz")
        gt_p = dataset_paths.find_label(cid)
        if not os.path.exists(pred_p) or not gt_p or not os.path.exists(gt_p):
            continue

        pred_img = sitk.ReadImage(pred_p)
        gt_img = sitk.ReadImage(gt_p)

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
            print(f"  Evaluated {idx + 1}/{len(test_ids)} cases (Dice: {m['dice']:.4f})...")

    df = pd.DataFrame(rows)
    cols = ["case_id", "dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall"]
    df = df[cols]
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved metrics to {out_csv} ({time.time() - t0:.1f}s)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ablation Models Suite (N=150 Cases)")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite of existing predictions and CSVs")
    parser.add_argument("--sw-batch-size", type=int, default=8, help="Sliding window batch size (default: 8, optimal for 16GB GPU)")
    args = parser.parse_args()

    test_ids = get_test_case_ids()
    ckpt_dir = os.path.join(REPO_ROOT, "Q1_Publication_Package", "matched_200ep_benchmark", "checkpoints")

    tasks = [
        {
            "name": "Step 2 (+ AttentionGate3D Only)",
            "ckpt": os.path.join(ckpt_dir, "ablation_attngate_best.pth"),
            "pred_dir": os.path.join(PRED_DIR, "step2_attngate"),
            "csv": os.path.join(OUTPUT_DIR, "metrics_ablation_step2_attngate_200ep.csv"),
            "use_tta": False
        },
        {
            "name": "Step 3 (+ Deep Supervision)",
            "ckpt": os.path.join(ckpt_dir, "ablation_deepsup_best.pth"),
            "pred_dir": os.path.join(PRED_DIR, "step3_deepsup"),
            "csv": os.path.join(OUTPUT_DIR, "metrics_ablation_step3_deepsup_200ep.csv"),
            "use_tta": False
        },
        {
            "name": "Step 4 (+ StenosisAwareLoss Raw Model)",
            "ckpt": os.path.join(ckpt_dir, "rasnet_best.pth"),
            "pred_dir": os.path.join(PRED_DIR, "step4_raw_model"),
            "csv": os.path.join(OUTPUT_DIR, "metrics_ablation_step4_raw_model_200ep.csv"),
            "use_tta": False
        },
        {
            "name": "Step 5 (+ 4-Pass TTA)",
            "ckpt": os.path.join(ckpt_dir, "rasnet_best.pth"),
            "pred_dir": os.path.join(PRED_DIR, "step5_tta"),
            "csv": os.path.join(OUTPUT_DIR, "metrics_ablation_step5_tta_200ep.csv"),
            "use_tta": True
        }
    ]

    for t in tasks:
        print("\n" + "=" * 80)
        print(f"[*] EVALUATING: {t['name']}")
        print("=" * 80)
        if not os.path.exists(t["ckpt"]):
            print(f"[!] Checkpoint not found: {t['ckpt']}. Skipping.")
            continue
        run_single_model_inference(t["ckpt"], t["pred_dir"], test_ids, use_tta=t["use_tta"], overwrite=args.overwrite, sw_batch_size=args.sw_batch_size)
        evaluate_predictions(t["pred_dir"], t["csv"], test_ids, overwrite=args.overwrite)

    print("\n[OK] Ablation suite evaluation finished.")


if __name__ == "__main__":
    main()
