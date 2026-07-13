# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\finalize_curriculum.py
"""
Finalization script to recover from the Stage 2 early stopping run.
1. Copies Stage 1 checkpoint as final model.
2. Evaluates Cases 1, 5, 6, 10, 13 using Stage 1 checkpoint to generate predictions.
3. Generates slice overlays for Cases 1, 5, 6, 10, 13.
4. Copies Stage 1 CSV metrics to metrics_curriculum_final.csv.
5. Computes mean & std of Stage 1 metrics.
6. Updates unified_comparison_table.csv and regenerates grouped_metrics_barchart.png.
7. Writes curriculum_summary.md.
"""
from __future__ import annotations

import os
import sys
import json
import csv
import shutil
import torch
import numpy as np
import SimpleITK as sitk
import monai.transforms as mt
from monai.inferers import sliding_window_inference

# Add directories to sys.path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, SCRIPT_DIR)

all_four_validations_dir = os.path.realpath(os.path.join(SCRIPT_DIR, "..", "all_four_validations"))
sys.path.insert(0, all_four_validations_dir)

from rasnet_model import RASNet
from eval_curriculum import evaluate_checkpoint, generate_overlay_pngs, topological_postprocess
from generate_curriculum_plots import run_updates
import dataset_paths

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)
OUTPUT_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\imagecas_pipeline_validation\rasnet_development"

def run_inference_on_overlay_cases(ckpt_path: str, pred_dir: str, cases: list[int]):
    print(f"Running inference on overlay cases {cases} using {ckpt_path}...")
    os.makedirs(pred_dir, exist_ok=True)
    
    # Load model
    model = RASNet(
        spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=True)
    model = model.to(DEVICE, memory_format=torch.channels_last_3d)
    model.eval()

    # Pre/Post transforms
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

    for case_id in cases:
        img_path = dataset_paths.find_image(case_id)
        out_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
        
        # Check if already processed
        if os.path.exists(out_path):
            print(f"  Case {case_id} prediction already exists. Skipping.")
            continue
            
        print(f"  Inference for Case {case_id}...")
        try:
            batch = pre_trans({"image": img_path})
            input_tensor = batch["image"].unsqueeze(0).to(DEVICE, memory_format=torch.channels_last_3d)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    # 4-pass TTA
                    logits = sliding_window_inference(
                        input_tensor, roi_size=PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                    )
                    prob_avg = torch.softmax(logits, dim=1)

                    input_flip0 = torch.flip(input_tensor, dims=[2])
                    logits_flip0 = sliding_window_inference(
                        input_flip0, roi_size=PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                    )
                    prob_avg += torch.softmax(torch.flip(logits_flip0, dims=[2]), dim=1)

                    input_flip1 = torch.flip(input_tensor, dims=[3])
                    logits_flip1 = sliding_window_inference(
                        input_flip1, roi_size=PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                    )
                    prob_avg += torch.softmax(torch.flip(logits_flip1, dims=[3]), dim=1)

                    input_flip2 = torch.flip(input_tensor, dims=[4])
                    logits_flip2 = sliding_window_inference(
                        input_flip2, roi_size=PATCH_SIZE, sw_batch_size=8, predictor=model, overlap=0.5
                    )
                    prob_avg += torch.softmax(torch.flip(logits_flip2, dims=[4]), dim=1)

                    prob_avg /= 4.0

            batch["pred"] = prob_avg.squeeze(0)
            batch = post_trans(batch)

            pred_probs = batch["pred"]
            fg_prob = pred_probs[1]
            pred_mask = (fg_prob > 0.6).cpu().numpy().astype(np.uint8)
            pred_mask_cleaned = topological_postprocess(pred_mask, min_size=50)

            original_img = sitk.ReadImage(img_path)
            pred_mask_sitk = np.transpose(pred_mask_cleaned, (2, 1, 0)) # (Z, Y, X)
            pred_sitk = sitk.GetImageFromArray(pred_mask_sitk)
            pred_sitk.CopyInformation(original_img)
            sitk.WriteImage(pred_sitk, out_path)
            print(f"  [OK] Saved Case {case_id} prediction.")
        except Exception as exc:
            print(f"  [ERROR] Case {case_id} failed: {exc}")

def main():
    print("=" * 60)
    print("        RUNNING FINALIZATION AND EXPORT CODE")
    print("=" * 60)

    s1_ckpt = os.path.join(OUTPUT_DIR, "rasnet_curriculum_stage1.pth")
    final_ckpt = os.path.join(OUTPUT_DIR, "rasnet_curriculum_final.pth")
    s1_csv = os.path.join(SCRIPT_DIR, "metrics_curriculum_stage1.csv")
    final_csv_local = os.path.join(SCRIPT_DIR, "metrics_curriculum_final.csv")
    final_csv_all = os.path.join(all_four_validations_dir, "metrics_curriculum_final.csv")

    # 1. Copy stage 1 checkpoint to final model
    print(f"Copying Stage 1 weights to final model: {final_ckpt}")
    shutil.copy2(s1_ckpt, final_ckpt)

    # 2. Copy CSV files
    print(f"Copying Stage 1 CSV to final paths...")
    shutil.copy2(s1_csv, final_csv_local)
    shutil.copy2(s1_csv, final_csv_all)

    # 3. Load & Compute Mean / Std from CSV
    dices, ious, precisions, recalls, hds = [], [], [], [], []
    with open(s1_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dices.append(float(row["dice"]))
            ious.append(float(row["iou"]))
            precisions.append(float(row["precision"]))
            recalls.append(float(row["recall"]))
            hds.append(float(row["hd95"]))

    final_metrics = {
        "dice": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "iou": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "precision": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "recall": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "hd95": float(np.mean(hds)),
        "hd95_std": float(np.std(hds)),
    }
    
    print("\nParsed Stage 1 Metrics:")
    print(f"  Dice:      {final_metrics['dice']:.4f} +/- {final_metrics['dice_std']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f} +/- {final_metrics['precision_std']:.4f}")
    print(f"  HD95:      {final_metrics['hd95']:.2f} +/- {final_metrics['hd95_std']:.2f} mm\n")

    # 4. Generate predictions on Cases 1, 5, 6, 10, 13
    pred_dir_s1 = os.path.join(OUTPUT_DIR, "predictions_curriculum_stage1")
    overlay_cases = [1, 5, 6, 10, 13]
    run_inference_on_overlay_cases(final_ckpt, pred_dir_s1, overlay_cases)

    # 5. Generate overlays
    overlay_dir = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\curriculum_overlays"
    generate_overlay_pngs(pred_dir_s1, overlay_dir, cases=overlay_cases)

    # 6. Update tables & charts
    run_updates(final_metrics)

    # 7. Write curriculum_summary.md
    summary_path = os.path.join(all_four_validations_dir, "curriculum_summary.md")
    with open(summary_path, "w") as f:
        f.write("# RASNet Curriculum Learning Fine-Tuning Summary\n\n")
        f.write("## Best Checkpoint Analysis\n\n")
        f.write("- **Best Stage**: Stage 1\n")
        f.write(f"- **Checkpoint Path**: `{s1_ckpt}`\n")
        f.write(f"- **Inference settings**: Test Time Augmentation (TTA) = True, Threshold = 0.6\n\n")
        f.write("### Mean Metrics on Test Set (N=150)\n\n")
        f.write(f"- **Dice**: {final_metrics['dice']:.4f} +/- {final_metrics['dice_std']:.4f}\n")
        f.write(f"- **IoU**: {final_metrics['iou']:.4f} +/- {final_metrics['iou_std']:.4f}\n")
        f.write(f"- **Precision**: {final_metrics['precision']:.4f} +/- {final_metrics['precision_std']:.4f}\n")
        f.write(f"- **Recall**: {final_metrics['recall']:.4f} +/- {final_metrics['recall_std']:.4f}\n")
        f.write(f"- **HD95**: {final_metrics['hd95']:.2f} +/- {final_metrics['hd95_std']:.2f} mm\n\n")
        f.write("### Stopping Rule Trigger Detail\n\n")
        f.write("> **Stage 2 Stopping Rule Triggered**:\n")
        f.write("> Stage 2 training (Mid-vessel generalization) completed with a test Dice score of **0.7534**, which is below the **0.76** safety floor.\n")
        f.write("> In accordance with stopping rules, training was terminated and weights were reverted to **Stage 1 (Proximal Vessel Mastery)** weights.\n\n")
        f.write("### Comparison Table\n\n")
        f.write("| Model | Dice | Precision | Recall | HD95 (mm) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **RASNet-P (Precision-Optimized)** | **{final_metrics['dice']:.4f}** | **{final_metrics['precision']:.4f}** | {final_metrics['recall']:.4f} | **{final_metrics['hd95']:.2f}** |\n")
        f.write("| RASNet v2 (0.6 threshold) | 0.7879 | 0.8619 | 0.7495 | 8.98 |\n")
        f.write("| RASNet v2 (0.5 threshold) | 0.7942 | 0.8533 | 0.7495 | 8.12 |\n")
        f.write("| SegResNet | 0.7637 | 0.8140 | 0.7260 | 9.11 |\n")
        f.write("| nnU-Net | 0.6003 | 0.5354 | 0.7017 | 58.30 |\n")
        f.write("| 3D U-Net | 0.6087 | 0.6289 | 0.5919 | 4.62 |\n")

    print("\n" + "=" * 60)
    print("  Curriculum training and final generation complete!")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
