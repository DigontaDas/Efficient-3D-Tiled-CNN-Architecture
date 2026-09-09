"""
Task 1.7 — Generate the Unified Comparison Table
=================================================
Aggregates performance metrics (Dice, IoU, Precision, Recall, HD95) across all
evaluated models (SegResNet, nnU-Net, 3D U-Net, V-Net) on the 150 test cases.
Outputs a Markdown table and saves the summary to unified_comparison_table.csv.

Run with:
    c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe 08_build_comparison_table.py
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "all_four_validations"))

model_paths = {
    "RASNet (Ours)": os.path.realpath(os.path.join(os.path.dirname(__file__), "metrics_rasnet.csv")),
    "SegResNet": os.path.join(ROOT_DIR, "mandatory_artifacts_segresnet", "metrics_segresnet.csv"),
    "nnU-Net":   os.path.join(ROOT_DIR, "mandatory_artifacts_nnunet", "metrics_nnunet.csv"),
    "3D U-Net":  os.path.join(ROOT_DIR, "mandatory_artifacts_3dunet", "metrics_3dunet.csv"),
    "V-Net":     None,  # Baseline with documented training instability / limits
}

OUT_CSV = os.path.join(ROOT_DIR, "unified_comparison_table.csv")


def main() -> None:
    print("Aggregating model evaluation results...")
    
    rows = []
    for model, path in model_paths.items():
        if path is None or not os.path.exists(path):
            note = "Instability Limits" if model == "V-Net" else "Missing results"
            rows.append({
                "Model": model,
                "N": 0 if model != "V-Net" else "N/A",
                "Dice": "-",
                "IoU": "-",
                "Precision": "-",
                "Recall": "-",
                "HD95 (mm)": "- (" + note + ")"
            })
            continue
            
        df = pd.read_csv(path)
        
        # Calculate mean and standard dev
        dice_m, dice_s = df['dice'].mean(), df['dice'].std()
        iou_m, iou_s   = df['iou'].mean(), df['iou'].std()
        prec_m, prec_s = df['precision'].mean(), df['precision'].std()
        rec_m, rec_s   = df['recall'].mean(), df['recall'].std()
        
        # HD95 can be infinite or NaN in some edge cases (e.g. empty prediction)
        # Filter out inf/nan values for mean calculation
        hd95_vals = df['hd95'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(hd95_vals) > 0:
            hd95_m, hd95_s = hd95_vals.mean(), hd95_vals.std()
            hd95_str = f"{hd95_m:.2f} +/- {hd95_s:.2f}"
        else:
            hd95_str = "-"
            
        rows.append({
            "Model": model,
            "N": len(df),
            "Dice":      f"{dice_m:.4f} +/- {dice_s:.4f}",
            "IoU":       f"{iou_m:.4f} +/- {iou_s:.4f}",
            "Precision": f"{prec_m:.4f} +/- {prec_s:.4f}",
            "Recall":    f"{rec_m:.4f} +/- {rec_s:.4f}",
            "HD95 (mm)": hd95_str,
        })
        
    out_df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("                          IMAGE CAS MODEL COMPARISON REPORT")
    print("=" * 80)
    print(out_df.to_markdown(index=False))
    print("=" * 80)
    
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nUnified comparison table saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
