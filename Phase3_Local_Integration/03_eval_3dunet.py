"""
Task 1.4 — Compute Metrics for 3D U-Net Predictions
===================================================
Evaluates the pre-trained 3D U-Net model on the 150 test cases (IDs 851-1000).
Uses the shared dataset_paths module to find the ground truth label files,
and computes evaluation metrics (Dice, IoU, Precision, Recall, HD95)
using eval_utils.

Run with:
    c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe 03_eval_3dunet.py
"""

import os
import sys
import csv
import json
import numpy as np

# Add parent and current directory to path so we can import eval_utils and dataset_paths
sys.path.insert(0, os.path.dirname(__file__))
import eval_utils
import dataset_paths

PRED_DIR = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__), "..",
        "all_four_validations", "mandatory_artifacts_3dunet", "predictions", "test"
    )
)

OUT_CSV = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__), "..",
        "all_four_validations", "mandatory_artifacts_3dunet", "metrics_3dunet.csv"
    )
)

GAP_REPORT = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__), "..",
        "all_four_validations", "mandatory_artifacts_3dunet", "eval_gap_report.txt"
    )
)

SPLITS_FILE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "splits_final.json")
)


def main() -> None:
    if not os.path.isdir(PRED_DIR):
        sys.exit(f"[ERROR] Prediction directory not found:\n  {PRED_DIR}")
        
    if not os.path.exists(SPLITS_FILE):
        sys.exit(f"[ERROR] Splits file not found. Run 01_create_splits.py first:\n  {SPLITS_FILE}")
        
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    test_ids = splits.get("test", [])
    
    if not test_ids:
        sys.exit("[ERROR] Test set in splits_final.json is empty!")
        
    print(f"Loaded test cases: {len(test_ids)} (IDs {min(test_ids)}..{max(test_ids)})")
    print(f"Prediction dir    : {PRED_DIR}")
    print()
    
    results = []
    skipped_no_pred = []
    skipped_no_gt = []
    
    for case_id in sorted(test_ids):
        pred_path = os.path.join(PRED_DIR, f"{case_id}_pred_mask.nii.gz")
        gt_path = dataset_paths.find_label(case_id)
        
        if not os.path.exists(pred_path):
            skipped_no_pred.append(case_id)
            print(f"[SKIP] No prediction found for case {case_id}")
            continue
            
        if not gt_path:
            skipped_no_gt.append(case_id)
            print(f"[SKIP] No ground truth label found for case {case_id}")
            continue
            
        try:
            m = eval_utils.compute_metrics(pred_path, gt_path)
            m["case_id"] = case_id
            results.append(m)
            print(
                f"[OK] Case {case_id:4d}  "
                f"Dice={m['dice']:.4f}  IoU={m['iou']:.4f}  HD95={m['hd95']:.1f}mm"
            )
        except Exception as exc:
            print(f"[WARNING] Error evaluating Case {case_id}: {exc}")
            
    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Expected test cases   : {len(test_ids)}")
    print(f"  Predictions found     : {len(test_ids) - len(skipped_no_pred)}")
    print(f"  GT labels found       : {len(test_ids) - len(skipped_no_gt)}")
    print(f"  Successfully evaluated: {len(results)}")
    if skipped_no_gt:
        print(f"  Cases lacking GT ({len(skipped_no_gt)}): {skipped_no_gt[:10]}...")
    print("=" * 60)
    
    # ── Save metrics CSV ──────────────────────────────────────────────────────
    if results:
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["case_id", "dice", "iou", "precision", "recall", "hd95"]
            )
            writer.writeheader()
            writer.writerows(results)
            
        dices = [r["dice"] for r in results]
        hd95s = [r["hd95"] for r in results]
        print(f"\n  Dice mean +/- std : {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
        print(f"  HD95 mean +/- std : {np.mean(hd95s):.2f} +/- {np.std(hd95s):.2f} mm")
        print(f"\n  Metrics saved -> {OUT_CSV}")
    else:
        print("\n  [WARNING] No cases could be evaluated.")
        
    # ── Gap report ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(GAP_REPORT), exist_ok=True)
    with open(GAP_REPORT, "w") as f:
        f.write("3D U-Net Evaluation Gap Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total test cases in split          : {len(test_ids)}\n")
        f.write(f"Predictions found on disk          : {len(test_ids) - len(skipped_no_pred)}\n")
        f.write(f"GT labels found                    : {len(test_ids) - len(skipped_no_gt)}\n")
        f.write(f"Cases evaluated successfully       : {len(results)}\n\n")
        f.write("Root cause / Resolution status:\n")
        if not skipped_no_gt:
            f.write("  All ground truth labels were found successfully. Gap resolved!\n")
        else:
            f.write(f"  Missing GT labels for {len(skipped_no_gt)} cases.\n")
            f.write("  Please ensure the full ImageCAS dataset is downloaded and mapped correctly.\n")
            
    print(f"  Gap report saved -> {GAP_REPORT}")


if __name__ == "__main__":
    main()
