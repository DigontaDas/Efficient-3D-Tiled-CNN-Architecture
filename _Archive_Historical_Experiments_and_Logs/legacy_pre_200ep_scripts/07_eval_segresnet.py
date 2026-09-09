"""
Task 1.5 — Compute Metrics for SegResNet Predictions (Parallelized)
====================================================================
Evaluates the SegResNet model predictions on the 150 test cases (IDs 851-1000)
in parallel using multiple CPU processes to speed up HD95 computation.
Uses the shared dataset_paths module to find the ground truth label files,
and computes evaluation metrics using eval_utils.

Run with:
    c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe -u 07_eval_segresnet.py
"""

import os
import sys
import csv
import json
import numpy as np
import concurrent.futures

# Add current directory to path so we can import eval_utils and dataset_paths
sys.path.insert(0, os.path.dirname(__file__))
import eval_utils
import dataset_paths

PRED_DIR = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__), "..",
        "all_four_validations", "mandatory_artifacts_segresnet", "predictions", "test"
    )
)

OUT_CSV = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__), "..",
        "all_four_validations", "mandatory_artifacts_segresnet", "metrics_segresnet.csv"
    )
)

SPLITS_FILE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "splits_final.json")
)


def evaluate_case(case_id: int) -> dict:
    # SegResNet predictions are named case_id.nii.gz
    pred_path = os.path.join(PRED_DIR, f"{case_id}.nii.gz")
    gt_path = dataset_paths.find_label(case_id)
    
    if not os.path.exists(pred_path):
        return {"case_id": case_id, "status": "missing_pred"}
        
    if not gt_path:
        return {"case_id": case_id, "status": "missing_gt"}
        
    try:
        m = eval_utils.compute_metrics(pred_path, gt_path)
        m["case_id"] = case_id
        m["status"] = "ok"
        return m
    except Exception as exc:
        return {"case_id": case_id, "status": "error", "error": str(exc)}


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
        
    print(f"Loaded test cases: {len(test_ids)}")
    print(f"Prediction dir    : {PRED_DIR}")
    print(f"Running evaluation in parallel...")
    print()
    
    results = []
    skipped_no_pred = []
    skipped_no_gt = []
    
    # Run evaluation in parallel using a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_case, cid): cid for cid in sorted(test_ids)}
        
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            cid = res["case_id"]
            status = res["status"]
            
            if status == "ok":
                results.append(res)
                print(
                    f"[{idx+1}/{len(test_ids)}] [OK] Case {cid:4d}  "
                    f"Dice={res['dice']:.4f}  IoU={res['iou']:.4f}  HD95={res['hd95']:.1f}mm"
                )
            elif status == "missing_pred":
                skipped_no_pred.append(cid)
            elif status == "missing_gt":
                skipped_no_gt.append(cid)
            elif status == "error":
                print(f"[{idx+1}/{len(test_ids)}] [WARNING] Error evaluating Case {cid}: {res['error']}")
                
    # Sort results by case_id for consistency in output
    results.sort(key=lambda x: x["case_id"])
    
    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Expected test cases   : {len(test_ids)}")
    print(f"  Predictions found     : {len(test_ids) - len(skipped_no_pred)}")
    print(f"  GT labels found       : {len(test_ids) - len(skipped_no_gt)}")
    print(f"  Successfully evaluated: {len(results)}")
    if skipped_no_pred:
        print(f"  Missing predictions ({len(skipped_no_pred)}): {skipped_no_pred[:10]}...")
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
            # Write metrics without the helper status key
            writer.writerows([{k: v for k, v in r.items() if k != "status"} for r in results])
            
        dices = [r["dice"] for r in results]
        hd95s = [r["hd95"] for r in results]
        print(f"\n  Dice mean +/- std : {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
        print(f"  HD95 mean +/- std : {np.mean(hd95s):.2f} +/- {np.std(hd95s):.2f} mm")
        print(f"\n  Metrics saved -> {OUT_CSV}")
    else:
        print("\n  [WARNING] No cases could be evaluated.")


if __name__ == "__main__":
    main()
