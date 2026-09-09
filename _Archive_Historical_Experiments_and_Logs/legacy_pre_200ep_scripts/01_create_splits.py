"""
Task 1.2 — Create Standardised Splits
=====================================
Generates a reproducible patient-level train/val/test split from the 934 valid
cases across the 1000-case ImageCAS dataset.

To maintain compatibility with the pre-trained 3D U-Net, the test split is
locked to cases 851-1000 (150 cases). The validation and training sets are
randomly split (seed 42) from the remaining 784 cases.

Run with:
    c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe 01_create_splits.py
"""

import os
import json
import random

AUDIT_FILE = os.path.join(os.path.dirname(__file__), "dataset_audit.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "splits_final.json")

# Refuse to overwrite an existing locked split.
if os.path.exists(OUTPUT_FILE):
    raise SystemExit(
        f"[ABORTED] {OUTPUT_FILE} already exists. "
        "Delete it manually if you truly intend to re-split."
    )


def load_valid_ids() -> list[int]:
    """Return case IDs that passed the audit (both img + label present)."""
    if not os.path.exists(AUDIT_FILE):
        raise FileNotFoundError(
            f"Run 00_dataset_audit.py first — {AUDIT_FILE} not found."
        )
    with open(AUDIT_FILE) as f:
        audit = json.load(f)
    return [c["id"] for c in audit["cases"] if c["img_ok"] and c["lbl_ok"]]


def main() -> None:
    valid_ids = load_valid_ids()
    n = len(valid_ids)
    print(f"Total valid cases: {n}")
    
    # Identify 3D U-Net test cases (851 to 1000)
    test_ids = [i for i in range(851, 1001) if i in valid_ids]
    n_test = len(test_ids)
    
    # Remaining cases for train/val pool
    train_val_pool = [i for i in valid_ids if i not in test_ids]
    
    random.seed(42)
    shuffled = train_val_pool.copy()
    random.shuffle(shuffled)
    
    # Target ~10% of total cases for validation (94 cases)
    n_val = 94
    n_train = len(shuffled) - n_val
    
    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train:])
    
    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": sorted(test_ids),
        "_meta": {
            "seed": 42,
            "total": n,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "note": "Locked split — test cases 851-1000 reserved to match 3D U-Net evaluation."
        }
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(splits, f, indent=2)
        
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"Test ID range: {min(test_ids)} .. {max(test_ids)}")
    print(f"Saved -> {OUTPUT_FILE}")
    print()
    print("[WARNING] This file is now LOCKED. Every model evaluation must use these test IDs.")


if __name__ == "__main__":
    main()
