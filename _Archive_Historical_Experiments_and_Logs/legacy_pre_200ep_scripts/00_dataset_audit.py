"""
Task 1.1 — Dataset Audit
========================
Scans all 1000 ImageCAS cases using the shared dataset_paths module.
Verifies that both image and label files exist, retrieves their size/spacing
using fast header-only reading (SimpleITK.ImageFileReader), and writes a
JSON audit report.

Run with:
    c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe 00_dataset_audit.py
"""

import os
import json
import sys
import SimpleITK as sitk

# Add current directory to path to import dataset_paths
sys.path.insert(0, os.path.dirname(__file__))
import dataset_paths

OUTPUT = os.path.join(os.path.dirname(__file__), "dataset_audit.json")


def audit_case(case_id: int) -> dict:
    img_path, lbl_path = dataset_paths.find_pair(case_id)
    
    row = {
        "id": case_id,
        "img_ok": False,
        "lbl_ok": False,
    }
    
    if img_path:
        try:
            # Fast header reading
            reader = sitk.ImageFileReader()
            reader.SetFileName(img_path)
            reader.ReadImageInformation()
            row.update({
                "img_ok": True,
                "img_size": list(reader.GetSize()),
                "img_spacing": [round(s, 4) for s in reader.GetSpacing()],
            })
        except Exception as exc:
            row["img_error"] = str(exc)
            
    if lbl_path:
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(lbl_path)
            reader.ReadImageInformation()
            row.update({
                "lbl_ok": True,
                "lbl_size": list(reader.GetSize()),
            })
        except Exception as exc:
            row["lbl_error"] = str(exc)
            
    return row


def main() -> None:
    print("Starting audit of all 1000 ImageCAS cases...")
    results = []
    ok_count = 0
    
    for case_id in range(1, 1001):
        row = audit_case(case_id)
        results.append(row)
        
        is_ok = row["img_ok"] and row["lbl_ok"]
        status = "[OK]" if is_ok else "[XX]"
        if is_ok:
            ok_count += 1
            
        # Print progress every 100 cases or if missing to avoid overwhelming output
        if case_id % 100 == 0 or not is_ok:
            size_str = str(row.get("img_size", "missing"))
            print(f"{status} Case {case_id:4d}  size={size_str}")

    missing = [r["id"] for r in results if not (r["img_ok"] and r["lbl_ok"])]
    
    print()
    print("=" * 45)
    print(f"  Total: 1000 | OK: {ok_count} | Missing: {len(missing)}")
    if missing:
        # Show a truncated list if too long
        show_missing = missing[:20]
        suffix = "..." if len(missing) > 20 else ""
        print(f"  Missing IDs ({len(missing)}): {show_missing}{suffix}")
    print(f"  Audit saved -> {OUTPUT}")
    print("=" * 45)
    
    with open(OUTPUT, "w") as f:
        json.dump({"ok_count": ok_count, "missing_ids": missing, "cases": results}, f, indent=2)


if __name__ == "__main__":
    main()
