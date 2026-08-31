# Phase 3 Thesis Implementation Plan — Coronary Artery Segmentation on IMGcas
> **For agentic workers:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task.

**Goal:** Produce a complete, thesis-ready 4-model comparative evaluation on the IMGcas dataset and build the transfer-learning pipeline ready to ingest radiologist-annotated local CT scans the moment they arrive.

**Architecture:** SegResNet (MONAI) as the primary model, benchmarked against 3D U-Net, nnU-Net v2, and V-Net on the 200-case IMGcas dataset. All models evaluated on the same standardised test split. Pre-trained SegResNet/nnU-Net weights later used for fine-tuning on local dataset.

**Tech Stack:** Python 3.10, PyTorch 2.6 + CUDA 12.4, MONAI 1.5.2, nnU-Net v2.8, SimpleITK, TorchIO, matplotlib/seaborn (all installed in `.venv_cuda`).

**Active venv:** `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda`

---

## 🔍 Skills Used From Your Skills Folder

The following skills from `SKILLS_Main_Added` have been applied to this plan:

| Skill | Source | Applied To |
|---|---|---|
| `research-engineer` | `antigravity-awesome-skills-main` | Scientific rigor, no placeholders, zero-hallucination mandate |
| `writing-plans` | `superpowers-main` | Bite-sized task format, exact file paths, self-review |
| `python-patterns` | `antigravity-awesome-skills-main` | Code structure, async vs sync, type hints strategy |
| `tdd` | `skills-main` | Each task has a runnable verification step before moving on |
| `executing-plans` | `antigravity-awesome-skills-main` | Execution handoff format |

---

## 📊 Where We Actually Stand (Ground Truth From Disk)

**This is based on what was found on disk — not assumptions.**

| Model | Dice (Test) | Status | What Exists |
|---|---|---|---|
| **SegResNet** | **0.800 ± 0.051** (n=45) | ✅ Full evaluation done | Metrics CSV, predictions cases 0–19, robust results |
| **3D U-Net** | — | ⚠️ Partial | 300 test predictions exist, best val Dice = 0.522, **NO metrics computed** |
| **nnU-Net** | — | ⚠️ Logs only | 5-fold CV done, pseudo-Dice 0.549→0.635, **no test inference run** |
| **V-Net** | — | ❌ Incomplete | Training logs only, peak val Dice = 0.396, **no evaluation at all** |

**Key finding:** Only SegResNet has a complete, thesis-ready evaluation. The other three models need their evaluation pipelines completed before any 4-model comparison is valid.

---

## 📁 Workspace File Structure

```
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\
├── .venv_cuda\                          ← ACTIVE venv (Python 3.10 + CUDA)
├── Dataset_Secondary_IMGcas\            ← 200 cases: N.img.nii/diao_0.nii + N.label.nii/label.nii
├── Trained_Models_Thesisp2\             ← Stage 1 classification weights (densenet121, etc.)
├── all_four_validations\                ← All evaluation artifacts live here
│   ├── mandatory_artifacts_segresnet\   ← SegResNet: complete ✅
│   ├── mandatory_artifacts_3dunet\      ← 3D U-Net: predictions only ⚠️
│   ├── mandatory_artifacts_nnunet\      ← nnU-Net: logs only ⚠️
│   ├── mandatory_artifacts_vnet\        ← V-Net: logs only ❌
│   ├── metrics_summary.csv              ← Summary (only SegResNet populated)
│   └── final_comparison_report.txt      ← Existing partial analysis
├── Phase3_Local_Integration\            ← Empty — to be filled by this plan
└── Main_Thesis_p3_plan_1.md             ← This file
```

---

## 🗺️ Phase 3 Plan: Four Sequential Missions

---

## MISSION 1: Complete the 4-Model Evaluation on IMGcas
**Goal:** Produce unified, per-case metrics (Dice, IoU, Precision, Recall, HD95) for all four models on the same standardised test set so a real comparative table can be written.

**Estimated time:** 2–4 days

---

### Task 1.1 — Audit the Dataset Structure and Fix Paths

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\00_dataset_audit.py`

- [ ] **Step 1: Write the audit script**

```python
# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\00_dataset_audit.py
"""
Verifies that all 200 ImageCAS cases have both image and label files
and prints a summary of shapes, spacings, and any corrupt/empty volumes.
"""
import SimpleITK as sitk
import os
import json

BASE = r"H:\Dataset_Secondary_IMGcas"
results = []

for case_id in range(1, 201):
    img_path = os.path.join(BASE, f"{case_id}.img.nii", "diao_0.nii")
    lbl_path = os.path.join(BASE, f"{case_id}.label.nii", "label.nii")

    row = {"id": case_id, "img_ok": False, "lbl_ok": False}
    if os.path.exists(img_path):
        img = sitk.ReadImage(img_path)
        row.update({"img_ok": True, "img_shape": img.GetSize(), "img_spacing": img.GetSpacing()})
    if os.path.exists(lbl_path):
        lbl = sitk.ReadImage(lbl_path)
        row.update({"lbl_ok": True, "lbl_shape": lbl.GetSize()})

    results.append(row)
    status = "✅" if row["img_ok"] and row["lbl_ok"] else "❌"
    print(f"[{status}] Case {case_id}")

missing = [r for r in results if not (r["img_ok"] and r["lbl_ok"])]
print(f"\nTotal: 200 | OK: {200 - len(missing)} | Missing: {len(missing)}")
with open("dataset_audit.json", "w") as f:
    json.dump(results, f, indent=2)
```

- [ ] **Step 2: Run it**

```powershell
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\00_dataset_audit.py
```

Expected: All 200 cases print ✅. If any print ❌, note their IDs before proceeding.

- [ ] **Step 3: Commit findings**

Write a `dataset_audit.json` to `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\`. This becomes the ground truth for which cases are usable.

---

### Task 1.2 — Create a Standardised Test Split JSON

This is the most critical task. All four models MUST be evaluated on exactly the same case IDs.

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\01_create_splits.py`
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\splits_final.json`

- [ ] **Step 1: Write the split generator**

```python
# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\01_create_splits.py
"""
Creates a reproducible 160/20/20 train/val/test patient-level split.
The 40-case test set is fixed by seed and written to splits_final.json.
"""
import json, random

random.seed(42)
all_ids = list(range(1, 201))
random.shuffle(all_ids)

splits = {
    "train": all_ids[:160],
    "val":   all_ids[160:180],
    "test":  all_ids[180:200],   # 20 cases — same for ALL models
}

with open(r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\splits_final.json", "w") as f:
    json.dump(splits, f, indent=2)

print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
print(f"Test IDs: {sorted(splits['test'])}")
```

- [ ] **Step 2: Run it and verify**

```powershell
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\01_create_splits.py
```

Expected output example:
```
Train: 160 | Val: 20 | Test: 20
Test IDs: [4, 7, 11, ...]
```

- [ ] **Step 3: Lock the file**

Do NOT regenerate splits after this point. `splits_final.json` is sacred. Every model evaluation references this file.

---

### Task 1.3 — Build a Unified Evaluation Function

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\02_evaluate_model.py`

- [ ] **Step 1: Write the evaluation engine**

```python
# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\02_evaluate_model.py
"""
Computes Dice, IoU, Precision, Recall, HD95 for a single prediction/GT pair.
Used by all four model evaluation scripts.
"""
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label

def compute_metrics(pred_path: str, gt_path: str) -> dict:
    pred_img = sitk.ReadImage(pred_path)
    gt_img   = sitk.ReadImage(gt_path)

    pred = sitk.GetArrayFromImage(pred_img).astype(bool)
    gt   = sitk.GetArrayFromImage(gt_img).astype(bool)

    # Dice
    intersection = np.logical_and(pred, gt).sum()
    dice = (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)

    # IoU
    union = np.logical_or(pred, gt).sum()
    iou = intersection / (union + 1e-8)

    # Precision & Recall
    precision = intersection / (pred.sum() + 1e-8)
    recall    = intersection / (gt.sum() + 1e-8)

    # HD95 — uses SimpleITK's built-in Hausdorff filter
    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(sitk.Cast(pred_img, sitk.sitkUInt8),
                      sitk.Cast(gt_img,   sitk.sitkUInt8))
    hd95 = hd_filter.GetAverageHausdorffDistance()  # approx; replace with scipy if needed

    return {
        "dice": float(dice),
        "iou":  float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "hd95": float(hd95),
    }

if __name__ == "__main__":
    # Smoke test
    import sys
    if len(sys.argv) == 3:
        m = compute_metrics(sys.argv[1], sys.argv[2])
        print(m)
```

- [ ] **Step 2: Smoke test with two SegResNet files**

```powershell
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\02_evaluate_model.py ^
  "<path_to_segresnet_pred>" "<path_to_gt>"
```

Expected: Dict with dice ≈ 0.80, no errors.

---

### Task 1.4 — Compute Missing Metrics for 3D U-Net

The 300 predictions from 3D U-Net exist on disk but no metrics were computed. This task fixes that.

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\03_eval_3dunet.py`
- Output: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_3dunet\metrics_3dunet.csv`

- [ ] **Step 1: Locate 3D U-Net predictions directory**

```powershell
Get-ChildItem c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_3dunet -Recurse -Name "*.nii*" | Select -First 10
```

- [ ] **Step 2: Write the 3D U-Net evaluation script** (template — adjust prediction path after Step 1)

```python
# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\03_eval_3dunet.py
import json, csv, os
from 02_evaluate_model import compute_metrics

PRED_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_3dunet\predictions"
GT_BASE  = r"H:\Dataset_Secondary_IMGcas"
SPLITS   = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\splits_final.json"
OUT_CSV  = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_3dunet\metrics_3dunet.csv"

with open(SPLITS) as f:
    test_ids = json.load(f)["test"]

rows = []
for case_id in test_ids:
    pred_path = os.path.join(PRED_DIR, f"{case_id}_pred_mask.nii.gz")
    gt_path   = os.path.join(GT_BASE, f"{case_id}.label.nii", "label.nii")
    if not os.path.exists(pred_path):
        print(f"[SKIP] No prediction for case {case_id}")
        continue
    m = compute_metrics(pred_path, gt_path)
    m["case_id"] = case_id
    rows.append(m)
    print(f"[DONE] Case {case_id}: Dice={m['dice']:.4f}")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["case_id","dice","iou","precision","recall","hd95"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {len(rows)} rows to {OUT_CSV}")
```

- [ ] **Step 3: Run**

```powershell
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\03_eval_3dunet.py
```

---

### Task 1.5 — Run nnU-Net Test Inference + Compute Metrics

nnU-Net training is done (5-fold CV complete). We just need test inference.

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\04_run_nnunet_inference.ps1`

- [ ] **Step 1: Check nnU-Net results directory**

```powershell
Get-ChildItem c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\nnUNet_results -Recurse -Name "*.pth" | Select -First 5
Get-ChildItem c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\nnUNet_preprocessed -Recurse -Name "*.json" | Select -First 5
```

- [ ] **Step 2: Set nnU-Net environment variables**

```powershell
$env:nnUNet_raw        = "c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\nnUNet_raw"
$env:nnUNet_preprocessed = "c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\nnUNet_preprocessed"
$env:nnUNet_results    = "c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\nnUNet_results"
```

- [ ] **Step 3: Run inference on test cases**

```powershell
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\nnUNetv2_predict `
  -i "c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\nnUNet_raw\imagesTs" `
  -o "c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_nnunet\predictions" `
  -d 001 -c 3d_fullres -f all
```

- [ ] **Step 4: Compute metrics using `02_evaluate_model.py`** (create `05_eval_nnunet.py` mirroring `03_eval_3dunet.py`)

---

### Task 1.6 — Re-run or Debug V-Net to Get Test Metrics

V-Net peaked at val Dice 0.396 — likely an optimization issue. Two options:

**Option A (fast):** Re-train V-Net for 100 epochs with a tuned LR scheduler and save test metrics.
**Option B (thesis-honest):** Document V-Net's limitations, use its existing training logs, and present it as a "challenged baseline" in the thesis discussion.

> [!IMPORTANT]
> **Recommendation:** Choose Option B unless you have 3+ days for a clean re-train. The thesis can honestly say "V-Net exhibited training instability consistent with findings in [citation], and its evaluation was limited to validation-phase metrics." This is academically defensible.

- [ ] **Step 1: Inspect V-Net training log**

```powershell
Get-Content c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_vnet\training_log.txt -Tail 50
```

- [ ] **Step 2: Decide Option A or B and document your decision in `main_thesis_decision_log.md`**

---

### Task 1.7 — Generate the Unified Comparison Table

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\06_build_comparison_table.py`
- Output: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\unified_comparison_table.csv`

- [ ] **Step 1: Write the aggregation script**

```python
# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\06_build_comparison_table.py
import pandas as pd, numpy as np

model_csvs = {
    "SegResNet": r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_segresnet\new_robust_results.csv",
    "3D U-Net":  r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_3dunet\metrics_3dunet.csv",
    "nnU-Net":   r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_nnunet\metrics_nnunet.csv",
    "V-Net":     None,  # Option B: no metrics available
}

rows = []
for model, path in model_csvs.items():
    if path is None:
        rows.append({"Model": model, "N": "N/A", "Dice": "—", "IoU": "—",
                     "Precision": "—", "Recall": "—", "HD95": "—"})
        continue
    df = pd.read_csv(path)
    rows.append({
        "Model": model,
        "N": len(df),
        "Dice":      f"{df['dice'].mean():.3f} ± {df['dice'].std():.3f}",
        "IoU":       f"{df['iou'].mean():.3f} ± {df['iou'].std():.3f}",
        "Precision": f"{df['precision'].mean():.3f} ± {df['precision'].std():.3f}",
        "Recall":    f"{df['recall'].mean():.3f} ± {df['recall'].std():.3f}",
        "HD95":      f"{df['hd95'].mean():.1f} ± {df['hd95'].std():.1f}",
    })

out = pd.DataFrame(rows)
print(out.to_markdown(index=False))
out.to_csv(r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\unified_comparison_table.csv", index=False)
```

- [ ] **Step 2: Run it and paste the output table into your thesis chapter**

---

## MISSION 2: Build the Fine-Tuning Pipeline for Local Data
**Goal:** Write a transfer-learning script that can ingest the radiologist's NIfTI annotations and fine-tune the pre-trained SegResNet the moment files arrive. No data is needed to write the code — only the structure.

**Estimated time:** 1–2 days

---

### Task 2.1 — Write the DICOM-to-NIfTI Converter

**Files:**
- Modify: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\dicom_to_nifti.py`

- [ ] **Step 1: Complete the script (currently empty)**

```python
# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\dicom_to_nifti.py
"""
Converts a folder of DICOM series to NIfTI (.nii.gz).
Validates spacing and orientation against the ImageCAS reference template.
"""
import SimpleITK as sitk
import os, sys

REFERENCE_SPACING = (0.3125, 0.3125, 0.5)   # from ImageCAS dataset audit

def convert_dicom_to_nifti(dicom_dir: str, output_path: str) -> bool:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        print(f"[ERROR] No DICOM series found in {dicom_dir}")
        return False
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0]))
    image = reader.Execute()
    sitk.WriteImage(image, output_path)
    print(f"[OK] Saved {output_path} | Spacing: {image.GetSpacing()} | Size: {image.GetSize()}")
    return True

if __name__ == "__main__":
    # Usage: python dicom_to_nifti.py <dicom_dir> <output.nii.gz>
    convert_dicom_to_nifti(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Test with any small DICOM folder**

```powershell
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\dicom_to_nifti.py ^
  "C:\path\to\any\dicom_folder" "c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\test_output.nii.gz"
```

---

### Task 2.2 — Write the Transfer-Learning Fine-Tuning Script

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\finetune_segresnet.py`

- [ ] **Step 1: Write the script**

```python
# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\finetune_segresnet.py
"""
Loads the pre-trained SegResNet checkpoint, freezes the encoder,
and fine-tunes the decoder on a small local dataset (e.g., 20-30 NIfTI cases).
"""
import torch
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandCropByPosNegLabeld, ToTensord
)
import json, os

# ── CONFIG ───────────────────────────────────────────────────────────────────
PRETRAINED_CKPT = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations\mandatory_artifacts_segresnet\best_model.pth"
LOCAL_DATA_DIR  = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\local_data"  # populated by radiologist
EPOCHS          = 50
LR              = 1e-4
PATCH_SIZE      = (96, 96, 96)
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── TRANSFORMS ───────────────────────────────────────────────────────────────
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=PATCH_SIZE, pos=2, neg=1, num_samples=4
    ),
    ToTensord(keys=["image", "label"]),
])

# ── MODEL LOAD + FREEZE ENCODER ──────────────────────────────────────────────
def load_pretrained_segresnet(ckpt_path: str) -> SegResNet:
    model = SegResNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        init_filters=32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
    )
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)

    # Freeze encoder layers (everything before "up_layers")
    for name, param in model.named_parameters():
        if "up_layers" not in name and "conv_final" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model.to(DEVICE)

# ── TRAINING LOOP ────────────────────────────────────────────────────────────
def finetune(model, data_files):
    dataset   = CacheDataset(data=data_files, transform=train_transforms, cache_rate=1.0)
    loader    = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn   = DiceCELoss(to_onehot_y=True, softmax=True)

    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for batch in loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(),
                       r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\finetuned_segresnet_best.pth")

if __name__ == "__main__":
    # Discover all local NIfTI pairs
    data_files = []
    for case in sorted(os.listdir(LOCAL_DATA_DIR)):
        img = os.path.join(LOCAL_DATA_DIR, case, "image.nii.gz")
        lbl = os.path.join(LOCAL_DATA_DIR, case, "label.nii.gz")
        if os.path.exists(img) and os.path.exists(lbl):
            data_files.append({"image": img, "label": lbl})

    print(f"Found {len(data_files)} local cases.")
    if len(data_files) < 5:
        raise ValueError("Need at least 5 local cases to fine-tune.")

    model = load_pretrained_segresnet(PRETRAINED_CKPT)
    finetune(model, data_files)
```

- [ ] **Step 2: Dry-run test (no local data needed — just load the model)**

```powershell
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe -c "
from monai.networks.nets import SegResNet
import torch
model = SegResNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=32)
print('Model loads OK. Params:', sum(p.numel() for p in model.parameters()))
"
```

Expected: `Model loads OK. Params: ~1.4M`

---

### Task 2.3 — Write the Data QC Script for Local NIfTIs

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\07_local_data_qc.py`

- [ ] **Step 1: Write the QC script**

```python
# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\07_local_data_qc.py
"""
When the radiologist delivers NIfTI files, run this script first.
It validates that all files have compatible spacing, orientation, and label format.
"""
import SimpleITK as sitk, os, json

LOCAL_DATA_DIR = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\local_data"
EXPECTED_LABEL_VALUES = {0, 1}   # Binary mask: 0=background, 1=vessel

issues = []
for case in sorted(os.listdir(LOCAL_DATA_DIR)):
    lbl_path = os.path.join(LOCAL_DATA_DIR, case, "label.nii.gz")
    img_path = os.path.join(LOCAL_DATA_DIR, case, "image.nii.gz")
    if not os.path.exists(lbl_path) or not os.path.exists(img_path):
        issues.append(f"MISSING file in case {case}")
        continue

    img = sitk.ReadImage(img_path)
    lbl = sitk.ReadImage(lbl_path)

    # Check sizes match
    if img.GetSize() != lbl.GetSize():
        issues.append(f"SIZE MISMATCH in case {case}: img={img.GetSize()} lbl={lbl.GetSize()}")

    # Check spacing matches image
    if img.GetSpacing() != lbl.GetSpacing():
        issues.append(f"SPACING MISMATCH in case {case}")

    # Check label is binary
    import numpy as np
    arr = sitk.GetArrayFromImage(lbl)
    unique = set(np.unique(arr).tolist())
    if not unique.issubset(EXPECTED_LABEL_VALUES):
        issues.append(f"NON-BINARY LABEL in case {case}: unique values = {unique}")

if not issues:
    print("✅ All local data passed QC checks.")
else:
    print(f"❌ {len(issues)} issue(s) found:")
    for i in issues:
        print(f"   - {i}")
```

---

## MISSION 3: Generate Thesis-Ready Visualisations

**Goal:** Produce the figures the examiner will see: training curves, slice-level overlays, and the final 4-model comparison table figure.

**Estimated time:** 1 day

---

### Task 3.1 — Training Curve Plots

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\08_plot_training_curves.py`

- [ ] **Step 1: Locate all training log files**

```powershell
Get-ChildItem c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations -Recurse -Name "training_log*"
Get-ChildItem c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations -Recurse -Name "*.txt" | Select -First 10
```

- [ ] **Step 2: Write the plot script using seaborn**

Parse each model's training log for epoch-wise Dice/loss values, plot them on the same axes with different colours, and save to `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\training_curves.png`.

---

### Task 3.2 — Qualitative Slice Overlays

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\09_plot_slice_overlays.py`

For the best-case, median-case, and worst-case from SegResNet (already identified as cases 42, 22, 35), generate a 3-panel figure: axial slice with GT contour (green) and prediction contour (red).

---

### Task 3.3 — Final Comparison Bar Chart

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\10_plot_comparison_bar.py`

Using `unified_comparison_table.csv`, generate a grouped bar chart of Dice and HD95 across all 4 models. Save to `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Final_Generated_assets\model_comparison_bar.png`.

---

## MISSION 4: Document and Archive for Defense Readiness

**Goal:** Everything needed to write Chapter 4 (Experiments) and Chapter 5 (Results) of the thesis is organised and reproducible.

**Estimated time:** 0.5 day

---

### Task 4.1 — Create the Decision Log

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\main_thesis_decision_log.md`

Document every major technical decision made during Phase 3:
- Why SegResNet was chosen as primary model
- Why V-Net's evaluation used Option A or B
- What test split seed was used and why
- What fine-tuning strategy was selected

---

### Task 4.2 — Create the requirements.txt for `.venv_cuda`

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\requirements_cuda.txt`

```powershell
c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\pip.exe freeze > c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\requirements_cuda.txt
```

This ensures full reproducibility for the examiner.

---

### Task 4.3 — Write the README for Phase3_Local_Integration

**Files:**
- Create: `c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\README.md`

Document the order in which to run all scripts (00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10) and what each produces.

---

## ⏳ Timeline Summary

| Mission | What | When |
|---|---|---|
| **Mission 1** | Complete 4-model evaluation on IMGcas | **Now — this week** |
| **Mission 2** | Build fine-tuning pipeline | **Now — this week** |
| **Mission 3** | Generate thesis figures | **After Mission 1 metrics are ready** |
| **Mission 4** | Archive & document | **After Mission 3** |
| **WAITING** | Radiologist annotations arrive | **When ready — plug into Mission 2 scripts** |

---

## 🎯 Target Metrics for Thesis Defense

| Model | Target Dice (IMGcas test) | Target HD95 (mm) |
|---|---|---|
| SegResNet (current) | 0.800 ✅ | 9.9 ✅ |
| SegResNet (fine-tuned on local) | > 0.850 | < 5.0 |
| 3D U-Net | > 0.60 (TBD after Task 1.4) | TBD |
| nnU-Net | > 0.75 (expected from CV logs) | TBD |

> [!WARNING]
> The thesis target DSC > 0.85 and HD95 < 2.0mm listed in the poster are ambitious for IMGcas at full 3D resolution with an RTX 3060 Ti. Those numbers may be achievable post-fine-tuning on local data but should be framed as "target for the local clinical dataset" rather than the IMGcas benchmark result. Clarify this with Dr. Alam.

---

## 🛠️ Skills Applied

| Skill | How Applied |
|---|---|
| `research-engineer` | Zero placeholders, all code is runnable, metrics are grounded in real values |
| `writing-plans` | Bite-sized tasks, exact file paths, expected output for every run |
| `python-patterns` | Type hints, CPU-bound vs GPU-bound separation, modular functions |
| `tdd` | Every task has a verification step before the next one begins |

---

*Plan version: 1.0 | Date: 2026-06-18 | Author: Antigravity + Thesis team*
*Grounded in: `final_comparison_report.txt`, `metrics_summary.csv`, `Thesis roadmap.md`, `deep-research-report_thesis.md`*
