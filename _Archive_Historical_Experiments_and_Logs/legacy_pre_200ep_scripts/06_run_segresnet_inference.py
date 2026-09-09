"""
Task 1.5 — Run SegResNet Test Inference (with Inverted Transforms)
==================================================================
Runs inference using the pre-trained SegResNet model on the 150 test cases
(IDs 851-1000). Preprocesses the inputs to 0.8mm spacing, runs sliding window
inference on the GPU, and applies MONAI's Invertd to properly align predictions
back to the original image space and orientation.

Run with:
    c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe -u 06_run_segresnet_inference.py
"""

import os
import sys
import json
import torch
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
import monai.transforms as mt

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))
import dataset_paths

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "all_four_validations"))
CKPT_PATH = os.path.join(ROOT_DIR, "mandatory_artifacts_segresnet", "best_resumed.pt")
OUTPUT_DIR = os.path.join(ROOT_DIR, "mandatory_artifacts_segresnet", "predictions", "test")
SPLITS_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), "splits_final.json"))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(SPLITS_FILE):
        sys.exit(f"[ERROR] Splits file not found: {SPLITS_FILE}")
        
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    test_ids = splits.get("test", [])
    
    if not test_ids:
        sys.exit("[ERROR] Test set in splits_final.json is empty!")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ── Load model ──
    print("Instantiating SegResNet...")
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        dropout_prob=0.1,
    ).to(device)
    
    print(f"Loading weights from {CKPT_PATH}...")
    state = torch.load(CKPT_PATH, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully.")
    
    # Define MONAI preprocessing transforms matching the notebook
    pre_trans = mt.Compose([
        mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.Orientationd(keys=["image"], axcodes="RAS"),
        mt.Spacingd(keys=["image"], pixdim=(0.8, 0.8, 0.8), mode="bilinear"),
        mt.ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=800, b_min=0.0, b_max=1.0, clip=True),
        mt.EnsureTyped(keys=["image"])
    ])
    
    # Define MONAI post-processing transforms to invert preprocessing
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
    
    print(f"Running inference on {len(test_ids)} test cases...")
    for idx, case_id in enumerate(sorted(test_ids)):
        out_path = os.path.join(OUTPUT_DIR, f"{case_id}.nii.gz")
        # NOTE: We overwrite to replace any old, misaligned predictions
        
        img_path = dataset_paths.find_image(case_id)
        if not img_path:
            print(f"[SKIP] Case {case_id}: Image not found.")
            continue
            
        print(f"[{idx+1}/{len(test_ids)}] Predicting Case {case_id}...")
        
        try:
            # 1. Apply MONAI preprocessing
            batch = pre_trans({"image": img_path})
            input_tensor = batch["image"].unsqueeze(0).to(device) # Shape: [1, 1, H, W, D]
            
            # 2. Sliding window inference on GPU
            with torch.no_grad():
                logits = sliding_window_inference(
                    input_tensor, 
                    roi_size=(96, 96, 96), 
                    sw_batch_size=4, 
                    predictor=model, 
                    overlap=0.5
                )
            
            # 3. Put prediction logits into batch dict (without batch dimension)
            batch["pred"] = logits.squeeze(0)
            
            # 4. Invert transforms to restore original spacing and orientation
            batch = post_trans(batch)
            
            # 5. Extract prediction probability tensor and argmax to get binary mask
            pred_probs = batch["pred"] # Shape: [2, H_orig, W_orig, D_orig]
            pred_mask = torch.argmax(pred_probs, dim=0).cpu().numpy().astype(np.uint8)
            
            # 6. Convert prediction to SimpleITK image and copy original metadata
            original_img = sitk.ReadImage(img_path)
            # Transpose from PyTorch (X, Y, Z) to SimpleITK (Z, Y, X)
            pred_mask_sitk = np.transpose(pred_mask, (2, 1, 0))
            
            pred_sitk = sitk.GetImageFromArray(pred_mask_sitk)
            pred_sitk.CopyInformation(original_img)
            
            # 7. Save predicted mask
            sitk.WriteImage(pred_sitk, out_path)
            
        except Exception as exc:
            print(f"[ERROR] Case {case_id} failed: {exc}")


if __name__ == "__main__":
    main()
