"""
Task 1.5 — Run nnU-Net Test Inference
=====================================
Runs inference using the pre-trained nnU-Net model (Fold 0) on the 150 test
cases (IDs 851-1000). Saves prediction masks as NIfTI files.

Run with:
    c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\.venv_cuda\Scripts\python.exe 04_run_nnunet_inference.py
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))
import dataset_paths

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "all_four_validations"))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "mandatory_artifacts_nnunet", "checkpoints")
OUTPUT_DIR = os.path.join(ROOT_DIR, "mandatory_artifacts_nnunet", "predictions", "test")
SPLITS_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), "splits_final.json"))


def main() -> None:
    # Set nnU-Net environment variables required for loading models
    os.environ['nnUNet_raw'] = os.path.join(ROOT_DIR, "nnUNet_raw")
    os.environ['nnUNet_preprocessed'] = os.path.join(ROOT_DIR, "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = os.path.join(ROOT_DIR, "nnUNet_results")
    
    # Create required environment directories
    for key in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
        os.makedirs(os.environ[key], exist_ok=True)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(SPLITS_FILE):
        sys.exit(f"[ERROR] Splits file not found: {SPLITS_FILE}")
        
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    test_ids = splits.get("test", [])
    
    if not test_ids:
        sys.exit("[ERROR] Test set in splits_final.json is empty!")
        
    print(f"Loaded test cases: {len(test_ids)}")
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        sys.exit("[ERROR] nnunetv2 is not installed in this environment!")
        
    print("Loading nnU-Net predictor from Fold 0 checkpoints...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,                   # Standard 50% overlap (much faster)
        use_gaussian=False,                   # Disable Gaussian for speed/memory
        use_mirroring=False,                  # Faster inference
        perform_everything_on_device=False,   # Preprocess on CPU to avoid OOM
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # Initialize predictor from trained model folder
    predictor.initialize_from_trained_model_folder(
        Path(CHECKPOINT_DIR),
        [0],
        'checkpoint_best.pth'
    )
    print("nnU-Net predictor successfully loaded.")
    
    # Filter test cases to run
    input_files = []
    output_files = []
    case_ids_to_run = []
    
    for case_id in sorted(test_ids):
        img_path = dataset_paths.find_image(case_id)
        out_path = os.path.join(OUTPUT_DIR, f"{case_id}.nii.gz")  # standard name matching ground truth or nnUNet output
        
        # Check if already processed
        if os.path.exists(out_path):
            continue
            
        if not img_path:
            print(f"[WARNING] Image for case {case_id} not found, skipping.")
            continue
            
        input_files.append([img_path])
        output_files.append(out_path)
        case_ids_to_run.append(case_id)
        
    if not input_files:
        print("All test cases have already been predicted.")
        return
        
    print(f"Starting nnU-Net prediction for {len(input_files)} cases...")
    
    # Run predictions in a single batch
    try:
        predictor.predict_from_files(
            input_files,
            output_files,
            save_probabilities=False,
            overwrite=True,
            num_processes_segmentation_export=1,
            num_processes_preprocessing=1
        )
        print("\nPrediction complete!")
    except Exception as exc:
        print(f"\n[ERROR] Prediction failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
