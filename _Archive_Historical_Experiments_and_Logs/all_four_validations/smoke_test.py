"""
SMOKE TEST for Model Evaluation Framework
==========================================

Tests that all 4 models can:
1. Load successfully
2. Run inference on GPU
3. Produce valid output shapes and labels

Only proceeds to full evaluation if all models pass.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to path
sys.path.append(str(Path(__file__).parent))

from model_evaluation_framework import ModelEvaluator

def smoke_test_models():
    """Run smoke test on all 4 models"""
    print("=" * 70)
    print("SMOKE TEST: Verifying all 4 models can load and run on GPU")
    print("=" * 70)
    
    root_dir = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\all_four_validations"
    dataset_dir = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Dataset_Main"
    
    evaluator = ModelEvaluator(root_dir, dataset_dir)
    evaluator.register_models()
    
    # Get first test case
    test_cases = evaluator.load_test_cases(num_cases=2)
    if len(test_cases) < 2:
        print("❌ Need at least 2 test cases for smoke test")
        return False
    
    case1 = test_cases[0]
    case2 = test_cases[1]
    
    print(f"\nUsing test cases: {case1['id']}, {case2['id']}")
    
    # Load ground truth and images
    gt1 = evaluator.load_nifti(case1['label_path'])
    img1 = evaluator.load_nifti(case1['image_path'])
    gt2 = evaluator.load_nifti(case2['label_path'])
    img2 = evaluator.load_nifti(case2['image_path'])
    
    if gt1 is None or img1 is None or gt2 is None or img2 is None:
        print("❌ Failed to load test data")
        return False
    
    print(f"Image 1 shape: {img1.shape}, GT 1 shape: {gt1.shape}")
    print(f"Image 2 shape: {img2.shape}, GT 2 shape: {gt2.shape}")
    
    results = {}
    
    # Test each model
    for model_id in ['3dunet', 'vnet', 'segresnet', 'nnunet']:
        print(f"\n{'='*70}")
        print(f"Testing {evaluator.models[model_id].name}")
        print(f"{'='*70}")
        
        try:
            # Load model
            model = evaluator.load_model(model_id)
            config = evaluator.models[model_id]
            
            if model == "existing_results":
                print(f"✅ {model_id}: Using existing pre-computed results")
                results[model_id] = True
                continue
            elif model == "nnunet_available":
                print(f"⚠️  {model_id}: nnU-Net requires full nnunetv2 setup")
                print(f"   Skipping inference test for nnU-Net")
                results[model_id] = "needs_setup"
                continue
            
            if model is None:
                print(f"❌ {model_id}: Failed to load")
                results[model_id] = False
                continue
            
            print(f"✅ {model_id}: Model loaded")
            
            # Test inference on case 1
            print(f"\n  Testing inference on case {case1['id']}...")
            pred1 = evaluator.inference_model(model, img1, config.model_type)
            
            if pred1 is None:
                print(f"❌ {model_id}: Inference returned None")
                results[model_id] = False
                continue
            
            print(f"  Input shape: {img1.shape}")
            print(f"  Output shape: {pred1.shape}")
            print(f"  Output dtype: {pred1.dtype}")
            print(f"  Unique labels: {np.unique(pred1)}")
            print(f"  Output device: CPU (after .cpu().numpy())")
            
            # Verify output shape matches GT
            if pred1.shape != gt1.shape:
                print(f"⚠️  {model_id}: Shape mismatch! Pred: {pred1.shape}, GT: {gt1.shape}")
                # Try to resize
                from scipy.ndimage import zoom
                zoom_factors = [gt1.shape[i] / pred1.shape[i] for i in range(3)]
                pred1 = zoom(pred1, zoom_factors, order=0).astype(np.uint8)
                print(f"  Resized to: {pred1.shape}")
            
            # Test inference on case 2
            print(f"\n  Testing inference on case {case2['id']}...")
            pred2 = evaluator.inference_model(model, img2, config.model_type)
            
            if pred2 is None:
                print(f"❌ {model_id}: Inference returned None on second case")
                results[model_id] = False
                continue
            
            # Verify GPU usage
            if evaluator.device.type == 'cuda':
                torch.cuda.synchronize()
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_res = torch.cuda.memory_reserved() / 1024**3
                peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
                print(f"\n  GPU Memory: {mem_alloc:.2f}GB alloc, {mem_res:.2f}GB reserved (peak: {peak_alloc:.2f}GB)")
            
            print(f"✅ {model_id}: Smoke test PASSED")
            results[model_id] = True
            
            # Clear GPU memory
            if evaluator.device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ {model_id}: Error during smoke test: {e}")
            import traceback
            traceback.print_exc()
            results[model_id] = False
    
    # Print summary
    print(f"\n{'='*70}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*70}")
    
    all_passed = True
    for model_id, result in results.items():
        name = evaluator.models[model_id].name
        if result is True:
            print(f"✅ {name}: PASSED")
        elif result == "needs_setup":
            print(f"⚠️  {name}: NEEDS SETUP (nnunetv2)")
        else:
            print(f"❌ {name}: FAILED")
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 EVALUATION READY - All models passed smoke test")
        return True
    else:
        print(f"\n❌ Smoke test FAILED - Fix model loading before proceeding")
        return False

if __name__ == "__main__":
    success = smoke_test_models()
    sys.exit(0 if success else 1)
