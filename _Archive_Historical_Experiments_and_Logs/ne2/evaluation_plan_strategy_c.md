
# EVALUATION PLAN: Strategy C - Use 3D U-Net Cases (851-1000)

## Current Status:
DONE 3D U-Net: 150 predictions (851-1000) - READY FOR METRICS
TODO V-Net: No predictions for 851-1000 - NEED INFERENCE
TODO SegResNet: No predictions for 851-1000 - NEED INFERENCE

## Required Actions:

### Step 1: Compute 3D U-Net Metrics (Immediate)
- Input: 150 predictions + 150 ground truth labels
- Output: metrics for 150 cases
- Command: python compute_3dunet_metrics_dunet_cases.py

### Step 2: Run V-Net Inference on 851-1000
- Need: V-Net model + training script
- Input: 150 images (851-1000)
- Output: 150 predictions
- Status: NEED MODEL ACCESS

### Step 3: Run SegResNet Inference on 851-1000  
- Need: SegResNet model + training script
- Input: 150 images (851-1000)
- Output: 150 predictions
- Status: NEED MODEL ACCESS

### Step 4: Compute V-Net and SegResNet Metrics
- Input: Their predictions + ground truth
- Output: metrics for 150 cases each

### Step 5: Create Unified Comparison
- All 3 models with 150 cases each
- Statistical analysis and plotting

## Expected Outcome:
- 150 cases for cross-model comparison (vs current 5)
- Robust statistical analysis
- Comprehensive thesis results

## Alternative: Partial Strategy
If V-Net/SegResNet models unavailable:
- Use 3D U-Net metrics alone (150 cases)
- Compare with existing V-Net/SegResNet metrics (different cases)
- Note limitation in thesis
