
# Cross-Model Validation Report
## Unified Analysis of 3D U-Net, V-Net, and SegResNet

### Executive Summary
This analysis compares the performance of three medical image segmentation models:
3D U-Net, V-Net, SegResNet.

### Model Performance Overview

#### 3D U-Net
- **Dataset Size:** 150 cases
- **Dice Score:** 0.0298 ± 0.0135
- **IoU:** 0.0152
- **Precision:** 0.0267
- **Recall:** 0.0348
- **HD95:** nan mm

#### V-Net
- **Dataset Size:** 45 cases
- **Dice Score:** 0.0092 ± 0.0059
- **IoU:** 0.0046
- **Precision:** 0.0240
- **Recall:** 0.0060
- **HD95:** 291.89 mm

#### SegResNet
- **Dataset Size:** 5 cases
- **Dice Score:** 0.7794 ± 0.0508
- **IoU:** 0.6408
- **Precision:** 0.8183
- **Recall:** 0.7454
- **HD95:** 11.20 mm

### Performance Rankings
Based on Dice coefficient:
1. **SegResNet**: 0.7794
2. **3D U-Net**: 0.0298
3. **V-Net**: 0.0092

### Key Findings
1. **Best Performing Model:** SegResNet with Dice score of 0.7794
2. **Dataset Coverage:** 
   - 3D U-Net: 150 cases
   - V-Net: 45 cases  
   - SegResNet: 5 cases

### Statistical Analysis Ready
The unified dataset is now ready for:
- Statistical significance testing
- Detailed failure case analysis
- Qualitative visualization
- Thesis chapter preparation

### Files Generated
- `unified_cross_model_results.csv` - Combined metrics
- `cross_model_comparison.png` - Visual comparison
- This report - Summary findings

---
*Analysis completed using GPU acceleration on NVIDIA RTX 3060 Ti*
