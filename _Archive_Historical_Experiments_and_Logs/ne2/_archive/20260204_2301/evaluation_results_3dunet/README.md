# 3D U-Net IMGcas Dataset Evaluation Results

This directory contains the complete evaluation artifacts for the 3D U-Net model trained on the IMGcas coronary artery segmentation dataset.

## 📁 Directory Structure

```
evaluation_results/
├── README.md                      # This file
├── evaluation_summary.md          # Comprehensive evaluation report
├── evaluation_results.csv         # Validation set metrics
├── case_mapping.csv               # ID mapping (Original → SegResNet format)
├── test_predictions_summary.csv   # Complete test predictions inventory
├── visualizations/                # Validation case visualizations
│   └── Case_701_visualization.png
└── test_visualizations/           # Test case examples
    ├── Test_Case_905_visualization.png
    ├── Test_Case_982_visualization.png
    └── Test_Case_993_visualization.png
```

## 📊 Key Files Description

### 🎯 Core Evaluation Results
- **`evaluation_results.csv`** - Per-case metrics (Dice, IoU, HD95, Precision, Recall)
- **`test_predictions_summary.csv`** - All 150 test cases with prediction file paths
- **`case_mapping.csv`** - Maps original IDs (851-1000) to SegResNet format (Case_0-Case_149)

### 📈 Reports & Documentation
- **`evaluation_summary.md`** - Detailed analysis and interpretation
- **`README.md`** - This overview file

### 🖼️ Visualizations
- **`visualizations/`** - Validation case with ground truth comparison
- **`test_visualizations/`** - Representative test cases (input + prediction)

## 🚀 Quick Start

### View Evaluation Metrics
```python
import pandas as pd
metrics = pd.read_csv('evaluation_results.csv')
print(metrics.describe())
```

### Load Case Mapping
```python
mapping = pd.read_csv('case_mapping.csv')
print(mapping.head())
```

### Access Test Predictions
```python
test_preds = pd.read_csv('test_predictions_summary.csv')
print(f"Total test cases: {len(test_preds)}")
```

## 📋 Model Information

- **Architecture:** 3D U-Net
- **Training Epochs:** 35 (best at epoch 15)
- **Input Size:** [96, 96, 96]
- **Dataset:** IMGcas coronary artery segmentation
- **Test Cases:** 150 (complete coverage)

## ⚠️ Important Notes

1. **Test Ground Truth:** Not available (typical for challenge datasets)
2. **Validation Coverage:** Limited to available cases in current workspace
3. **Performance Metrics:** Based on single validation case (case 701)

## 🔧 Generated Scripts

The following scripts were used to generate these artifacts:

1. **`evaluation_3dunet.py`** - Main evaluation pipeline
2. **`generate_test_visualizations.py`** - Test case visualization generator

## 📞 Usage

These evaluation artifacts are ready for:
- ✅ Model comparison with other architectures
- ✅ Integration into 4-model comparison framework
- ✅ Case ID mapping for cross-model consistency
- ✅ Visual quality assessment

## 📈 Status

- ✅ All required evaluation artifacts generated
- ✅ Case mapping complete for 4-model comparison
- ✅ Visualizations created for representative cases
- ⏳ Awaiting test ground truth for complete metrics

---

**Generated:** 2026-02-05  
**Model:** 3D U-Net  
**Status:** Ready for 4-model comparison
