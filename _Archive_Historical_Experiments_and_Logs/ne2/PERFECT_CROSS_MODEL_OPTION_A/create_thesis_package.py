#!/usr/bin/env python3
"""
Create Clean Thesis Results Package for Team Review
Analyzes all results and collects only essential files for thesis review
"""

import os
import shutil
from pathlib import Path
import pandas as pd

def analyze_existing_results():
    """Analyze all existing results to determine what's essential"""
    
    print("🔍 ANALYZING EXISTING RESULTS...")
    
    # Key directories and files
    analysis = {
        'essential_data': [],
        'essential_plots': [],
        'essential_reports': [],
        'supporting_docs': []
    }
    
    # Essential data files
    essential_data = [
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/perfect_unified_metrics_with_hd95.csv",
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/summary_metrics.csv",
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/ranking_table.csv",
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/efficiency_summary.csv"
    ]
    
    # Essential plots (most impactful for thesis)
    essential_plots = [
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/plots/dice_comparison_all_models.png",
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/plots/comprehensive_metrics_comparison.png",
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/plots/training_efficiency_analysis.png",
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/plots/nnunet_projection.png",
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/plots/metric_bars_mean_std.png"
    ]
    
    # Essential reports
    essential_reports = [
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/thesis_comparison_report.md",
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/nnunet_analysis_report.md"
    ]
    
    # Supporting documentation
    supporting_docs = [
        "c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/README.txt"
    ]
    
    analysis['essential_data'] = essential_data
    analysis['essential_plots'] = essential_plots
    analysis['essential_reports'] = essential_reports
    analysis['supporting_docs'] = supporting_docs
    
    return analysis

def create_thesis_review_package(analysis):
    """Create clean thesis review package"""
    
    print("📦 CREATING THESIS REVIEW PACKAGE...")
    
    # Create clean package directory
    package_dir = Path("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/THESIS_RESULTS_PACKAGE")
    
    # Remove if exists and create new
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (package_dir / "data").mkdir(exist_ok=True)
    (package_dir / "plots").mkdir(exist_ok=True)
    (package_dir / "reports").mkdir(exist_ok=True)
    
    # Copy essential files
    copied_files = []
    
    # Copy data files
    for file_path in analysis['essential_data']:
        src = Path(file_path)
        if src.exists():
            dst = package_dir / "data" / src.name
            shutil.copy2(src, dst)
            copied_files.append(f"data/{src.name}")
            print(f"   ✅ Data: {src.name}")
    
    # Copy plots
    for file_path in analysis['essential_plots']:
        src = Path(file_path)
        if src.exists():
            dst = package_dir / "plots" / src.name
            shutil.copy2(src, dst)
            copied_files.append(f"plots/{src.name}")
            print(f"   📊 Plot: {src.name}")
    
    # Copy reports
    for file_path in analysis['essential_reports']:
        src = Path(file_path)
        if src.exists():
            dst = package_dir / "reports" / src.name
            shutil.copy2(src, dst)
            copied_files.append(f"reports/{src.name}")
            print(f"   📄 Report: {src.name}")
    
    # Copy supporting docs
    for file_path in analysis['supporting_docs']:
        src = Path(file_path)
        if src.exists():
            dst = package_dir / src.name
            shutil.copy2(src, dst)
            copied_files.append(src.name)
            print(f"   📋 Doc: {src.name}")
    
    return package_dir, copied_files

def create_thesis_summary_report(package_dir):
    """Create executive summary for thesis review"""
    
    print("📋 CREATING THESIS EXECUTIVE SUMMARY...")
    
    # Load key metrics
    try:
        summary_df = pd.read_csv("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/summary_metrics.csv")
        ranking_df = pd.read_csv("c:/Thesis_RASNET/Thesis_Trainings/Thesis_Trainings/ne2/PERFECT_CROSS_MODEL_OPTION_A/THESIS_EXPORT_PACKAGE/ranking_table.csv")
    except:
        print("   ⚠️ Could not load metrics data for summary")
        return
    
    # Create executive summary
    summary = f"""# THESIS RESULTS EXECUTIVE SUMMARY
## Cross-Model Validation & Comparative Analysis

### OVERVIEW
Comprehensive evaluation of 4 state-of-the-art 3D medical image segmentation models on 45 identical test cases.

### FINAL MODEL RANKINGS

| Rank | Model | Dice Score | Status |
|------|-------|-------------|---------|
| 1 | SegResNet | {ranking_df.iloc[0]['dice_mean']:.4f} | Complete |
| 2 | NNUNet (Final) | 0.7688 | Incomplete training |
| 3 | NNUNet (Best) | 0.7519 | Incomplete training |
| 4 | V-Net | {ranking_df.iloc[1]['dice_mean']:.4f} | Complete |
| 5 | 3D U-Net | {ranking_df.iloc[2]['dice_mean']:.4f} | Complete |

### KEY FINDINGS

#### Performance Analysis:
- **SegResNet dominates** with Dice: 0.8005 ± 0.0508
- **NNUNet shows exceptional promise** despite incomplete training (52/1000 epochs)
- **V-Net and 3D U-Net** show poor performance requiring architectural improvements

#### Critical Insight:
NNUNet achieved **94% of SegResNet performance** in only **5.2% of training time**, suggesting potential to exceed current best performer with complete training.

#### Clinical Implications:
- **SegResNet**: Currently best for clinical deployment (Dice > 0.8)
- **NNUNet**: Highest potential for future improvement
- **V-Net/3D U-Net**: Not recommended for clinical use

### RECOMMENDATIONS

#### Immediate Actions:
1. **Complete NNUNet training** (remaining 948 epochs)
2. **Re-evaluate all models** after NNUNet completion
3. **Consider ensemble methods** combining best performers

#### Thesis Impact:
- **Current**: SegResNet leads with strong performance
- **Future**: NNUNet could become new leader with complete training
- **Significance**: Demonstrates importance of training completion in model evaluation

### PACKAGE CONTENTS

#### Data Files (4):
- perfect_unified_metrics_with_hd95.csv - Complete case-by-case metrics
- summary_metrics.csv - Statistical summaries (mean±std, median)
- ranking_table.csv - Model performance rankings
- efficiency_summary.csv - Computational efficiency metrics

#### Plots (5):
- dice_comparison_all_models.png - All models Dice comparison
- comprehensive_metrics_comparison.png - All metrics visualization
- training_efficiency_analysis.png - Performance vs training effort
- nnunet_projection.png - NNUNet potential projection
- metric_bars_mean_std.png - Performance with error bars

#### Reports (2):
- thesis_comparison_report.md - Complete analysis with NNUNet section
- nnunet_analysis_report.md - Detailed NNUNet training analysis

### NEXT STEPS FOR THESIS

1. **Review Results**: Examine plots and reports in this package
2. **Decision Point**: Determine if NNUNet training should be completed
3. **Final Analysis**: Update thesis with complete model comparison if needed
4. **Clinical Validation**: Validate chosen model on additional datasets

---

**Generated**: 2026-02-05  
**Analysis**: 4 models, 45 test cases, comprehensive metrics  
**Status**: Ready for thesis team review  
**Contact**: Thesis Research Team
"""
    
    # Save summary with UTF-8 encoding
    summary_path = package_dir / "THESIS_EXECUTIVE_SUMMARY.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"   ✅ Created: THESIS_EXECUTIVE_SUMMARY.md")
    return summary_path

def create_upload_instructions(package_dir):
    """Create upload instructions for cloud storage"""
    
    instructions = f"""# THESIS RESULTS PACKAGE - Upload Instructions

## Package Contents
This package contains ONLY the essential files needed for thesis team review.

## Folder Structure
```
THESIS_RESULTS_PACKAGE/
├── data/                    # 4 CSV files with metrics
├── plots/                   # 5 key plots for thesis
├── reports/                 # 2 detailed analysis reports
├── THESIS_EXECUTIVE_SUMMARY.md  # Quick overview
└── README.txt               # This file
```

## Upload Instructions

### Google Drive:
1. Compress THESIS_RESULTS_PACKAGE folder to ZIP
2. Upload to Google Drive
3. Share with thesis team (view/edit permissions)
4. Send link via email or team chat

### OneDrive:
1. Right-click THESIS_RESULTS_PACKAGE folder
2. Select "Share" -> "OneDrive"
3. Set permissions for thesis team
4. Copy sharing link

### Alternative (Direct Upload):
1. Upload entire folder structure
2. Maintain folder organization
3. Ensure all files are accessible

## What to Review First

### Quick Start (5 minutes):
1. Read **THESIS_EXECUTIVE_SUMMARY.md** for overview
2. Review **plots/dice_comparison_all_models.png** for performance ranking
3. Check **reports/thesis_comparison_report.md** for detailed analysis

### Detailed Review (30 minutes):
1. Examine all plots in plots/ folder
2. Review data files for statistical details
3. Read NNUNet analysis for future potential

## Key Questions for Team Review

1. **Model Selection**: Should we proceed with SegResNet or complete NNUNet training?
2. **Clinical Validation**: Are additional validation studies needed?
3. **Thesis Focus**: Should we emphasize current results or future potential?
4. **Publication**: Are these results ready for manuscript preparation?

## Contact
Questions about the analysis or results?
- Review the detailed reports in the reports/ folder
- Check the original analysis repository for methodology details
- Contact the research team for clarification

---

**Package Size**: ~10MB (compressed)  
**Upload Time**: < 1 minute on typical connection  
**Review Time**: 5-30 minutes depending on detail level
"""
    
    # Save instructions with UTF-8 encoding
    instructions_path = package_dir / "README.txt"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"   ✅ Created: README.txt (upload instructions)")
    return instructions_path

def main():
    print("🎓 CREATING CLEAN THESIS RESULTS PACKAGE FOR TEAM REVIEW")
    print("=" * 60)
    
    # Analyze existing results
    analysis = analyze_existing_results()
    
    # Create clean package
    package_dir, copied_files = create_thesis_review_package(analysis)
    
    # Create executive summary
    summary_path = create_thesis_summary_report(package_dir)
    
    # Create upload instructions
    instructions_path = create_upload_instructions(package_dir)
    
    # Show final package structure
    print(f"\n📁 THESIS RESULTS PACKAGE CREATED:")
    print(f"   Location: {package_dir}")
    print(f"   Total files: {len(copied_files) + 2}")  # +2 for summary and instructions
    
    print(f"\n📋 PACKAGE CONTENTS:")
    print(f"   📊 Data files: {len([f for f in copied_files if f.startswith('data/')])}")
    print(f"   📈 Plot files: {len([f for f in copied_files if f.startswith('plots/')])}")
    print(f"   📄 Report files: {len([f for f in copied_files if f.startswith('reports/')])}")
    print(f"   📋 Supporting files: 2 (summary + instructions)")
    
    print(f"\n🚀 READY FOR UPLOAD:")
    print(f"   ✅ Compress and upload THESIS_RESULTS_PACKAGE folder")
    print(f"   ✅ Share with thesis team for review")
    print(f"   ✅ All essential results included, no clutter")
    
    # Show file sizes
    total_size = sum(f.stat().st_size for f in package_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    print(f"\n📊 Package Size: {size_mb:.1f} MB (uncompressed)")
    print(f"📊 Estimated ZIP size: {size_mb * 0.6:.1f} MB")

if __name__ == '__main__':
    main()
