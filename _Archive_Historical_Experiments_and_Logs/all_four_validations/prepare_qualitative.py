"""
Qualitative Analysis Setup

Note: SegResNet qualitative images exist only for cases 0-19.
Representative cases selected based on metrics:
- Best: Case_42 (Dice 0.895) - image NOT available
- Median: Case_22 (Dice 0.812) - image NOT available  
- Worst: Case_35 (Dice 0.698) - image NOT available

Available qualitative images (cases 0-19) do not include the selected representative cases.
This is a limitation for qualitative comparison.
"""

import shutil
import os

os.makedirs('qualitative_results', exist_ok=True)

# Copy available SegResNet images as examples
# Note: These are NOT the representative cases due to limited available images
example_cases = [0, 10, 13]  # approximating best, median, worst from available
labels = ['example_approx_best', 'example_approx_median', 'example_approx_worst']

for case_num, label in zip(example_cases, labels):
    src = f'mandatory_artifacts_segresnet/new_robust_Images/case_{case_num}_result.png'
    dst = f'qualitative_results/segresnet_{label}_case_{case_num}.png'
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"Missing: {src}")

# Create note about limitation
with open('qualitative_results/README.txt', 'w') as f:
    f.write("""QUALITATIVE RESULTS - LIMITATION NOTICE
========================================

Representative cases selected from metrics:
- Best performing: Case_42 (Dice: 0.895)
- Median performing: Case_22 (Dice: 0.812)  
- Worst performing: Case_35 (Dice: 0.698)

However, qualitative visualization images are only available for cases 0-19
in the SegResNet artifacts. The images included here are approximations:
- case_0 (Dice 0.753) - closest available to worst category
- case_10 (Dice 0.838) - closest available to median/best category
- case_13 (Dice 0.715) - closest available to worst category

3D U-Net predictions exist (300 test cases) but without:
1. Computed metrics to identify representative cases
2. Clear mapping to SegResNet case numbering
3. Qualitative rendering/slices

Therefore, meaningful cross-model qualitative comparison is NOT possible
with existing artifacts alone.
""")

print("\nQualitative results prepared with noted limitations.")
