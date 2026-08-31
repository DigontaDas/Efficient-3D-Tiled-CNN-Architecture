QUALITATIVE RESULTS - LIMITATION NOTICE
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
