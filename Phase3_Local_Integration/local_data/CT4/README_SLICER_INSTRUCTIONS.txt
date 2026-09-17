================================================================================
CLINICAL CCTA STENOSIS ANNOTATION PACKAGE — CASE CT4
================================================================================
Designed for: Ibrahim Cardiac Hospital Cohort / Thesis RASNet Clinical Evaluation

FILES IN THIS PACKAGE:
1. image.nii.gz:
   Native 3D CCTA scan volume.
   
2. pred_multilabel.nii.gz:
   Dual-layer segmentation mask:
     - Value 1 = Full Coronary Artery Tree (Green)
     - Value 2 = Stenosis Lesion Zone (Red)

3. pred_mask.nii.gz:
   Original whole-artery lumen segmentation (AI baseline).

4. stenosis_mask.nii.gz:
   Standalone binary mask of the focal stenosis lesion.

5. load_case_in_slicer.py:
   1-click startup script for 3D Slicer.

6. stenosis_annotation_preview.png:
   Multi-planar CT reconstruction showing the exact green vessel contour,
   red stenosis lesion highlight, and lumen caliber diameter profile.

--------------------------------------------------------------------------------
HOW THE CARDIOLOGIST OPENS THIS IN 3D SLICER:
--------------------------------------------------------------------------------
Method A (Automated 1-Click):
1. Open 3D Slicer (version 5.0+).
2. Press `Ctrl + ~` (or View -> Python Console).
3. Copy-paste or run:
   exec(open(r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data\CT4\load_case_in_slicer.py").read())
   Both CT volume and styled multi-layer segmentation will load immediately,
   centered on the stenosis lesion!

Method B (Manual Drag & Drop):
1. Drag and drop `image.nii.gz` into 3D Slicer -> Select 'Volume' -> OK.
2. In Volumes module, set Window=700, Level=250 (Preset: CT-Cardiac).
3. Drag and drop `pred_multilabel.nii.gz` -> Select 'Segmentation' -> OK.
4. In Segment Editor:
   - Segment 1 is the Coronary Artery Lumen (Green).
   - Segment 2 is the Stenosis Lesion (Red).
5. The doctor can use 'Paint' / 'Scissors' / 'Erase' on Segment 2 to touch up
   the lesion boundary in 1-2 minutes!
================================================================================
