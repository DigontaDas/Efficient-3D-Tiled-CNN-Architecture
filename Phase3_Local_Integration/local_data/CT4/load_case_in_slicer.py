# 3D Slicer Automated Loader for CT4
# ===============================================
# Run this script inside 3D Slicer Python Console (Ctrl + `)
# or launch: Slicer.exe --python-script "H:\Thesis_Trainings\Phase3_Local_Integration\local_data\CT4\load_case_in_slicer.py"

import slicer
import os

case_dir = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data\CT4"
img_path = os.path.join(case_dir, "image.nii.gz")
seg_path = os.path.join(case_dir, "pred_multilabel.nii.gz")

# 1. Close existing scene
slicer.mrmlScene.Clear(0)

# 2. Load CT Volume
print("Loading CCTA volume...")
vol_node = slicer.util.loadVolume(img_path)
vol_node.SetName("CT4_CCTA_Scan")

# Configure Cardiac Window/Level (Window: 700 HU, Level: 250 HU)
vol_disp = vol_node.GetDisplayNode()
if vol_disp:
    vol_disp.AutoWindowLevelOff()
    vol_disp.SetWindow(700)
    vol_disp.SetLevel(250)

# 3. Load Multi-label Segmentation
print("Loading Stenosis & Artery Segmentation...")
seg_node = slicer.util.loadSegmentation(seg_path)
seg_node.SetName("CT4_Coronary_and_Stenosis")

# Configure Segment Names & Colors
segmentation = seg_node.GetSegmentation()
if segmentation.GetNumberOfSegments() >= 2:
    # Segment 1: Full Artery Tree
    seg1 = segmentation.GetNthSegment(0)
    seg1.SetName("1. Coronary Artery Lumen")
    seg1.SetColor(0.15, 0.85, 0.25) # Medical Green
    
    # Segment 2: Stenosis Lesion
    seg2 = segmentation.GetNthSegment(1)
    seg2.SetName("2. Stenosis Lesion (LCx 48%)")
    seg2.SetColor(0.95, 0.15, 0.15) # Focal Alert Red

# 4. Center 2D and 3D Views on Stenosis Notch
# Notch coordinates in IJK: (354, 322, 75)
print("Centering views on stenosis lesion (LCx notch at slice 75)...")
ras_pt = [0, 0, 0]
vol_node.TransformIndexToPhysicalPoint([354, 322, 75], ras_pt)

for view_name in ["Red", "Yellow", "Green"]:
    slice_logic = slicer.app.layoutManager().sliceWidget(view_name).sliceLogic()
    slice_logic.SetSliceOffset(ras_pt[2] if view_name == "Red" else (ras_pt[0] if view_name == "Yellow" else ras_pt[1]))
    slice_logic.FitSliceToBackground()

# Switch to 3D Segment Editor module for instant doctor review
slicer.util.selectModule("SegmentEditor")
print("\nSUCCESS: CT4 loaded!")
print("  - Green: Coronary Artery Lumen")
print("  - Red: Stenosis Lesion (LCx, %DS = 48.1%)")
print("  - Cardiologist can now inspect or edit Segment 2 in Segment Editor.")
