"""
generate_stenosis_annotation_package.py — 3D Stenosis Lesion Annotation & 3D Slicer Packaging
=============================================================================================
Extracts 3D stenosis lesion masks on top of the coronary artery tree for clinical validation.
Outputs:
  1. pred_multilabel.nii.gz (0: Background, 1: Artery Lumen, 2: Stenosis Lesion)
  2. stenosis_mask.nii.gz (Standalone binary lesion mask)
  3. load_in_slicer.py (1-click python loader for 3D Slicer with preset windowing and styled labels)
  4. stenosis_annotation_preview.png (Multi-planar diagnostic image centered on the stenosis notch)
  5. Standalone ZIP package ready to send to Mihir and the cardiologist
"""

import os
import sys
import zipfile
import argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import authoritative QCA geometry routines
sys.path.insert(0, os.path.dirname(__file__))
from production_qca_engine import extract_case_geometry, get_cad_rads_str

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "local_data")

# Known verified lesions from Step 0 provenance audit
VERIFIED_TARGETS = {
    "CT4":  {"artery": "LCx", "target_str": "LCX", "rad_stenosis": 85.0, "badge": "70-99% (CAD-RADS 4)"},
    "CT61": {"artery": "RCA", "target_str": "RCA", "rad_stenosis": 60.0, "badge": "50-69% (CAD-RADS 3)"},
    "CT63": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 80.0, "badge": "70-99% (CAD-RADS 4)"},
    "CT64": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 85.0, "badge": "70-99% (CAD-RADS 4)"},
    "CT65": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 40.0, "badge": "25-49% (CAD-RADS 2)"},
    "CT68": {"artery": "RCA", "target_str": "RCA", "rad_stenosis": 85.0, "badge": "70-99% (CAD-RADS 4)"},
    "CT70": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 95.0, "badge": "90-99% (CAD-RADS 4C)"},
    "CT71": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 35.0, "badge": "25-49% (CAD-RADS 2)"},
    "CT74": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 40.0, "badge": "25-49% (CAD-RADS 2)"},
    "CT75": {"artery": "RCA", "target_str": "RCA", "rad_stenosis": 60.0, "badge": "50-69% (CAD-RADS 3)"},
    "CT76": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 60.0, "badge": "50-69% (CAD-RADS 3)"},
    "CT78": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 95.0, "badge": "90-99% (CAD-RADS 4C)"},
    "CT79": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 35.0, "badge": "25-49% (CAD-RADS 2)"},
    "CT80": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 95.0, "badge": "90-99% (CAD-RADS 4C)"},
    "CT81": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 85.0, "badge": "70-99% (CAD-RADS 4)"},
    "CT83": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 95.0, "badge": "90-99% (CAD-RADS 4C)"},
    "CT85": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 85.0, "badge": "70-99% (CAD-RADS 4)"},
    "CT87": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 40.0, "badge": "25-49% (CAD-RADS 2)"},
    "CT88": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 60.0, "badge": "50-69% (CAD-RADS 3)"},
    "CT89": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 85.0, "badge": "70-99% (CAD-RADS 4)"},
    "CT90": {"artery": "LAD", "target_str": "LAD", "rad_stenosis": 35.0, "badge": "25-49% (CAD-RADS 2)"},
}

def analyze_and_extract_lesion(case_geom, target_info=None):
    """Identifies the target stenosis lesion on the centerline and extracts 3D voxel mask."""
    branches = case_geom['branches']
    vox_size = case_geom['vox_size']
    pred_arr = case_geom['pred_arr'] # (Z, Y, X) boolean
    
    target_artery = target_info.get("artery") if target_info else None
    
    evaluated_branches = []
    for b_idx, pb in enumerate(branches):
        raw_diams = pb['diams']
        b_pts = pb['pts']
        artery = pb['artery']
        
        ref_d = float(np.percentile(raw_diams, 85))
        if ref_d < 1.4 or len(raw_diams) < 7:
            continue
            
        win = min(7, len(raw_diams) if len(raw_diams) % 2 != 0 else len(raw_diams) - 1)
        s_diams = savgol_filter(raw_diams, window_length=max(5, win), polyorder=2)
        n_pts = len(s_diams)
        
        i_start = 2 if n_pts >= 7 else 1
        i_end = n_pts - 2 if n_pts >= 7 else n_pts - 1
        if i_end <= i_start:
            continue
            
        eval_diams = s_diams[i_start:i_end]
        min_idx = int(np.argmin(eval_diams)) + i_start
        mld_raw = float(raw_diams[min_idx])
        mld = min(mld_raw, float(s_diams[min_idx]))
        
        # Proximal & distal reference calibers
        w_nodes = max(3, int(round(8.0 / vox_size)))
        prox_start = max(0, min_idx - 2 * w_nodes)
        prox_end = max(0, min_idx - w_nodes // 2)
        if prox_end > prox_start and (prox_end - prox_start) >= 2:
            prox_ref_d = float(np.percentile(s_diams[prox_start:prox_end], 85))
        else:
            prox_ref_d = ref_d
        prox_ref_d = min(5.0, max(prox_ref_d, 1.5))
        
        dist_start = min(n_pts, min_idx + int(round(8.0 / vox_size)))
        dist_end = min(n_pts, min_idx + int(round(16.0 / vox_size)))
        if dist_end > dist_start:
            dist_ref_d = min(prox_ref_d, float(np.percentile(s_diams[dist_start:dist_end], 85)))
        else:
            dist_ref_d = max(1.2, prox_ref_d - 0.020 * (min_idx * vox_size))
            
        alpha = max(0.0, min(1.0, (min_idx - prox_start) / max(1, (dist_start - prox_start))))
        interp_ref_d = (1.0 - alpha) * prox_ref_d + alpha * dist_ref_d
        
        ds = max(0.0, min(99.0, (1.0 - mld / interp_ref_d) * 100.0))
        as_pct = max(0.0, min(99.9, (1.0 - (mld / interp_ref_d)**2) * 100.0))
        
        # Lesion extent along centerline: where caliber is < 85% of reference,
        # or at least ±5 mm around the MLD notch
        lesion_margin_nodes = max(3, int(round(4.0 / vox_size))) # ±4mm minimum
        l_start = max(0, min_idx - lesion_margin_nodes)
        l_end = min(n_pts, min_idx + lesion_margin_nodes + 1)
        
        # Expand while caliber < 0.85 * interp_ref_d
        while l_start > 0 and s_diams[l_start] < 0.85 * interp_ref_d and (min_idx - l_start) < 2 * lesion_margin_nodes:
            l_start -= 1
        while l_end < n_pts and s_diams[l_end - 1] < 0.85 * interp_ref_d and (l_end - min_idx) < 2 * lesion_margin_nodes:
            l_end += 1
            
        lesion_pts = b_pts[l_start:l_end]
        notch_pt = b_pts[min_idx] # (z, y, x)
        
        evaluated_branches.append({
            'b_idx': b_idx,
            'artery': artery,
            'ds': ds,
            'as': as_pct,
            'mld': mld,
            'ref_d': interp_ref_d,
            'min_idx': min_idx,
            'notch_pt': notch_pt,
            'lesion_pts': lesion_pts,
            'length_mm': len(pb['nodes']) * vox_size,
            's_diams': s_diams,
            'raw_diams': raw_diams
        })
        
    if not evaluated_branches:
        raise ValueError("No valid coronary branches could be evaluated for this case.")
        
    # Select lesion: match target artery if specified, otherwise pick highest %DS
    target_matches = [eb for eb in evaluated_branches if eb['artery'] == target_artery] if target_artery else []
    if target_matches:
        best_lesion = max(target_matches, key=lambda x: x['ds'])
    else:
        best_lesion = max(evaluated_branches, key=lambda x: x['ds'])
        
    # Now construct the 3D volumetric lesion mask in native patient space
    # Find all voxels in pred_arr that are within distance of the lesion centerline points
    lesion_mask = np.zeros_like(pred_arr, dtype=bool)
    lesion_pts = best_lesion['lesion_pts'] # array of (z, y, x)
    
    # Bounding box around lesion
    z_min, y_min, x_min = np.min(lesion_pts, axis=0)
    z_max, y_max, x_max = np.max(lesion_pts, axis=0)
    pad = 8
    z0, z1 = max(0, z_min - pad), min(pred_arr.shape[0], z_max + pad + 1)
    y0, y1 = max(0, y_min - pad), min(pred_arr.shape[1], y_max + pad + 1)
    x0, x1 = max(0, x_min - pad), min(pred_arr.shape[2], x_max + pad + 1)
    
    sub_pred = pred_arr[z0:z1, y0:y1, x0:x1]
    sub_coords = np.argwhere(sub_pred) # relative to (z0, y0, x0)
    if len(sub_coords) > 0:
        global_coords = sub_coords + np.array([z0, y0, x0])
        from scipy.spatial import cKDTree
        tree = cKDTree(lesion_pts)
        dists, _ = tree.query(global_coords)
        # Dilate by up to local lumen radius
        max_dist_vox = max(3.0, (best_lesion['ref_d'] / 2.0) / vox_size + 1.0)
        within_lesion = dists <= max_dist_vox
        
        for gc in global_coords[within_lesion]:
            lesion_mask[gc[0], gc[1], gc[2]] = True
            
    # Stenosis lesion MUST be a strict subset of pred_arr
    lesion_mask = lesion_mask & pred_arr
    
    # Construct multi-label array:
    # 0 = Background
    # 1 = Artery Lumen (Full vessel)
    # 2 = Stenosis Lesion
    multilabel_arr = np.zeros_like(pred_arr, dtype=np.uint8)
    multilabel_arr[pred_arr] = 1
    multilabel_arr[lesion_mask] = 2
    
    return {
        'best_lesion': best_lesion,
        'lesion_mask': lesion_mask,
        'multilabel_arr': multilabel_arr,
        'notch_pt': best_lesion['notch_pt']
    }

def generate_case_package(case_name: str):
    print(f"\n================================================================================")
    print(f">> PROCESSING STENOSIS ANNOTATION PACKAGE: {case_name}")
    print(f"================================================================================")
    
    case_dir = os.path.join(LOCAL_DATA_DIR, case_name)
    img_path = os.path.join(case_dir, "image.nii.gz")
    pred_path = os.path.join(case_dir, "pred_mask.nii.gz")
    
    if not os.path.exists(img_path) or not os.path.exists(pred_path):
        print(f"Error: Missing image.nii.gz or pred_mask.nii.gz in {case_dir}")
        return False
        
    target_info = VERIFIED_TARGETS.get(case_name, None)
    if target_info:
        print(f"   [Step 0 Provenance Target]: Artery={target_info['artery']}, Rad Stenosis={target_info['rad_stenosis']}%, Badge={target_info['badge']}")
    else:
        print(f"   [Autonomous Whole-Tree Mode]: Highest %DS across all branches")
        
    # 1. Extract geometry
    geom = extract_case_geometry(case_name)
    if not geom:
        print("Failed to extract case geometry.")
        return False
        
    # 2. Extract lesion mask & multi-label volume
    lesion_data = analyze_and_extract_lesion(geom, target_info)
    bl = lesion_data['best_lesion']
    multilabel_arr = lesion_data['multilabel_arr']
    lesion_mask = lesion_data['lesion_mask']
    notch_z, notch_y, notch_x = lesion_data['notch_pt']
    
    print(f"   [Stenosis Notch Localization]: Voxel (Z={notch_z}, Y={notch_y}, X={notch_x})")
    print(f"   [Quantification]: Artery={bl['artery']}, %DS={bl['ds']:.1f}%, %AS={bl['as']:.1f}%")
    print(f"   [Caliber Profile]: MLD={bl['mld']:.2f} mm, Reference={bl['ref_d']:.2f} mm")
    print(f"   [CAD-RADS Grade]: {get_cad_rads_str(bl['ds'])}")
    print(f"   [Lesion Voxel Count]: {int(np.sum(lesion_mask))} voxels (Total vessel: {int(np.sum(geom['pred_arr']))})")
    
    # 3. Save NIfTI files with identical metadata
    pred_sitk = sitk.ReadImage(pred_path)
    
    # Save multi-label NIfTI
    multilabel_sitk = sitk.GetImageFromArray(multilabel_arr)
    multilabel_sitk.CopyInformation(pred_sitk)
    multilabel_path = os.path.join(case_dir, "pred_multilabel.nii.gz")
    sitk.WriteImage(multilabel_sitk, multilabel_path)
    print(f"   [Saved Multi-label NIfTI]: {multilabel_path}")
    
    # Save standalone lesion mask
    lesion_sitk = sitk.GetImageFromArray(lesion_mask.astype(np.uint8))
    lesion_sitk.CopyInformation(pred_sitk)
    lesion_path = os.path.join(case_dir, "stenosis_mask.nii.gz")
    sitk.WriteImage(lesion_sitk, lesion_path)
    print(f"   [Saved Stenosis Mask]: {lesion_path}")
    
    # 4. Generate 1-Click 3D Slicer Startup Script
    slicer_script_path = os.path.join(case_dir, "load_case_in_slicer.py")
    slicer_code = f'''# 3D Slicer Automated Loader for {case_name}
# ===============================================
# Run this script inside 3D Slicer Python Console (Ctrl + `)
# or launch: Slicer.exe --python-script "{slicer_script_path}"

import slicer
import os

case_dir = r"{case_dir}"
img_path = os.path.join(case_dir, "image.nii.gz")
seg_path = os.path.join(case_dir, "pred_multilabel.nii.gz")

# 1. Close existing scene
slicer.mrmlScene.Clear(0)

# 2. Load CT Volume
print("Loading CCTA volume...")
vol_node = slicer.util.loadVolume(img_path)
vol_node.SetName("{case_name}_CCTA_Scan")

# Configure Cardiac Window/Level (Window: 700 HU, Level: 250 HU)
vol_disp = vol_node.GetDisplayNode()
if vol_disp:
    vol_disp.AutoWindowLevelOff()
    vol_disp.SetWindow(700)
    vol_disp.SetLevel(250)

# 3. Load Multi-label Segmentation
print("Loading Stenosis \u0026 Artery Segmentation...")
seg_node = slicer.util.loadSegmentation(seg_path)
seg_node.SetName("{case_name}_Coronary_and_Stenosis")

# Configure Segment Names & Colors
segmentation = seg_node.GetSegmentation()
if segmentation.GetNumberOfSegments() >= 2:
    # Segment 1: Full Artery Tree
    seg1 = segmentation.GetNthSegment(0)
    seg1.SetName("1. Coronary Artery Lumen")
    seg1.SetColor(0.15, 0.85, 0.25) # Medical Green
    
    # Segment 2: Stenosis Lesion
    seg2 = segmentation.GetNthSegment(1)
    seg2.SetName("2. Stenosis Lesion ({bl['artery']} {bl['ds']:.0f}%)")
    seg2.SetColor(0.95, 0.15, 0.15) # Focal Alert Red

# 4. Center 2D and 3D Views on Stenosis Notch
# Notch coordinates in IJK: ({notch_x}, {notch_y}, {notch_z})
print("Centering views on stenosis lesion ({bl['artery']} notch at slice {notch_z})...")
ras_pt = [0, 0, 0]
vol_node.TransformIndexToPhysicalPoint([{notch_x}, {notch_y}, {notch_z}], ras_pt)

for view_name in ["Red", "Yellow", "Green"]:
    slice_logic = slicer.app.layoutManager().sliceWidget(view_name).sliceLogic()
    slice_logic.SetSliceOffset(ras_pt[2] if view_name == "Red" else (ras_pt[0] if view_name == "Yellow" else ras_pt[1]))
    slice_logic.FitSliceToBackground()

# Switch to 3D Segment Editor module for instant doctor review
slicer.util.selectModule("SegmentEditor")
print("\\nSUCCESS: {case_name} loaded!")
print("  - Green: Coronary Artery Lumen")
print("  - Red: Stenosis Lesion ({bl['artery']}, %DS = {bl['ds']:.1f}%)")
print("  - Cardiologist can now inspect or edit Segment 2 in Segment Editor.")
'''
    with open(slicer_script_path, "w", encoding="utf-8") as f:
        f.write(slicer_code)
    print(f"   [Generated 3D Slicer Script]: {slicer_script_path}")
    
    # 5. Generate High-Resolution Diagnostic Visualization (PNG)
    img_sitk = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    
    fig = plt.figure(figsize=(18, 10), dpi=200)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.25, wspace=0.15)
    
    # Axial Slice at Stenosis Notch
    ax_axial = fig.add_subplot(gs[0, 0])
    axial_slice = np.clip(img_arr[notch_z], -100, 700)
    ax_axial.imshow(axial_slice, cmap='gray', origin='lower')
    if geom['pred_arr'][notch_z].any():
        ax_axial.contour(geom['pred_arr'][notch_z], colors='#22c55e', levels=[0.5], linewidths=1.8)
    if lesion_mask[notch_z].any():
        ax_axial.contourf(lesion_mask[notch_z], levels=[0.5, 1.5], colors=['#ef4444'], alpha=0.75)
    ax_axial.scatter([notch_x], [notch_y], color='yellow', s=50, edgecolors='black', zorder=5, label='Notch (MLD)')
    ax_axial.set_title(f"A. Axial View (Slice Z={notch_z})\nRed = Stenosis Lesion | Green = Artery", fontsize=11, fontweight='bold')
    ax_axial.legend(loc='upper right', fontsize=8)
    ax_axial.axis('off')
    
    # Coronal Slice at Stenosis Notch
    ax_coronal = fig.add_subplot(gs[0, 1])
    coronal_slice = np.clip(img_arr[:, notch_y, :], -100, 700)
    ax_coronal.imshow(coronal_slice, cmap='gray', origin='lower')
    if geom['pred_arr'][:, notch_y, :].any():
        ax_coronal.contour(geom['pred_arr'][:, notch_y, :], colors='#22c55e', levels=[0.5], linewidths=1.8)
    if lesion_mask[:, notch_y, :].any():
        ax_coronal.contourf(lesion_mask[:, notch_y, :], levels=[0.5, 1.5], colors=['#ef4444'], alpha=0.75)
    ax_coronal.scatter([notch_x], [notch_z], color='yellow', s=50, edgecolors='black', zorder=5)
    ax_coronal.set_title(f"B. Coronal View (Y={notch_y})\nLongitudinal Vessel Course", fontsize=11, fontweight='bold')
    ax_coronal.axis('off')
    
    # Sagittal Slice at Stenosis Notch
    ax_sagittal = fig.add_subplot(gs[0, 2])
    sagittal_slice = np.clip(img_arr[:, :, notch_x], -100, 700)
    ax_sagittal.imshow(sagittal_slice, cmap='gray', origin='lower')
    if geom['pred_arr'][:, :, notch_x].any():
        ax_sagittal.contour(geom['pred_arr'][:, :, notch_x], colors='#22c55e', levels=[0.5], linewidths=1.8)
    if lesion_mask[:, :, notch_x].any():
        ax_sagittal.contourf(lesion_mask[:, :, notch_x], levels=[0.5, 1.5], colors=['#ef4444'], alpha=0.75)
    ax_sagittal.scatter([notch_y], [notch_z], color='yellow', s=50, edgecolors='black', zorder=5)
    ax_sagittal.set_title(f"C. Sagittal View (X={notch_x})\nNarrowing Profile", fontsize=11, fontweight='bold')
    ax_sagittal.axis('off')
    
    # Stenosis Centerline Diameter Profile Graph
    ax_prof = fig.add_subplot(gs[1, :2])
    pts_x = np.arange(len(bl['raw_diams'])) * geom['vox_size']
    ax_prof.plot(pts_x, bl['raw_diams'], color='#94a3b8', linestyle=':', label='Raw Caliber (Distance Transform)', alpha=0.8)
    ax_prof.plot(pts_x, bl['s_diams'], color='#2563eb', lw=2.5, label='Filtered Vessel Caliber Profile')
    ax_prof.axhline(bl['ref_d'], color='#10b981', linestyle='--', lw=2, label=f'Reference Caliber: {bl["ref_d"]:.2f} mm')
    ax_prof.axhline(bl['mld'], color='#ef4444', linestyle='--', lw=2, label=f'Minimal Lumen Diam (MLD): {bl["mld"]:.2f} mm')
    ax_prof.scatter([bl['min_idx'] * geom['vox_size']], [bl['mld']], color='#dc2626', s=100, zorder=6, label='Stenosis Trough (MLD)')
    
    # Highlight lesion zone on profile
    l_pts_len = len(bl['lesion_pts']) * geom['vox_size']
    l_start_x = max(0, bl['min_idx'] * geom['vox_size'] - l_pts_len / 2.0)
    l_end_x = min(pts_x[-1], bl['min_idx'] * geom['vox_size'] + l_pts_len / 2.0)
    ax_prof.axvspan(l_start_x, l_end_x, color='#ef4444', alpha=0.15, label='Annotated Stenosis Lesion Zone')
    
    ax_prof.set_title(f"D. {bl['artery']} Lumen Diameter Trajectory & Quantitative Stenosis Notch", fontsize=12, fontweight='bold')
    ax_prof.set_xlabel("Distance Along Artery Centerline (mm)", fontsize=10, fontweight='bold')
    ax_prof.set_ylabel("Physical Lumen Caliber (mm)", fontsize=10, fontweight='bold')
    ax_prof.grid(True, alpha=0.3)
    ax_prof.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Clinical Summary Card
    ax_card = fig.add_subplot(gs[1, 2])
    ax_card.axis('off')
    summary_text = (
        f"PATIENT / CASE: {case_name}\n"
        f"─────────────────────────────────────\n"
        f"• Target Artery:       {bl['artery']}\n"
        f"• Minimal Lumen Diam:  {bl['mld']:.2f} mm\n"
        f"• Reference Caliber:   {bl['ref_d']:.2f} mm\n"
        f"• Diameter Stenosis:   {bl['ds']:.1f}%\n"
        f"• Area Stenosis:       {bl['as']:.1f}%\n"
        f"• CAD-RADS Class:      {get_cad_rads_str(bl['ds'])}\n"
        f"─────────────────────────────────────\n"
        f"GROUND TRUTH PACS BADGE:\n"
        f"• Verified Badge:      {target_info['badge'] if target_info else 'N/A'}\n"
        f"• Verified %DS:        {target_info['rad_stenosis'] if target_info else 'N/A'}%\n"
        f"─────────────────────────────────────\n"
        f"ANNOTATION LAYERS (3D SLICER):\n"
        f"  [1] Green: Full Artery Lumen (RASNet)\n"
        f"  [2] Red:   Focal Stenosis Lesion\n"
        f"Cardiologist: Touch up Red zone in\n"
        f"Segment Editor (~1-2 min)."
    )
    ax_card.text(0.05, 0.95, summary_text, transform=ax_card.transAxes, fontsize=10.5,
                 fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1', lw=1.5))
                 
    plt.suptitle(f"Clinical Stenosis Lesion Annotation Package — Case {case_name}", fontsize=15, fontweight='bold', y=0.98)
    preview_png_path = os.path.join(case_dir, "stenosis_annotation_preview.png")
    plt.savefig(preview_png_path, bbox_inches='tight')
    plt.close()
    print(f"   [Generated Visual Preview]: {preview_png_path}")
    
    # 6. Build Portable ZIP Package
    zip_filename = f"{case_name}_Stenosis_Annotation_Package.zip"
    zip_path = os.path.join(case_dir, zip_filename)
    
    readme_text = f"""================================================================================
CLINICAL CCTA STENOSIS ANNOTATION PACKAGE — CASE {case_name}
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
   exec(open(r"{slicer_script_path}").read())
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
"""
    readme_path = os.path.join(case_dir, "README_SLICER_INSTRUCTIONS.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_text)
        
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(multilabel_path, arcname="pred_multilabel.nii.gz")
        zipf.write(lesion_path, arcname="stenosis_mask.nii.gz")
        zipf.write(pred_path, arcname="pred_mask.nii.gz")
        zipf.write(slicer_script_path, arcname="load_case_in_slicer.py")
        zipf.write(preview_png_path, arcname="stenosis_annotation_preview.png")
        zipf.write(readme_path, arcname="README_SLICER_INSTRUCTIONS.txt")
        # Notice: image.nii.gz is included so package is 100% self-contained
        zipf.write(img_path, arcname="image.nii.gz")
        
    print(f"   [Built Portable ZIP Package]: {zip_path} ({round(os.path.getsize(zip_path) / 1e6, 1)} MB)")
    print(f"================================================================================\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D Stenosis Annotation Packages")
    parser.add_argument("--case", type=str, default="CT61", help="Case name (e.g. CT61, CT4)")
    args = parser.parse_args()
    
    generate_case_package(args.case)
