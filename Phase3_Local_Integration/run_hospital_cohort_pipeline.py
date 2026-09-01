"""
run_hospital_cohort_pipeline.py — Targeted CCTA Series Selection, GPU Inference & Clinical Stenosis Profiling
=============================================================================================================
Processes clinical CCTA scans from H:\CT_Scans_Thesis\CT {i}\A:
  1. Stage 1: Smart-matches and converts the Radiologist's CCTA Series (SS-Freeze / Temporal 75%/45%) to NIfTI
  2. Stage 2: Executes 3D RASNet Deep Learning inference on NVIDIA GeForce RTX 3060 Ti GPU
  3. Stage 3: Topology-Aware 3D Connected Component Cleaning & Native Spatial Grid Alignment
"""
import os
import sys
import time
import pydicom
import torch
import numpy as np
import SimpleITK as sitk
import monai.transforms as mt
from monai.inferers import sliding_window_inference
import cc3d

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))
from rasnet_model import RASNet

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Force CUDA Tensor Core acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 2  # Optimized for 8 GB VRAM RTX 3060 Ti

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_PATH = os.path.join(BASE_DIR, "results-after-hallucin-fix", "checkpoints", "rasnet_best.pth")
DICOM_ROOT = r"H:\CT_Scans_Thesis"
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "local_data")

# Transforms
pre_trans = mt.Compose([
    mt.LoadImaged(keys=["image"]),
    mt.EnsureChannelFirstd(keys=["image"]),
    mt.Orientationd(keys=["image"], axcodes="RAS"),
    mt.Spacingd(keys=["image"], pixdim=(0.5, 0.5, 0.5), mode="bilinear"),
    mt.ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=800, b_min=0.0, b_max=1.0, clip=True),
    mt.EnsureTyped(keys=["image"])
])

post_trans = mt.Compose([
    mt.Invertd(
        keys=["pred"],
        transform=pre_trans,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True,
    )
])


def topological_postprocess(pred_mask: np.ndarray, min_size=50) -> np.ndarray:
    """Removes isolated floating background components and keeps top-2 largest 3D trees."""
    labeled, N = cc3d.connected_components(pred_mask, return_N=True)
    if N == 0:
        return np.zeros_like(pred_mask, dtype=np.uint8)
    sizes = np.bincount(labeled.flat)
    valid_components = []
    for comp_id in range(1, len(sizes)):
        if sizes[comp_id] >= min_size:
            valid_components.append((comp_id, sizes[comp_id]))
    valid_components.sort(key=lambda x: x[1], reverse=True)
    top_components = valid_components[:2]
    cleaned = np.zeros_like(pred_mask, dtype=np.uint8)
    for comp_id, _ in top_components:
        cleaned[labeled == comp_id] = 1
    return cleaned


def run_gpu_inference(model: RASNet, img_path: str, pred_path: str):
    """Executes 4-Pass TTA 3D Inference with MONAI Invertd coordinate restoration."""
    start_t = time.time()
    torch.cuda.empty_cache()
    batch = pre_trans({"image": img_path})
    input_tensor = batch["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = sliding_window_inference(
                input_tensor, roi_size=PATCH_SIZE, sw_batch_size=SW_BATCH_SIZE,
                predictor=model, overlap=0.5
            )
            prob_avg = torch.softmax(logits, dim=1)

            for flip_dim in (2, 3, 4):
                input_flip = torch.flip(input_tensor, dims=[flip_dim]).contiguous()
                logits_flip = sliding_window_inference(
                    input_flip, roi_size=PATCH_SIZE, sw_batch_size=SW_BATCH_SIZE,
                    predictor=model, overlap=0.5
                )
                prob_avg += torch.softmax(torch.flip(logits_flip, dims=[flip_dim]).contiguous(), dim=1)

            prob_avg /= 4.0

    # Invert transforms to restore original physical coordinate grid
    batch["pred"] = prob_avg.squeeze(0)
    batch = post_trans(batch)

    # Free GPU VRAM
    del input_tensor, logits, prob_avg
    torch.cuda.empty_cache()

    # Threshold foreground confidence at 0.55
    pred_probs = batch["pred"]
    fg_prob = pred_probs[1]
    pred_mask = (fg_prob > 0.55).cpu().numpy().astype(np.uint8)

    # Topology component cleaning
    pred_mask_cleaned = topological_postprocess(pred_mask, min_size=50)

    # Save NIfTI image with identical spatial coordinate metadata
    original_img = sitk.ReadImage(img_path)
    pred_mask_sitk = np.transpose(pred_mask_cleaned, (2, 1, 0)) # (Z, Y, X)
    pred_sitk = sitk.GetImageFromArray(pred_mask_sitk)
    pred_sitk.CopyInformation(original_img)

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    sitk.WriteImage(pred_sitk, pred_path)

    # Automatically generate 3-slice verified overlay plot and 3D centerline profile
    try:
        import matplotlib.pyplot as plt
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import skeletonize

        arr_img = sitk.GetArrayFromImage(original_img)
        arr_mask = sitk.GetArrayFromImage(pred_sitk).astype(bool)
        spacing = pred_sitk.GetSpacing() # (x, y, z)
        z_counts = np.sum(arr_mask, axis=(1, 2))
        best_z = int(np.argmax(z_counts))
        
        # 1. 3-Slice Overlay Plot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=180)
        for idx, z in enumerate([max(0, best_z - 5), best_z, min(len(arr_img) - 1, best_z + 5)]):
            axs[idx].imshow(np.clip(arr_img[z], -100, 700), cmap='gray', origin='lower')
            if arr_mask[z].any():
                axs[idx].contour(arr_mask[z], colors='#ef4444', levels=[0.5], linewidths=1.5)
            axs[idx].set_title(f'Axial Slice z={z} (Verified Inverted)', fontsize=11, fontweight='bold')
            axs[idx].axis('off')
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(pred_path), "invertd_perfect_overlay.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        # 2. 3D Centerline & Stenosis Profile Plot
        coords = np.argwhere(arr_mask)
        if len(coords) > 0:
            zmin, ymin, xmin = coords.min(axis=0)
            zmax, ymax, xmax = coords.max(axis=0) + 1
            pad = 5
            z0, z1 = max(0, zmin - pad), min(arr_mask.shape[0], zmax + pad)
            y0, y1 = max(0, ymin - pad), min(arr_mask.shape[1], ymax + pad)
            x0, x1 = max(0, xmin - pad), min(arr_mask.shape[2], xmax + pad)

            sub_pred = arr_mask[z0:z1, y0:y1, x0:x1]
            sub_edt = distance_transform_edt(sub_pred, sampling=(spacing[2], spacing[1], spacing[0]))
            sub_skel = skeletonize(sub_pred)
            radii = sub_edt[sub_skel]
            diameters = 2.0 * radii

            if len(diameters) > 0:
                mean_d = float(np.mean(diameters))
                min_d = float(np.min(diameters))
                ref_d = float(np.percentile(diameters, 90))
                max_ds = float((1.0 - min_d / ref_d) * 100.0) if ref_d > 0 else 0.0

                cname = os.path.basename(os.path.dirname(pred_path))
                fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
                axs[0].imshow(np.clip(arr_img[best_z], -100, 700), cmap='gray', origin='lower')
                if arr_mask[best_z].any():
                    axs[0].contour(arr_mask[best_z], colors='#ef4444', levels=[0.5], linewidths=1.5)
                    skel_slice = np.zeros_like(arr_mask[best_z], dtype=bool)
                    if z0 <= best_z < z1:
                        skel_slice[y0:y1, x0:x1] = sub_skel[best_z - z0]
                        if skel_slice.any():
                            axs[0].scatter(np.where(skel_slice)[1], np.where(skel_slice)[0], c='#3b82f6', s=4, label='Centerline')
                            axs[0].legend(loc='upper right')
                axs[0].set_title(f'{cname} Native CT & 3D Centerline (Slice {best_z})', fontsize=10, fontweight='bold')
                axs[0].axis('off')

                axs[1].hist(diameters, bins=25, color='#3b82f6', edgecolor='black', alpha=0.7)
                axs[1].axvline(mean_d, color='#10b981', linestyle='--', lw=2, label=f'Mean Diam: {mean_d:.2f} mm')
                axs[1].axvline(min_d, color='#ef4444', linestyle='--', lw=2, label=f'Min Diam (MLD): {min_d:.2f} mm')
                axs[1].set_title(f'{cname} Lumen Diameter Profile (Max %DS: {max_ds:.1f}%)', fontsize=10, fontweight='bold')
                axs[1].set_xlabel('Physical Lumen Diameter (mm)')
                axs[1].set_ylabel('Voxel Centerline Count')
                axs[1].legend(loc='upper right')
                axs[1].grid(True, alpha=0.3)

                plt.tight_layout()
                skel_plot = os.path.join(os.path.dirname(pred_path), f"vessel_centerline_overlay_{cname}.png")
                plt.savefig(skel_plot, bbox_inches='tight')
                plt.close()
                print(f"   [Stage 3] Profiled {cname}: Mean={mean_d:.2f}mm, MLD={min_d:.2f}mm, %DS={max_ds:.1f}%", flush=True)

    except Exception as plot_err:
        print(f"   [WARN] Could not save overlay plots: {plot_err}", flush=True)

    elapsed = time.time() - start_t
    print(f"   [Stage 2] RTX 3060 Ti Inference: Saved {pred_path} in {elapsed:.2f}s (Foreground Voxels: {int(np.sum(pred_mask_cleaned))})", flush=True)


def select_best_ccta_series(dicom_dir: str):
    """Identifies and extracts the contrast-enhanced single-phase CCTA volume matching the radiologist's view."""
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if not os.path.isdir(os.path.join(dicom_dir, f))]
    series_map = {}
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            if hasattr(ds, 'SeriesNumber'):
                se_num = int(ds.SeriesNumber)
                se_desc = str(getattr(ds, 'SeriesDescription', ''))
                thick = float(getattr(ds, 'SliceThickness', 999.0))
                z_pos = float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else (float(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else 0.0)
                if se_num not in series_map:
                    series_map[se_num] = {'desc': se_desc, 'thick': thick, 'files': []}
                series_map[se_num]['files'].append((z_pos, f))
        except Exception:
            pass

    EXCLUDE_KEYWORDS = ['scout', 'calcium', 'scoring', 'plain', 'report', 'film', 'pages', 'dose', 'prep', '0-90', 'auto state', 'saved state']
    
    candidates = []
    for se_num, info in series_map.items():
        desc = info['desc'].lower()
        count = len(info['files'])
        thick = info['thick']
        
        # Skip excluded series or partial reconstructions (< 50 slices)
        if any(kw in desc for kw in EXCLUDE_KEYWORDS) or count < 50:
            continue
            
        score = 0
        if '75%' in desc:
            score += 60
        elif '45%' in desc:
            score += 10

        if 'ss-freeze' in desc or 'freeze' in desc:
            score += 100
        elif 'temporal' in desc:
            score += 80
        elif 'sseg' in desc and '0-90' not in desc:
            score += 70
        elif 'coronary' in desc:
            score += 50
            
        if thick <= 0.75:
            score += 15
            
        candidates.append((score, count, se_num, info))
        
    if not candidates:
        return None, None
        
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_candidate = candidates[0]
    return best_candidate[2], best_candidate[3]


def convert_targeted_dicom_to_nifti(dicom_dir: str, output_path: str) -> bool:
    """Extracts the matched CCTA series and writes to NIfTI."""
    se_num, info = select_best_ccta_series(dicom_dir)
    if se_num is None:
        print(f"   [ERROR] No suitable CCTA series found in {dicom_dir}!", flush=True)
        return False

    file_tuples = info['files']
    file_tuples.sort(key=lambda x: x[0])
    sorted_files = [f for _, f in file_tuples]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_files)
    image = reader.Execute()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(image, output_path)
    print(f"   [Stage 1] Matched Se #{se_num} ({info['desc']}) -> Saved {output_path} ({len(sorted_files)} slices, Spacing={image.GetSpacing()}, Size={image.GetSize()})", flush=True)
    return True





def process_case(case_id: int, model: RASNet, force_recompute=True):
    """Processes a single patient case through Stages 1 and 2."""
    case_name = f"CT{case_id}"
    dicom_dir = os.path.join(DICOM_ROOT, f"CT {case_id}", "A")
    case_out_dir = os.path.join(LOCAL_DATA_DIR, case_name)
    img_path = os.path.join(case_out_dir, "image.nii.gz")
    pred_path = os.path.join(case_out_dir, "pred_mask.nii.gz")

    print(f"\n==================== PROCESSING {case_name} ====================", flush=True)
    if not os.path.exists(dicom_dir):
        print(f"[ERROR] Missing DICOM folder: {dicom_dir}", flush=True)
        return False

    # Stage 1: DICOM -> NIfTI (Recompute with targeted CCTA series)
    if force_recompute or not os.path.exists(img_path):
        success = convert_targeted_dicom_to_nifti(dicom_dir, img_path)
        if not success:
            return False
    else:
        print(f"   [Stage 1] NIfTI image already exists: {img_path}", flush=True)

    # Stage 2: GPU Inference
    if force_recompute or not os.path.exists(pred_path):
        run_gpu_inference(model, img_path, pred_path)
    else:
        print(f"   [Stage 2] Prediction mask already exists: {pred_path}", flush=True)

    return True


def main():
    print("================================================================================", flush=True)
    print(">> HOSPITAL CLINICAL COHORT PIPELINE (TARGETED CCTA SERIES: CT61 - CT90)", flush=True)
    print(f"   Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"   VRAM Total: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB", flush=True)
    print(f"   Sliding Window Batch Size: {SW_BATCH_SIZE} (AMP FP16 Enabled)", flush=True)
    print(f"   Champion Checkpoint: {CKPT_PATH}", flush=True)
    print("================================================================================", flush=True)

    # Load Model on GPU
    print("Loading RASNet champion model onto GPU...", flush=True)
    model = RASNet(
        spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1
    )
    weights = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(weights, strict=True)
    model = model.to(DEVICE, memory_format=torch.channels_last_3d)
    model.eval()
    print("RASNet loaded successfully on RTX 3060 Ti GPU!\n", flush=True)

    # Process remaining cohort cases: CT78 to CT90
    target_ids = list(range(78, 91))
    success_count = 0

    for cid in target_ids:
        if process_case(cid, model, force_recompute=True):
            success_count += 1

    print(f"\n[DONE] Finished processing {success_count}/{len(target_ids)} cases successfully!", flush=True)


if __name__ == "__main__":
    main()
