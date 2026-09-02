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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 2  # Optimized for 8 GB VRAM RTX 3060 Ti

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_PATH = os.path.join(BASE_DIR, "results-after-hallucin-fix", "checkpoints", "rasnet_best.pth")
DICOM_ROOT = r"H:\CT_Scans_Thesis"
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "local_data")

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

    batch["pred"] = prob_avg.squeeze(0)
    batch = post_trans(batch)

    del input_tensor, logits, prob_avg
    torch.cuda.empty_cache()

    pred_probs = batch["pred"]
    fg_prob = pred_probs[1]
    pred_mask = (fg_prob > 0.55).cpu().numpy().astype(np.uint8)
    pred_mask_cleaned = topological_postprocess(pred_mask, min_size=50)

    original_img = sitk.ReadImage(img_path)
    pred_mask_sitk = np.transpose(pred_mask_cleaned, (2, 1, 0)) # (Z, Y, X)
    pred_sitk = sitk.GetImageFromArray(pred_mask_sitk)
    pred_sitk.CopyInformation(original_img)

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    sitk.WriteImage(pred_sitk, pred_path)

    elapsed = time.time() - start_t
    print(f"   [Stage 2] RTX 3060 Ti GPU Inference: Saved {pred_path} in {elapsed:.2f}s (Foreground Voxels: {int(np.sum(pred_mask_cleaned))})", flush=True)

def select_best_ccta_series(dicom_dir: str):
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
        if any(kw in desc for kw in EXCLUDE_KEYWORDS) or count < 50:
            continue
        score = 0
        if 'ss-freeze' in desc or 'freeze' in desc:
            score += 100
        elif 'temporal' in desc:
            score += 80
        elif 'sseg' in desc and '0-90' not in desc:
            score += 70
        elif 'coronary' in desc:
            score += 50
        if '75%' in desc:
            score += 20
        elif '45%' in desc:
            score += 10
        if thick <= 0.75:
            score += 15
        candidates.append((score, count, se_num, info))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[0]
    return best[2], best[3]

def convert_targeted_dicom_to_nifti(dicom_dir: str, output_nii: str):
    se_num, se_info = select_best_ccta_series(dicom_dir)
    if se_num is None:
        print(f"   [ERROR] No valid CCTA series found in {dicom_dir}", flush=True)
        return False
    print(f"   [Stage 1] Matched CCTA Series #{se_num}: '{se_info['desc']}' ({len(se_info['files'])} slices, thick={se_info['thick']}mm)", flush=True)
    sorted_files = [f for _, f in sorted(se_info['files'], key=lambda x: x[0])]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_files)
    image = reader.Execute()
    os.makedirs(os.path.dirname(output_nii), exist_ok=True)
    sitk.WriteImage(image, output_nii)
    print(f"   [Stage 1] Converted and saved: {output_nii} (Size: {image.GetSize()}, Spacing: {image.GetSpacing()})", flush=True)
    return True

def process_case(cid: int, model: RASNet):
    cname = f"CT{cid}"
    sub_a = os.path.join(DICOM_ROOT, f"CT {cid}", "A")
    dicom_dir = sub_a if os.path.exists(sub_a) else os.path.join(DICOM_ROOT, f"CT {cid}")
    case_out_dir = os.path.join(LOCAL_DATA_DIR, cname)
    img_path = os.path.join(case_out_dir, "image.nii.gz")
    pred_path = os.path.join(case_out_dir, "pred_mask.nii.gz")

    print(f"\n==================== PROCESSING {cname} ====================", flush=True)
    if not os.path.exists(dicom_dir):
        print(f"[ERROR] Missing DICOM folder: {dicom_dir}", flush=True)
        return False

    success = convert_targeted_dicom_to_nifti(dicom_dir, img_path)
    if not success:
        return False

    run_gpu_inference(model, img_path, pred_path)
    return True

def main():
    print("================================================================================", flush=True)
    print(">> RUNNING GPU INFERENCE ON NEW DATASETS: CT1, CT4, CT5", flush=True)
    print(f"   Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"   Checkpoint: {CKPT_PATH}", flush=True)
    print("================================================================================", flush=True)

    model = RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)
    weights = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(weights, strict=True)
    model = model.to(DEVICE, memory_format=torch.channels_last_3d)
    model.eval()
    print("Loaded RASNet on RTX 3060 Ti GPU!\n", flush=True)

    for cid in [1, 4, 5]:
        process_case(cid, model)

    print("\n[ALL DONE] CT1, CT4, CT5 processed successfully on GPU!", flush=True)

if __name__ == "__main__":
    main()
