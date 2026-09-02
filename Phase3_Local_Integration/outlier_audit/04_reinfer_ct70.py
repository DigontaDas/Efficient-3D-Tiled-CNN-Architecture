"""
04_reinfer_ct70.py
==================
Re-run GPU inference on CT70 using the correctly converted Series 108 NIfTI.
CT70 image.nii.gz was already overwritten in 03_apply_case_corrections.py with Series 108 (365 slices).
"""

import os, sys, time, torch, numpy as np, SimpleITK as sitk
import monai.transforms as mt
from monai.inferers import sliding_window_inference
import cc3d

sys.path.insert(0, r'H:\Thesis_Trainings\Phase3_Local_Integration')
from rasnet_model import RASNet

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
SW_BATCH_SIZE = 2

CKPT_PATH = r"H:\Thesis_Trainings\results-after-hallucin-fix\checkpoints\rasnet_best.pth"
IMG_PATH   = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data\CT70\image.nii.gz"
PRED_PATH  = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data\CT70\pred_mask.nii.gz"

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
    valid_components = [(cid, sizes[cid]) for cid in range(1, len(sizes)) if sizes[cid] >= min_size]
    valid_components.sort(key=lambda x: x[1], reverse=True)
    top_components = valid_components[:2]
    cleaned = np.zeros_like(pred_mask, dtype=np.uint8)
    for cid, _ in top_components:
        cleaned[labeled == cid] = 1
    return cleaned

print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CT70 image path: {IMG_PATH}")
print(f"CT70 image size: {sitk.ReadImage(IMG_PATH).GetSize()}")

model = RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)
weights = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(weights, strict=True)
model = model.to(DEVICE, memory_format=torch.channels_last_3d)
model.eval()
print("RASNet loaded on GPU.")

start_t = time.time()
torch.cuda.empty_cache()
batch = pre_trans({"image": IMG_PATH})
input_tensor = batch["image"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    with torch.amp.autocast('cuda'):
        logits = sliding_window_inference(input_tensor, roi_size=PATCH_SIZE, sw_batch_size=SW_BATCH_SIZE, predictor=model, overlap=0.5)
        prob_avg = torch.softmax(logits, dim=1)
        for flip_dim in (2, 3, 4):
            input_flip = torch.flip(input_tensor, dims=[flip_dim]).contiguous()
            logits_flip = sliding_window_inference(input_flip, roi_size=PATCH_SIZE, sw_batch_size=SW_BATCH_SIZE, predictor=model, overlap=0.5)
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

original_img = sitk.ReadImage(IMG_PATH)
pred_mask_sitk = np.transpose(pred_mask_cleaned, (2, 1, 0))
pred_sitk = sitk.GetImageFromArray(pred_mask_sitk)
pred_sitk.CopyInformation(original_img)
sitk.WriteImage(pred_sitk, PRED_PATH)

elapsed = time.time() - start_t
print(f"\n[DONE] CT70 pred_mask saved: {PRED_PATH}")
print(f"       Elapsed: {elapsed:.1f}s | Foreground Voxels: {int(np.sum(pred_mask_cleaned))}")
