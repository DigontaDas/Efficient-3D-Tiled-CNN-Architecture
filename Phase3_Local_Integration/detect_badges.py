from PIL import Image
import numpy as np
import os

ROOT = r"H:\Thesis_CT_scans_Labeled"

print(f"{'Case':<6} | {'Image':<10} | {'Shape':<12} | {'Green Pixels':<14} | {'White Pixels':<14} | {'Has Badge?'}")
print("-" * 75)

for cid in range(61, 91):
    cname = f"CT{cid}"
    fdir = os.path.join(ROOT, cname)
    if not os.path.exists(fdir):
        continue
    for img_name in sorted(os.listdir(fdir)):
        if not img_name.endswith(".png"):
            continue
        ipath = os.path.join(fdir, img_name)
        im = Image.open(ipath).convert("RGB")
        arr = np.array(im)
        green_mask = (arr[:, :, 1] > 170) & (arr[:, :, 0] < 100) & (arr[:, :, 2] < 170)
        white_mask = (arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245)
        n_green = int(np.sum(green_mask))
        n_white = int(np.sum(white_mask))
        has_badge = (n_green > 100 or n_white > 1000)
        print(f"{cname:<6} | {img_name:<10} | {str(arr.shape):<12} | {n_green:<14} | {n_white:<14} | {has_badge}")
