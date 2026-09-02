import os
from PIL import Image
import numpy as np
import scipy.ndimage as ndi

ROOT = r"H:\Thesis_CT_scans_Labeled"
OUT = r"H:\Thesis_Trainings\Phase3_Local_Integration\extracted_badges"
os.makedirs(OUT, exist_ok=True)

count = 0
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
        white_mask = (arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235)
        labeled, num = ndi.label(white_mask)
        if num == 0:
            continue
        sizes = ndi.sum(white_mask, labeled, range(1, num + 1))
        for i, sz in enumerate(sizes):
            if sz > 600:
                coords = np.argwhere(labeled == (i + 1))
                ymin, xmin = coords.min(axis=0)
                ymax, xmax = coords.max(axis=0)
                h = ymax - ymin
                w = xmax - xmin
                # Badge aspect ratio: usually w > h, e.g. 80x40 or 150x60
                if 15 < h < 200 and 35 < w < 400:
                    badge_crop = im.crop((max(0, xmin - 2), max(0, ymin - 2), min(im.width, xmax + 2), min(im.height, ymax + 2)))
                    bname = f"{cname}_{img_name[:-4]}_b{i+1}.png"
                    badge_crop.save(os.path.join(OUT, bname))
                    count += 1

print(f"Done! Extracted {count} badges to {OUT}")
