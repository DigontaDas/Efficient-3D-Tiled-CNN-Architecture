import os
from PIL import Image
import numpy as np
import pandas as pd

OUT = r"H:\Thesis_Trainings\Phase3_Local_Integration\extracted_badges"

# Load reference templates
t_25 = Image.open(os.path.join(OUT, "CT65_65.1_b334.png")).convert("L")
t_50 = Image.open(os.path.join(OUT, "CT61_61.1_b7.png")).convert("L")
t_70 = Image.open(os.path.join(OUT, "CT63_63.2_b395.png")).convert("L")

target_size = (120, 50)
t_25_arr = np.array(t_25.resize(target_size)) < 128
t_50_arr = np.array(t_50.resize(target_size)) < 128
t_70_arr = np.array(t_70.resize(target_size)) < 128

templates = {
    "25-49%": t_25_arr,
    "50-69%": t_50_arr,
    "70-99%": t_70_arr
}

def match_badge(img_path):
    im = Image.open(img_path).convert("L")
    arr = np.array(im.resize(target_size)) < 128
    
    # Check if there is actual text
    if np.sum(arr) < 50:
        return "BLANK / NONE"
        
    best_match = None
    best_score = -1.0
    for label, t_arr in templates.items():
        # Intersection over union of black text pixels
        intersection = np.logical_and(arr, t_arr).sum()
        union = np.logical_or(arr, t_arr).sum()
        iou = intersection / union if union > 0 else 0
        if iou > best_score:
            best_score = iou
            best_match = label
            
    return best_match if best_score > 0.35 else f"UNKNOWN (IoU={best_score:.2f})"

print(f"{'Badge File':<28} | {'Recognized Label'}")
print("-" * 50)

results = {}
for f in sorted(os.listdir(OUT)):
    if not f.endswith(".png"): continue
    res = match_badge(os.path.join(OUT, f))
    print(f"{f:<28} | {res}")
    cname = f.split("_")[0]
    if cname not in results:
        results[cname] = []
    if "UNKNOWN" not in res and "BLANK" not in res:
        results[cname].append(res)

print("\n" + "=" * 50)
print("CASE-LEVEL BADGE SUMMARY:")
print("=" * 50)
for cname, badges in sorted(results.items()):
    unique = list(set(badges))
    print(f"{cname}: {unique}")
