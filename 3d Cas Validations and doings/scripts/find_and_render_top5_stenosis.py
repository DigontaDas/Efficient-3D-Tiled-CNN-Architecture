r"""
find_and_render_top5_stenosis.py — Render Top 5 Authentic Focal Stenoses
Identifies true in-vessel focal luminal narrowings (excluding boundary ends)
where both Ground Truth and RASNet accurately capture the stenosis contour,
and renders multi-panel clinical figures with spatial overlay and longitudinal curves.
"""

import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil

WORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(WORK_DIR, "results", "unseen_66_cohort")
DATASET_ROOT = r"H:\3D CT Images for Coronary Artery Segmentation (200 Samples)"
PRED_DIR = os.path.join(RESULTS_DIR, "predictions", "rasnet")
OUT_DIR = os.path.join(RESULTS_DIR, "stenosis_blocks", "top5_true_stenosis")
ARTIFACTS_DIR = r"C:\Users\Mehedi D Nafis\.gemini\antigravity-ide\brain\8d05635b-8fbc-476b-a200-a8af231abd07"

os.makedirs(OUT_DIR, exist_ok=True)


def find_candidates():
    print("[*] Scanning 66 unseen cases for authentic in-lumen focal stenoses...")
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "case_*_pred.nii.gz")))
    candidates = []

    for idx, pf in enumerate(pred_files, 1):
        cid = int(os.path.basename(pf).replace("case_", "").replace("_pred.nii.gz", ""))
        lbl_p = os.path.join(DATASET_ROOT, f"{cid}.label.nii", "label.nii")
        img_p = os.path.join(DATASET_ROOT, f"{cid}.img.nii", "dia_0.nii")

        if not os.path.exists(lbl_p) or not os.path.exists(img_p):
            continue

        gt_img = sitk.ReadImage(lbl_p)
        pred_img = sitk.ReadImage(pf)

        gt_arr = sitk.GetArrayFromImage(gt_img)
        pred_arr = sitk.GetArrayFromImage(pred_img)

        z_gt = gt_arr.sum(axis=(1, 2))
        z_pred = pred_arr.sum(axis=(1, 2))

        valid = np.where(z_gt > 15)[0]
        if len(valid) < 30:
            continue

        # Strictly exclude the first 15 and last 15 slices to eliminate boundary edge artifacts
        interior = valid[15:-15]
        if len(interior) < 10:
            continue

        for z in interior:
            gt_a = z_gt[z]
            pred_a = z_pred[z]
            if gt_a < 15 or pred_a < 15:
                continue

            # Local reference window (+/- 8 slices)
            w_start = max(valid[0], z - 8)
            w_end = min(valid[-1], z + 8)
            local_gt = z_gt[w_start:w_end]
            ref_gt = float(np.percentile(local_gt, 90))
            min_local_gt = float(np.min(local_gt))

            if abs(gt_a - min_local_gt) <= 2:  # Is a local constriction
                drop_ratio = (1.0 - (gt_a / max(ref_gt, 1.0))) * 100.0

                local_pred = z_pred[w_start:w_end]
                ref_pred = float(np.percentile(local_pred, 90)) if (local_pred > 0).any() else 1.0
                pred_drop = (1.0 - (pred_a / max(ref_pred, 1.0))) * 100.0

                # Focal narrowing between 35% and 85% area reduction
                if 35.0 <= drop_ratio <= 85.0 and 30.0 <= pred_drop <= 90.0:
                    slice_gt = gt_arr[z] > 0
                    slice_pred = pred_arr[z] > 0
                    inter = np.logical_and(slice_gt, slice_pred).sum()
                    dice_z = float(2.0 * inter / (slice_gt.sum() + slice_pred.sum() + 1e-6))

                    if dice_z >= 0.70:
                        candidates.append({
                            "case_id": cid,
                            "z": int(z),
                            "gt_area": int(gt_a),
                            "pred_area": int(pred_a),
                            "gt_stenosis_pct": round(drop_ratio, 1),
                            "pred_stenosis_pct": round(pred_drop, 1),
                            "abs_error_pct": round(abs(drop_ratio - pred_drop), 1),
                            "slice_dice": round(dice_z, 3),
                            "valid_slices": valid,
                            "z_gt_area": z_gt,
                            "z_pred_area": z_pred,
                            "img_p": img_p,
                            "lbl_p": lbl_p,
                            "pred_p": pf
                        })

    df = pd.DataFrame(candidates)
    if df.empty:
        print("[!] No focal narrowing candidates matched strict criteria.")
        return []

    # Sort by slice dice (best visual match) and lowest stenosis error
    df = df.sort_values(by=["slice_dice", "abs_error_pct"], ascending=[False, True])
    # Keep one best slice per case
    df = df.drop_duplicates(subset=["case_id"])
    print(f"[+] Found {len(df)} authentic focal stenosis candidate cases.")
    return df.head(5).to_dict("records")


def render_figure(cand: dict, rank: int):
    case_id = cand["case_id"]
    z = cand["z"]
    print(f"[*] Rendering Rank {rank}: Case {case_id} at focal stenosis slice Z={z} (Dice: {cand['slice_dice']}, GT Stenosis: {cand['gt_stenosis_pct']}%, Pred: {cand['pred_stenosis_pct']}%)")

    ct_img = sitk.ReadImage(cand["img_p"])
    gt_img = sitk.ReadImage(cand["lbl_p"])
    pred_img = sitk.ReadImage(cand["pred_p"])

    ct_arr = sitk.GetArrayFromImage(ct_img)
    gt_arr = sitk.GetArrayFromImage(gt_img)
    pred_arr = sitk.GetArrayFromImage(pred_img)

    ct_slice = ct_arr[z]
    ct_win = np.clip(ct_slice, -100, 800)
    ct_norm = (ct_win - (-100)) / (800 - (-100))

    gt_slice = gt_arr[z] > 0
    pred_slice = pred_arr[z] > 0

    pts = np.argwhere(gt_slice | pred_slice)
    if len(pts) > 0:
        y_min, x_min = pts.min(axis=0)
        y_max, x_max = pts.max(axis=0)
        pad = 40
        y_min = max(0, y_min - pad)
        y_max = min(ct_arr.shape[1], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(ct_arr.shape[2], x_max + pad)
    else:
        y_min, y_max = 150, 350
        x_min, x_max = 150, 350

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=220)

    # Panel 1: CT Slice
    axes[0].imshow(ct_norm[y_min:y_max, x_min:x_max], cmap="bone")
    axes[0].set_title(f"Case {case_id} — Axial CT\nSlice Z={z} (Cardiac Window)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Ground Truth Lumen
    axes[1].imshow(ct_norm[y_min:y_max, x_min:x_max], cmap="bone")
    if gt_slice[y_min:y_max, x_min:x_max].any():
        axes[1].contour(gt_slice[y_min:y_max, x_min:x_max], colors=["#00FF66"], linewidths=2.0)
    axes[1].set_title(f"Ground Truth Lumen\nConstriction Area Loss: {cand['gt_stenosis_pct']}%", fontsize=11, fontweight="bold", color="#008833")
    axes[1].axis("off")

    # Panel 3: RASNet Predicted Lumen
    axes[2].imshow(ct_norm[y_min:y_max, x_min:x_max], cmap="bone")
    if pred_slice[y_min:y_max, x_min:x_max].any():
        axes[2].contour(pred_slice[y_min:y_max, x_min:x_max], colors=["#FF2255"], linewidths=2.0)
    axes[2].set_title(f"RASNet Predicted Lumen\nDetected Area Loss: {cand['pred_stenosis_pct']}%", fontsize=11, fontweight="bold", color="#CC0033")
    axes[2].axis("off")

    # Panel 4: Spatial Co-Registration (Overlay)
    axes[3].imshow(ct_norm[y_min:y_max, x_min:x_max], cmap="bone")
    if gt_slice[y_min:y_max, x_min:x_max].any():
        axes[3].contour(gt_slice[y_min:y_max, x_min:x_max], colors=["#00FF66"], linewidths=2.0)
    if pred_slice[y_min:y_max, x_min:x_max].any():
        axes[3].contour(pred_slice[y_min:y_max, x_min:x_max], colors=["#FF2255"], linewidths=1.5, linestyles="--")
    axes[3].set_title(f"Spatial Alignment Overlay\nSlice Dice: {cand['slice_dice']:.3f} (Green=GT, Red=RASNet)", fontsize=11, fontweight="bold", color="#1155CC")
    axes[3].axis("off")

    # Panel 5: Longitudinal Lumen Area Profile
    valid_z = cand["valid_slices"]
    gt_curve = cand["z_gt_area"][valid_z]
    pred_curve = cand["z_pred_area"][valid_z]

    axes[4].plot(valid_z, gt_curve, label="Ground Truth", color="#00AA44", linewidth=2.2)
    axes[4].plot(valid_z, pred_curve, label="RASNet Prediction", color="#FF2255", linewidth=2.0, linestyle="--")
    axes[4].axvline(z, color="black", linestyle=":", linewidth=2.0, label=f"Stenosis Locus (Z={z})")
    axes[4].set_title(f"Longitudinal Lumen Area Profile\nStenosis Δ Error: {cand['abs_error_pct']}%", fontsize=11, fontweight="bold")
    axes[4].set_xlabel("Slice Index (Z)", fontsize=10)
    axes[4].set_ylabel("Lumen Area (Voxels)", fontsize=10)
    axes[4].legend(loc="upper right", frameon=True, fontsize=8)
    axes[4].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, f"top{rank}_case_{case_id}_focal_stenosis.png")
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close()

    # Also copy to artifacts directory for markdown embedding
    artifact_copy = os.path.join(ARTIFACTS_DIR, f"top{rank}_case_{case_id}_focal_stenosis.png")
    shutil.copy2(out_file, artifact_copy)
    cand["output_png"] = out_file
    cand["artifact_png"] = artifact_copy
    return cand


def main():
    top5 = find_candidates()
    if not top5:
        return

    rendered = []
    for rank, cand in enumerate(top5, 1):
        r = render_figure(cand, rank)
        rendered.append(r)

    summary_csv = os.path.join(OUT_DIR, "top5_stenosis_summary.csv")
    pd.DataFrame(rendered)[["case_id", "z", "slice_dice", "gt_stenosis_pct", "pred_stenosis_pct", "abs_error_pct", "output_png"]].to_csv(summary_csv, index=False)
    print(f"\n[+] Successfully rendered all top 5 focal stenosis comparisons -> {OUT_DIR}\n")


if __name__ == "__main__":
    main()
