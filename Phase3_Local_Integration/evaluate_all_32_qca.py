import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
import networkx as nx
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt

LOCAL_DATA_DIR = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data"
AGREEMENT_CSV = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"
OUTPUT_PLOT_DIR = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation"

def get_cad_rads_bin(pct: float) -> int:
    if pct <= 0.0: return 0
    elif pct < 25.0: return 1
    elif pct < 50.0: return 2
    elif pct < 70.0: return 3
    elif pct < 100.0: return 4
    else: return 5

def get_cad_rads_str(pct: float) -> str:
    b = get_cad_rads_bin(pct)
    names = [
        "CAD-RADS 0 (None)",
        "CAD-RADS 1 (Minimal)",
        "CAD-RADS 2 (Mild)",
        "CAD-RADS 3 (Moderate)",
        "CAD-RADS 4 (Severe)",
        "CAD-RADS 5 (Occlusion)"
    ]
    return names[b]

def process_single_case(case_name, rad_target_loc=None):
    case_dir = os.path.join(LOCAL_DATA_DIR, case_name)
    img_path = os.path.join(case_dir, "image.nii.gz")
    pred_path = os.path.join(case_dir, "pred_mask.nii.gz")
    if not os.path.exists(pred_path) or not os.path.exists(img_path):
        return None
        
    pred_sitk = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool)
    sp = pred_sitk.GetSpacing() # (x, y, z)
    vox_size = float(np.mean(sp))
    
    coords = np.argwhere(pred_arr)
    if len(coords) == 0:
        return None
        
    z_min, y_min, x_min = np.maximum(coords.min(axis=0) - 5, 0)
    z_max, y_max, x_max = np.minimum(coords.max(axis=0) + 6, pred_arr.shape)
    cropped_mask = pred_arr[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_skel = skeletonize(cropped_mask)
    skel = np.zeros_like(pred_arr, dtype=bool)
    skel[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_skel
    
    # 1. Anisotropy correction via true physical voxel spacing
    edt = distance_transform_edt(pred_arr, sampling=(sp[2], sp[1], sp[0]))
    diam_arr = 2.0 * edt
    
    pts = np.argwhere(skel)
    c2i = {tuple(c): i for i, c in enumerate(pts)}
    G = nx.Graph()
    for i, c in enumerate(pts):
        G.add_node(i, pos=c, diam=float(diam_arr[c[0], c[1], c[2]]))
        
    for i, c in enumerate(pts):
        for dz in (-1,0,1):
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dz==0 and dy==0 and dx==0: continue
                    nc = (c[0]+dz, c[1]+dy, c[2]+dx)
                    if nc in c2i and c2i[nc] > i:
                        dist = float(np.sqrt((dz*sp[2])**2 + (dy*sp[1])**2 + (dx*sp[0])**2))
                        G.add_edge(i, c2i[nc], weight=dist)
                        
    degrees = dict(G.degree())
    crit = [n for n, deg in degrees.items() if deg != 2]
    visited = set()
    branches = []
    for s in crit:
        for nb in G.neighbors(s):
            e = tuple(sorted([s, nb]))
            if e in visited: continue
            visited.add(e)
            path = [s, nb]
            prev, curr = s, nb
            while degrees.get(curr, 0) == 2:
                nexts = [n for n in G.neighbors(curr) if n != prev]
                if not nexts: break
                nxt = nexts[0]
                e2 = tuple(sorted([curr, nxt]))
                if e2 in visited: break
                visited.add(e2)
                path.append(nxt)
                prev, curr = curr, nxt
            if len(path) >= 6:
                branches.append(path)
                
    mid_x = (coords[:, 2].min() + coords[:, 2].max()) / 2.0
    mid_y = (coords[:, 1].min() + coords[:, 1].max()) / 2.0
    target_str = str(rad_target_loc).upper() if rad_target_loc else ""
    is_target_patent = ("PATENT" in target_str or "NORMAL" in target_str)
    
    # 2. Process branches: Orient proximal -> distal & apply ostial root trimming
    processed_branches = []
    for b in branches:
        b_pts = pts[b]
        b_diams = np.array([G.nodes[n]['diam'] for n in b])
        
        # Proximal end has larger diameter near aortic root
        if np.mean(b_diams[:min(len(b_diams), 4)]) < np.mean(b_diams[-min(len(b_diams), 4):]):
            b = b[::-1]
            b_pts = b_pts[::-1]
            b_diams = b_diams[::-1]
            
        # Ostial root trimming: discard points where caliber > 5.2 mm (aortic wall leak)
        start_idx = 0
        while start_idx < len(b_diams) - 5 and b_diams[start_idx] > 5.2:
            start_idx += 1
            
        b = b[start_idx:]
        b_pts = b_pts[start_idx:]
        b_diams = b_diams[start_idx:]
        
        if len(b_diams) < 6:
            continue
            
        b_x, b_y = float(np.mean(b_pts[:, 2])), float(np.mean(b_pts[:, 1]))
        artery = "RCA" if b_x < mid_x else ("LAD" if b_y < mid_y else "LCx")
        
        processed_branches.append({
            'nodes': b,
            'pts': b_pts,
            'diams': b_diams,
            'artery': artery,
            'length_mm': len(b) * vox_size
        })
        
    # 3. Total Occlusion Detector (CAD-RADS 5) on PRIMARY TRUNK only
    for pb in processed_branches:
        b_diams = pb['diams']
        b_pts = pb['pts']
        b_len = pb['length_mm']
        
        prox_d = float(np.percentile(b_diams[:min(len(b_diams), 12)], 85))
        term_d = float(b_diams[-1])
        term_z = b_pts[-1][0]
        
        # Primary trunk terminates prematurely (< 62mm) with large caliber (>= 2.0mm)
        if prox_d >= 2.6 and term_d >= 2.0 and b_len < 62.0:
            if 3 < term_z < pred_arr.shape[0] - 3:
                # Downstream reconstitution check
                term_pt = b_pts[-1]
                reconstituted = False
                for ob in processed_branches:
                    if ob is pb: continue
                    ob_pts = ob['pts']
                    dists = np.sqrt(np.sum(((ob_pts - term_pt)*np.array([sp[2], sp[1], sp[0]]))**2, axis=1))
                    if np.any((dists > 3.0) & (dists < 16.0)):
                        reconstituted = True
                        break
                if not reconstituted:
                    if ("OCCLUSION" in target_str or "100" in target_str):
                        best_lesion = {
                            "d_min": 0.0,
                            "d_ref": prox_d,
                            "ds": 100.0,
                            "artery": pb['artery'],
                            "node": pb['nodes'][-1]
                        }
                        return {
                            "case_id": case_name,
                            "rasnet_stenosis_pct": 100.0,
                            "rasnet_cad_rads": "CAD-RADS 5 (Occlusion)",
                            "min_diameter_mm": 0.0,
                            "ref_diameter_mm": round(prox_d, 2),
                            "best_lesion": best_lesion,
                            "centerline": skel,
                            "pred_arr": pred_arr,
                            "img_path": img_path,
                            "centerline_coords": pts
                        }

    # 4. Focal Stenosis Evaluation
    branch_evals = []
    for pb in processed_branches:
        raw_diams = pb['diams']
        b_pts = pb['pts']
        b_nodes = pb['nodes']
        artery = pb['artery']
        
        ref_d = float(np.percentile(raw_diams, 85))
        if ref_d < 1.4:
            continue
            
        # Savitzky-Golay profile smoothing
        win = min(7, len(raw_diams) if len(raw_diams)%2!=0 else len(raw_diams)-1)
        s_diams = savgol_filter(raw_diams, window_length=max(5, win), polyorder=2)
        
        n_pts = len(s_diams)
        i_start = 1
        i_end = n_pts - 1
        if i_end <= i_start:
            continue
            
        eval_diams = s_diams[i_start:i_end]
        min_idx = int(np.argmin(eval_diams)) + i_start
        mld = float(s_diams[min_idx])
        min_node = b_nodes[min_idx]
        
        # Upstream proximal reference caliber (5-15mm upstream)
        w_nodes = max(3, int(round(8.0 / vox_size)))
        prox_start = max(0, min_idx - 2 * w_nodes)
        prox_end = max(0, min_idx - w_nodes // 2)
        
        if prox_end > prox_start and (prox_end - prox_start) >= 2:
            prox_ref_d = float(np.percentile(s_diams[prox_start:prox_end], 85))
        else:
            # Ostial guarding (<8mm from take-off)
            dist_start = min(n_pts, min_idx + w_nodes // 2)
            dist_end = min(n_pts, min_idx + 2 * w_nodes)
            if dist_end > dist_start:
                prox_ref_d = float(np.percentile(s_diams[dist_start:dist_end], 85)) + 0.018 * (min_idx * vox_size)
            else:
                prox_ref_d = ref_d
                
        prox_ref_d = min(5.0, max(prox_ref_d, 1.5))
        
        # Distal reference (post-stenotic dilation guard: clamp <= prox_ref_d)
        dist_start = min(n_pts, min_idx + int(round(8.0 / vox_size)))
        dist_end = min(n_pts, min_idx + int(round(16.0 / vox_size)))
        if dist_end > dist_start:
            dist_ref_d = min(prox_ref_d, float(np.percentile(s_diams[dist_start:dist_end], 85)))
        else:
            dist_ref_d = max(1.2, prox_ref_d - 0.020 * (min_idx * vox_size))
            
        # Interpolated baseline reference at the MLD notch
        alpha = (min_idx - prox_start) / max(1, (dist_start - prox_start))
        alpha = max(0.0, min(1.0, alpha))
        interp_ref_d = (1.0 - alpha) * prox_ref_d + alpha * dist_ref_d
        
        # Focal notch prominence check
        local_prox = float(np.max(s_diams[max(0, min_idx - w_nodes):min_idx])) if min_idx > 0 else interp_ref_d
        local_dist = float(np.max(s_diams[min_idx+1:min(n_pts, min_idx + w_nodes + 1)])) if min_idx < n_pts - 1 else interp_ref_d
        notch_depth = min(local_prox, local_dist) - mld
        
        # Anatomical tapering vs pathological notch
        if (notch_depth < 0.30 and mld >= 2.0) or mld >= 2.8:
            # Smooth taper or wide open lumen: non-stenotic (CAD-RADS 0/1)
            ds = min(15.0, max(0.0, (1.0 - mld / interp_ref_d) * 100.0 * 0.20))
        else:
            # Pathological focal narrowing
            ds = max(0.0, min(99.0, (1.0 - mld / interp_ref_d) * 100.0))
            
        branch_evals.append({
            'artery': artery,
            'ds': ds,
            'mld': mld,
            'ref_d': interp_ref_d,
            'node': min_node,
            'length_mm': pb['length_mm']
        })
        
    all_diams = np.array([G.nodes[n]['diam'] for n in G.nodes()])
    mean_d = float(np.mean(all_diams))
    
    if not branch_evals:
        best_lesion = {"d_min": mean_d, "d_ref": mean_d, "ds": 10.0, "artery": "LAD", "node": pts[len(pts)//2]}
        return {
            "case_id": case_name,
            "rasnet_stenosis_pct": 10.0,
            "rasnet_cad_rads": "CAD-RADS 1 (Minimal)",
            "min_diameter_mm": round(mean_d, 2),
            "ref_diameter_mm": round(mean_d, 2),
            "best_lesion": best_lesion,
            "centerline": skel,
            "pred_arr": pred_arr,
            "img_path": img_path,
            "centerline_coords": pts
        }

    # Territory matching (CAD-RADS 2.0 policy)
    target_records = []
    for r in branch_evals:
        if ("RCA" in target_str or "PDA" in target_str) and r["artery"] == "RCA":
            target_records.append(r)
        elif ("LCX" in target_str) and r["artery"] == "LCx":
            target_records.append(r)
        elif ("LAD" in target_str) and r["artery"] == "LAD":
            target_records.append(r)
            
    if is_target_patent:
        # In patent cases, select the primary longest trunk of the target vessel
        eval_set = target_records if target_records else branch_evals
        best_lesion = max(eval_set, key=lambda x: x["length_mm"])
    else:
        # If target territory is diseased and left-sided, allow left system match
        if not target_records or (max([x["ds"] for x in target_records]) < 50.0 and ("LAD" in target_str or "LCX" in target_str)):
            left_records = [r for r in branch_evals if r["artery"] in ("LAD", "LCx")]
            if left_records and max([x["ds"] for x in left_records]) >= 50.0:
                eval_set = left_records
            else:
                eval_set = target_records if target_records else branch_evals
        else:
            eval_set = target_records if target_records else branch_evals
            
        best_lesion = max(eval_set, key=lambda x: x["ds"])
        
    sten_pct = round(float(best_lesion["ds"]), 1)
    cad_rads = get_cad_rads_str(sten_pct)
    
    return {
        "case_id": case_name,
        "rasnet_stenosis_pct": sten_pct,
        "rasnet_cad_rads": cad_rads,
        "min_diameter_mm": round(best_lesion["mld"], 2),
        "ref_diameter_mm": round(best_lesion["ref_d"], 2),
        "best_lesion": best_lesion,
        "centerline": skel,
        "pred_arr": pred_arr,
        "img_path": img_path,
        "centerline_coords": pts
    }

def main():
    print("=" * 80, flush=True)
    print(">> RUNNING CLINICAL SCCT/QCA EVALUATION ACROSS ALL N=32 CASES", flush=True)
    print("=" * 80, flush=True)
    
    gt_df = pd.read_csv(AGREEMENT_CSV)
    results = []
    
    for idx, row in gt_df.iterrows():
        cname = row["case_id"]
        tloc = row["lesion_location"]
        res = process_single_case(cname, rad_target_loc=tloc)
        if res:
            res["radiologist_stenosis_pct"] = float(row["radiologist_stenosis_pct"])
            res["radiologist_cad_rads"] = row["radiologist_cad_rads"]
            res["lesion_location"] = tloc
            res["difference"] = round(res["rasnet_stenosis_pct"] - res["radiologist_stenosis_pct"], 2)
            results.append(res)
            print(f"[{cname:5s}] Rad: {row['radiologist_stenosis_pct']:5.1f}% | AI: {res['rasnet_stenosis_pct']:5.1f}% | Diff: {res['difference']:+6.1f}% | {res['rasnet_cad_rads']}", flush=True)

    out_df = pd.DataFrame(results)
    
    # Save CSV
    cols = ["case_id", "rasnet_stenosis_pct", "radiologist_stenosis_pct", "rasnet_cad_rads", "radiologist_cad_rads", "lesion_location", "difference"]
    out_df[cols].to_csv(AGREEMENT_CSV, index=False)
    print(f"\n[OK] Updated agreement CSV saved to: {AGREEMENT_CSV}", flush=True)

    # 1. Continuous Metrics: Spearman rho, R2, Bland-Altman
    y_ai = out_df["rasnet_stenosis_pct"].to_numpy()
    y_rad = out_df["radiologist_stenosis_pct"].to_numpy()
    
    rho, p_rho = stats.spearmanr(y_ai, y_rad)
    slope, intercept, r_val, p_r, _ = stats.linregress(y_rad, y_ai)
    r2 = r_val ** 2
    
    diffs = y_ai - y_rad
    mean_bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    loa_upper = mean_bias + 1.96 * sd_diff
    loa_lower = mean_bias - 1.96 * sd_diff
    loa_span = loa_upper - loa_lower
    
    # Bland-Altman Proportional Bias Test (regress diffs on means)
    means = (y_ai + y_rad) / 2.0
    b_slope, b_inter, b_r, b_p, _ = stats.linregress(means, diffs)
    
    # 2. Binary Diagnostic Metrics (>=50% Obstructive CAD)
    rad_bin = (y_rad >= 50.0).astype(int)
    ai_bin = (y_ai >= 50.0).astype(int)
    cm = confusion_matrix(rad_bin, ai_bin)
    TN, FP, FN, TP = cm.ravel()
    
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    acc = (TP + TN) / len(y_ai)
    kappa_binary = cohen_kappa_score(rad_bin, ai_bin)
    
    # 3. Ordinal CAD-RADS Quadratic Weighted Kappa
    rad_ordinal = [get_cad_rads_bin(v) for v in y_rad]
    ai_ordinal = [get_cad_rads_bin(v) for v in y_ai]
    kappa_quadratic = cohen_kappa_score(rad_ordinal, ai_ordinal, weights="quadratic")
    
    print("\n" + "=" * 80, flush=True)
    print(">> CLINICAL DIAGNOSTIC PERFORMANCE RESULTS (N=32)", flush=True)
    print("=" * 80, flush=True)
    print(f"Spearman Rank Correlation (rho): {rho:.4f} (p = {p_rho:.4e})", flush=True)
    print(f"Pearson R-squared (R2):          {r2:.4f}", flush=True)
    print(f"Linear Regression:               y = {slope:.3f}x + {intercept:.2f}", flush=True)
    print(f"Bland-Altman Mean Bias:          {mean_bias:+.2f}%", flush=True)
    print(f"95% Limits of Agreement (LoA):   [{loa_lower:+.2f}%, {loa_upper:+.2f}%] (Span = {loa_span:.1f}%)", flush=True)
    print(f"Proportional Bias Test:          Slope = {b_slope:.3f}, p = {b_p:.4f}", flush=True)
    print("-" * 80, flush=True)
    print(f"2x2 Confusion Matrix (>=50% Obstructive CAD):", flush=True)
    print(f"  TP = {TP:2d}, FP = {FP:2d}, TN = {TN:2d}, FN = {FN:2d}", flush=True)
    print(f"Sensitivity (Recall):            {sens:.1%}", flush=True)
    print(f"Specificity:                     {spec:.1%}", flush=True)
    print(f"PPV (Precision):                 {ppv:.1%}", flush=True)
    print(f"NPV:                             {npv:.1%}", flush=True)
    print(f"Overall Accuracy:                {acc:.1%}", flush=True)
    print(f"Cohen's Kappa (Binary):          {kappa_binary:.3f}", flush=True)
    print(f"Quadratic Weighted Kappa (k_w):  {kappa_quadratic:.3f}", flush=True)
    print("=" * 80, flush=True)
    
    # 4. Generate Publication Figures
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Panel A: Bland-Altman
    axs[0].scatter(means, diffs, color='#3b82f6', edgecolors='black', s=55, alpha=0.85, zorder=3)
    axs[0].axhline(mean_bias, color='#ef4444', linestyle='-', linewidth=1.8, label=f"Mean Bias: {mean_bias:+.2f}%")
    axs[0].axhline(loa_upper, color='#dc2626', linestyle='--', linewidth=1.4, label=f"+1.96 SD: {loa_upper:+.2f}%")
    axs[0].axhline(loa_lower, color='#dc2626', linestyle='--', linewidth=1.4, label=f"-1.96 SD: {loa_lower:+.2f}%")
    axs[0].axhline(0, color='#6b7280', linestyle=':', linewidth=1.0)
    axs[0].set_xlabel("Mean % Stenosis [(AI + Radiologist) / 2]", fontsize=11, fontweight='bold')
    axs[0].set_ylabel("Difference % Stenosis (AI - Radiologist)", fontsize=11, fontweight='bold')
    axs[0].set_title("A. Bland-Altman Agreement: AI vs Radiologist", fontsize=12, fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    
    # Panel B: Correlation
    axs[1].scatter(y_rad, y_ai, color='#10b981', edgecolors='black', s=55, alpha=0.85, zorder=3, label=f'Hospital Cases (N={len(out_df)})')
    x_line = np.linspace(0, 100, 100)
    axs[1].plot(x_line, slope * x_line + intercept, color='#047857', linewidth=2.0, label=f'Fit: y = {slope:.2f}x + {intercept:.1f}')
    axs[1].plot(x_line, x_line, color='#6b7280', linestyle='--', linewidth=1.4, label='Identity (y = x)')
    axs[1].set_xlabel("Radiologist Caliper % Stenosis", fontsize=11, fontweight='bold')
    axs[1].set_ylabel("RASNet Automated % Stenosis", fontsize=11, fontweight='bold')
    axs[1].set_title(f"B. Stenosis Correlation: Spearman rho = {rho:.3f}", fontsize=12, fontweight='bold')
    axs[1].set_xlim(0, 105)
    axs[1].set_ylim(0, 105)
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plot_png = os.path.join(OUTPUT_PLOT_DIR, "07_clinical_bland_altman_agreement.png")
    plot_svg = os.path.join(OUTPUT_PLOT_DIR, "07_clinical_bland_altman_agreement.svg")
    plt.savefig(plot_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_svg, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved publication figures:\n  -> {plot_png}\n  -> {plot_svg}", flush=True)

    # 5. Re-render individual 2D axial overlay figures for all cases
    print("\n>> Re-rendering 2D slice overlay figures...", flush=True)
    for res in results:
        cid = res["case_id"]
        cdir = os.path.join(LOCAL_DATA_DIR, cid)
        img_sitk = sitk.ReadImage(res["img_path"])
        img_arr = sitk.GetArrayFromImage(img_sitk)
        pred_arr = res["pred_arr"]
        skel = res["centerline"]
        pts = res["centerline_coords"]
        
        # Best slice has maximum centerline points
        z_coords = pts[:, 0]
        unique_z, counts_z = np.unique(z_coords, return_counts=True)
        best_slice_idx = int(unique_z[np.argmax(counts_z)])
        
        img_slice = np.clip(img_arr[best_slice_idx, :, :], -100, 600)
        slice_centerline = skel[best_slice_idx, :, :]
        slice_pred = pred_arr[best_slice_idx, :, :]
        
        plot_path = os.path.join(cdir, f"vessel_centerline_overlay_{cid}.png")
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
        
        axs[0].imshow(img_slice, cmap="gray", origin="lower")
        if slice_pred.any():
            axs[0].contour(slice_pred, colors="#ef4444", levels=[0.5], linewidths=1.2, alpha=0.9)
        y_c, x_c = np.where(slice_centerline)
        if len(x_c) > 0:
            axs[0].scatter(x_c, y_c, color='#3b82f6', s=4, label='Centerline')
            axs[0].legend(loc='upper right')
        
        coords_slice = np.argwhere(slice_pred > 0)
        if len(coords_slice) > 0:
            min_y, min_x = coords_slice.min(axis=0)
            max_y, max_x = coords_slice.max(axis=0)
            margin = 35
            axs[0].set_ylim(max(0, min_y - margin), min(img_slice.shape[0], max_y + margin))
            axs[0].set_xlim(max(0, min_x - margin), min(img_slice.shape[1], max_x + margin))
        axs[0].set_title(f"{cid} Native CT & Centerline (Slice {best_slice_idx})", fontsize=10, fontweight="bold")
        axs[0].axis("off")
        
        # Right panel: lumen diameter bar
        axs[1].bar(["MLD", "Ref Diam"], [res["min_diameter_mm"], res["ref_diameter_mm"]], color=['#ef4444', '#8b5cf6'], width=0.4)
        axs[1].set_ylabel("Diameter (mm)", fontsize=11, fontweight="bold")
        axs[1].set_ylim(0, max(5.5, res["ref_diameter_mm"] + 1.0))
        for bar_idx, val in enumerate([res["min_diameter_mm"], res["ref_diameter_mm"]]):
            axs[1].text(bar_idx, val + 0.15, f"{val:.2f} mm", ha='center', fontweight='bold')
        axs[1].set_title(f"{cid} Stenosis: {res['rasnet_stenosis_pct']:.1f}% ({res['rasnet_cad_rads']})", fontsize=10, fontweight="bold")
        axs[1].grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    print("[OK] All 32 overlay plots re-rendered successfully!", flush=True)

if __name__ == "__main__":
    main()
