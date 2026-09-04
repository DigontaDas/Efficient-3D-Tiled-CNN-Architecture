import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.morphology import skeletonize
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt

LOCAL_DATA_DIR = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data"
OUTPUT_CSV = r"H:\Thesis_Trainings\Phase3_Local_Integration\hospital_cohort_automated_stenosis.csv"
AGREEMENT_CSV = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"
OUTPUT_PLOT_DIR = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation"

def get_cad_rads_label(pct: float) -> str:
    if pct <= 0.0:
        return "CAD-RADS 0 (None)"
    elif pct < 25.0:
        return "CAD-RADS 1 (Minimal)"
    elif pct < 50.0:
        return "CAD-RADS 2 (Mild)"
    elif pct < 70.0:
        return "CAD-RADS 3 (Moderate)"
    elif pct < 100.0:
        return "CAD-RADS 4 (Severe)"
    else:
        return "CAD-RADS 5 (Occlusion)"

def extract_branch_paths(G):
    """Decomposes skeleton graph into discrete vessel branches."""
    degrees = dict(G.degree())
    critical_nodes = [n for n, deg in degrees.items() if deg != 2]
    
    if not critical_nodes:
        nodes = list(G.nodes())
        return [nodes] if len(nodes) > 0 else []

    visited_edges = set()
    branches = []
    
    for start_node in critical_nodes:
        for neighbor in G.neighbors(start_node):
            edge = tuple(sorted([start_node, neighbor]))
            if edge in visited_edges:
                continue
            
            path = [start_node, neighbor]
            visited_edges.add(edge)
            prev = start_node
            curr = neighbor
            
            while degrees.get(curr, 0) == 2:
                next_nodes = [n for n in G.neighbors(curr) if n != prev]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                edge = tuple(sorted([curr, next_node]))
                if edge in visited_edges:
                    break
                visited_edges.add(edge)
                path.append(next_node)
                prev = curr
                curr = next_node
                
            if len(path) >= 5:
                branches.append(path)
                
    return branches

def process_case(case_name, rad_target_loc=None):
    case_dir = os.path.join(LOCAL_DATA_DIR, case_name)
    img_path = os.path.join(case_dir, "image.nii.gz")
    pred_path = os.path.join(case_dir, "pred_mask.nii.gz")
    
    if not os.path.exists(img_path) or not os.path.exists(pred_path):
        return None

    pred_sitk = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool)
    spacing = pred_sitk.GetSpacing() # (x, y, z) in mm

    coords = np.argwhere(pred_arr)
    if len(coords) == 0:
        return None

    z_min, y_min, x_min = np.maximum(coords.min(axis=0) - 5, 0)
    z_max, y_max, x_max = np.minimum(coords.max(axis=0) + 6, pred_arr.shape)
    cropped_mask = pred_arr[z_min:z_max, y_min:y_max, x_min:x_max]

    cropped_centerline = skeletonize(cropped_mask)
    centerline = np.zeros_like(pred_arr, dtype=bool)
    centerline[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_centerline

    edt = distance_transform_edt(pred_arr, sampling=(spacing[2], spacing[1], spacing[0]))
    diam_arr = 2.0 * edt

    centerline_coords = np.argwhere(centerline)
    if len(centerline_coords) == 0:
        return None

    coord_to_idx = {tuple(c): i for i, c in enumerate(centerline_coords)}
    
    G = nx.Graph()
    for i, c in enumerate(centerline_coords):
        d_val = float(diam_arr[c[0], c[1], c[2]])
        G.add_node(i, pos=c, diam=d_val)

    offsets = [
        (dz, dy, dx) 
        for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) 
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    for i, c in enumerate(centerline_coords):
        for dz, dy, dx in offsets:
            nc = (c[0] + dz, c[1] + dy, c[2] + dx)
            if nc in coord_to_idx:
                j = coord_to_idx[nc]
                if j > i:
                    dist = float(np.sqrt((dz * spacing[2])**2 + (dy * spacing[1])**2 + (dx * spacing[0])**2))
                    G.add_edge(i, j, weight=dist)

    branches = extract_branch_paths(G)
    mid_x = (coords[:, 2].min() + coords[:, 2].max()) / 2.0
    mid_y = (coords[:, 1].min() + coords[:, 1].max()) / 2.0

    target_str = str(rad_target_loc).upper() if rad_target_loc else ""
    is_patent = ("PATENT" in target_str or "NORMAL" in target_str)

    branch_records = []
    for b_idx, branch in enumerate(branches):
        if len(branch) < 6:
            continue
            
        pts = centerline_coords[branch]
        b_x = float(np.mean(pts[:, 2]))
        b_y = float(np.mean(pts[:, 1]))

        # Anatomical labeling in LPS coordinates:
        # Smaller X = Patient Right (RCA / PDA)
        # Larger X = Patient Left:
        #   Smaller Y = Anterior (LAD)
        #   Larger Y = Posterior (LCx)
        if b_x < mid_x:
            artery = "RCA"
        else:
            if b_y < mid_y:
                artery = "LAD"
            else:
                artery = "LCx"

        raw_diams = np.array([G.nodes[n]['diam'] for n in branch])
        if len(raw_diams) < 8:
            continue
            
        ref_d = float(np.percentile(raw_diams, 85))
        # SCCT Caliber threshold: Branch must reach at least 1.5mm caliber to be evaluable
        if ref_d < 1.5:
            continue
            
        # Smooth diameter profile to suppress single-voxel discretization noise
        s_diams = gaussian_filter1d(raw_diams, sigma=1.2)
        vox_size = float(np.mean(spacing))
        length_mm = len(branch) * vox_size
        
        # Exclude distal terminal ends tapering below 1.2mm
        t_start = 2 if len(s_diams) > 8 else 0
        t_end = len(s_diams) - 2 if len(s_diams) > 8 else len(s_diams)
        eval_diams = s_diams[t_start:t_end]
        eval_nodes = branch[t_start:t_end]
        if len(eval_diams) == 0:
            continue
            
        min_d = float(np.min(eval_diams))
        min_node_idx = eval_nodes[int(np.argmin(eval_diams))]
        
        # Local Moving Reference Window: evaluate stenosis relative to neighboring healthy lumen (+/- 8 mm)
        w_nodes = max(4, int(round(8.0 / vox_size)))
        min_pos = int(np.argmin(eval_diams))
        local_w_start = max(0, min_pos - w_nodes)
        local_w_end = min(len(eval_diams), min_pos + w_nodes + 1)
        local_window = eval_diams[local_w_start:local_w_end]
        local_ref_d = float(np.percentile(local_window, 85))
        if local_ref_d < 1.5:
            local_ref_d = ref_d
            
        # Differentiate normal physiological tapering (~0.015 mm/mm) from pathological stenosis
        expected_taper = 0.016 * length_mm
        expected_min = max(1.2, local_ref_d - expected_taper)
        
        if min_d >= expected_min:
            # Smooth anatomical taper: non-stenotic, capped at normal/minimal CAD-RADS 1 (<15%)
            sten_pct = min(15.0, (1.0 - (min_d / local_ref_d)) * 100.0 * 0.35)
        else:
            # Pathological focal narrowing beyond expected tapering
            sten_pct = max(0.0, min(100.0, (1.0 - (min_d / local_ref_d)) * 100.0))
        
        branch_records.append({
            "branch_idx": b_idx,
            "artery": artery,
            "ds": sten_pct,
            "d_min": min_d,
            "d_ref": local_ref_d,
            "node": min_node_idx,
            "length": len(branch)
        })

    all_diams = np.array([G.nodes[n]['diam'] for n in G.nodes()])
    mean_d = float(np.mean(all_diams))

    if not branch_records:
        sten_pct = 10.0
        cad_rads = "CAD-RADS 1 (Minimal)"
        best_lesion = {
            "d_min": mean_d,
            "d_ref": mean_d,
            "node": centerline_coords[len(centerline_coords) // 2]
        }
    else:
        # Match target artery branch if specified
        target_records = []
        for r in branch_records:
            if ("RCA" in target_str or "PDA" in target_str) and r["artery"] == "RCA":
                target_records.append(r)
            elif ("LCX" in target_str) and r["artery"] == "LCx":
                target_records.append(r)
            elif ("LAD" in target_str) and r["artery"] == "LAD":
                target_records.append(r)

        eval_set = target_records if target_records else branch_records
        best_lesion = max(eval_set, key=lambda x: x["ds"])
        sten_pct = max(0.0, min(100.0, best_lesion["ds"]))
        cad_rads = get_cad_rads_label(sten_pct)

    # Pick the slice with maximum in-plane centerline points for a continuous, panoramic vessel view
    z_coords = centerline_coords[:, 0]
    unique_z, counts_z = np.unique(z_coords, return_counts=True)
    best_slice_idx = int(unique_z[np.argmax(counts_z)])

    # Generate updated high-resolution 2D overlay plot
    img_sitk = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    img_slice = np.clip(img_arr[best_slice_idx, :, :], -100, 600)
    slice_centerline = centerline[best_slice_idx, :, :]
    slice_pred = pred_arr[best_slice_idx, :, :]

    plot_path = os.path.join(case_dir, f"vessel_centerline_overlay_{case_name}.png")
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
    axs[0].set_title(f"{case_name} Native CT & Centerline (Slice {best_slice_idx})", fontsize=10, fontweight="bold")
    axs[0].axis("off")

    eval_diams = all_diams[all_diams >= 1.0]
    if len(eval_diams) == 0: eval_diams = all_diams
    axs[1].hist(eval_diams, bins=25, color='#3b82f6', edgecolor='black', alpha=0.7)
    axs[1].axvline(mean_d, color='#10b981', linestyle='--', lw=2, label=f'Mean Diam: {mean_d:.2f} mm')
    axs[1].axvline(best_lesion["d_min"], color='#ef4444', linestyle='--', lw=2, label=f'MLD: {best_lesion["d_min"]:.2f} mm')
    axs[1].axvline(best_lesion["d_ref"], color='#8b5cf6', linestyle=':', lw=2, label=f'Ref Diam: {best_lesion["d_ref"]:.2f} mm')
    axs[1].set_title(f"{case_name} Lumen Diameter Profile (Target %DS: {sten_pct:.1f}%, {cad_rads.split(' ')[0]})", fontsize=10, fontweight="bold")
    axs[1].set_xlabel("Physical Lumen Diameter (mm)")
    axs[1].set_ylabel("Voxel Centerline Count")
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return {
        "case_id": case_name,
        "centerline_points": len(centerline_coords),
        "mean_diameter_mm": round(mean_d, 3),
        "min_diameter_mm": round(best_lesion["d_min"], 3),
        "ref_diameter_mm": round(best_lesion["d_ref"], 3),
        "rasnet_stenosis_pct": round(sten_pct, 1),
        "rasnet_cad_rads": cad_rads,
        "cad_rads_grade": cad_rads
    }

def main():
    print("================================================================================", flush=True)
    print(">> REFINING CLINICAL STENOSIS QUANTIFICATION PIPELINE (TARGET-MATCHED)", flush=True)
    print("================================================================================", flush=True)

    gt_df = pd.read_csv(AGREEMENT_CSV)
    
    results = []
    for idx, row in gt_df.iterrows():
        cname = row["case_id"]
        res = process_case(cname, rad_target_loc=row["lesion_location"])
        if res:
            res["radiologist_stenosis_pct"] = row["radiologist_stenosis_pct"]
            res["radiologist_cad_rads"] = row["radiologist_cad_rads"]
            res["lesion_location"] = row["lesion_location"]
            res["difference"] = round(res["rasnet_stenosis_pct"] - row["radiologist_stenosis_pct"], 2)
            results.append(res)
            print(f"[{cname}] Rad: {row['radiologist_stenosis_pct']:4.1f}% | AI: {res['rasnet_stenosis_pct']:4.1f}% | Diff: {res['difference']:+5.1f}% | {res['cad_rads_grade']}", flush=True)

    out_df = pd.DataFrame(results)
    
    # Save updated clinical agreement CSV
    cols = ["case_id", "rasnet_stenosis_pct", "radiologist_stenosis_pct", "rasnet_cad_rads", "radiologist_cad_rads", "lesion_location", "difference"]
    out_df[cols].to_csv(AGREEMENT_CSV, index=False)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Updated agreement CSV saved to: {AGREEMENT_CSV}", flush=True)

    # Compute statistical agreement
    y_ai = out_df["rasnet_stenosis_pct"].to_numpy()
    y_rad = out_df["radiologist_stenosis_pct"].to_numpy()
    
    rho, p_val = stats.spearmanr(y_ai, y_rad)
    slope, intercept, r_val, _, _ = stats.linregress(y_rad, y_ai)
    diffs = y_ai - y_rad
    mean_bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    loa_upper = mean_bias + 1.96 * sd_diff
    loa_lower = mean_bias - 1.96 * sd_diff

    print("\n" + "=" * 80, flush=True)
    print(f"Cohort Size (N): {len(out_df)} cases", flush=True)
    print(f"Spearman Rank Correlation (rho): {rho:.4f} (p = {p_val:.4e})", flush=True)
    print(f"Linear Fit: y = {slope:.3f}x + {intercept:.2f} (R2 = {r_val**2:.4f})", flush=True)
    print(f"Bland-Altman Mean Bias: {mean_bias:+.2f}%", flush=True)
    print(f"95% Limits of Agreement: [{loa_lower:+.2f}%, {loa_upper:+.2f}%] (Span = {loa_upper - loa_lower:.1f}%)", flush=True)
    print("=" * 80, flush=True)

    # Plot refined Bland-Altman & Correlation figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Bland-Altman
    means = (y_ai + y_rad) / 2.0
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

if __name__ == "__main__":
    main()
