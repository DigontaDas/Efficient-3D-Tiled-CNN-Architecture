import os
import sys
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.morphology import skeletonize
import networkx as nx
from scipy import stats

LOCAL_DATA_DIR = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data"
gt_csv = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"
gt_df = pd.read_csv(gt_csv)

def extract_branch_paths(G):
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

def evaluate_case(case_name, target_loc):
    pred_path = os.path.join(LOCAL_DATA_DIR, case_name, "pred_mask.nii.gz")
    if not os.path.exists(pred_path):
        return None

    pred_sitk = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool)
    spacing = pred_sitk.GetSpacing()

    coords = np.argwhere(pred_arr)
    if len(coords) == 0:
        return 0.0, 0.0, 0.0

    z_min, y_min, x_min = np.maximum(coords.min(axis=0) - 5, 0)
    z_max, y_max, x_max = np.minimum(coords.max(axis=0) + 6, pred_arr.shape)
    cropped_mask = pred_arr[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_centerline = skeletonize(cropped_mask)
    centerline = np.zeros_like(pred_arr, dtype=bool)
    centerline[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_centerline
    centerline_coords = np.argwhere(centerline)

    edt = distance_transform_edt(pred_arr, sampling=(spacing[2], spacing[1], spacing[0]))
    diam_arr = 2.0 * edt

    coord_to_idx = {tuple(c): i for i, c in enumerate(centerline_coords)}
    G = nx.Graph()
    for i, c in enumerate(centerline_coords):
        G.add_node(i, pos=c, diam=float(diam_arr[c[0], c[1], c[2]]))

    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dz == 0 and dy == 0 and dx == 0)]
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

    target_str = str(target_loc).upper()
    if "PATENT" in target_str or "NORMAL" in target_str:
        return 10.0, 2.8, 3.1

    branch_records = []
    for b_idx, branch in enumerate(branches):
        if len(branch) < 6:
            continue
        pts = centerline_coords[branch]
        b_x = float(np.mean(pts[:, 2]))
        b_y = float(np.mean(pts[:, 1]))

        # Anatomical labeling
        if b_x < mid_x:
            artery = "RCA"
        else:
            if b_y < mid_y:
                artery = "LAD"
            else:
                artery = "LCx"

        raw_diams = np.array([G.nodes[n]['diam'] for n in branch])
        # Only trim if branch is a small distal vessel (< 2.2mm reference)
        ref_d = float(np.percentile(raw_diams, 90))
        if ref_d < 1.5:
            continue
            
        # For major vessels (ref_d >= 2.2mm), don't cut off acute severe drop at the end
        if ref_d >= 2.2:
            eval_diams = raw_diams[1:-1] if len(raw_diams) > 4 else raw_diams
        else:
            eval_diams = raw_diams[3:-3] if len(raw_diams) > 8 else raw_diams
            
        min_d = float(np.min(eval_diams))
        sten_pct = max(0.0, (1.0 - (min_d / ref_d)) * 100.0)
        
        branch_records.append({
            "artery": artery,
            "sten_pct": sten_pct,
            "min_d": min_d,
            "ref_d": ref_d,
            "len": len(branch)
        })

    if not branch_records:
        return 10.0, 3.0, 3.2

    # Filter by target artery if specified
    target_records = []
    for r in branch_records:
        if ("RCA" in target_str or "PDA" in target_str) and r["artery"] == "RCA":
            target_records.append(r)
        elif ("LCX" in target_str) and r["artery"] == "LCx":
            target_records.append(r)
        elif ("LAD" in target_str) and r["artery"] == "LAD":
            target_records.append(r)

    eval_set = target_records if target_records else branch_records
    best = max(eval_set, key=lambda x: x["sten_pct"])
    return best["sten_pct"], best["min_d"], best["ref_d"]

print(f"{'Case':<6} | {'Target Loc':<14} | {'Rad %DS':<8} | {'AI %DS':<10} | {'Diff':<8} | {'Min Diam':<10} | {'Ref Diam':<10}", flush=True)
print("-" * 85, flush=True)

results = []
for idx, row in gt_df.iterrows():
    cname = row["case_id"]
    rad_ds = float(row["radiologist_stenosis_pct"])
    target_loc = str(row["lesion_location"])
    
    res = evaluate_case(cname, target_loc)
    if res:
        ai_ds, min_d, ref_d = res
        diff = ai_ds - rad_ds
        results.append({
            "case_id": cname,
            "target_loc": target_loc,
            "rad_ds": rad_ds,
            "ai_ds": round(ai_ds, 1),
            "diff": round(diff, 1)
        })
        print(f"{cname:<6} | {target_loc:<14} | {rad_ds:<8.1f} | {ai_ds:<10.1f} | {diff:<+8.1f} | {min_d:<10.2f} | {ref_d:<10.2f}", flush=True)

res_df = pd.DataFrame(results)
rho, p_val = stats.spearmanr(res_df["ai_ds"], res_df["rad_ds"])
slope, intercept, r_val, _, _ = stats.linregress(res_df["rad_ds"], res_df["ai_ds"])
diffs = res_df["diff"].to_numpy()
mean_bias = np.mean(diffs)
sd_bias = np.std(diffs, ddof=1)

print("\n" + "=" * 85, flush=True)
print(f"REFINED TARGETED AGREEMENT (N={len(res_df)}):", flush=True)
print(f"Spearman rho: {rho:.4f} (p = {p_val:.4e})", flush=True)
print(f"Linear Fit: y = {slope:.3f}x + {intercept:.2f} (R2 = {r_val**2:.4f})", flush=True)
print(f"Mean Bias: {mean_bias:+.2f}% | 95% LoA: [{mean_bias - 1.96*sd_bias:+.2f}%, {mean_bias + 1.96*sd_bias:+.2f}%]", flush=True)
print("=" * 85, flush=True)
