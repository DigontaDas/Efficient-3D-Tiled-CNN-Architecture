import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import networkx as nx

LOCAL_DATA_DIR = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data"

def compute_stenosis_for_case(case_name, min_tip_dist_mm=8.0, local_window_mm=12.0, min_eval_diam_mm=1.5):
    case_dir = os.path.join(LOCAL_DATA_DIR, case_name)
    pred_path = os.path.join(case_dir, "pred_mask.nii.gz")
    if not os.path.exists(pred_path):
        return None

    pred_sitk = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool)
    spacing = pred_sitk.GetSpacing() # (x, y, z) in mm

    # Crop to ROI
    coords = np.argwhere(pred_arr)
    if len(coords) == 0:
        return None
    z_min, y_min, x_min = np.maximum(coords.min(axis=0) - 5, 0)
    z_max, y_max, x_max = np.minimum(coords.max(axis=0) + 6, pred_arr.shape)
    cropped_mask = pred_arr[z_min:z_max, y_min:y_max, x_min:x_max]

    cropped_centerline = skeletonize(cropped_mask)
    centerline = np.zeros_like(pred_arr, dtype=bool)
    centerline[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_centerline

    # 3D EDT in mm
    edt = distance_transform_edt(pred_arr, sampling=(spacing[2], spacing[1], spacing[0]))
    diam_arr = 2.0 * edt

    centerline_coords = np.argwhere(centerline) # (N, 3) in (Z, Y, X)
    if len(centerline_coords) == 0:
        return None

    coord_to_idx = {tuple(c): i for i, c in enumerate(centerline_coords)}
    
    # Build 26-connectivity graph
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

    # 1. Identify leaf endpoints (degree == 1)
    degrees = dict(G.degree())
    leaf_nodes = [n for n, deg in degrees.items() if deg == 1]

    # 2. Compute shortest path distance (in mm) from each node to nearest leaf node
    # Use multi-source Dijkstra from all leaf nodes
    dist_to_leaf = {}
    if leaf_nodes:
        dist_to_leaf = nx.multi_source_dijkstra_path_length(G, leaf_nodes, weight='weight')
    else:
        for n in G.nodes():
            dist_to_leaf[n] = 999.0

    # 3. For each candidate node (distance from leaf >= min_tip_dist_mm):
    # Find all nodes within local_window_mm along graph
    candidate_stenosis = []
    
    for n in G.nodes():
        # Exclude distal tapering tips
        if dist_to_leaf.get(n, 0) < min_tip_dist_mm:
            continue
            
        d_curr = G.nodes[n]['diam']
        
        # Get neighborhood within local_window_mm (using single_source_dijkstra_path_length)
        lengths = nx.single_source_dijkstra_path_length(G, n, cutoff=local_window_mm, weight='weight')
        neigh_nodes = list(lengths.keys())
        neigh_diams = [G.nodes[v]['diam'] for v in neigh_nodes]
        
        # Local healthy reference diameter: 90th percentile of local neighborhood
        local_ref = float(np.percentile(neigh_diams, 90))
        
        # Only evaluate if local vessel segment is of clinically evaluable caliber (>= min_eval_diam_mm)
        if local_ref >= min_eval_diam_mm:
            ds = max(0.0, (1.0 - (d_curr / local_ref)) * 100.0)
            candidate_stenosis.append({
                "node": n,
                "diam": d_curr,
                "ref_diam": local_ref,
                "stenosis_pct": ds
            })

    if not candidate_stenosis:
        # If no major vessel candidate exceeds tip margin, check if whole tree is small or healthy
        return 0.0, 0.0, 0.0

    # Sort by stenosis
    candidate_stenosis.sort(key=lambda x: x["stenosis_pct"], reverse=True)
    
    # Take top 5% or maximum robust focal constriction
    sten_values = [x["stenosis_pct"] for x in candidate_stenosis]
    max_ds = float(np.percentile(sten_values, 99)) if len(sten_values) >= 20 else float(np.max(sten_values))
    top_case = candidate_stenosis[0]
    
    return max_ds, top_case["diam"], top_case["ref_diam"]


# Load radiologist ground truth to compare directly
gt_csv = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"
gt_df = pd.read_csv(gt_csv)

print(f"{'Case':<8} | {'Radiologist %DS':<15} | {'Old AI %DS':<12} | {'New AI %DS':<12} | {'Min Diam':<10} | {'Ref Diam':<10}", flush=True)
print("-" * 80, flush=True)

results = []
for idx, row in gt_df.iterrows():
    cname = row["case_id"]
    rad_ds = row["radiologist_stenosis_pct"]
    old_ai = row["rasnet_stenosis_pct"]
    
    res = compute_stenosis_for_case(cname, min_tip_dist_mm=6.0, local_window_mm=10.0, min_eval_diam_mm=1.5)
    if res:
        new_ai, min_d, ref_d = res
    else:
        new_ai, min_d, ref_d = 0.0, 0.0, 0.0
        
    results.append({
        "case_id": cname,
        "radiologist_ds": rad_ds,
        "old_ai_ds": old_ai,
        "new_ai_ds": round(new_ai, 1),
        "min_diam": round(min_d, 2),
        "ref_diam": round(ref_d, 2)
    })
    print(f"{cname:<8} | {rad_ds:<15.1f} | {old_ai:<12.1f} | {new_ai:<12.1f} | {min_d:<10.2f} | {ref_d:<10.2f}", flush=True)

res_df = pd.DataFrame(results)
from scipy import stats
rho_old, p_old = stats.spearmanr(res_df["old_ai_ds"], res_df["radiologist_ds"])
rho_new, p_new = stats.spearmanr(res_df["new_ai_ds"], res_df["radiologist_ds"])

slope_old, intercept_old, _, _, _ = stats.linregress(res_df["radiologist_ds"], res_df["old_ai_ds"])
slope_new, intercept_new, _, _, _ = stats.linregress(res_df["radiologist_ds"], res_df["new_ai_ds"])

print("\n" + "=" * 80, flush=True)
print(f"OLD Spearman rho: {rho_old:.4f} (p={p_old:.4e}) | Fit: y = {slope_old:.2f}x + {intercept_old:.1f}", flush=True)
print(f"NEW Spearman rho: {rho_new:.4f} (p={p_new:.4e}) | Fit: y = {slope_new:.2f}x + {intercept_new:.1f}", flush=True)
print("=" * 80, flush=True)
