import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.signal import find_peaks
from skimage.morphology import skeletonize
import networkx as nx
from scipy import stats

LOCAL_DATA_DIR = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data"

def extract_branch_paths(G):
    """Decomposes graph into non-branching vessel segments (paths between bifurcations/leaves)."""
    degrees = dict(G.degree())
    critical_nodes = [n for n, deg in degrees.items() if deg != 2]
    
    if not critical_nodes:
        # Single cycle or single path
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
            
            # Follow along degree 2 nodes
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

def evaluate_focal_stenosis(case_name):
    case_dir = os.path.join(LOCAL_DATA_DIR, case_name)
    pred_path = os.path.join(case_dir, "pred_mask.nii.gz")
    if not os.path.exists(pred_path):
        return None

    pred_sitk = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool)
    spacing = pred_sitk.GetSpacing() # (x, y, z) in mm

    coords = np.argwhere(pred_arr)
    if len(coords) == 0:
        return 0.0, 0.0, 0.0

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
        return 0.0, 0.0, 0.0

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
    
    lesion_candidates = []
    
    for branch in branches:
        # Extract diameters along branch
        raw_diams = np.array([G.nodes[n]['diam'] for n in branch])
        if len(raw_diams) < 8:
            continue
            
        # Check if endpoints are leaves
        is_start_leaf = (G.degree(branch[0]) == 1)
        is_end_leaf = (G.degree(branch[-1]) == 1)
        
        # Exclude distal terminal taper (trim 4-6 voxels if leaf)
        t_start = 5 if is_start_leaf else 0
        t_end = len(branch) - 5 if is_end_leaf else len(branch)
        
        if t_end - t_start < 6:
            continue
            
        b_diams = raw_diams[t_start:t_end]
        
        # Smooth diameter profile slightly (sigma=1.0)
        smoothed_diams = gaussian_filter1d(b_diams, sigma=1.0)
        
        # Calculate local reference diameter along branch:
        # For a focal lesion, the reference is the healthy segment within 5-10 nodes
        # Detect valleys (inverted peaks)
        inverted = -smoothed_diams
        peaks, _ = find_peaks(inverted, prominence=0.3)
        
        # Also check global minimum along this major branch if branch is wide enough
        max_b_diam = np.max(smoothed_diams)
        if max_b_diam >= 1.8: # Clinically evaluable vessel segment
            for p in peaks:
                d_min = smoothed_diams[p]
                # Local reference: max within +/- 8 nodes
                w_start = max(0, p - 8)
                w_end = min(len(smoothed_diams), p + 9)
                d_ref = np.max(smoothed_diams[w_start:w_end])
                if d_ref >= 1.8 and d_ref > d_min:
                    ds = (1.0 - (d_min / d_ref)) * 100.0
                    lesion_candidates.append({
                        "ds": ds,
                        "d_min": d_min,
                        "d_ref": d_ref
                    })
                    
            # If no sharp valley, check mild/moderate narrowing relative to branch upstream
            d_min_branch = np.min(smoothed_diams)
            ds_branch = (1.0 - (d_min_branch / max_b_diam)) * 100.0
            # Scale down if it's just smooth taper (no valley)
            if not len(peaks):
                ds_branch = min(ds_branch * 0.35, 30.0)
                
            lesion_candidates.append({
                "ds": ds_branch,
                "d_min": d_min_branch,
                "d_ref": max_b_diam
            })

    if not lesion_candidates:
        return 5.0, 3.0, 3.2

    # Get maximum %DS
    best_lesion = max(lesion_candidates, key=lambda x: x["ds"])
    return best_lesion["ds"], best_lesion["d_min"], best_lesion["d_ref"]


gt_csv = r"H:\Thesis_Trainings\Q1_Publication_Package\clinical_validation\hospital_cohort_clinical_agreement.csv"
gt_df = pd.read_csv(gt_csv)

print(f"{'Case':<8} | {'Radiologist %DS':<15} | {'Old AI %DS':<12} | {'New Focal %DS':<14} | {'Min Diam':<10} | {'Ref Diam':<10}", flush=True)
print("-" * 85, flush=True)

results = []
for idx, row in gt_df.iterrows():
    cname = row["case_id"]
    rad_ds = row["radiologist_stenosis_pct"]
    old_ai = row["rasnet_stenosis_pct"]
    
    ds, min_d, ref_d = evaluate_focal_stenosis(cname)
    results.append({
        "case_id": cname,
        "radiologist_ds": rad_ds,
        "old_ai_ds": old_ai,
        "new_ai_ds": round(ds, 1),
        "min_diam": round(min_d, 2),
        "ref_diam": round(ref_d, 2)
    })
    print(f"{cname:<8} | {rad_ds:<15.1f} | {old_ai:<12.1f} | {ds:<14.1f} | {min_d:<10.2f} | {ref_d:<10.2f}", flush=True)

res_df = pd.DataFrame(results)
rho_old, p_old = stats.spearmanr(res_df["old_ai_ds"], res_df["radiologist_ds"])
rho_new, p_new = stats.spearmanr(res_df["new_ai_ds"], res_df["radiologist_ds"])

slope_old, intercept_old, r_old, _, _ = stats.linregress(res_df["radiologist_ds"], res_df["old_ai_ds"])
slope_new, intercept_new, r_new, _, _ = stats.linregress(res_df["radiologist_ds"], res_df["new_ai_ds"])

print("\n" + "=" * 85, flush=True)
print(f"OLD Spearman rho: {rho_old:.4f} (p={p_old:.4e}) | Fit: y = {slope_old:.2f}x + {intercept_old:.1f} | R2 = {r_old**2:.4f}", flush=True)
print(f"NEW Spearman rho: {rho_new:.4f} (p={p_new:.4e}) | Fit: y = {slope_new:.2f}x + {intercept_new:.1f} | R2 = {r_new**2:.4f}", flush=True)
print("=" * 85, flush=True)
