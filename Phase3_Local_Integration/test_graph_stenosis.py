import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import networkx as nx

LOCAL_DATA_DIR = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data"

def extract_centerline_branches(centerline_mask, diameters, spacing):
    """
    Converts 3D boolean centerline mask into a graph of branch segments.
    Filters out distal taper near terminal leaf endpoints.
    """
    coords = np.argwhere(centerline_mask) # (N, 3) in (Z, Y, X)
    if len(coords) == 0:
        return []

    # Map coord tuple to index
    coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
    
    # Build 26-connectivity graph
    G = nx.Graph()
    for i, c in enumerate(coords):
        G.add_node(i, pos=c, diam=diameters[c[0], c[1], c[2]])

    # 26-connectivity offsets
    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                offsets.append((dz, dy, dx))

    for i, c in enumerate(coords):
        for dz, dy, dx in offsets:
            nc = (c[0] + dz, c[1] + dy, c[2] + dx)
            if nc in coord_to_idx:
                j = coord_to_idx[nc]
                if not G.has_edge(i, j):
                    # physical distance in mm
                    dist = np.sqrt((dz * spacing[2])**2 + (dy * spacing[1])**2 + (dx * spacing[0])**2)
                    G.add_edge(i, j, weight=dist)

    # Find degrees
    degrees = dict(G.degree())
    endpoints = [n for n, deg in degrees.items() if deg == 1]
    junctions = [n for n, deg in degrees.items() if deg >= 3]

    # For each branch between junctions/endpoints:
    # Decompose graph into simple paths (branches)
    # A branch is a path between nodes of degree != 2
    critical_nodes = set(endpoints + junctions)
    if not critical_nodes:
        # It's a single loop or simple path
        branches = [list(G.nodes())]
    else:
        visited_edges = set()
        branches = []
        for start_node in critical_nodes:
            for neighbor in G.neighbors(start_node):
                edge = tuple(sorted([start_node, neighbor]))
                if edge in visited_edges:
                    continue
                # Traverse path
                path = [start_node, neighbor]
                visited_edges.add(edge)
                curr = neighbor
                prev = start_node
                while degrees[curr] == 2:
                    next_nodes = [n for n in G.neighbors(curr) if n != prev]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0]
                    edge = tuple(sorted([curr, next_node]))
                    visited_edges.add(edge)
                    path.append(next_node)
                    prev = curr
                    curr = next_node
                branches.append(path)

    return G, branches, endpoints

def evaluate_case_stenosis(case_name):
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

    G, branches, endpoints = extract_centerline_branches(centerline, diam_arr, spacing)
    
    # Analyze each branch:
    # 1. Ignore tiny spurious branches (< 5 mm length)
    # 2. On valid branches, trim the distal tip (last 10 mm / 15 nodes) if it terminates at a leaf endpoint (physiological taper)
    # 3. Along the remaining branch, calculate % stenosis relative to the branch's healthy reference diameter (90th percentile)
    
    branch_results = []
    for b_idx, branch in enumerate(branches):
        if len(branch) < 6:
            continue
        
        # Branch physical length
        b_diameters = np.array([G.nodes[n]['diam'] for n in branch])
        
        # Check if endpoints are leaves (degree == 1)
        is_start_leaf = (G.degree(branch[0]) == 1)
        is_end_leaf = (G.degree(branch[-1]) == 1)
        
        # Trim distal ends that terminate as leaves (tapering zone: trim 6-8 voxels ~3-4mm from leaf)
        trim_start = 6 if is_start_leaf else 0
        trim_end = len(branch) - 6 if is_end_leaf else len(branch)
        
        if trim_end - trim_start < 4:
            continue
            
        trimmed_branch = branch[trim_start:trim_end]
        trimmed_diams = b_diameters[trim_start:trim_end]
        
        # We only evaluate clinically relevant vessels (reference diameter >= 1.5 mm according to SCCT guidelines)
        ref_d = np.percentile(trimmed_diams, 90)
        if ref_d < 1.5:
            continue
            
        min_d = np.min(trimmed_diams)
        ds = max(0.0, (1.0 - (min_d / ref_d)) * 100.0)
        
        branch_results.append({
            "branch_idx": b_idx,
            "length_nodes": len(trimmed_branch),
            "ref_diam": ref_d,
            "min_diam": min_d,
            "stenosis_pct": ds
        })

    if not branch_results:
        print(f"[{case_name}] No significant major vessel lesions detected. Default: 0.0%")
        return 0.0

    max_ds = max(b["stenosis_pct"] for b in branch_results)
    best_branch = max(branch_results, key=lambda x: x["stenosis_pct"])
    print(f"[{case_name}] Max Stenosis: {max_ds:.1f}% (Branch Ref: {best_branch['ref_diam']:.2f}mm, Min: {best_branch['min_diam']:.2f}mm, len: {best_branch['length_nodes']})")
    return max_ds

test_cases = ["CT61", "CT62", "CT63", "CT64", "CT65", "CT66", "CT67", "CT68", "CT70", "CT71", "CT72", "CT73", "CT74", "CT75", "CT76", "CT77", "CT78", "CT79", "CT80", "CT81", "CT82", "CT83", "CT84", "CT85", "CT86", "CT87", "CT88", "CT89", "CT90"]

for c in test_cases:
    evaluate_case_stenosis(c)
