import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.morphology import skeletonize
import networkx as nx

pred_path = r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data\CT80\pred_mask.nii.gz"
pred_sitk = sitk.ReadImage(pred_path)
pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool)
spacing = pred_sitk.GetSpacing()
edt = distance_transform_edt(pred_arr, sampling=(spacing[2], spacing[1], spacing[0]))
diam_arr = 2.0 * edt

coords = np.argwhere(pred_arr)
z_min, y_min, x_min = np.maximum(coords.min(axis=0) - 5, 0)
z_max, y_max, x_max = np.minimum(coords.max(axis=0) + 6, pred_arr.shape)
cropped_mask = pred_arr[z_min:z_max, y_min:y_max, x_min:x_max]
cropped_centerline = skeletonize(cropped_mask)
centerline = np.zeros_like(pred_arr, dtype=bool)
centerline[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_centerline
centerline_coords = np.argwhere(centerline)

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
                G.add_edge(i, j, weight=float(np.sqrt((dz * spacing[2])**2 + (dy * spacing[1])**2 + (dx * spacing[0])**2)))

# Extract branches
degrees = dict(G.degree())
critical_nodes = [n for n, deg in degrees.items() if deg != 2]
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

print(f"Total branches in CT80: {len(branches)}")
for b_idx, b in enumerate(branches):
    raw_diams = np.array([G.nodes[n]['diam'] for n in b])
    pts = centerline_coords[b]
    b_x = float(np.mean(pts[:, 2]))
    b_y = float(np.mean(pts[:, 1]))
    
    t_start = 4 if G.degree(b[0]) == 1 else 0
    t_end = len(b) - 4 if G.degree(b[-1]) == 1 else len(b)
    trimmed = raw_diams[t_start:t_end] if t_end - t_start >= 4 else raw_diams
    
    ref_d = float(np.percentile(trimmed, 90))
    min_d = float(np.min(trimmed))
    sten_pct = (1.0 - (min_d / ref_d)) * 100.0 if ref_d > 0 else 0.0
    
    print(f"Branch {b_idx}: len={len(b)} | X={b_x:.1f} | RawMin={min(raw_diams):.2f} | TrimMin={min_d:.2f} | Ref={ref_d:.2f} | %DS={sten_pct:.1f}%")
