import os, sys, SimpleITK as sitk, numpy as np, networkx as nx
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

def inspect_ct5_ct82():
    for cid in ["CT5", "CT82"]:
        case_dir = os.path.join(r"H:\Thesis_Trainings\Phase3_Local_Integration\local_data", cid)
        pred_path = os.path.join(case_dir, "pred_mask.nii.gz")
        pred_sitk = sitk.ReadImage(pred_path)
        pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(bool)
        sp = pred_sitk.GetSpacing()
        vox_size = float(np.mean(sp))
        coords = np.argwhere(pred_arr)
        z_min, y_min, x_min = np.maximum(coords.min(axis=0) - 5, 0)
        z_max, y_max, x_max = np.minimum(coords.max(axis=0) + 6, pred_arr.shape)
        cropped_mask = pred_arr[z_min:z_max, y_min:y_max, x_min:x_max]
        cropped_skel = skeletonize(cropped_mask)
        skel = np.zeros_like(pred_arr, dtype=bool)
        skel[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_skel
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
                if len(path) >= 8:
                    branches.append(path)
        print(f"=== {cid} ===")
        for idx, b in enumerate(branches[:10]):
            b_diams = [G.nodes[n]["diam"] for n in b]
            b_len_mm = len(b) * vox_size
            print(f"  Branch {idx}: len={len(b)} ({b_len_mm:.1f}mm), start_d={b_diams[0]:.2f}, end_d={b_diams[-1]:.2f}, min_d={min(b_diams):.2f}, max_d={max(b_diams):.2f}")

if __name__ == "__main__":
    inspect_ct5_ct82()
