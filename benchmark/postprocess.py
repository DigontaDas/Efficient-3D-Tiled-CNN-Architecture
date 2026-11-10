"""Connected-component post-processing with retained/removed volume logging."""
import cc3d
import numpy as np


def prune_components(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    keep_top_k: int | None = 2,
    connectivity: int = 26,
) -> tuple[np.ndarray, dict]:
    """
    Keep at most the top-k largest 3D connected components (26-connectivity) and log
    retained/removed component counts, voxel counts, and physical volumes in mm^3.
    """
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    sp = np.asarray(spacing_zyx, dtype=float)
    if mask_u8.ndim != 3 or sp.shape != (3,) or np.any(sp <= 0) or not np.all(np.isfinite(sp)):
        raise ValueError("Expected 3D binary mask and positive finite 3D spacing_zyx")

    voxel_vol_mm3 = float(np.prod(sp))
    total_voxels = int(mask_u8.sum())
    if total_voxels == 0:
        return mask_u8, {
            "components_total": 0,
            "components_retained": 0,
            "components_removed": 0,
            "voxels_retained": 0,
            "voxels_removed": 0,
            "volume_retained_mm3": 0.0,
            "volume_removed_mm3": 0.0,
        }

    labeled_sub, n_comp = cc3d.connected_components(
        mask_u8, connectivity=connectivity, return_N=True
    )
    n_comp = int(n_comp)
    if keep_top_k is None or n_comp <= keep_top_k:
        return mask_u8, {
            "components_total": n_comp,
            "components_retained": n_comp,
            "components_removed": 0,
            "voxels_retained": total_voxels,
            "voxels_removed": 0,
            "volume_retained_mm3": float(total_voxels * voxel_vol_mm3),
            "volume_removed_mm3": 0.0,
        }

    if keep_top_k < 1:
        raise ValueError("keep_top_k must be >= 1 or None")

    counts = np.bincount(labeled_sub.ravel())
    counts[0] = 0
    comp_ids = list(range(1, len(counts)))
    comp_ids.sort(key=lambda cid: (-int(counts[cid]), cid))
    keep_ids = comp_ids[:keep_top_k]

    lut = np.zeros(len(counts), dtype=np.uint8)
    for k_id in keep_ids:
        lut[k_id] = 1
    cleaned = lut[labeled_sub]
    retained_voxels = int(sum(int(counts[k]) for k in keep_ids))
    removed_voxels = int(total_voxels - retained_voxels)
    return cleaned, {
        "components_total": n_comp,
        "components_retained": int(len(keep_ids)),
        "components_removed": int(n_comp - len(keep_ids)),
        "voxels_retained": retained_voxels,
        "voxels_removed": removed_voxels,
        "volume_retained_mm3": float(retained_voxels * voxel_vol_mm3),
        "volume_removed_mm3": float(removed_voxels * voxel_vol_mm3),
    }

