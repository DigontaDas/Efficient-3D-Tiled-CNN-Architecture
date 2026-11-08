"""Binary volume metrics; array spacing is ordered as array axes (z, y, x)."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize

METRIC_VERSION = 'surface-v1'


def check_geometry(a, b, atol=1e-5):
    for key in ('size', 'origin', 'spacing', 'direction'):
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or not np.allclose(x, y, atol=atol, rtol=0):
            raise ValueError(f'Physical geometry mismatch: {key}: {x} vs {y}')


def surface_distances(a, b, spacing):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if a.shape != b.shape or a.ndim != 3 or spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Expected same 3D shape and positive finite axis spacing')
    if not a.any() or not b.any():
        raise ValueError('Surface distance is undefined for an empty mask')
    sa = a & ~binary_erosion(a, border_value=0)
    sb = b & ~binary_erosion(b, border_value=0)
    return distance_transform_edt(~sb, sampling=spacing)[sa]


def compute_metrics(pred, gt, spacing):
    p, g = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if p.shape != g.shape or p.ndim != 3:
        raise ValueError('Masks must have equal 3D shape')
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Invalid spacing')
    ps, gs = int(p.sum()), int(g.sum())
    intersection = int((p & g).sum()); union = ps + gs - intersection
    out = {'dice': 2*intersection/(ps+gs) if ps+gs else np.nan,
           'iou': intersection/union if union else np.nan,
           'precision': intersection/ps if ps else np.nan,
           'recall': intersection/gs if gs else np.nan,
           'hd95': np.nan, 'asd': np.nan, 'cldice': np.nan, 'centerline_recall': np.nan,
           'status': 'ok' if ps and gs else ('both_empty' if not ps and not gs else ('prediction_empty' if not ps else 'ground_truth_empty'))}
    if ps and gs:
        d = np.concatenate((surface_distances(p,g,spacing),surface_distances(g,p,spacing)))
        out['hd95'], out['asd'] = float(np.percentile(d,95)), float(np.mean(d))
        sp, sg = skeletonize(p).astype(bool), skeletonize(g).astype(bool)
        if not sp.any() or not sg.any():
            out['status'] = 'empty_skeleton'
        else:
            tprec, tsens = (sp & g).sum()/sp.sum(), (sg & p).sum()/sg.sum()
            out['centerline_recall'] = float(tsens)
            out['cldice'] = float(2*tprec*tsens/(tprec+tsens)) if tprec+tsens else 0.0
    return out
