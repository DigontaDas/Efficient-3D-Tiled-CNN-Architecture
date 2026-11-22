"""Strategy A deterministic preprocessing, transform hashing, patch sampling, and full 3D augmentations.

Matches nnU-Net's ImageCAS fingerprint without raw-HU CropForegroundd(x > 0):
  - Native median spacing (MONAI XYZ): (0.349609375, 0.349609375, 0.50) mm
  - Patch size (MONAI XYZ):            (160, 160, 96)
  - CTNormalization:                   clip [-166.0, 727.0] HU, z-score (x - 147.97113) / 179.68488
  - Full 3D augmentations:             3-axis flips, 3D affine rot/scale, Gaussian noise,
                                       Gaussian blur, intensity scaling, contrast/gamma.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import torch
from monai import transforms as mt
from monai.transforms import MapTransform


CT_CLIP_LO = -166.0
CT_CLIP_HI = 727.0
CT_MEAN = 147.97113
CT_STD = 179.68488
PAD_IMAGE_VAL = float((CT_CLIP_LO - CT_MEAN) / CT_STD)  # -1.747343
PAD_LABEL_VAL = 0

_AUG_PIPE_CACHE: dict[str, mt.Compose] = {}


class CTStandardizeD(MapTransform):
    """Exact nnU-Net CTNormalization: clip [clip_lo, clip_hi] then (x - mean) / std in float32."""

    def __init__(
        self,
        keys: Sequence[str] = ("image",),
        clip_lo: float = CT_CLIP_LO,
        clip_hi: float = CT_CLIP_HI,
        mean: float = CT_MEAN,
        std: float = CT_STD,
    ):
        super().__init__(keys)
        self.clip_lo = float(clip_lo)
        self.clip_hi = float(clip_hi)
        self.mean = float(mean)
        self.std = max(float(std), 1e-8)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            if isinstance(val, torch.Tensor):
                out = torch.clamp(val.float(), min=self.clip_lo, max=self.clip_hi)
                d[key] = (out - self.mean) / self.std
            else:
                arr = np.asarray(val, dtype=np.float32)
                np.clip(arr, self.clip_lo, self.clip_hi, out=arr)
                arr -= self.mean
                arr /= self.std
                d[key] = arr
        return d


def compute_transform_hash(preproc_cfg: dict[str, Any]) -> str:
    """Deterministic SHA-256 of all preprocessing parameters keying the PersistentDataset cache."""
    canonical = {
        "axis_order_monai": str(preproc_cfg.get("axis_order_monai", "XYZ")),
        "pixdim_xyz": [float(v) for v in preproc_cfg["pixdim_xyz"]],
        "crop_foreground_hu_gt_0": bool(preproc_cfg.get("crop_foreground_hu_gt_0", False)),
        "ct_clip_hu": [float(v) for v in preproc_cfg["ct_clip_hu"]],
        "ct_mean": float(preproc_cfg["ct_mean"]),
        "ct_std": float(preproc_cfg["ct_std"]),
        "bg_coords_max_per_case": int(preproc_cfg.get("bg_coords_max_per_case", 50000)),
        "cache_dtype_image": str(preproc_cfg.get("cache_dtype_image", "float16")),
        "cache_dtype_label": str(preproc_cfg.get("cache_dtype_label", "uint8")),
        "version": "strategy_a_no_crop_v1_1",
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_deterministic_preprocess(
    preproc_cfg: dict[str, Any],
    with_label: bool = True,
    track_meta: bool = False,
) -> mt.Compose:
    """Build deterministic spacing + CTNormalization pipeline (strictly NO CropForegroundd)."""
    if preproc_cfg.get("crop_foreground_hu_gt_0", False):
        raise ValueError("Strategy A prohibits raw-HU CropForegroundd(x > 0)")

    pixdim = tuple(float(v) for v in preproc_cfg["pixdim_xyz"])
    clip_lo, clip_hi = float(preproc_cfg["ct_clip_hu"][0]), float(preproc_cfg["ct_clip_hu"][1])
    ct_mean = float(preproc_cfg["ct_mean"])
    ct_std = float(preproc_cfg["ct_std"])

    keys = ["image", "label"] if with_label else ["image"]
    spacing_mode: Any = ("bilinear", "nearest") if with_label else "bilinear"

    ops: list[Any] = [
        mt.LoadImaged(keys=keys),
        mt.EnsureChannelFirstd(keys=keys),
        mt.Spacingd(keys=keys, pixdim=pixdim, mode=spacing_mode),
        CTStandardizeD(keys=["image"], clip_lo=clip_lo, clip_hi=clip_hi, mean=ct_mean, std=ct_std),
        mt.EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=track_meta),
    ]
    if with_label:
        ops.append(mt.EnsureTyped(keys=["label"], dtype=torch.uint8, track_meta=track_meta))
    return mt.Compose(ops)


def pad_patch_if_needed(
    patch_img: np.ndarray,
    patch_lbl: np.ndarray,
    patch_size_xyz: tuple[int, int, int],
    pad_image_val: float = PAD_IMAGE_VAL,
    pad_label_val: int = PAD_LABEL_VAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a (1, sx, sy, sz) crop cleanly if a volume axis is smaller than patch_size_xyz."""
    _, sx, sy, sz = patch_img.shape
    px, py, pz = (int(v) for v in patch_size_xyz)
    if sx == px and sy == py and sz == pz:
        return patch_img, patch_lbl
    pads = []
    for cur, target in ((sx, px), (sy, py), (sz, pz)):
        diff = max(target - cur, 0)
        b = diff // 2
        a = diff - b
        pads.append((b, a))
    spec = ((0, 0), pads[0], pads[1], pads[2])
    img_p = np.pad(patch_img, spec, mode="constant", constant_values=float(pad_image_val))
    lbl_p = np.pad(patch_lbl, spec, mode="constant", constant_values=int(pad_label_val))
    return img_p, lbl_p


def compute_crop_slices(
    vol_shape_xyz: tuple[int, int, int],
    fg_coords: np.ndarray,
    bg_coords: np.ndarray | None,
    rng: np.random.Generator,
    patch_size_xyz: tuple[int, int, int] = (160, 160, 96),
    fg_prob: float = 1.0 / 3.0,
) -> tuple[slice, slice, slice]:
    """Select deterministic pos/neg crop bounding box (O(1) memory, works on chunked blosc2 arrays)."""
    sx, sy, sz = (int(v) for v in vol_shape_xyz)
    px, py, pz = (int(v) for v in patch_size_xyz)

    use_fg = (len(fg_coords) > 0) and (rng.random() < float(fg_prob))
    if use_fg:
        idx = int(rng.integers(0, len(fg_coords)))
        cx, cy, cz = int(fg_coords[idx, 0]), int(fg_coords[idx, 1]), int(fg_coords[idx, 2])
    elif bg_coords is not None and len(bg_coords) > 0:
        idx = int(rng.integers(0, len(bg_coords)))
        cx, cy, cz = int(bg_coords[idx, 0]), int(bg_coords[idx, 1]), int(bg_coords[idx, 2])
    else:
        cx = int(rng.integers(0, max(sx, 1)))
        cy = int(rng.integers(0, max(sy, 1)))
        cz = int(rng.integers(0, max(sz, 1)))

    def _axis_slice(c: int, s: int, p: int) -> slice:
        if s <= p:
            return slice(0, s)
        start = min(max(c - (p // 2), 0), s - p)
        return slice(start, start + p)

    return _axis_slice(cx, sx, px), _axis_slice(cy, sy, py), _axis_slice(cz, sz, pz)


def _get_aug_pipeline(aug_cfg: dict[str, Any]) -> mt.Compose:
    key = json.dumps(aug_cfg, sort_keys=True, separators=(",", ":"))
    pipe = _AUG_PIPE_CACHE.get(key)
    if pipe is not None:
        return pipe

    rot = tuple(float(v) for v in aug_cfg.get("rotate_range_rad", [0.2617993878] * 3))
    scale = tuple(float(v) for v in aug_cfg.get("scale_range", [0.15, 0.15, 0.15]))
    sigma = tuple(float(v) for v in aug_cfg.get("gaussian_smooth_sigma", [0.5, 1.0]))
    gamma = tuple(float(v) for v in aug_cfg.get("adjust_contrast_gamma", [0.7, 1.5]))
    flip_p = float(aug_cfg.get("flip_prob", 0.5))

    pipe = mt.Compose([
        mt.RandFlipd(keys=["image", "label"], spatial_axis=0, prob=flip_p),
        mt.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=flip_p),
        mt.RandFlipd(keys=["image", "label"], spatial_axis=2, prob=flip_p),
        mt.RandAffined(
            keys=["image", "label"],
            prob=float(aug_cfg.get("affine_prob", 0.30)),
            rotate_range=rot,
            scale_range=scale,
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        mt.RandGaussianNoised(
            keys=["image"],
            prob=float(aug_cfg.get("gaussian_noise_prob", 0.15)),
            mean=0.0,
            std=float(aug_cfg.get("gaussian_noise_std", 0.10)),
        ),
        mt.RandGaussianSmoothd(
            keys=["image"],
            prob=float(aug_cfg.get("gaussian_smooth_prob", 0.15)),
            sigma_x=sigma,
            sigma_y=sigma,
            sigma_z=sigma,
        ),
        mt.RandScaleIntensityd(
            keys=["image"],
            factors=float(aug_cfg.get("scale_intensity_factors", 0.15)),
            prob=float(aug_cfg.get("scale_intensity_prob", 0.15)),
        ),
        mt.RandAdjustContrastd(
            keys=["image"],
            prob=float(aug_cfg.get("adjust_contrast_prob", 0.15)),
            gamma=gamma,
            retain_stats=True,
        ),
    ])
    _AUG_PIPE_CACHE[key] = pipe
    return pipe


def apply_3d_train_augmentations(
    patch_img: np.ndarray,
    patch_lbl: np.ndarray,
    aug_cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply full deterministic 3D spatial & intensity augmentations to a (1, X, Y, Z) patch."""
    pipe = _get_aug_pipeline(aug_cfg)
    pipe.set_random_state(seed=int(seed) & 0x7FFFFFFF)
    sample = {
        "image": torch.from_numpy(np.ascontiguousarray(patch_img, dtype=np.float32)),
        "label": torch.from_numpy(np.ascontiguousarray(patch_lbl, dtype=np.float32)),
    }
    out = pipe(sample)
    out_img = np.ascontiguousarray(
        out["image"].detach().cpu().numpy() if isinstance(out["image"], torch.Tensor) else np.asarray(out["image"]),
        dtype=np.float32,
    )
    out_lbl_raw = (
        out["label"].detach().cpu().numpy() if isinstance(out["label"], torch.Tensor) else np.asarray(out["label"])
    )
    out_lbl = np.ascontiguousarray((out_lbl_raw > 0.5).astype(np.uint8))
    if not np.all(np.isfinite(out_img)):
        # Guard against any pathological zero-range gamma NaN on constant synthetic patches
        out_img = np.nan_to_num(out_img, nan=PAD_IMAGE_VAL, posinf=4.0, neginf=PAD_IMAGE_VAL)
    return out_img, out_lbl
