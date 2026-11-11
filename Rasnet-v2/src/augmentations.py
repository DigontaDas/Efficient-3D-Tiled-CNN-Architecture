"""
Shared MONAI Preprocessing, Per-Key SpatialPadd Fix, and Shared 3D Patch Augmentations.
Matches nnUNetPlans.json CTNormalization & spacing in MONAI (X, Y, Z) axis order:
  pixdim = (0.349609375, 0.349609375, 0.50), patch_size = (160, 160, 96).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from monai import transforms as t
from monai.transforms import MapTransform


CT_CLIP_LO = -166.0
CT_CLIP_HI = 727.0
CT_MEAN = 147.97113
CT_STD = 179.68488
BODY_CROP_HU = -950.0
PAD_IMAGE_VAL = float((CT_CLIP_LO - CT_MEAN) / CT_STD)  # -1.747343
PAD_LABEL_VAL = 0


class CTStandardizeD(MapTransform):
    """Exact nnU-Net CTNormalization: clip [p00_5, p99_5] then (x - mean) / std in float32."""

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
                # Preserve MetaTensor metadata cleanly if present
                out = torch.clamp(val.float(), min=self.clip_lo, max=self.clip_hi)
                out = (out - self.mean) / self.std
                d[key] = out
            else:
                arr = np.asarray(val, dtype=np.float32)
                np.clip(arr, self.clip_lo, self.clip_hi, out=arr)
                arr -= self.mean
                arr /= self.std
                d[key] = arr
        return d


def build_cache_preprocess_transform(
    pixdim_xyz: tuple[float, float, float] = (0.349609375, 0.349609375, 0.50),
    body_crop_hu: float = BODY_CROP_HU,
) -> t.Compose:
    """
    Shared deterministic preprocessing for cache_fp16/ construction (no training SpatialPadd).
    Reads raw .nii.gz pair -> (1, X, Y, Z) normalized image & uint8 binary label.
    """
    return t.Compose([
        t.LoadImaged(keys=["image", "label"]),
        t.EnsureChannelFirstd(keys=["image", "label"]),
        t.CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            select_fn=lambda x: x > body_crop_hu,
            allow_smaller=True,
        ),
        t.Spacingd(
            keys=["image", "label"],
            pixdim=pixdim_xyz,
            mode=("bilinear", "nearest"),
        ),
        CTStandardizeD(keys=["image"]),
        t.EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
        t.EnsureTyped(keys=["label"], dtype=torch.uint8, track_meta=False),
    ])


def build_raw_eval_transform(
    pixdim_xyz: tuple[float, float, float] = (0.349609375, 0.349609375, 0.50),
    body_crop_hu: float = BODY_CROP_HU,
) -> t.Compose:
    """
    Raw-.nii.gz evaluation transform keeping MONAI MetaTensor history for Invertd back onto
    the native SimpleITK (Z, Y, X) physical grid. Strictly omits SpatialPadd.
    """
    return t.Compose([
        t.LoadImaged(keys=["image"]),
        t.EnsureChannelFirstd(keys=["image"]),
        t.CropForegroundd(
            keys=["image"],
            source_key="image",
            select_fn=lambda x: x > body_crop_hu,
            allow_smaller=True,
        ),
        t.Spacingd(keys=["image"], pixdim=pixdim_xyz, mode="bilinear"),
        CTStandardizeD(keys=["image"]),
        t.EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=True),
    ])


def build_per_key_spatial_pad(
    patch_size_xyz: tuple[int, int, int] = (160, 160, 96),
    pad_image_val: float = PAD_IMAGE_VAL,
    pad_label_val: int = PAD_LABEL_VAL,
) -> t.Compose:
    """
    Safe separate SpatialPadd per key avoiding MONAI/NumPy's (before, after) tuple pitfall
    where passing (-1.747, 0) wraps -1.747 into uint8(255) inside label tensors.
    """
    return t.Compose([
        t.SpatialPadd(
            keys="image",
            spatial_size=patch_size_xyz,
            mode="constant",
            constant_values=float(pad_image_val),
        ),
        t.SpatialPadd(
            keys="label",
            spatial_size=patch_size_xyz,
            mode="constant",
            constant_values=int(pad_label_val),
        ),
    ])


def pad_volume_if_needed(
    img: np.ndarray,
    lbl: np.ndarray,
    patch_size_xyz: tuple[int, int, int] = (160, 160, 96),
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """
    Pad (1, X, Y, Z) arrays only along axes smaller than patch_size_xyz using
    per-key constants. Returns padded (img, lbl, before_offsets_xyz).
    """
    _, sx, sy, sz = img.shape
    px, py, pz = patch_size_xyz
    if sx >= px and sy >= py and sz >= pz:
        return img, lbl, (0, 0, 0)

    pads = []
    offsets = []
    for cur, target in ((sx, px), (sy, py), (sz, pz)):
        if cur < target:
            diff = target - cur
            b = diff // 2
            a = diff - b
            pads.append((b, a))
            offsets.append(b)
        else:
            pads.append((0, 0))
            offsets.append(0)
    pad_spec = ((0, 0), pads[0], pads[1], pads[2])
    img_p = np.pad(img, pad_spec, mode="constant", constant_values=float(PAD_IMAGE_VAL))
    lbl_p = np.pad(lbl, pad_spec, mode="constant", constant_values=int(PAD_LABEL_VAL))
    return img_p, lbl_p, (offsets[0], offsets[1], offsets[2])


def sample_pos_neg_patch(
    img: np.ndarray,
    lbl: np.ndarray,
    fg_coords: np.ndarray,
    bg_coords: np.ndarray | None,
    rng: np.random.Generator,
    patch_size_xyz: tuple[int, int, int] = (160, 160, 96),
    fg_prob: float = 1.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    O(1) deterministic RandCropByPosNegLabel(pos=1, neg=2) equivalent using precomputed
    foreground/background center indices. Handles < patch_size axes safely via pad_volume_if_needed.
    """
    img_p, lbl_p, (ox, oy, oz) = pad_volume_if_needed(img, lbl, patch_size_xyz)
    _, sx, sy, sz = img_p.shape
    px, py, pz = patch_size_xyz

    use_fg = (len(fg_coords) > 0) and (rng.random() < fg_prob)
    if use_fg:
        idx = int(rng.integers(0, len(fg_coords)))
        cx, cy, cz = int(fg_coords[idx, 0]) + ox, int(fg_coords[idx, 1]) + oy, int(fg_coords[idx, 2]) + oz
    elif bg_coords is not None and len(bg_coords) > 0:
        idx = int(rng.integers(0, len(bg_coords)))
        cx, cy, cz = int(bg_coords[idx, 0]) + ox, int(bg_coords[idx, 1]) + oy, int(bg_coords[idx, 2]) + oz
    else:
        cx = int(rng.integers(0, sx))
        cy = int(rng.integers(0, sy))
        cz = int(rng.integers(0, sz))

    hx, hy, hz = px // 2, py // 2, pz // 2
    x0 = min(max(cx - hx, 0), sx - px)
    y0 = min(max(cy - hy, 0), sy - py)
    z0 = min(max(cz - hz, 0), sz - pz)

    patch_img = np.array(img_p[:, x0:x0 + px, y0:y0 + py, z0:z0 + pz], dtype=np.float32, copy=True)
    patch_lbl = np.array(lbl_p[:, x0:x0 + px, y0:y0 + py, z0:z0 + pz], dtype=np.uint8, copy=True)
    return patch_img, patch_lbl


def apply_shared_patch_augmentations(
    patch_img: np.ndarray,
    patch_lbl: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identical 3D spatial & light CT intensity augmentations shared across all MONAI runs:
      - Random axis flips along X, Y, Z (prob=0.5 each)
      - Random 90-deg in-plane XY rotation (prob=0.25)
      - Random multiplicative scale +/-10% (prob=0.15)
      - Random additive brightness shift +/-0.10 std (prob=0.15)
    """
    for axis in (1, 2, 3):
        if rng.random() < 0.5:
            patch_img = np.flip(patch_img, axis=axis)
            patch_lbl = np.flip(patch_lbl, axis=axis)
    if rng.random() < 0.25:
        k = int(rng.integers(1, 4))
        patch_img = np.rot90(patch_img, k=k, axes=(1, 2))
        patch_lbl = np.rot90(patch_lbl, k=k, axes=(1, 2))
    if rng.random() < 0.15:
        scale = float(rng.uniform(0.90, 1.10))
        patch_img = patch_img * scale
    if rng.random() < 0.15:
        shift = float(rng.uniform(-0.10, 0.10))
        patch_img = patch_img + shift
    return np.ascontiguousarray(patch_img, dtype=np.float32), np.ascontiguousarray(patch_lbl, dtype=np.uint8)

