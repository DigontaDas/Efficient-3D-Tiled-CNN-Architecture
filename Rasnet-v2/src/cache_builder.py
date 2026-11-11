"""
Crash-Safe Atomic float16 Image + uint8 Label Cache Builder (`cache_fp16/`)
with Precomputed fg/bg Center Coordinates, Parallel Processing, Shape Parity Report,
and Visual Patch Overlays.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augmentations import (
    BODY_CROP_HU,
    CT_MEAN,
    CT_STD,
    apply_shared_patch_augmentations,
    build_cache_preprocess_transform,
    sample_pos_neg_patch,
)
from dataset_pairs import (
    build_val20_stratified_ids,
    load_validated_splits,
    resolve_case_pairs,
)

_WORKER_TRANSFORM = None


def _init_cache_worker(pixdim_xyz: tuple[float, float, float], body_crop_hu: float) -> None:
    global _WORKER_TRANSFORM
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    _WORKER_TRANSFORM = build_cache_preprocess_transform(
        pixdim_xyz=pixdim_xyz,
        body_crop_hu=float(body_crop_hu),
    )


def atomic_write_bytes(path: Path, writer_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer_fn(tmp)
    tmp.replace(path)


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    def _write(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            np.save(f, arr, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
    atomic_write_bytes(path, _write)


def _save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    atomic_write_bytes(path, _write)


def case_cache_paths(cache_dir: Path, cid: int) -> dict[str, Path]:
    stem = f"case_{int(cid):04d}"
    return {
        "img": cache_dir / f"{stem}_img.npy",
        "lbl": cache_dir / f"{stem}_lbl.npy",
        "fg": cache_dir / f"{stem}_fg.npy",
        "bg": cache_dir / f"{stem}_bg.npy",
        "meta": cache_dir / f"{stem}_meta.json",
    }


def is_case_cached(cache_dir: Path, cid: int) -> bool:
    ps = case_cache_paths(cache_dir, cid)
    if not all(p.is_file() and p.stat().st_size > 0 for p in ps.values()):
        return False
    try:
        meta = json.loads(ps["meta"].read_text(encoding="utf-8"))
        return int(meta.get("case_id", -1)) == int(cid)
    except Exception:
        return False


def _build_single_case(
    cid: int,
    img_path_str: str,
    lbl_path_str: str,
    cache_dir_str: str,
    pixdim_xyz: tuple[float, float, float],
    body_crop_hu: float,
    max_bg: int,
) -> dict[str, Any]:
    global _WORKER_TRANSFORM
    if _WORKER_TRANSFORM is None:
        _init_cache_worker(pixdim_xyz, body_crop_hu)

    cache_dir = Path(cache_dir_str)
    ps = case_cache_paths(cache_dir, cid)
    out = _WORKER_TRANSFORM({"image": img_path_str, "label": lbl_path_str})
    img_t = out["image"].detach().cpu().numpy().astype(np.float16, copy=False)
    lbl_t = (out["label"].detach().cpu().numpy() > 0).astype(np.uint8, copy=False)
    if img_t.shape != lbl_t.shape or img_t.ndim != 4 or img_t.shape[0] != 1:
        raise RuntimeError(f"Case {cid}: unexpected preprocessed shape {img_t.shape} vs {lbl_t.shape}")

    # Precompute int16 center coordinates in MONAI (X, Y, Z) order for O(1) pos=1:neg=2 sampling
    _, sx, sy, sz = lbl_t.shape
    fg_xyz = np.argwhere(lbl_t[0] > 0).astype(np.int16)
    if len(fg_xyz) == 0:
        raise RuntimeError(f"Case {cid}: empty ground-truth mask after preprocessing")

    fg_vals = img_t[0][lbl_t[0] > 0].astype(np.float32)

    # Per-case deterministic RNG guarantees identical bg_coords regardless of parallel scheduling or resume
    rng = np.random.default_rng(42 + int(cid))
    body_mask = (img_t[0] > np.float16((BODY_CROP_HU - CT_MEAN) / CT_STD + 0.05)) & (lbl_t[0] == 0)
    bg_all = np.argwhere(body_mask)
    if len(bg_all) == 0:
        bg_all = np.argwhere(lbl_t[0] == 0)
    if len(bg_all) > max_bg:
        sel = rng.choice(len(bg_all), size=max_bg, replace=False)
        bg_coords = bg_all[sel].astype(np.int16)
    else:
        bg_coords = bg_all.astype(np.int16)

    _save_npy_atomic(ps["img"], img_t)
    _save_npy_atomic(ps["lbl"], lbl_t)
    _save_npy_atomic(ps["fg"], fg_xyz)
    _save_npy_atomic(ps["bg"], bg_coords)
    meta = {
        "case_id": int(cid),
        "shape_xyz": [int(sx), int(sy), int(sz)],
        "pixdim_xyz": list(pixdim_xyz),
        "n_fg": int(len(fg_xyz)),
        "n_bg_indexed": int(len(bg_coords)),
        "fg_norm_mean": float(fg_vals.mean()),
        "fg_norm_std": float(fg_vals.std()),
        "rng_seed": int(42 + int(cid)),
    }
    _save_json_atomic(ps["meta"], meta)
    return meta


def build_cache(
    config_path: str | Path,
    case_ids_override: list[int] | None = None,
    bg_max_per_case: int | None = None,
    limit: int | None = None,
    max_workers: int = 6,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    cache_dir = root_dir / "cache_fp16"
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = load_validated_splits(cfg["study"]["splits_file"])
    if case_ids_override is not None:
        target_ids = [int(c) for c in case_ids_override]
    else:
        target_ids = sorted(int(c) for c in (splits["train"] + splits["val"]))
    if limit is not None:
        target_ids = target_ids[: int(limit)]

    # Freeze val20_curves_ids.json deterministically if missing
    val20_file = Path(cfg["study"]["val20_ids_file"])
    if not val20_file.is_file():
        build_val20_stratified_ids(
            cfg["study"]["val_armA_baseline_csv"],
            cfg["study"]["splits_file"],
            output_json=val20_file,
            seed=42,
        )

    pairs = resolve_case_pairs(cfg["study"]["dataset_root"], target_ids, check_headers=True)
    pixdim_xyz = tuple(float(v) for v in cfg["preprocessing"]["pixdim_xyz"])
    body_crop_hu = float(cfg["preprocessing"]["body_crop_hu"])
    max_bg = int(bg_max_per_case or cfg["preprocessing"]["bg_coords_max_per_case"])

    built, skipped = 0, 0
    shapes: list[list[int]] = []
    fg_means: list[float] = []
    fg_stds: list[float] = []
    to_build: list[int] = []

    for cid in target_ids:
        ps = case_cache_paths(cache_dir, cid)
        if is_case_cached(cache_dir, cid):
            meta = json.loads(ps["meta"].read_text(encoding="utf-8"))
            shapes.append(meta["shape_xyz"])
            fg_means.append(float(meta["fg_norm_mean"]))
            fg_stds.append(float(meta["fg_norm_std"]))
            skipped += 1
        else:
            to_build.append(cid)

    if to_build:
        workers = max(1, int(max_workers))
        if workers == 1 or len(to_build) == 1:
            _init_cache_worker(pixdim_xyz, body_crop_hu)
            for idx, cid in enumerate(to_build, 1):
                img_path, lbl_path = pairs[cid]
                meta = _build_single_case(
                    cid, str(img_path), str(lbl_path), str(cache_dir), pixdim_xyz, body_crop_hu, max_bg
                )
                shapes.append(meta["shape_xyz"])
                fg_means.append(float(meta["fg_norm_mean"]))
                fg_stds.append(float(meta["fg_norm_std"]))
                built += 1
                if (skipped + built) % 50 == 0 or idx == len(to_build):
                    print(f"[cache_builder] {skipped + built}/{len(target_ids)} | built={built}, cached={skipped}")
        else:
            n_pool = min(workers, len(to_build))
            with ProcessPoolExecutor(
                max_workers=n_pool,
                initializer=_init_cache_worker,
                initargs=(pixdim_xyz, body_crop_hu),
            ) as pool:
                futs = {
                    pool.submit(
                        _build_single_case,
                        cid,
                        str(pairs[cid][0]),
                        str(pairs[cid][1]),
                        str(cache_dir),
                        pixdim_xyz,
                        body_crop_hu,
                        max_bg,
                    ): cid
                    for cid in to_build
                }
                for idx, fut in enumerate(as_completed(futs), 1):
                    meta = fut.result()
                    shapes.append(meta["shape_xyz"])
                    fg_means.append(float(meta["fg_norm_mean"]))
                    fg_stds.append(float(meta["fg_norm_std"]))
                    built += 1
                    if (skipped + built) % 50 == 0 or idx == len(to_build):
                        print(f"[cache_builder] {skipped + built}/{len(target_ids)} | built={built}, cached={skipped}")

    summary = {
        "n_target": len(target_ids),
        "built": built,
        "skipped": skipped,
        "min_shape_xyz": np.min(shapes, axis=0).tolist() if shapes else None,
        "max_shape_xyz": np.max(shapes, axis=0).tolist() if shapes else None,
        "fg_norm_mean_avg": float(np.mean(fg_means)) if fg_means else None,
        "fg_norm_std_avg": float(np.mean(fg_stds)) if fg_stds else None,
    }
    _save_json_atomic(cache_dir / "CACHE_MANIFEST.json", summary)
    return summary


def generate_patch_overlays(
    config_path: str | Path,
    case_ids: list[int],
    output_dir: str | Path,
) -> list[Path]:
    """Generate multi-slice Gate 2 PNG overlays inspecting sampled 160x160x96 pos/neg patches."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cache_dir = Path(cfg["study"]["root_dir"]) / "cache_fp16"
    patch_xyz = tuple(int(v) for v in cfg["preprocessing"]["patch_size_xyz"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    saved: list[Path] = []

    for cid in case_ids:
        ps = case_cache_paths(cache_dir, cid)
        img = np.load(ps["img"])
        lbl = np.load(ps["lbl"])
        fg = np.load(ps["fg"])
        bg = np.load(ps["bg"])

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for row_idx, force_fg in enumerate((True, False)):
            p_img, p_lbl = sample_pos_neg_patch(
                img,
                lbl,
                fg,
                bg,
                rng,
                patch_size_xyz=patch_xyz,
                fg_prob=1.0 if force_fg else 0.0,
            )
            p_img, p_lbl = apply_shared_patch_augmentations(p_img, p_lbl, rng)
            assert set(np.unique(p_lbl).tolist()) <= {0, 1}, f"Corrupt label values: {np.unique(p_lbl)}"
            z_mid = int(np.argmax(p_lbl[0].sum(axis=(0, 1)))) if p_lbl[0].sum() > 0 else patch_xyz[2] // 2
            y_mid = patch_xyz[1] // 2
            x_mid = patch_xyz[0] // 2
            slices = [
                ("Axial (XY)", p_img[0, :, :, z_mid].T, p_lbl[0, :, :, z_mid].T),
                ("Coronal (XZ)", p_img[0, :, y_mid, :].T, p_lbl[0, :, y_mid, :].T),
                ("Sagittal (YZ)", p_img[0, x_mid, :, :].T, p_lbl[0, x_mid, :, :].T),
            ]
            for col_idx, (title, sl_i, sl_l) in enumerate(slices):
                ax = axes[row_idx, col_idx]
                ax.imshow(sl_i, cmap="gray", origin="lower", vmin=-1.75, vmax=3.2)
                if np.any(sl_l > 0):
                    ax.contour(sl_l, levels=[0.5], colors=["lime"], linewidths=1.0)
                kind = "FG-Centered" if force_fg else "BG-Centered"
                ax.set_title(f"Case {cid} ({kind}) {title}\nfg_vox={int(p_lbl.sum())}")
                ax.axis("off")

        fig.tight_layout()
        out_png = out_dir / f"patch_overlay_case_{int(cid):04d}.png"
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        saved.append(out_png)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cases", type=int, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overlays-dir", default=None)
    args = parser.parse_args()
    res = build_cache(args.config, case_ids_override=args.cases, limit=args.limit, max_workers=args.workers)
    print(json.dumps(res, indent=2))
    if args.overlays_dir:
        cids = args.cases if args.cases else [14, 45, 47]
        paths = generate_patch_overlays(args.config, cids[:3], args.overlays_dir)
        print("Saved overlays:", [str(p) for p in paths])


if __name__ == "__main__":
    main()
