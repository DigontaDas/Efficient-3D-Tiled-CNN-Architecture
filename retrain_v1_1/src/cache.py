"""Transform-hashed, crash-safe blosc2 ZSTD-1 Persistent Cache for Strategy A (`retrain_v1_1/cache/`).

Keyed strictly by `compute_transform_hash(preproc_cfg)[:16]`, making v1's LAS/RAS cache-reuse bug
impossible. Stores chunked `.b2nd` float16 images and uint8 labels (~90 MB/case, ~71 GB for 784
train+val scans vs ~154 GB uncompressed) with fast direct patch slicing during training.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import blosc2
import numpy as np

from benchmark.dataset import resolve_case_pairs
from benchmark.preflight import validate_splits
from retrain_v1_1.src.transforms import (
    CT_MEAN,
    CT_STD,
    build_deterministic_preprocess,
    compute_transform_hash,
)

_WORKER_TF = None
_WORKER_PREPROC_CFG: dict[str, Any] | None = None
_WORKER_HASH: str | None = None


def case_cache_dir(cache_root: str | Path, transform_sha256: str) -> Path:
    return Path(cache_root) / f"tf_{str(transform_sha256)[:16]}"


def case_cache_paths(cache_dir: Path, cid: int) -> dict[str, Path]:
    stem = f"case_{int(cid):04d}"
    return {
        "img": cache_dir / f"{stem}_img.b2nd",
        "lbl": cache_dir / f"{stem}_lbl.b2nd",
        "fg": cache_dir / f"{stem}_fg.npy",
        "bg": cache_dir / f"{stem}_bg.npy",
        "meta": cache_dir / f"{stem}_meta.json",
    }


def _save_b2nd_atomic(path: Path, arr: np.ndarray, chunks: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    eff_chunks = tuple(min(int(c), int(s)) for c, s in zip(chunks, arr.shape))
    blosc2.asarray(
        np.ascontiguousarray(arr),
        urlpath=str(tmp),
        chunks=eff_chunks,
        cparams=blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=1),
        mode="w",
    )
    tmp.replace(path)


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def save_cached_case_atomic(
    cache_dir: Path,
    cid: int,
    img_fp16: np.ndarray,
    lbl_u8: np.ndarray,
    fg_xyz: np.ndarray,
    bg_xyz: np.ndarray,
    meta: dict[str, Any],
) -> None:
    """Atomically persist one preprocessed volume (meta written last as commit marker)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ps = case_cache_paths(cache_dir, cid)
    _save_b2nd_atomic(ps["img"], np.asarray(img_fp16, dtype=np.float16), chunks=(1, 64, 64, 48))
    _save_b2nd_atomic(ps["lbl"], np.asarray(lbl_u8, dtype=np.uint8), chunks=(1, 64, 64, 48))
    _save_npy_atomic(ps["fg"], np.asarray(fg_xyz, dtype=np.int16))
    _save_npy_atomic(ps["bg"], np.asarray(bg_xyz, dtype=np.int16))
    _save_json_atomic(ps["meta"], meta)


def is_case_cached(cache_dir: Path, cid: int, expected_transform_sha256: str) -> bool:
    ps = case_cache_paths(cache_dir, cid)
    if not all(p.is_file() and p.stat().st_size > 0 for p in ps.values()):
        return False
    try:
        meta = json.loads(ps["meta"].read_text(encoding="utf-8"))
        return (
            int(meta.get("case_id", -1)) == int(cid)
            and str(meta.get("transform_sha256", "")) == str(expected_transform_sha256)
        )
    except Exception:
        return False


def load_cached_case(
    cache_dir: Path,
    cid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load full cached arrays for one case."""
    ps = case_cache_paths(cache_dir, cid)
    img = blosc2.open(str(ps["img"]), mode="r")[:]
    lbl = blosc2.open(str(ps["lbl"]), mode="r")[:]
    fg = np.load(ps["fg"], allow_pickle=False)
    bg = np.load(ps["bg"], allow_pickle=False)
    meta = json.loads(ps["meta"].read_text(encoding="utf-8"))
    return img, lbl, fg, bg, meta


def open_cached_case_handles(cache_dir: Path, cid: int):
    """Open chunked blosc2 handles for fast sub-volume patch slicing without full decompression."""
    ps = case_cache_paths(cache_dir, cid)
    img_h = blosc2.open(str(ps["img"]), mode="r")
    lbl_h = blosc2.open(str(ps["lbl"]), mode="r")
    fg = np.load(ps["fg"], mmap_mode="r", allow_pickle=False)
    bg = np.load(ps["bg"], mmap_mode="r", allow_pickle=False)
    return img_h, lbl_h, fg, bg


def _init_worker(preproc_cfg: dict[str, Any], tf_hash: str) -> None:
    global _WORKER_TF, _WORKER_PREPROC_CFG, _WORKER_HASH
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    blosc2.set_nthreads(2)
    _WORKER_PREPROC_CFG = preproc_cfg
    _WORKER_HASH = tf_hash
    _WORKER_TF = build_deterministic_preprocess(preproc_cfg, with_label=True, track_meta=False)


def _process_one_case(
    cid: int,
    img_path: str,
    lbl_path: str,
    cache_dir_str: str,
    preproc_cfg: dict[str, Any],
    tf_hash: str,
) -> dict[str, Any]:
    global _WORKER_TF
    if _WORKER_TF is None:
        _init_worker(preproc_cfg, tf_hash)

    out = _WORKER_TF({"image": img_path, "label": lbl_path})
    img_t = out["image"].detach().cpu().numpy().astype(np.float16, copy=False)
    lbl_t = (out["label"].detach().cpu().numpy() > 0).astype(np.uint8, copy=False)
    if img_t.shape != lbl_t.shape or img_t.ndim != 4 or img_t.shape[0] != 1:
        raise RuntimeError(f"Case {cid}: unexpected shape {img_t.shape} vs {lbl_t.shape}")

    _, sx, sy, sz = lbl_t.shape
    fg_xyz = np.argwhere(lbl_t[0] > 0).astype(np.int16)
    if len(fg_xyz) == 0:
        raise RuntimeError(f"Case {cid}: empty GT mask after spacing")

    # Deterministic per-case background index sampling (prefer body voxels above air floor)
    rng = np.random.default_rng(42 + int(cid))
    max_bg = int(preproc_cfg.get("bg_coords_max_per_case", 50000))
    air_floor = np.float16((-950.0 - CT_MEAN) / CT_STD + 0.05)
    body_bg = np.argwhere((img_t[0] > air_floor) & (lbl_t[0] == 0))
    if len(body_bg) == 0:
        body_bg = np.argwhere(lbl_t[0] == 0)
    if len(body_bg) > max_bg:
        sel = rng.choice(len(body_bg), size=max_bg, replace=False)
        bg_xyz = body_bg[sel].astype(np.int16)
    else:
        bg_xyz = body_bg.astype(np.int16)

    meta = {
        "case_id": int(cid),
        "transform_sha256": str(tf_hash),
        "shape_xyz": [int(sx), int(sy), int(sz)],
        "pixdim_xyz": [float(v) for v in preproc_cfg["pixdim_xyz"]],
        "n_fg": int(len(fg_xyz)),
        "n_bg_indexed": int(len(bg_xyz)),
    }
    save_cached_case_atomic(Path(cache_dir_str), cid, img_t, lbl_t, fg_xyz, bg_xyz, meta)
    return meta


def check_disk_headroom_gb(path: Path, min_free_gb: float = 25.0) -> float:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(str(path)).free / (1024 ** 3)
    if free_gb < float(min_free_gb):
        raise RuntimeError(
            f"Insufficient free disk space at {path}: {free_gb:.2f} GB < required {min_free_gb:.2f} GB"
        )
    return float(free_gb)


def build_transform_hashed_cache(
    cfg: dict[str, Any],
    repo_root: Path,
    dataset_root: str | Path,
    case_ids: list[int] | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Build or resume the transform-hashed blosc2 cache for train + val cases."""
    preproc_cfg = cfg["preprocessing"]
    tf_hash = compute_transform_hash(preproc_cfg)
    study_dir = repo_root / cfg["study"]["study_dir"]
    cache_dir = case_cache_dir(study_dir / "cache", tf_hash)
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits_raw = json.loads((repo_root / cfg["study"]["splits_file"]).read_text(encoding="utf-8"))
    parts = validate_splits(splits_raw)
    if case_ids is None:
        target_ids = sorted(int(c) for c in (parts["train"] | parts["val"]))
    else:
        target_ids = sorted(int(c) for c in case_ids)

    pairs = resolve_case_pairs(str(dataset_root), target_ids, check_headers=False)
    to_build: list[int] = []
    skipped = 0
    for cid in target_ids:
        if is_case_cached(cache_dir, cid, tf_hash):
            skipped += 1
        else:
            to_build.append(cid)

    min_free = float(cfg["training"].get("min_free_disk_gb", 25.0))
    free_gb = check_disk_headroom_gb(cache_dir, min_free_gb=min_free)
    print(
        f"[cache] dir={cache_dir.name} | total={len(target_ids)}, already_cached={skipped}, "
        f"todo={len(to_build)} | free_disk={free_gb:.1f} GB",
        flush=True,
    )

    built = 0
    t0 = time.time()
    if to_build:
        workers = max(1, int(max_workers))
        if workers == 1 or len(to_build) == 1:
            _init_worker(preproc_cfg, tf_hash)
            for idx, cid in enumerate(to_build, 1):
                if idx % 25 == 1:
                    check_disk_headroom_gb(cache_dir, min_free_gb=min_free)
                img_p, lbl_p = pairs[cid]
                _process_one_case(cid, str(img_p), str(lbl_p), str(cache_dir), preproc_cfg, tf_hash)
                built += 1
                if (skipped + built) % 50 == 0 or idx == len(to_build):
                    free_now = shutil.disk_usage(str(cache_dir)).free / (1024 ** 3)
                    print(
                        f"[cache] {skipped + built}/{len(target_ids)} ({time.time() - t0:.0f}s) | "
                        f"free={free_now:.1f} GB",
                        flush=True,
                    )
        else:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(to_build)),
                initializer=_init_worker,
                initargs=(preproc_cfg, tf_hash),
            ) as pool:
                futs = {
                    pool.submit(
                        _process_one_case,
                        cid,
                        str(pairs[cid][0]),
                        str(pairs[cid][1]),
                        str(cache_dir),
                        preproc_cfg,
                        tf_hash,
                    ): cid
                    for cid in to_build
                }
                for idx, fut in enumerate(as_completed(futs), 1):
                    fut.result()
                    built += 1
                    if idx % 25 == 0:
                        check_disk_headroom_gb(cache_dir, min_free_gb=min_free)
                    if (skipped + built) % 50 == 0 or idx == len(to_build):
                        free_now = shutil.disk_usage(str(cache_dir)).free / (1024 ** 3)
                        print(
                            f"[cache] {skipped + built}/{len(target_ids)} ({time.time() - t0:.0f}s) | "
                            f"free={free_now:.1f} GB",
                            flush=True,
                        )

    total_bytes = sum(p.stat().st_size for p in cache_dir.iterdir() if p.is_file())
    summary = {
        "transform_sha256": tf_hash,
        "cache_dir": str(cache_dir),
        "n_target": len(target_ids),
        "already_cached": skipped,
        "built_now": built,
        "cache_size_gb": round(total_bytes / (1024 ** 3), 3),
        "free_disk_gb": round(shutil.disk_usage(str(cache_dir)).free / (1024 ** 3), 2),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save_json_atomic(cache_dir / "CACHE_MANIFEST.json", summary)
    return summary
