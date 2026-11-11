"""Deterministic ImageCAS split, header geometry, and stratified val20 helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk


def image_geometry(image: sitk.Image) -> dict[str, Any]:
    return {
        "size": image.GetSize(),
        "origin": image.GetOrigin(),
        "spacing": image.GetSpacing(),
        "direction": image.GetDirection(),
    }


def check_geometry(a: dict[str, Any], b: dict[str, Any], atol: float = 1e-5) -> None:
    for key in ("size", "origin", "spacing", "direction"):
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or not np.allclose(x, y, atol=atol, rtol=0):
            raise ValueError(f"Physical geometry mismatch: {key}: {x} vs {y}")


def _read_header_geometry(path: Path) -> dict[str, Any]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return {
        "size": reader.GetSize(),
        "origin": reader.GetOrigin(),
        "spacing": reader.GetSpacing(),
        "direction": reader.GetDirection(),
    }


def validate_splits(splits: dict[str, Any]) -> dict[str, set[int]]:
    parts: dict[str, set[int]] = {}
    for name in ("train", "val", "test"):
        values = splits.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(isinstance(x, bool) or not isinstance(x, int) for x in values)
        ):
            raise ValueError(f"Invalid or empty {name} IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name} IDs")
        parts[name] = set(values)
    if parts["train"] & parts["val"] or parts["train"] & parts["test"] or parts["val"] & parts["test"]:
        raise ValueError("Overlapping splits")
    if (parts["train"] | parts["val"]) & set(range(851, 1001)):
        raise ValueError("Locked test ID in train or validation")
    if not parts["test"] <= set(range(851, 1001)):
        raise ValueError("ImageCAS test IDs must be in locked range")
    return parts


def load_validated_splits(splits_file: str | Path) -> dict[str, Any]:
    p = Path(splits_file)
    data = json.loads(p.read_text(encoding="utf-8"))
    validate_splits(data)
    return data


def resolve_case_pairs(
    dataset_root: str | Path,
    case_ids: list[int],
    check_headers: bool = True,
) -> dict[int, tuple[Path, Path]]:
    """
    Scan dataset_root for {cid}.img.nii.gz and {cid}.label.nii.gz pairs.
    Fails on duplicates, missing files, or mismatched physical geometry.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not case_ids:
        raise ValueError("Empty case_ids list")

    catalog: dict[int, tuple[Path, Path]] = {}
    for img_path in sorted(root.rglob("*.img.nii.gz")):
        stem = img_path.name[: -len(".img.nii.gz")]
        try:
            cid = int(stem)
        except ValueError:
            continue
        lbl_path = img_path.with_name(f"{cid}.label.nii.gz")
        if not lbl_path.is_file():
            raise FileNotFoundError(f"Missing label for case {cid}: {lbl_path}")
        if cid in catalog:
            raise ValueError(f"Duplicate case ID {cid} found at {img_path} and {catalog[cid][0]}")
        catalog[cid] = (img_path, lbl_path)

    resolved: dict[int, tuple[Path, Path]] = {}
    for cid in case_ids:
        cid_int = int(cid)
        if cid_int not in catalog:
            raise FileNotFoundError(f"Missing case ID {cid_int} under {root}")
        img_path, lbl_path = catalog[cid_int]
        if check_headers:
            check_geometry(_read_header_geometry(img_path), _read_header_geometry(lbl_path))
        resolved[cid_int] = (img_path, lbl_path)
    return resolved


def validate_case_files(pairs: dict[int, tuple[Path, Path]]) -> None:
    if not pairs:
        raise ValueError("Empty case list")
    for cid, (image, label) in pairs.items():
        for f in (image, label):
            if not Path(f).is_file():
                raise FileNotFoundError(f"Case {cid}: {f}")
        check_geometry(
            _read_header_geometry(Path(image)),
            _read_header_geometry(Path(label)),
        )


def build_val20_stratified_ids(
    val_arm_a_csv: str | Path,
    splits_file: str | Path,
    output_json: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Construct the frozen N=20 val20_curves subset from N=94 validation cases:
    4 quartiles over reconstructed GT coronary voxel count sorted secondary by v1 Dice,
    picking 5 representative cases per quartile cleanly with fixed seed=42.
    """
    splits = load_validated_splits(splits_file)
    val_set = set(int(x) for x in splits["val"])
    df = pd.read_csv(val_arm_a_csv)
    sub = df[df["model"] == "rasnet"].copy()
    sub["case_id"] = sub["case_id"].astype(int)
    if set(sub["case_id"]) != val_set:
        raise ValueError("val_arm_a_csv rasnet case_ids do not match canonical val split")

    # Reconstruct exact GT foreground voxel count from TP = precision * pred_vox, GT = TP / recall
    sub["pred_vox"] = sub["voxels_retained"].astype(float)
    sub["gt_vox_est"] = (sub["pred_vox"] * sub["precision"] / sub["recall"]).round().astype(int)
    sub["vol_quartile"] = pd.qcut(sub["gt_vox_est"], q=4, labels=[0, 1, 2, 3]).astype(int)

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    strata_summary: list[dict[str, Any]] = []
    for q in range(4):
        grp = sub[sub["vol_quartile"] == q].sort_values(["dice", "case_id"]).reset_index(drop=True)
        # Span difficulty evenly within each volume quartile (low -> high Dice)
        idx = np.linspace(0, len(grp) - 1, num=5, dtype=int)
        # Add deterministic tie-safe verification that indices are unique
        if len(set(idx.tolist())) < 5:
            idx = rng.choice(len(grp), size=5, replace=False)
        picks = grp.iloc[idx]
        cids = [int(c) for c in picks["case_id"].tolist()]
        chosen.extend(cids)
        strata_summary.append({
            "quartile": int(q),
            "n_pool": int(len(grp)),
            "gt_vox_range": [int(grp["gt_vox_est"].min()), int(grp["gt_vox_est"].max())],
            "dice_range": [round(float(grp["dice"].min()), 4), round(float(grp["dice"].max()), 4)],
            "selected_case_ids": sorted(cids),
        })

    chosen = sorted(chosen)
    if len(chosen) != 20 or len(set(chosen)) != 20 or not (set(chosen) <= val_set):
        raise RuntimeError("Invalid val20_curves selection")

    payload = {
        "n_cases": 20,
        "seed": int(seed),
        "method": "4 GT-voxel quartiles x 5 evenly spaced v1-Dice difficulty ranks",
        "case_ids": chosen,
        "strata": strata_summary,
        "sha256_case_ids": hashlib.sha256(json.dumps(chosen).encode()).hexdigest(),
    }
    if output_json is not None:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.with_suffix(".tmp")
        tmp_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_p.replace(out_p)
    return payload

