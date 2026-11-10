"""Deterministic ImageCAS dataset pair resolution and header geometry verification."""
from pathlib import Path
import SimpleITK as sitk
from benchmark.metrics import check_geometry


def _read_header_geometry(path: Path) -> dict:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return {
        "size": reader.GetSize(),
        "origin": reader.GetOrigin(),
        "spacing": reader.GetSpacing(),
        "direction": reader.GetDirection(),
    }


def resolve_case_pairs(dataset_root: str, case_ids: list[int], check_headers: bool = True) -> dict[int, tuple[Path, Path]]:
    """
    Scan dataset_root for {cid}.img.nii.gz and {cid}.label.nii.gz pairs.
    Fails on duplicate case IDs across subfolders, missing files, or mismatched physical geometry.
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
        if cid not in catalog:
            raise FileNotFoundError(f"Case ID {cid} not found under {root}")
        img_p, lbl_p = catalog[cid]
        if check_headers:
            check_geometry(_read_header_geometry(img_p), _read_header_geometry(lbl_p), atol=1e-4)
        resolved[int(cid)] = (img_p, lbl_p)

    return resolved

