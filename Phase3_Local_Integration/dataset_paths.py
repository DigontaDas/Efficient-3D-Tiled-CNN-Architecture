"""
dataset_paths.py — Shared ImageCAS Path Resolver
=================================================
Handles the three file-format variants that exist across the 1000-case
ImageCAS dataset on this machine:

  Format A (cases 1-200, nested):
      <BASE>/N.img.nii/diao_0.nii
      <BASE>/N.label.nii/label.nii

  Format B (cases 201-1000, flat .nii.gz in range-subdirs):
      <BASE>/201-400/N.img.nii.gz
      <BASE>/201-400/N.label.nii.gz
      <BASE>/401-600/N.img.nii.gz   ... etc.

Import this module from every evaluation script so path logic
never gets duplicated.
"""

from __future__ import annotations
import os
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dataset_root_from_config(config_path: Path) -> str | None:
    """Read `dataset_root` from config.lab.yaml without requiring hardcoded paths in code."""
    if not config_path.is_file():
        return None
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        cfg = yaml.safe_load(text) or {}
        val = cfg.get("dataset_root")
        if val:
            return str(val)
    except Exception:
        pass
    m = re.search(r"^\s*dataset_root\s*:\s*['\"]?([^'\"#\r\n]+?)['\"]?\s*(?:#.*)?$", text, re.M)
    return m.group(1).strip() if m else None


def _resolve_candidate_bases() -> list[str]:
    """Resolve dataset root(s) from IMAGECAS_DATA_ROOT env var or config.lab.yaml (Rule 1)."""
    bases: list[str] = []
    env_root = os.environ.get("IMAGECAS_DATA_ROOT", "").strip()
    if env_root:
        bases.append(env_root)
    cfg_root = _load_dataset_root_from_config(_REPO_ROOT / "config.lab.yaml")
    if cfg_root and cfg_root not in bases:
        bases.append(cfg_root)
    return bases


# ── Root of the ImageCAS dataset (configured via IMAGECAS_DATA_ROOT or config.lab.yaml) ──
CANDIDATE_BASES = _resolve_candidate_bases()
IMGCAS_BASE = next(
    (b for b in CANDIDATE_BASES if os.path.exists(b)),
    CANDIDATE_BASES[0] if CANDIDATE_BASES else "",
)

# Range-subdir mapping for format B (inclusive ranges)
_RANGE_SUBDIRS: list[tuple[int, int, str]] = [
    (1,    200,  "1-200"),
    (201,  400,  "201-400"),
    (401,  600,  "401-600"),
    (601,  800,  "601-800"),
    (801, 1000,  "801-1000"),
]


def find_image(case_id: int) -> str | None:
    """Return the absolute path to the image NIfTI, or None if not found."""
    for base in CANDIDATE_BASES:
        if not os.path.exists(base):
            continue
        # Format A — nested directory (check diao_0.nii and dia_0.nii)
        for fname in ["diao_0.nii", "dia_0.nii", "diao_0.nii.gz", "dia_0.nii.gz"]:
            p = os.path.join(base, f"{case_id}.img.nii", fname)
            if os.path.exists(p):
                return p

        # Format B — flat .nii.gz in root, range subdirectory, or nested range/range subdirectory
        subdir = _subdir_for(case_id)
        for rel in (
            (f"{case_id}.img.nii.gz",),
            (subdir, f"{case_id}.img.nii.gz") if subdir else (),
            (subdir, subdir, f"{case_id}.img.nii.gz") if subdir else (),
        ):
            if not rel:
                continue
            p = os.path.join(base, *rel)
            if os.path.exists(p):
                return p

    return None


def find_label(case_id: int) -> str | None:
    """Return the absolute path to the label NIfTI, or None if not found."""
    for base in CANDIDATE_BASES:
        if not os.path.exists(base):
            continue
        # Format A — nested directory
        for fname in ["label.nii", "label.nii.gz"]:
            p = os.path.join(base, f"{case_id}.label.nii", fname)
            if os.path.exists(p):
                return p

        # Format B — flat .nii.gz in root, range subdirectory, or nested range/range subdirectory
        subdir = _subdir_for(case_id)
        for rel in (
            (f"{case_id}.label.nii.gz",),
            (subdir, f"{case_id}.label.nii.gz") if subdir else (),
            (subdir, subdir, f"{case_id}.label.nii.gz") if subdir else (),
        ):
            if not rel:
                continue
            p = os.path.join(base, *rel)
            if os.path.exists(p):
                return p

    return None


def find_pair(case_id: int) -> tuple[str | None, str | None]:
    """Return (image_path, label_path) — either may be None."""
    return find_image(case_id), find_label(case_id)


def case_ok(case_id: int) -> bool:
    """True if both image and label exist for this case."""
    img, lbl = find_pair(case_id)
    return img is not None and lbl is not None


def all_valid_ids(max_id: int = 1000) -> list[int]:
    """Return sorted list of all case IDs (1..max_id) that have both files."""
    return [i for i in range(1, max_id + 1) if case_ok(i)]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _subdir_for(case_id: int) -> str | None:
    """Return the range-subdir name for a format-B case_id, or None."""
    for lo, hi, name in _RANGE_SUBDIRS:
        if lo <= case_id <= hi:
            return name
    return None


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("dataset_paths.py self-test")
    print(f"  IMGCAS_BASE = {IMGCAS_BASE}")
    print()

    # Spot-check a few known cases
    tests = [
        (1,   True,  "Format A, first case"),
        (200, True,  "Format A, last"),
        (201, True,  "Format B, first"),
        (400, True,  "Format B, boundary"),
        (851, True,  "Format B, 3D U-Net test set starts"),
        (1000, True, "Format B, last"),
    ]

    all_pass = True
    for case_id, expect_ok, note in tests:
        img = find_image(case_id)
        lbl = find_label(case_id)
        ok  = (img is not None) and (lbl is not None)
        status = "[PASS]" if ok == expect_ok else "[FAIL]"
        if ok != expect_ok:
            all_pass = False
        print(f"  {status} Case {case_id:4d}  ({note})")
        if img:
            print(f"           img: {img}")
        if lbl:
            print(f"           lbl: {lbl}")
        if not img:
            print(f"           img: NOT FOUND")
        if not lbl:
            print(f"           lbl: NOT FOUND")
        print()

    valid = all_valid_ids()
    print(f"  Total valid pairs (1-1000): {len(valid)}")
    print(f"  ID range: {min(valid)} .. {max(valid)}")
    print()
    if all_pass:
        print("[PASS] dataset_paths.py is working correctly.")
    else:
        print("[FAIL] Some cases not found - check paths above.")
        sys.exit(1)
