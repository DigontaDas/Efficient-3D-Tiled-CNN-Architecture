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

# ── Root of the ImageCAS dataset ─────────────────────────────────────────────
IMGCAS_BASE = r"F:\All 3d tile cnn dataset\ImageCas with V-net"

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
    # Format A — nested directory
    p = os.path.join(IMGCAS_BASE, f"{case_id}.img.nii", "diao_0.nii")
    if os.path.exists(p):
        return p

    # Format B — flat .nii.gz in range subdirectory
    subdir = _subdir_for(case_id)
    if subdir:
        p = os.path.join(IMGCAS_BASE, subdir, f"{case_id}.img.nii.gz")
        if os.path.exists(p):
            return p

    return None


def find_label(case_id: int) -> str | None:
    """Return the absolute path to the label NIfTI, or None if not found."""
    # Format A — nested directory
    p = os.path.join(IMGCAS_BASE, f"{case_id}.label.nii", "label.nii")
    if os.path.exists(p):
        return p

    # Format B — flat .nii.gz in range subdirectory
    subdir = _subdir_for(case_id)
    if subdir:
        p = os.path.join(IMGCAS_BASE, subdir, f"{case_id}.label.nii.gz")
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
