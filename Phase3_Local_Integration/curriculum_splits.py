# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\curriculum_splits.py
"""
Stage 0 — Curriculum Splits Preparation
========================================
Reads splits_final.json to get all 784 training case IDs.
Loads each GT label NIfTI via SimpleITK (raw, NO resampling).
Counts foreground voxels (label > 0) per case.
Sorts descending → assigns to 3 tiers:
  Tier 1 (Easy):   Top    30% (~235 cases)  — large proximal vessels
  Tier 2 (Medium): Middle 40% (~314 cases)  — mid-vessel
  Tier 3 (Hard):   Bottom 30% (~235 cases)  — thin distal / stenosed

Saves:  Phase3_Local_Integration/curriculum_splits.json
Prints: Tier statistics table (min / max / mean / median / std voxel counts)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import SimpleITK as sitk

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import dataset_paths  # noqa: E402

SPLITS_FILE    = SCRIPT_DIR / "splits_final.json"
OUTPUT_JSON    = SCRIPT_DIR / "curriculum_splits.json"

# ── Tier fractions ────────────────────────────────────────────────────────────
TIER1_FRAC = 0.30   # Easy   — top 30%
TIER2_FRAC = 0.40   # Medium — middle 40%
# Tier 3 = remaining 30%


# ─────────────────────────────────────────────────────────────────────────────

class TierStats(NamedTuple):
    name: str
    n: int
    min_v: int
    max_v: int
    mean_v: float
    median_v: float
    std_v: float


def count_label_voxels(label_path: str) -> int:
    """Load a NIfTI label file via SimpleITK and return foreground voxel count."""
    img = sitk.ReadImage(label_path)
    arr = sitk.GetArrayFromImage(img)   # (Z, Y, X), values 0 or 1
    return int(np.sum(arr > 0))


def compute_tier_stats(name: str, counts: list[int]) -> TierStats:
    a = np.array(counts, dtype=np.int64)
    return TierStats(
        name=name,
        n=len(a),
        min_v=int(a.min()),
        max_v=int(a.max()),
        mean_v=float(a.mean()),
        median_v=float(np.median(a)),
        std_v=float(a.std()),
    )


def print_stats_table(stats_list: list[TierStats]) -> None:
    header = f"{'Tier':<20} {'N':>6} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10} {'Std':>10}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for s in stats_list:
        print(
            f"{s.name:<20} {s.n:>6} {s.min_v:>10,} {s.max_v:>10,} "
            f"{s.mean_v:>10,.1f} {s.median_v:>10,.1f} {s.std_v:>10,.1f}"
        )
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  STAGE 0 — Curriculum Splits Preparation")
    print("=" * 60)

    # 1. Load training IDs from splits_final.json
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(f"Splits file not found: {SPLITS_FILE}")

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    all_train_ids: list[int] = splits.get("train", [])
    print(f"\n[INFO] Total training cases in splits_final.json: {len(all_train_ids)}")

    # 2. Collect voxel counts — skip cases where label file is missing
    print("\n[INFO] Counting foreground voxels per case label (SimpleITK, raw, no resampling)...")
    print("       This may take a few minutes for 784 NIfTI files.\n")

    voxel_counts: dict[int, int] = {}
    skipped: list[int] = []

    for idx, cid in enumerate(all_train_ids):
        label_path = dataset_paths.find_label(cid)
        if label_path is None:
            skipped.append(cid)
            if idx % 50 == 0:
                print(f"  [{idx+1}/{len(all_train_ids)}] Case {cid}: SKIPPED (label not found)")
            continue

        try:
            vc = count_label_voxels(label_path)
            voxel_counts[cid] = vc
        except Exception as exc:
            skipped.append(cid)
            print(f"  [{idx+1}/{len(all_train_ids)}] Case {cid}: ERROR — {exc}")
            continue

        # Progress every 50 cases
        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_train_ids):
            print(f"  [{idx+1}/{len(all_train_ids)}] Case {cid}: {vc:,} voxels")

    n_valid = len(voxel_counts)
    print(f"\n[INFO] Valid cases: {n_valid}  |  Skipped: {len(skipped)}")
    if skipped:
        print(f"[WARN] Skipped case IDs: {skipped}")

    # 3. Sort by voxel count descending (most foreground = easiest)
    sorted_cases: list[tuple[int, int]] = sorted(
        voxel_counts.items(), key=lambda x: x[1], reverse=True
    )
    sorted_ids   = [c for c, _ in sorted_cases]
    sorted_vcnts = [v for _, v in sorted_cases]

    # 4. Assign tiers
    n1 = round(n_valid * TIER1_FRAC)   # top 30%
    n2 = round(n_valid * TIER2_FRAC)   # next 40%
    # n3 = remainder

    tier1_ids = sorted_ids[:n1]
    tier2_ids = sorted_ids[n1 : n1 + n2]
    tier3_ids = sorted_ids[n1 + n2 :]

    tier1_vcnts = sorted_vcnts[:n1]
    tier2_vcnts = sorted_vcnts[n1 : n1 + n2]
    tier3_vcnts = sorted_vcnts[n1 + n2 :]

    print(f"\n[INFO] Tier assignments:")
    print(f"  Tier 1 (Easy)   - top {TIER1_FRAC*100:.0f}%: {len(tier1_ids):4d} cases")
    print(f"  Tier 2 (Medium) - mid {TIER2_FRAC*100:.0f}%: {len(tier2_ids):4d} cases")
    print(f"  Tier 3 (Hard)   - bot {(1-TIER1_FRAC-TIER2_FRAC)*100:.0f}%: {len(tier3_ids):4d} cases")

    # 5. Statistics per tier
    stats_list = [
        compute_tier_stats("Tier 1 - Easy",   tier1_vcnts),
        compute_tier_stats("Tier 2 - Medium",  tier2_vcnts),
        compute_tier_stats("Tier 3 - Hard",    tier3_vcnts),
    ]
    print("\n[TIER STATISTICS - voxel counts per GT mask]")
    print_stats_table(stats_list)

    # Confirm monotonic separation
    t1_mean = stats_list[0].mean_v
    t2_mean = stats_list[1].mean_v
    t3_mean = stats_list[2].mean_v
    if t1_mean > t2_mean > t3_mean:
        print("[OK] Mean voxel counts are monotonically decreasing: Tier1 > Tier2 > Tier3 (PASS)")
    else:
        print("[WARN] Mean voxel counts are NOT monotonically decreasing - check tier fractions!")

    # Boundary values between tiers
    print(f"\n[INFO] Tier boundaries (voxel count thresholds):")
    print(f"  Tier 1 / Tier 2 boundary: {sorted_vcnts[n1-1]:,} -> {sorted_vcnts[n1]:,} voxels")
    print(f"  Tier 2 / Tier 3 boundary: {sorted_vcnts[n1+n2-1]:,} -> {sorted_vcnts[n1+n2]:,} voxels")

    # 6. Save curriculum_splits.json
    tier_stats_dict = {
        f"tier{i+1}": {
            "n":      s.n,
            "min":    s.min_v,
            "max":    s.max_v,
            "mean":   round(s.mean_v, 1),
            "median": round(s.median_v, 1),
            "std":    round(s.std_v, 1),
        }
        for i, s in enumerate(stats_list)
    }

    output = {
        "description": "Curriculum tier assignments for ImageCAS training cases",
        "n_total_valid": n_valid,
        "n_skipped": len(skipped),
        "skipped_ids": skipped,
        "tier_fractions": {
            "tier1_easy": TIER1_FRAC,
            "tier2_medium": TIER2_FRAC,
            "tier3_hard": round(1.0 - TIER1_FRAC - TIER2_FRAC, 2),
        },
        "tier_stats": tier_stats_dict,
        "tier1_easy":   tier1_ids,
        "tier2_medium": tier2_ids,
        "tier3_hard":   tier3_ids,
        "voxel_counts": {str(cid): vc for cid, vc in voxel_counts.items()},
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Saved curriculum_splits.json -> {OUTPUT_JSON}")
    print("\n" + "=" * 60)
    print("  Stage 0 complete. Review tier stats above before")
    print("  proceeding to Stage 1 training.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
