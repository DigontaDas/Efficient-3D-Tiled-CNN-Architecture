import json
import numpy as np
import pytest
from scipy.ndimage import binary_propagation
from benchmark.postprocess import prune_components, hysteresis_threshold_3d
from benchmark.thresholds import select_validation_thresholds, save_thresholds_json, load_and_validate_thresholds


def test_prune_components_keeps_top_k_and_logs_volumes():
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    # Component 1: 3x3x3 = 27 voxels
    mask[1:4, 1:4, 1:4] = 1
    # Component 2: 2x2x3 = 12 voxels
    mask[6:8, 6:8, 6:9] = 1
    # Component 3: 1x2x2 = 4 voxels
    mask[10:11, 9:11, 9:11] = 1

    spacing_zyx = (0.5, 0.4, 0.4)  # voxel volume = 0.08 mm^3
    cleaned, stats = prune_components(mask, spacing_zyx=spacing_zyx, keep_top_k=2)

    assert cleaned.shape == mask.shape
    assert int(cleaned.sum()) == 39
    assert int(cleaned[10:11, 9:11, 9:11].sum()) == 0
    assert stats["components_total"] == 3
    assert stats["components_retained"] == 2
    assert stats["components_removed"] == 1
    assert stats["voxels_retained"] == 39
    assert stats["voxels_removed"] == 4
    assert stats["volume_retained_mm3"] == pytest.approx(39 * 0.08)
    assert stats["volume_removed_mm3"] == pytest.approx(4 * 0.08)


def test_validation_threshold_selection_and_freezing(tmp_path):
    splits = {"train": [1, 2], "val": [10, 11], "test": [851, 852]}
    gt = np.zeros((6, 6, 6), dtype=bool)
    gt[2:4, 2:4, 2:4] = True

    # Construct probability maps where threshold 0.60 perfectly recovers gt
    prob = np.full((6, 6, 6), 0.20, dtype=np.float32)
    prob[1:5, 1:5, 1:5] = 0.52  # false positive halo at tau=0.50
    prob[2:4, 2:4, 2:4] = 0.80  # true core at tau=0.60

    val_probs = {
        "rasnet": {10: prob, 11: prob},
    }
    val_gts = {10: (gt, (0.5, 0.5, 0.5)), 11: (gt, (0.5, 0.5, 0.5))}
    grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    sel = select_validation_thresholds(
        val_probs=val_probs,
        val_gts=val_gts,
        splits=splits,
        grid=grid,
        keep_top_k=2,
    )
    assert sel["split"] == "val"
    assert sel["case_ids"] == [10, 11]
    assert sel["thresholds"]["rasnet"] == pytest.approx(0.55)  # tie between 0.55..0.70 broken closest to 0.50

    out_json = tmp_path / "thresholds.json"
    digest = save_thresholds_json(sel, str(out_json))
    loaded, loaded_sha = load_and_validate_thresholds(str(out_json), splits, ["rasnet"])
    assert loaded_sha == digest
    assert loaded["thresholds"]["rasnet"] == pytest.approx(0.55)


def test_hysteresis_matches_scipy_binary_propagation_and_rejects_unseeded_blobs():
    prob = np.full((16, 16, 16), 0.05, dtype=np.float32)
    # 1. Proximal main vessel core (high probability = 0.85, 18 voxels)
    prob[2:4, 2:5, 2:5] = 0.85
    # 2. Connected distal vessel branch (low probability = 0.24, connected diagonally at (4,5,5))
    prob[4, 5, 5] = 0.24
    prob[5, 6, 6] = 0.22
    prob[6, 7, 7] = 0.21
    # 3. Disconnected background false-positive island (medium probability = 0.42 < tau_high=0.50)
    prob[11:14, 11:14, 11:14] = 0.42

    tau_low, tau_high = 0.20, 0.50
    cleaned, stats = hysteresis_threshold_3d(
        prob, spacing_zyx=(0.5, 0.35, 0.35), tau_low=tau_low, tau_high=tau_high,
        min_seed_voxels=1, keep_top_k=None, connectivity=26
    )

    expected_scipy = binary_propagation(
        prob >= tau_high,
        structure=np.ones((3, 3, 3), dtype=bool),
        mask=(prob >= tau_low),
    ).astype(np.uint8)

    np.testing.assert_array_equal(cleaned, expected_scipy)
    assert int(cleaned.sum()) == 18 + 3  # core + 3 distal voxels recovered
    assert int(cleaned[11:14, 11:14, 11:14].sum()) == 0  # unseeded island rejected
    assert stats["components_total"] == 2
    assert stats["components_seeded"] == 1
    assert stats["components_retained"] == 1
    assert stats["components_removed"] == 1
    assert stats["voxels_retained"] == 21
    assert stats["voxels_removed"] == 27


def test_hysteresis_min_seed_voxels_and_top_k():
    prob = np.zeros((16, 16, 16), dtype=np.float32)
    # Component A (LCA): 20 low voxels, 10 high-seed voxels
    prob[1:3, 1:3, 1:6] = 0.30
    prob[1:3, 1:3, 1:3] = 0.70  # 8 high voxels
    prob[1, 1, 3:5] = 0.70      # +2 = 10 high voxels
    # Component B (RCA): 12 low voxels, 6 high-seed voxels
    prob[6:8, 6:8, 6:9] = 0.30
    prob[6:8, 6:8, 6:7] = 0.75  # 4 high voxels
    prob[6, 6, 7:9] = 0.75      # +2 = 6 high voxels
    # Component C (Speckle): 8 low voxels, only 1 high-seed voxel
    prob[12:14, 12:14, 12:14] = 0.30
    prob[12, 12, 12] = 0.65     # 1 high voxel

    cleaned, stats = hysteresis_threshold_3d(
        prob, spacing_zyx=(0.5, 0.5, 0.5), tau_low=0.25, tau_high=0.50,
        min_seed_voxels=5, keep_top_k=2, connectivity=26
    )
    assert stats["components_total"] == 3
    assert stats["components_seeded"] == 2
    assert stats["components_retained"] == 2
    assert int(cleaned.sum()) == 20 + 12
    assert int(cleaned[12:14, 12:14, 12:14].sum()) == 0


