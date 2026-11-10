import json
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from benchmark.dataset import resolve_case_pairs
from benchmark.comparison import compare_with_historical


def _write_nifti(path, arr, spacing=(0.5, 0.4, 0.4), origin=(0.0, 0.0, 0.0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    tmp_p = path.with_name(path.name.replace(".img.nii.gz", "_tmp.nii.gz")) if path.name.endswith(".img.nii.gz") else path
    sitk.WriteImage(img, str(tmp_p))
    if tmp_p != path:
        tmp_p.replace(path)


def test_resolve_case_pairs_and_fail_on_duplicates_or_missing(tmp_path):
    sub1 = tmp_path / "1-200" / "1-200"
    sub1.mkdir(parents=True)
    arr = np.zeros((4, 5, 6), dtype=np.float32)
    _write_nifti(sub1 / "14.img.nii.gz", arr)
    _write_nifti(sub1 / "14.label.nii.gz", arr.astype(np.uint8))

    pairs = resolve_case_pairs(str(tmp_path), [14])
    assert 14 in pairs

    # Missing case must fail loudly
    with pytest.raises(FileNotFoundError):
        resolve_case_pairs(str(tmp_path), [14, 99])

    # Duplicate case across subfolders must fail loudly
    sub2 = tmp_path / "201-400" / "201-400"
    sub2.mkdir(parents=True)
    _write_nifti(sub2 / "14.img.nii.gz", arr)
    _write_nifti(sub2 / "14.label.nii.gz", arr.astype(np.uint8))
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_case_pairs(str(tmp_path), [14])


def test_compare_with_historical_matches_exact_ids_and_flags_mismatches():
    new_df = pd.DataFrame({
        "case_id": [851, 852, 853],
        "dice": [0.8000, 0.7500, 0.7000],
        "hd95": [10.0, 12.0, 15.0],
    })
    hist_df = pd.DataFrame({
        "case_id": [852, 851, 854],
        "dice": [0.75005, 0.7000, 0.9000],
        "hd95": [12.0, 10.0, 5.0],
    })
    diff_df, summary = compare_with_historical(new_df, hist_df, metrics=["dice", "hd95"], tolerance=1e-4)
    assert summary["matched_cases"] == 2
    assert summary["only_in_new"] == [853]
    assert summary["only_in_historical"] == [854]
    row_851 = diff_df[diff_df["case_id"] == 851].iloc[0]
    row_852 = diff_df[diff_df["case_id"] == 852].iloc[0]
    assert bool(row_851["dice_exceeds_tol"]) is True
    assert bool(row_852["dice_exceeds_tol"]) is False


def test_execute_benchmark_synthetic_val_and_test_end_to_end(tmp_path):
    import yaml
    from benchmark.run_benchmark import execute_benchmark

    dset = tmp_path / "dataset" / "1-200"
    dset.mkdir(parents=True)
    arr = np.zeros((16, 16, 16), dtype=np.float32)
    arr[4:12, 4:12, 4:12] = 200.0
    lbl = np.zeros((16, 16, 16), dtype=np.uint8)
    lbl[5:11, 5:11, 5:11] = 1

    for cid in (1, 10, 11, 851, 852):
        _write_nifti(dset / f"{cid}.img.nii.gz", arr)
        _write_nifti(dset / f"{cid}.label.nii.gz", lbl)

    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps({"train": [1], "val": [10, 11], "test": [851, 852]}), encoding="utf-8")

    hist_csv = tmp_path / "hist_rasnet.csv"
    pd.DataFrame({
        "case_id": [851, 852],
        "dice": [0.5, 0.5],
        "iou": [0.3, 0.3],
        "precision": [0.5, 0.5],
        "recall": [0.5, 0.5],
        "hd95": [5.0, 5.0],
        "asd": [1.0, 1.0],
        "cldice": [0.5, 0.5],
        "centerline_recall": [0.5, 0.5],
    }).to_csv(hist_csv, index=False)

    cfg_file = tmp_path / "config.yaml"
    cfg = {
        "dataset_root": str(tmp_path / "dataset"),
        "split_file": str(splits_file),
        "output_root": str(tmp_path / "runs"),
        "thresholds_file": str(tmp_path / "thresholds.json"),
        "seed": 42,
        "sw_batch_size": 1,
        "historical_tolerance": 1e-4,
        "threshold_grid": [0.30, 0.50, 0.70],
        "postprocess": {"keep_top_k": 2, "connectivity": 26},
        "historical_csvs": {"rasnet": str(hist_csv)},
        "hypotheses": [{"a": "rasnet", "b": "nnunet", "metric": "dice", "direction": "higher"}],
    }
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # 1. Attempting to run Arm C on test before validation thresholds exist must fail loudly
    with pytest.raises(FileNotFoundError):
        execute_benchmark(cfg_file, split="test", arm_arg="C", models_arg="rasnet,nnunet", synthetic_dry_run=True)

    # 2. Run validation split to select and freeze thresholds.json
    val_dir = execute_benchmark(
        cfg_file, split="val", arm_arg="A", models_arg="rasnet,nnunet", run_id="val_run", synthetic_dry_run=True
    )
    assert (val_dir / "manifest.json").is_file()
    assert (val_dir / "thresholds.json").is_file()
    # Copy thresholds.json to configured path for the 2-model subset test
    (tmp_path / "thresholds.json").write_bytes((val_dir / "thresholds.json").read_bytes())

    # 3. Run test split across Arms A, B, C
    test_dir = execute_benchmark(
        cfg_file, split="test", arm_arg="all", models_arg="rasnet,nnunet", run_id="test_run", synthetic_dry_run=True, export_masks=True
    )
    manifest = json.loads((test_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "test"
    assert manifest["arms"] == ["A", "B", "C"]
    assert manifest["thresholds_sha256"] is not None
    for arm in ("A", "B", "C"):
        assert (test_dir / f"per_case_arm_{arm}.csv").is_file()
        assert (test_dir / f"summary_arm_{arm}.csv").is_file()
    assert (test_dir / "diff_vs_historical_summary.json").is_file()
