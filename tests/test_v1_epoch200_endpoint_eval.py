"""Synthetic CPU unit tests for V1 epoch-200 endpoint evaluation (tests-first)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark import endpoint_eval as ee

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_img_nifti(sitk, arr_zyx: np.ndarray, out_img_path: Path, spacing=(0.5, 0.5, 0.5),
                     origin=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
    img = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    tmp = out_img_path.with_name(out_img_path.name.replace(".img.nii.gz", "_img_tmp.nii.gz"))
    sitk.WriteImage(img, str(tmp))
    tmp.replace(out_img_path)
    return img


def _write_synthetic_case(sitk, root: Path, cid: int, shape=(24, 28, 32),
                          spacing=(0.45, 0.65, 1.25),
                          origin=(15.2, -34.8, 102.0),
                          direction=(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
                          gt_slice=(slice(7, 11), slice(19, 23), slice(23, 28))):
    arr = np.full(shape, -100.0, dtype=np.float32)
    arr[2:-2, 3:-3, 4:-2] = 50.0
    arr[gt_slice] = 750.0

    case_dir = root / "1-200"
    case_dir.mkdir(parents=True, exist_ok=True)
    img_path = case_dir / f"{cid}.img.nii.gz"
    img = _write_img_nifti(sitk, arr, img_path, spacing=spacing, origin=origin, direction=direction)

    gt = np.zeros(shape, dtype=np.uint8)
    gt[gt_slice] = 1
    lbl = sitk.GetImageFromArray(gt)
    lbl.CopyInformation(img)
    lbl_path = case_dir / f"{cid}.label.nii.gz"
    sitk.WriteImage(lbl, str(lbl_path))
    return img_path, lbl_path


def _make_endpoint_fixture(tmp_path: Path):
    import torch
    from Phase3_Local_Integration.rasnet_model import RASNet

    ckpt_dir = tmp_path / "recovered"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    net = RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)
    rasnet_ckpt = ckpt_dir / "rasnet_last.pth"
    torch.save(net.state_dict(), rasnet_ckpt)

    plans = {
        "plans_name": "nnUNetPlans",
        "configurations": {
            "3d_fullres": {
                "patch_size": [96, 160, 160],
                "spacing": [0.5, 0.349609375, 0.349609375],
                "normalization_schemes": ["CTNormalization"],
                "architecture": {"network_class_name": "PlainConvUNet"},
            }
        },
    }
    dataset_json = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "vessel": 1}, "file_ending": ".nii.gz"}

    nnunet_folder = ckpt_dir / "nnunet_native_model" / "Dataset501_CoronarySeg" / "nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres"
    fold_dir = nnunet_folder / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    plans_path = nnunet_folder / "plans.json"
    dataset_path = nnunet_folder / "dataset.json"
    plans_path.write_text(json.dumps(plans, indent=2), encoding="utf-8")
    dataset_path.write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    nnunet_ckpt_obj = {
        "current_epoch": 200,
        "trainer_name": "nnUNetTrainer_200epochs",
        "init_args": {
            "plans": plans,
            "configuration": "3d_fullres",
            "fold": 0,
            "dataset_json": dataset_json,
        },
        "network_weights": {"dummy": torch.zeros(1)},
    }
    nnunet_last_pt = ckpt_dir / "nnunet_last.pt"
    ckpt_final = fold_dir / "checkpoint_final.pth"
    torch.save(nnunet_ckpt_obj, nnunet_last_pt)
    ckpt_final.write_bytes(nnunet_last_pt.read_bytes())

    hashes = {
        "rasnet_last": _sha256_file(rasnet_ckpt),
        "nnunet_last": _sha256_file(nnunet_last_pt),
        "nnunet_plans": _sha256_file(plans_path),
        "nnunet_dataset": _sha256_file(dataset_path),
    }
    return {
        "rasnet_ckpt": rasnet_ckpt,
        "nnunet_last_pt": nnunet_last_pt,
        "nnunet_folder": nnunet_folder,
        "ckpt_final": ckpt_final,
        "plans_path": plans_path,
        "dataset_path": dataset_path,
        "plans": plans,
        "dataset_json": dataset_json,
        "hashes": hashes,
    }


# ---------------------------------------------------------------- 1. Existing BEST-only guards
def test_existing_best_only_guards_remain_intact(tmp_path):
    import yaml
    from benchmark.adapters.nnunet_adapter import build_nnunet_predictor
    from benchmark.run_benchmark import execute_benchmark
    from benchmark import symmetric_eval as se

    sitk = pytest.importorskip("SimpleITK")
    ds = tmp_path / "ds"
    _write_synthetic_case(sitk, ds, 14)
    _write_synthetic_case(sitk, ds, 851)
    splits = {"train": [14], "val": [14], "test": [851]}
    # Make disjoint splits for validate_splits
    _write_synthetic_case(sitk, ds, 1)
    splits = {"train": [1], "val": [14], "test": [851]}
    splits_p = tmp_path / "splits.json"
    splits_p.write_text(json.dumps(splits), encoding="utf-8")

    last_ckpt = tmp_path / "rasnet_last.pth"
    last_ckpt.write_bytes(b"not-a-best-ckpt")
    cfg = {
        "dataset_root": str(ds),
        "split_file": str(splits_p),
        "output_root": str(tmp_path / "bench_out"),
        "cache_dir": str(tmp_path / "cache"),
        "thresholds_file": str(tmp_path / "thresholds.json"),
        "seed": 42,
        "sw_batch_size": 1,
        "threshold_grid": [0.5],
        "postprocess": {"keep_top_k": 2, "connectivity": 26},
        "checkpoints": {"rasnet": str(last_ckpt)},
        "hypotheses": [],
    }
    cfg_p = tmp_path / "cfg.yaml"
    cfg_p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing non-best checkpoint"):
        execute_benchmark(
            config_path=str(cfg_p),
            split="val",
            arm_arg="A",
            models_arg="rasnet",
            run_id="best_guard_test",
            device="cpu",
            synthetic_dry_run=False,
        )

    nn_dir = tmp_path / "nn_folder"
    (nn_dir / "fold_0").mkdir(parents=True)
    (nn_dir / "plans.json").write_text("{}", encoding="utf-8")
    (nn_dir / "dataset.json").write_text("{}", encoding="utf-8")
    (nn_dir / "fold_0" / "checkpoint_final.pth").write_bytes(b"final")
    with pytest.raises(ValueError, match="Refusing non-best nnU-Net checkpoint"):
        build_nnunet_predictor(str(nn_dir), fold=0, checkpoint_name="checkpoint_final.pth", device="cpu")

    assert se.EXPECTED_CHECKPOINT_SHA256["rasnet"] != ee.EXPECTED_RASNET_LAST_SHA256
    assert se.EXPECTED_CHECKPOINT_SHA256["nnunet"] != ee.EXPECTED_NNUNET_LAST_SHA256
    for prot in se.PROTECTED_RESULT_DIRS:
        with pytest.raises(ValueError, match="protected"):
            se.check_out_dir(REPO / prot, REPO, resume=False)


# ---------------------------------------------------------------- 2. Frozen constants & checkpoint provenance
def test_endpoint_constants_match_m1_provenance():
    assert ee.CHECKPOINT_ROLE == "final_epoch200"
    assert ee.TARGET_EPOCH == 200
    assert ee.EXPECTED_RASNET_LAST_SHA256 == "71c9e5ef4a02849686c99ccb2d1544ff78c78187e1c0d53e7b22884c708c1b95"
    assert ee.EXPECTED_NNUNET_LAST_SHA256 == "2105bd7238fff7485f1a8d72622f977f635f441e6d44863920c85709b02ffec2"
    assert ee.EXPECTED_NNUNET_PLANS_SHA256 == "c67af4aaeb24f5a72b58c83e351a9c62cf026f8baf1bbb727e3022491d5733d8"
    assert ee.EXPECTED_NNUNET_DATASET_SHA256 == "d81f40253d7c2d88b9d1cdbd4bd7319a7a0206fce5b2350287317bd34f9c7389"
    assert ee.FROZEN_PROTOCOL["tta"] is False
    assert ee.FROZEN_PROTOCOL["threshold"] == 0.50
    assert ee.FROZEN_PROTOCOL["postprocess"] == "none"
    assert ee.FROZEN_PROTOCOL["keep_top_k"] is None
    assert ee.FROZEN_PROTOCOL["sw_overlap"] == 0.50
    assert ee.FROZEN_PROTOCOL["sw_mode"] == "gaussian"
    assert ee.FROZEN_PROTOCOL["bootstrap_seed"] == 42
    assert ee.FROZEN_PROTOCOL["bootstrap_n_boot"] == 2000
    assert ee.FROZEN_PROTOCOL["tie_tolerance"] == 1e-12


def test_endpoint_checkpoint_and_metadata_verification(tmp_path):
    import torch

    fx = _make_endpoint_fixture(tmp_path)

    # Default path enforcement rejects non-recovered paths when require_default_paths=True
    with pytest.raises(ValueError, match="recovered"):
        ee.verify_endpoint_checkpoints(
            rasnet_ckpt=fx["rasnet_ckpt"],
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            repo_root=REPO,
            expected_hashes=fx["hashes"],
            require_default_paths=True,
        )

    prov = ee.verify_endpoint_checkpoints(
        rasnet_ckpt=fx["rasnet_ckpt"],
        nnunet_last_ckpt=fx["nnunet_last_pt"],
        nnunet_model_folder=fx["nnunet_folder"],
        repo_root=REPO,
        expected_hashes=fx["hashes"],
        require_default_paths=False,
    )
    assert prov["checkpoint_role"] == "final_epoch200"
    assert prov["rasnet"]["role"] == "final_epoch200"
    assert prov["rasnet"]["embedded_epoch_metadata_present"] is False
    assert prov["rasnet"]["epoch_identity_supported_by"] == "script_log_archive_provenance"
    assert prov["nnunet"]["role"] == "final_epoch200"
    assert prov["nnunet"]["current_epoch"] == 200

    # Rejects "best" filenames
    best_r = tmp_path / "rasnet_best.pth"
    best_r.write_bytes(fx["rasnet_ckpt"].read_bytes())
    with pytest.raises(ValueError, match="best"):
        ee.verify_endpoint_checkpoints(
            rasnet_ckpt=best_r,
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            expected_hashes={**fx["hashes"], "rasnet_last": _sha256_file(best_r)},
            require_default_paths=False,
        )
    with pytest.raises(ValueError, match="checkpoint_final.pth"):
        ee.verify_endpoint_checkpoints(
            rasnet_ckpt=fx["rasnet_ckpt"],
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            nnunet_checkpoint_name="checkpoint_best.pth",
            expected_hashes=fx["hashes"],
            require_default_paths=False,
        )

    # Rejects wrong hash
    bad_hashes = {**fx["hashes"], "rasnet_last": "0" * 64}
    with pytest.raises(ValueError, match="SHA-256"):
        ee.verify_endpoint_checkpoints(
            rasnet_ckpt=fx["rasnet_ckpt"],
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            expected_hashes=bad_hashes,
            require_default_paths=False,
        )

    # Rejects current_epoch == 199
    bad_ckpt = torch.load(fx["ckpt_final"], map_location="cpu", weights_only=False)
    bad_ckpt["current_epoch"] = 199
    torch.save(bad_ckpt, fx["ckpt_final"])
    fx["nnunet_last_pt"].write_bytes(fx["ckpt_final"].read_bytes())
    h_199 = _sha256_file(fx["ckpt_final"])
    with pytest.raises(ValueError, match="current_epoch"):
        ee.verify_endpoint_checkpoints(
            rasnet_ckpt=fx["rasnet_ckpt"],
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            expected_hashes={**fx["hashes"], "nnunet_last": h_199},
            require_default_paths=False,
        )

    # Restore epoch 200, test plans mismatch with init_args
    bad_ckpt["current_epoch"] = 200
    torch.save(bad_ckpt, fx["ckpt_final"])
    fx["nnunet_last_pt"].write_bytes(fx["ckpt_final"].read_bytes())
    h_200 = _sha256_file(fx["ckpt_final"])
    modified_plans = dict(fx["plans"])
    modified_plans["plans_name"] = "tamperedPlans"
    fx["plans_path"].write_text(json.dumps(modified_plans), encoding="utf-8")
    with pytest.raises(ValueError, match="plans.json"):
        ee.verify_endpoint_checkpoints(
            rasnet_ckpt=fx["rasnet_ckpt"],
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            expected_hashes={
                **fx["hashes"],
                "nnunet_last": h_200,
                "nnunet_plans": _sha256_file(fx["plans_path"]),
            },
            require_default_paths=False,
        )

    # Restore plans, test dataset.json mismatch with init_args
    fx["plans_path"].write_text(json.dumps(fx["plans"], indent=2), encoding="utf-8")
    modified_ds = dict(fx["dataset_json"])
    modified_ds["file_ending"] = ".mha"
    fx["dataset_path"].write_text(json.dumps(modified_ds), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset.json"):
        ee.verify_endpoint_checkpoints(
            rasnet_ckpt=fx["rasnet_ckpt"],
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            expected_hashes={
                **fx["hashes"],
                "nnunet_last": h_200,
                "nnunet_plans": _sha256_file(fx["plans_path"]),
                "nnunet_dataset": _sha256_file(fx["dataset_path"]),
            },
            require_default_paths=False,
        )


def test_build_nnunet_endpoint_predictor_guards(tmp_path):
    import torch

    fx = _make_endpoint_fixture(tmp_path)
    with pytest.raises(ValueError, match="checkpoint_final.pth"):
        ee.build_nnunet_endpoint_predictor(fx["nnunet_folder"], fold=0, checkpoint_name="checkpoint_best.pth", device="cpu")
    with pytest.raises(ValueError, match="fold==0"):
        ee.build_nnunet_endpoint_predictor(fx["nnunet_folder"], fold=1, checkpoint_name="checkpoint_final.pth", device="cpu")

    ckpt = torch.load(fx["ckpt_final"], map_location="cpu", weights_only=False)
    ckpt["current_epoch"] = 199
    torch.save(ckpt, fx["ckpt_final"])
    with pytest.raises(ValueError, match="current_epoch"):
        ee.build_nnunet_endpoint_predictor(fx["nnunet_folder"], fold=0, checkpoint_name="checkpoint_final.pth", device="cpu")


# ---------------------------------------------------------------- 3. Case IDs, protected dirs, stale & incomplete guards
def test_protected_dirs_and_val94_case_id_guards(tmp_path):
    protected = [
        "results/thesis_outputs",
        "results/thesis_outputs/nested",
        "results/benchmark_runs/val_full_94",
        "results/benchmark_runs/test_retrospective_150",
        "results/rasnet_fixed_weight_optimization",
        "results/optionA_v1_symmetric_sw050",
        "results",
        ".",
    ]
    for rel in protected:
        target = REPO if rel == "." else REPO / rel
        with pytest.raises(ValueError, match="protected"):
            ee.validate_endpoint_out_dir(target, REPO, resume=False)

    ok_dir = tmp_path / "new_endpoint_out"
    assert ee.validate_endpoint_out_dir(ok_dir, REPO, resume=False) == ok_dir.resolve()

    splits_94 = {
        "train": list(range(1, 11)),
        "val": list(range(100, 194)),  # 94 cases
        "test": [851, 852],
    }
    assert len(ee.validate_endpoint_case_ids(splits_94, None, expected_val_count=94)) == 94
    assert ee.validate_endpoint_case_ids(splits_94, [105, 100], expected_val_count=94) == [100, 105]

    for bad_ids in ([851], [100, 1000], [1], [9999], [100, 100], []):
        with pytest.raises(ValueError):
            ee.validate_endpoint_case_ids(splits_94, bad_ids, expected_val_count=94)

    incomplete_splits = {"train": [1], "val": [14, 45], "test": [851]}
    with pytest.raises(ValueError, match="94"):
        ee.validate_endpoint_case_ids(incomplete_splits, None, expected_val_count=94)


def test_rejects_stale_mismatched_and_incomplete_outputs(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    ds = tmp_path / "ds"
    p14 = _write_synthetic_case(sitk, ds, 14)
    p45 = _write_synthetic_case(sitk, ds, 45)
    pairs = {14: p14, 45: p45}

    out = tmp_path / "out"
    out.mkdir()
    (out / "junk.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ee.validate_endpoint_out_dir(out, REPO, resume=False)

    expected_state = {
        "run_id": "run_A",
        "protocol_version": ee.PROTOCOL_VERSION,
        "checkpoint_role": ee.CHECKPOINT_ROLE,
        "case_ids": [14, 45],
        "splits_sha256": "abc",
        "rasnet_sha256": "r_hash",
        "nnunet_sha256": "n_hash",
        "plans_sha256": "p_hash",
        "dataset_sha256": "d_hash",
    }

    # Wrong run_id / checkpoint hash in existing manifest.json
    (out / "manifest.json").write_text(json.dumps({**expected_state, "run_id": "run_OLD"}), encoding="utf-8")
    with pytest.raises(ValueError, match="run_id"):
        ee.validate_run_state_and_reject_stale(out, expected_state, [14, 45], pairs, resume=True)

    (out / "manifest.json").write_text(json.dumps({**expected_state, "rasnet_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="rasnet_sha256"):
        ee.validate_run_state_and_reject_stale(out, expected_state, [14, 45], pairs, resume=True)

    (out / "manifest.json").write_text(json.dumps(expected_state), encoding="utf-8")

    # Stale NIfTI outside case_ids
    stale_mask = out / "predictions" / "rasnet" / "999.nii.gz"
    stale_mask.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((24, 28, 32), np.uint8)), str(stale_mask))
    with pytest.raises(ValueError, match="Stale"):
        ee.validate_run_state_and_reject_stale(out, expected_state, [14, 45], pairs, resume=True)
    stale_mask.unlink()

    # Existing mask with mismatched geometry
    bad_mask = out / "predictions" / "rasnet" / "14.nii.gz"
    bad_img = sitk.GetImageFromArray(np.zeros((10, 10, 10), np.uint8))
    sitk.WriteImage(bad_img, str(bad_mask))
    with pytest.raises(ValueError, match="geometry"):
        ee.validate_run_state_and_reject_stale(out, expected_state, [14, 45], pairs, resume=True)
    bad_mask.unlink()

    # Incomplete outputs (missing case ID from one model)
    df_r = pd.DataFrame([{"case_id": 14, "dice": 0.8}, {"case_id": 45, "dice": 0.9}])
    df_n = pd.DataFrame([{"case_id": 14, "dice": 0.7}])
    with pytest.raises(ValueError, match="Incomplete"):
        ee.validate_complete_results(df_r, df_n, [14, 45])


# ---------------------------------------------------------------- 4. End-to-end physical alignment, manifest, errors, paired stats
def test_end_to_end_physical_grid_alignment_manifest_and_paired_stats(tmp_path):
    import torch
    import yaml
    from benchmark.metrics import check_geometry
    from benchmark.preflight import image_geometry
    from benchmark.stats import bootstrap_ci

    sitk = pytest.importorskip("SimpleITK")
    fx = _make_endpoint_fixture(tmp_path)

    ds = tmp_path / "ds"
    p14_img, p14_lbl = _write_synthetic_case(
        sitk,
        ds,
        14,
        shape=(28, 32, 36),
        gt_slice=(slice(8, 13), slice(14, 20), slice(16, 28)),
    )
    p45_img, p45_lbl = _write_synthetic_case(
        sitk,
        ds,
        45,
        shape=(28, 32, 36),
        gt_slice=(slice(8, 13), slice(14, 20), slice(16, 28)),
    )
    _write_synthetic_case(sitk, ds, 1)
    _write_synthetic_case(sitk, ds, 851)

    splits = {"train": [1], "val": [14, 45], "test": [851]}
    splits_p = tmp_path / "splits.json"
    splits_p.write_text(json.dumps(splits), encoding="utf-8")

    cfg = {
        "dataset_root": str(ds),
        "split_file": str(splits_p),
        "output_root": str(tmp_path / "unused"),
        "cache_dir": str(tmp_path / "unused_cache"),
        "thresholds_file": str(tmp_path / "unused_thresh.json"),
        "seed": 42,
        "sw_batch_size": 2,
    }
    cfg_p = tmp_path / "cfg.yaml"
    cfg_p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class IntensityEchoNet(torch.nn.Module):
        def forward(self, x):
            fg_logit = (x - 0.5) * 12.0
            return torch.cat([-fg_logit, fg_logit], dim=1)

    class AlignedFakeNNUNetPredictor:
        def __init__(self, cid_to_lbl: dict[int, Path]):
            self.use_mirroring = False
            self.cid_to_lbl = cid_to_lbl

        def run_batch(self, case_ids, pairs):
            # Emulate _run_nnunet_batch_probabilities with aligned (Z, Y, X) GT-shifted probs
            for cid in case_ids:
                gt = sitk.GetArrayFromImage(sitk.ReadImage(str(self.cid_to_lbl[cid]))).astype(np.float32)
                prob = np.where(gt > 0, 0.85, 0.05).astype(np.float32)
                yield cid, prob, None

    out_dir = tmp_path / "smoke_out"
    fake_nn = AlignedFakeNNUNetPredictor({14: p14_lbl, 45: p45_lbl})

    result_dir = ee.run_endpoint_evaluation(
        repo_root=REPO,
        config_path=cfg_p,
        out_dir=out_dir,
        rasnet_ckpt=fx["rasnet_ckpt"],
        nnunet_last_ckpt=fx["nnunet_last_pt"],
        nnunet_model_folder=fx["nnunet_folder"],
        case_ids=[14, 45],
        smoke=True,
        device="cpu",
        sw_batch_size=2,
        use_amp=False,
        roi_size=(32, 32, 32),
        export_masks=True,
        resume=False,
        run_id="test_smoke",
        expected_val_count=2,
        expected_hashes=fx["hashes"],
        require_default_paths=False,
        rasnet_model_builder=lambda *a, **k: IntensityEchoNet().eval(),
        nnunet_predictor_builder=lambda *a, **k: fake_nn,
        nnunet_batch_runner=lambda pred, cids, pairs, need_tta, num_processes=4: pred.run_batch(cids, pairs),
    )

    assert result_dir == out_dir.resolve()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "test_smoke"
    assert manifest["smoke"] is True
    assert manifest["checkpoints"]["rasnet"]["role"] == "final_epoch200"
    assert manifest["checkpoints"]["rasnet"]["sha256"] == fx["hashes"]["rasnet_last"]
    assert manifest["checkpoints"]["rasnet"]["embedded_epoch_metadata_present"] is False
    assert manifest["checkpoints"]["rasnet"]["epoch_identity_supported_by"] == "script_log_archive_provenance"
    assert manifest["checkpoints"]["nnunet"]["role"] == "final_epoch200"
    assert manifest["checkpoints"]["nnunet"]["current_epoch"] == 200
    assert manifest["checkpoints"]["nnunet"]["checkpoint_final_sha256"] == fx["hashes"]["nnunet_last"]
    assert manifest["checkpoints"]["nnunet"]["plans_sha256"] == fx["hashes"]["nnunet_plans"]
    assert manifest["checkpoints"]["nnunet"]["dataset_sha256"] == fx["hashes"]["nnunet_dataset"]
    for k in ("source", "config", "env", "git", "split", "checkpoints", "protocol"):
        assert k in manifest
    proto = manifest["protocol"]
    assert proto["overlap"] == 0.50
    assert proto["sw_mode"] == "gaussian"
    assert proto["tta"] is False
    assert proto["threshold"] == 0.50
    assert proto["postprocess"] == "none"
    for rk in ("patch_size", "normalization", "spacing", "inversion", "precision_mode", "sw_batch_size"):
        assert rk in proto["rasnet"]
        assert rk in proto["nnunet"]

    # Geometry & landmark alignment
    geom_df = pd.read_csv(out_dir / "geometry_verification.csv")
    assert len(geom_df) == 4
    assert geom_df["geometry_ok"].all()
    for cid, img_p, lbl_p in ((14, p14_img, p14_lbl), (45, p45_img, p45_lbl)):
        ref = sitk.ReadImage(str(img_p))
        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_p))) > 0
        gt_com = np.argwhere(gt_arr).mean(axis=0)
        for model in ("rasnet", "nnunet"):
            m_img = sitk.ReadImage(str(out_dir / "predictions" / model / f"{cid}.nii.gz"))
            check_geometry(image_geometry(ref), image_geometry(m_img))
            m_arr = sitk.GetArrayFromImage(m_img) > 0
            assert m_arr.any()
            assert np.allclose(np.argwhere(m_arr).mean(axis=0), gt_com, atol=1.0)

    df_r = pd.read_csv(out_dir / "per_case_metrics_rasnet_epoch200.csv")
    df_n = pd.read_csv(out_dir / "per_case_metrics_nnunet_epoch200.csv")
    assert list(df_r["case_id"]) == [14, 45]
    assert list(df_n["case_id"]) == [14, 45]
    assert (df_r["dice"] > 0.7).all()
    assert (df_n["dice"] == 1.0).all()
    assert (df_r["status"] == "ok").all()

    paired_df = pd.read_csv(out_dir / "paired_stats_epoch200.csv")
    dice_row = paired_df[paired_df["metric"] == "dice"].iloc[0]
    expected_diffs = df_r["dice"].to_numpy() - df_n["dice"].to_numpy()
    lo, hi = bootstrap_ci(expected_diffs, seed=42, n_boot=2000, confidence=0.95)
    assert dice_row["mean_difference"] == pytest.approx(expected_diffs.mean())
    assert dice_row["diff_ci95_lo"] == pytest.approx(lo)
    assert dice_row["diff_ci95_hi"] == pytest.approx(hi)


def test_non_finite_probs_and_nnunet_ofile_mismatch_fail_loudly(tmp_path):
    import torch
    import yaml
    from benchmark.run_benchmark import _run_nnunet_batch_probabilities

    sitk = pytest.importorskip("SimpleITK")
    fx = _make_endpoint_fixture(tmp_path)
    ds = tmp_path / "ds"
    p14 = _write_synthetic_case(sitk, ds, 14)
    _write_synthetic_case(sitk, ds, 1)
    _write_synthetic_case(sitk, ds, 851)
    splits_p = tmp_path / "splits.json"
    splits_p.write_text(json.dumps({"train": [1], "val": [14], "test": [851]}), encoding="utf-8")
    cfg_p = tmp_path / "cfg.yaml"
    cfg_p.write_text(yaml.safe_dump({"dataset_root": str(ds), "split_file": str(splits_p), "sw_batch_size": 1}), encoding="utf-8")

    class NaNNet(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 2, *x.shape[2:]), float("nan"))

    with pytest.raises(RuntimeError, match="Non-finite"):
        ee.run_endpoint_evaluation(
            repo_root=REPO,
            config_path=cfg_p,
            out_dir=tmp_path / "nan_rasnet",
            rasnet_ckpt=fx["rasnet_ckpt"],
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            case_ids=[14],
            device="cpu",
            sw_batch_size=1,
            use_amp=False,
            roi_size=(32, 32, 32),
            export_masks=False,
            expected_val_count=1,
            expected_hashes=fx["hashes"],
            require_default_paths=False,
            rasnet_model_builder=lambda *a, **k: NaNNet().eval(),
            nnunet_predictor_builder=lambda *a, **k: object(),
            nnunet_batch_runner=lambda pred, cids, pairs, need_tta, num_processes=4: iter([(14, np.zeros((24, 28, 32), np.float32), None)]),
        )

    # Inf from nnU-Net fails loudly
    with pytest.raises(RuntimeError, match="Non-finite"):
        ee.run_endpoint_evaluation(
            repo_root=REPO,
            config_path=cfg_p,
            out_dir=tmp_path / "inf_nnunet",
            rasnet_ckpt=fx["rasnet_ckpt"],
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            case_ids=[14],
            device="cpu",
            sw_batch_size=1,
            use_amp=False,
            roi_size=(32, 32, 32),
            export_masks=False,
            expected_val_count=1,
            expected_hashes=fx["hashes"],
            require_default_paths=False,
            rasnet_model_builder=lambda *a, **k: torch.nn.Conv3d(1, 2, 1).eval(),
            nnunet_predictor_builder=lambda *a, **k: object(),
            nnunet_batch_runner=lambda pred, cids, pairs, need_tta, num_processes=4: iter([(14, np.full((24, 28, 32), np.inf, np.float32), None)]),
        )

    # Verify _run_nnunet_batch_probabilities rejects ofile != cid
    import nnunetv2.inference.data_iterators as di

    class DummyPredictor:
        plans_manager = None
        dataset_json = None
        configuration_manager = None
        label_manager = None
        use_mirroring = False

    orig_iter = di.preprocessing_iterator_fromfiles
    try:
        di.preprocessing_iterator_fromfiles = lambda **kwargs: iter([{"ofile": "999", "data": None, "data_properties": {}}])
        with pytest.raises(RuntimeError, match="case identity mismatch"):
            list(_run_nnunet_batch_probabilities(DummyPredictor(), [14], {14: p14}, need_tta=False, num_processes=1))
    finally:
        di.preprocessing_iterator_fromfiles = orig_iter


def test_empty_prediction_status_and_paired_stats_with_tie_tolerance():
    from benchmark.stats import bootstrap_ci

    df_r = pd.DataFrame([
        {"case_id": 14, "status": "prediction_empty", "dice": 0.0, "iou": 0.0, "precision": np.nan, "recall": 0.0, "hd95": np.nan, "asd": np.nan, "cldice": np.nan, "centerline_recall": np.nan},
        {"case_id": 45, "status": "ok", "dice": 0.80 + 5e-13, "iou": 0.70, "precision": 0.85, "recall": 0.75, "hd95": 5.0, "asd": 1.0, "cldice": 0.82, "centerline_recall": 0.80},
        {"case_id": 47, "status": "ok", "dice": 0.85, "iou": 0.75, "precision": 0.88, "recall": 0.82, "hd95": 4.0, "asd": 0.8, "cldice": 0.86, "centerline_recall": 0.85},
        {"case_id": 51, "status": "ok", "dice": 0.70, "iou": 0.55, "precision": 0.72, "recall": 0.68, "hd95": 8.0, "asd": 1.6, "cldice": 0.71, "centerline_recall": 0.70},
    ])
    df_n = pd.DataFrame([
        {"case_id": 14, "status": "ok", "dice": 0.75, "iou": 0.60, "precision": 0.80, "recall": 0.70, "hd95": 6.0, "asd": 1.2, "cldice": 0.76, "centerline_recall": 0.74},
        {"case_id": 45, "status": "ok", "dice": 0.80, "iou": 0.70, "precision": 0.85, "recall": 0.75, "hd95": 5.0, "asd": 1.0, "cldice": 0.82, "centerline_recall": 0.80},
        {"case_id": 47, "status": "ok", "dice": 0.75, "iou": 0.60, "precision": 0.78, "recall": 0.72, "hd95": 6.0, "asd": 1.2, "cldice": 0.76, "centerline_recall": 0.75},
        {"case_id": 51, "status": "ok", "dice": 0.75, "iou": 0.60, "precision": 0.78, "recall": 0.72, "hd95": 6.0, "asd": 1.2, "cldice": 0.76, "centerline_recall": 0.75},
    ])

    summary_df = ee.summarize_endpoint_metrics(pd.concat([df_r.assign(model="rasnet"), df_n.assign(model="nnunet")], ignore_index=True))
    r_sum = summary_df[summary_df["model"] == "rasnet"].iloc[0]
    assert r_sum["n_cases"] == 4
    assert r_sum["n_empty_pred"] == 1
    assert r_sum["n_ok"] == 3

    meta, paired_df = ee.compute_endpoint_paired_stats(df_r, df_n, seed=42, n_boot=2000, tie_tolerance=1e-12)
    dice_row = paired_df[paired_df["metric"] == "dice"].iloc[0]
    diffs = np.array([0.0 - 0.75, 5e-13, 0.85 - 0.75, 0.70 - 0.75])
    lo, hi = bootstrap_ci(diffs, seed=42, n_boot=2000, confidence=0.95)
    assert dice_row["n"] == 4
    assert dice_row["n_excluded_nan"] == 0
    assert dice_row["wins"] == 1
    assert dice_row["ties"] == 1
    assert dice_row["losses"] == 2
    assert dice_row["diff_ci95_lo"] == pytest.approx(lo)
    assert dice_row["diff_ci95_hi"] == pytest.approx(hi)

    hd_row = paired_df[paired_df["metric"] == "hd95"].iloc[0]
    assert hd_row["n"] == 3
    assert hd_row["n_excluded_nan"] == 1
    # Lower is better for hd95: case 45 tie (5==5), case 47 win (4<6), case 51 loss (8>6)
    assert hd_row["wins"] == 1 and hd_row["ties"] == 1 and hd_row["losses"] == 1
    assert meta["seed"] == 42 and meta["n_boot"] == 2000 and meta["tie_tolerance"] == 1e-12


# ---------------------------------------------------------------- 5. CLI guardrail test
def test_cli_rejects_locked_test_ids_and_protected_dirs(tmp_path):
    script = REPO / "scripts" / "run_v1_epoch200_endpoint_eval.py"
    res1 = subprocess.run(
        [PYTHON, str(script), "--case-ids", "851", "--out", str(tmp_path / "bad_out"), "--device", "cpu"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert res1.returncode != 0
    assert "851" in (res1.stderr + res1.stdout) or "locked test" in (res1.stderr + res1.stdout).lower()

    res2 = subprocess.run(
        [PYTHON, str(script), "--case-ids", "14", "--out", str(REPO / "results" / "thesis_outputs"), "--device", "cpu"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert res2.returncode != 0
    assert "protected" in (res2.stderr + res2.stdout).lower()


# ---------------------------------------------------------------- 6. Challenger 1 & Reviewer 1/2 Remediation Tests
def _setup_two_case_synthetic_env(tmp_path: Path):
    import torch
    import yaml

    sitk = pytest.importorskip("SimpleITK")
    fx = _make_endpoint_fixture(tmp_path)
    ds = tmp_path / "ds"
    _, p14_lbl = _write_synthetic_case(
        sitk,
        ds,
        14,
        shape=(28, 32, 36),
        gt_slice=(slice(8, 13), slice(14, 20), slice(16, 28)),
    )
    _, p45_lbl = _write_synthetic_case(
        sitk,
        ds,
        45,
        shape=(28, 32, 36),
        gt_slice=(slice(8, 13), slice(14, 20), slice(16, 28)),
    )
    _write_synthetic_case(sitk, ds, 1)
    _write_synthetic_case(sitk, ds, 851)

    splits_p = tmp_path / "splits.json"
    splits_p.write_text(json.dumps({"train": [1], "val": [14, 45], "test": [851]}), encoding="utf-8")
    cfg_p = tmp_path / "cfg.yaml"
    cfg_p.write_text(
        yaml.safe_dump({"dataset_root": str(ds), "split_file": str(splits_p), "sw_batch_size": 2}),
        encoding="utf-8",
    )

    class EchoNet(torch.nn.Module):
        def forward(self, x):
            fg = (x - 0.5) * 12.0
            return torch.cat([-fg, fg], dim=1)

    def _run_nn(pred, cids, pairs, need_tta, num_processes=4):
        for cid in cids:
            lbl_p = p14_lbl if int(cid) == 14 else p45_lbl
            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_p))).astype(np.float32)
            yield cid, np.where(gt > 0, 0.85, 0.05).astype(np.float32), None

    def _run(out_dir: Path, resume: bool = False):
        return ee.run_endpoint_evaluation(
            repo_root=REPO,
            config_path=cfg_p,
            out_dir=out_dir,
            rasnet_ckpt=fx["rasnet_ckpt"],
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            case_ids=[14, 45],
            smoke=True,
            device="cpu",
            sw_batch_size=2,
            use_amp=False,
            roi_size=(32, 32, 32),
            export_masks=True,
            resume=resume,
            run_id="repro_run",
            expected_val_count=2,
            expected_hashes=fx["hashes"],
            require_default_paths=False,
            rasnet_model_builder=lambda *a, **k: EchoNet().eval(),
            nnunet_predictor_builder=lambda *a, **k: object(),
            nnunet_batch_runner=_run_nn,
        )

    return fx, _run


def test_resume_rejects_tampered_manifest_nested_fields_with_and_without_run_state(tmp_path):
    _, run_eval = _setup_two_case_synthetic_env(tmp_path)
    out_dir = tmp_path / "out_manifest"
    run_eval(out_dir, resume=False)

    manifest_p = out_dir / "manifest.json"
    state_p = out_dir / "endpoint_run_state.json"
    orig_manifest_text = manifest_p.read_text(encoding="utf-8")
    orig_state_text = state_p.read_text(encoding="utf-8")

    def _mutate_case_ids(m):
        m["split"]["case_ids"] = [14, 47]

    def _mutate_splits_sha(m):
        m["split"]["splits_sha256"] = "0" * 64

    def _mutate_plans_sha(m):
        m["checkpoints"]["nnunet"]["plans_sha256"] = "0" * 64

    def _mutate_dataset_sha(m):
        m["checkpoints"]["nnunet"]["dataset_sha256"] = "0" * 64

    def _mutate_proto_ver(m):
        m["protocol"]["protocol_version"] = "stale-proto"

    def _mutate_proto_role(m):
        m["protocol"]["checkpoint_role"] = "best"

    mutations = [
        ("split.case_ids", _mutate_case_ids),
        ("split.splits_sha256", _mutate_splits_sha),
        ("checkpoints.nnunet.plans_sha256", _mutate_plans_sha),
        ("checkpoints.nnunet.dataset_sha256", _mutate_dataset_sha),
        ("protocol.protocol_version", _mutate_proto_ver),
        ("protocol.checkpoint_role", _mutate_proto_role),
    ]

    for keep_run_state in (True, False):
        for label, mutator in mutations:
            manifest_p.write_text(orig_manifest_text, encoding="utf-8")
            if keep_run_state:
                state_p.write_text(orig_state_text, encoding="utf-8")
            elif state_p.exists():
                state_p.unlink()

            tampered = json.loads(orig_manifest_text)
            mutator(tampered)
            tampered_text = json.dumps(tampered, indent=2)
            manifest_p.write_text(tampered_text, encoding="utf-8")

            with pytest.raises(ValueError):
                run_eval(out_dir, resume=True)
            assert manifest_p.read_text(encoding="utf-8") == tampered_text, (
                f"manifest.json was overwritten on failed {label} (keep_run_state={keep_run_state})"
            )


def test_resume_rejects_incomplete_completed_run_csvs_and_missing_provenance_columns(tmp_path):
    _, run_eval = _setup_two_case_synthetic_env(tmp_path)
    out_dir = tmp_path / "out_csv"
    run_eval(out_dir, resume=False)

    files = {
        "rasnet": out_dir / "per_case_metrics_rasnet_epoch200.csv",
        "nnunet": out_dir / "per_case_metrics_nnunet_epoch200.csv",
        "combined": out_dir / "per_case_metrics_combined.csv",
        "geom": out_dir / "geometry_verification.csv",
        "manifest": out_dir / "manifest.json",
        "state": out_dir / "endpoint_run_state.json",
    }
    originals = {k: p.read_text(encoding="utf-8") for k, p in files.items()}

    def _restore():
        for k, txt in originals.items():
            files[k].write_text(txt, encoding="utf-8")

    # 1. Deleting a row from rasnet / nnunet / combined / geometry CSVs of a completed run must raise ValueError
    for key in ("rasnet", "nnunet", "combined", "geom"):
        _restore()
        df = pd.read_csv(files[key])
        df.iloc[:-1].to_csv(files[key], index=False)
        before = files[key].read_text(encoding="utf-8")
        with pytest.raises(ValueError):
            run_eval(out_dir, resume=True)
        assert files[key].read_text(encoding="utf-8") == before

    # 2. Dropping checkpoint_sha256 or checkpoint_role column from non-empty per-case CSV must raise ValueError
    for csv_key in ("rasnet", "nnunet"):
        for col in ("checkpoint_sha256", "checkpoint_role"):
            _restore()
            df = pd.read_csv(files[csv_key]).drop(columns=[col])
            df.to_csv(files[csv_key], index=False)
            with pytest.raises(ValueError, match=col):
                run_eval(out_dir, resume=True)

    # 3. Even without manifest.json/endpoint_run_state.json, orphaned mask on disk without CSV row must raise ValueError
    _restore()
    files["manifest"].unlink()
    files["state"].unlink()
    df_r = pd.read_csv(files["rasnet"])
    df_r[df_r["case_id"] != 45].to_csv(files["rasnet"], index=False)
    assert (out_dir / "predictions" / "rasnet" / "45.nii.gz").is_file()
    with pytest.raises(ValueError, match="45"):
        run_eval(out_dir, resume=True)


def test_verify_endpoint_checkpoints_rejects_nan_nnunet_weights(tmp_path):
    import torch

    fx = _make_endpoint_fixture(tmp_path)
    ckpt = torch.load(fx["ckpt_final"], map_location="cpu", weights_only=False)
    ckpt["network_weights"]["dummy"] = torch.tensor([float("nan")])
    torch.save(ckpt, fx["ckpt_final"])
    fx["nnunet_last_pt"].write_bytes(fx["ckpt_final"].read_bytes())
    bad_hash = _sha256_file(fx["ckpt_final"])

    with pytest.raises(ValueError, match="non-finite"):
        ee.verify_endpoint_checkpoints(
            rasnet_ckpt=fx["rasnet_ckpt"],
            nnunet_last_ckpt=fx["nnunet_last_pt"],
            nnunet_model_folder=fx["nnunet_folder"],
            expected_hashes={**fx["hashes"], "nnunet_last": bad_hash},
            require_default_paths=False,
        )

    with pytest.raises(ValueError, match="non-finite"):
        ee.build_nnunet_endpoint_predictor(
            fx["nnunet_folder"], fold=0, checkpoint_name="checkpoint_final.pth", device="cpu"
        )


def test_resume_handles_zero_byte_csv_header_correctly(tmp_path):
    _, run_eval = _setup_two_case_synthetic_env(tmp_path)
    out_dir = tmp_path / "out_zero_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Simulate crash creating empty 0-byte CSVs before writing rows
    (out_dir / "per_case_metrics_rasnet_epoch200.csv").write_bytes(b"")
    (out_dir / "per_case_metrics_nnunet_epoch200.csv").write_bytes(b"")

    run_eval(out_dir, resume=True)
    df_r = pd.read_csv(out_dir / "per_case_metrics_rasnet_epoch200.csv")
    df_n = pd.read_csv(out_dir / "per_case_metrics_nnunet_epoch200.csv")
    assert list(df_r["case_id"]) == [14, 45]
    assert list(df_n["case_id"]) == [14, 45]
    assert "checkpoint_sha256" in df_r.columns and "checkpoint_role" in df_r.columns
    assert "checkpoint_sha256" in df_n.columns and "checkpoint_role" in df_n.columns
