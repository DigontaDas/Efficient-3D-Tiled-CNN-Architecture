"""Isolated V1 epoch-200 endpoint evaluation (RASNet vs nnU-Net V2 on Val94).

Preserves all existing BEST-only benchmark safeguards untouched by using an explicit
endpoint-only validation/checkpoint contract (`checkpoint_role = "final_epoch200"`).
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from benchmark.adapters.monai_models import (
    build_model,
    build_transform,
    model_constructor_args,
    transform_recipe,
)
from benchmark.adapters.nnunet_adapter import (
    ensure_nnunet_trainer_registered,
    export_mask_like,
)
from benchmark.dataset import resolve_case_pairs
from benchmark.metrics import METRIC_VERSION, check_geometry, compute_metrics
from benchmark.postprocess import prune_components
from benchmark.preflight import image_geometry, validate_case_files, validate_splits
from benchmark.run_benchmark import (
    ALL_METRICS,
    METRIC_DIRECTIONS,
    _run_nnunet_batch_probabilities,
    package_versions,
    sha256_file,
    source_tree_sha256,
)
from benchmark.stats import bootstrap_ci, paired_analysis
from benchmark.symmetric_eval import (
    MODES as SYM_MODES,
    SIGMA_SCALE,
    SW_MODE,
    SW_OVERLAP,
    env_info,
    git_info,
    infer_monai_symmetric,
)

PROTOCOL_VERSION = "v1-epoch200-endpoint-1.0"
CHECKPOINT_ROLE = "final_epoch200"
TARGET_EPOCH = 200
MODELS = ("rasnet", "nnunet")
LOCKED_TEST_RANGE = range(851, 1001)

DEFAULT_RECOVERED_DIR = Path("recovered_checkpoints/v1_epoch200_20261117_041500Z")
DEFAULT_RASNET_CKPT = DEFAULT_RECOVERED_DIR / "rasnet_last.pth"
DEFAULT_NNUNET_LAST_CKPT = DEFAULT_RECOVERED_DIR / "nnunet_last.pt"
DEFAULT_NNUNET_MODEL_FOLDER = (
    DEFAULT_RECOVERED_DIR
    / "nnunet_native_model"
    / "Dataset501_CoronarySeg"
    / "nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres"
)
DEFAULT_NNUNET_FOLD = 0
DEFAULT_NNUNET_CKPT_NAME = "checkpoint_final.pth"
DEFAULT_PROVENANCE_JSON = DEFAULT_RECOVERED_DIR / "provenance_verification.json"

EXPECTED_RASNET_LAST_SHA256 = "71c9e5ef4a02849686c99ccb2d1544ff78c78187e1c0d53e7b22884c708c1b95"
EXPECTED_NNUNET_LAST_SHA256 = "2105bd7238fff7485f1a8d72622f977f635f441e6d44863920c85709b02ffec2"
EXPECTED_NNUNET_PLANS_SHA256 = "c67af4aaeb24f5a72b58c83e351a9c62cf026f8baf1bbb727e3022491d5733d8"
EXPECTED_NNUNET_DATASET_SHA256 = "d81f40253d7c2d88b9d1cdbd4bd7319a7a0206fce5b2350287317bd34f9c7389"

EXPECTED_ENDPOINT_HASHES = {
    "rasnet_last": EXPECTED_RASNET_LAST_SHA256,
    "nnunet_last": EXPECTED_NNUNET_LAST_SHA256,
    "nnunet_plans": EXPECTED_NNUNET_PLANS_SHA256,
    "nnunet_dataset": EXPECTED_NNUNET_DATASET_SHA256,
}

PROTECTED_DIRS = (
    "results/thesis_outputs",
    "results/benchmark_runs",
    "results/benchmark_runs/val_full_94",
    "results/benchmark_runs/test_retrospective_150",
    "results/rasnet_fixed_weight_optimization",
    "results/optionA_v1_symmetric_sw050",
    "results/audits",
    "results/thresholds.json",
    "Q1_Publication_Package",
    "recovered_checkpoints",
    "Rasnet-v2",
    "benchmark",
    "scripts",
    "tests",
    "Phase3_Local_Integration",
    "secondary-imagcas-dataset",
    "docs",
)

FROZEN_PROTOCOL = {
    "protocol_version": PROTOCOL_VERSION,
    "checkpoint_role": CHECKPOINT_ROLE,
    "split": "val",
    "tta": False,
    "passes": 1,
    "threshold": 0.50,
    "postprocess": "none",
    "keep_top_k": None,
    "connectivity": 26,
    "sw_overlap": float(SW_OVERLAP),
    "sw_mode": str(SW_MODE),
    "sigma_scale": float(SIGMA_SCALE),
    "rasnet_roi_size": [96, 96, 96],
    "metric_version": METRIC_VERSION,
    "bootstrap_seed": 42,
    "bootstrap_n_boot": 2000,
    "tie_tolerance": 1e-12,
}


def validate_endpoint_out_dir(out_dir: Path, repo_root: Path, resume: bool = False) -> Path:
    """Validate that out_dir does not collide with repo_root, results root, or any protected path."""
    out = Path(out_dir).resolve()
    repo = Path(repo_root).resolve()
    results_root = (repo / "results").resolve()

    if out == repo or out == results_root:
        raise ValueError(f"Refusing to write into protected root directory: {out}")

    for rel in PROTECTED_DIRS:
        prot = (repo / rel).resolve()
        if out == prot or prot in out.parents or out in prot.parents:
            raise ValueError(f"Refusing to write into protected path: {out} (conflicts with {prot})")

    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty (pass --resume to continue): {out}")
    return out


def validate_endpoint_case_ids(
    splits: dict,
    requested_case_ids: list[int] | None = None,
    expected_val_count: int = 94,
) -> list[int]:
    """Validate validation-only case IDs; refuse test IDs, non-val IDs, duplicates, and empty lists."""
    parts = validate_splits(splits)
    val_ids = parts["val"]
    if len(val_ids) != int(expected_val_count):
        raise ValueError(
            f"Validation split must contain {expected_val_count} cases, found {len(val_ids)}"
        )

    if requested_case_ids is None:
        return sorted(int(c) for c in val_ids)

    if not requested_case_ids:
        raise ValueError("Requested case_ids list cannot be empty")

    cleaned: list[int] = []
    seen: set[int] = set()
    for raw in requested_case_ids:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"Invalid non-integer case ID: {raw!r}")
        cid = int(raw)
        if cid in seen:
            raise ValueError(f"Duplicate case ID requested: {cid}")
        seen.add(cid)
        if cid in LOCKED_TEST_RANGE:
            raise ValueError(f"Refusing locked test case ID {cid} (851-1000 is locked)")
        if cid not in val_ids:
            raise ValueError(f"Refusing non-validation case ID {cid}; endpoint evaluation is Val-only")
        cleaned.append(cid)

    return sorted(cleaned)


def _check_nnunet_checkpoint_metadata(
    ckpt_path: Path,
    plans_data: dict,
    dataset_data: dict,
    expected_fold: int = DEFAULT_NNUNET_FOLD,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid nnU-Net checkpoint payload type: {type(ckpt).__name__}")

    if "current_epoch" not in ckpt:
        raise ValueError(f"Missing current_epoch metadata in nnU-Net checkpoint: {ckpt_path}")
    current_epoch = int(ckpt["current_epoch"])
    if current_epoch != TARGET_EPOCH:
        raise ValueError(
            f"nnU-Net checkpoint current_epoch={current_epoch} != required {TARGET_EPOCH}: {ckpt_path}"
        )

    trainer_name = ckpt.get("trainer_name")
    if trainer_name != "nnUNetTrainer_200epochs":
        raise ValueError(
            f"Unexpected nnU-Net trainer_name={trainer_name!r} (expected 'nnUNetTrainer_200epochs')"
        )

    init_args = ckpt.get("init_args")
    if not isinstance(init_args, dict):
        raise ValueError("Missing init_args dictionary in nnU-Net checkpoint")

    config_name = init_args.get("configuration")
    if config_name != "3d_fullres":
        raise ValueError(f"Unexpected nnU-Net configuration={config_name!r} (expected '3d_fullres')")

    fold_val = int(init_args.get("fold", -1))
    if fold_val != int(expected_fold):
        raise ValueError(f"Unexpected nnU-Net fold={fold_val} (expected {expected_fold})")

    if init_args.get("plans") != plans_data:
        raise ValueError("Disk plans.json does not match nnU-Net checkpoint init_args['plans']")
    if init_args.get("dataset_json") != dataset_data:
        raise ValueError("Disk dataset.json does not match nnU-Net checkpoint init_args['dataset_json']")

    network_weights = ckpt.get("network_weights")
    if not isinstance(network_weights, Mapping) or len(network_weights) == 0:
        raise ValueError(f"Missing or empty network_weights in nnU-Net checkpoint: {ckpt_path}")
    for k, v in network_weights.items():
        if not isinstance(v, torch.Tensor) or not torch.isfinite(v).all():
            raise ValueError(
                f"Non-tensor or non-finite parameter in nnU-Net checkpoint network_weights key {k!r}"
            )

    return ckpt


def verify_endpoint_checkpoints(
    rasnet_ckpt: Path,
    nnunet_last_ckpt: Path,
    nnunet_model_folder: Path,
    nnunet_fold: int = DEFAULT_NNUNET_FOLD,
    nnunet_checkpoint_name: str = DEFAULT_NNUNET_CKPT_NAME,
    repo_root: Path | None = None,
    expected_hashes: dict[str, str] | None = None,
    require_default_paths: bool = True,
) -> dict:
    """Strictly verify recovered epoch-200 checkpoints, hashes, paths, and embedded metadata."""
    rasnet_p = Path(rasnet_ckpt)
    nnunet_last_p = Path(nnunet_last_ckpt)
    nnunet_folder_p = Path(nnunet_model_folder)

    if "best" in rasnet_p.name.lower():
        raise ValueError(f"Refusing 'best' checkpoint in epoch-200 endpoint eval: {rasnet_p}")
    if "best" in nnunet_last_p.name.lower():
        raise ValueError(f"Refusing 'best' checkpoint in epoch-200 endpoint eval: {nnunet_last_p}")
    if "best" in nnunet_checkpoint_name.lower() or nnunet_checkpoint_name != DEFAULT_NNUNET_CKPT_NAME:
        raise ValueError(
            f"Endpoint evaluation requires nnunet_checkpoint_name=='{DEFAULT_NNUNET_CKPT_NAME}', got {nnunet_checkpoint_name!r}"
        )
    if int(nnunet_fold) != DEFAULT_NNUNET_FOLD:
        raise ValueError(f"Endpoint evaluation requires fold=={DEFAULT_NNUNET_FOLD}, got {nnunet_fold}")

    if require_default_paths:
        if repo_root is None:
            raise ValueError("repo_root is required when require_default_paths=True")
        repo = Path(repo_root).resolve()
        exp_r = (repo / DEFAULT_RASNET_CKPT).resolve()
        exp_n = (repo / DEFAULT_NNUNET_LAST_CKPT).resolve()
        exp_f = (repo / DEFAULT_NNUNET_MODEL_FOLDER).resolve()
        if rasnet_p.resolve() != exp_r:
            raise ValueError(f"RASNet path must match recovered checkpoint {exp_r}, got {rasnet_p.resolve()}")
        if nnunet_last_p.resolve() != exp_n:
            raise ValueError(f"nnU-Net last path must match recovered checkpoint {exp_n}, got {nnunet_last_p.resolve()}")
        if nnunet_folder_p.resolve() != exp_f:
            raise ValueError(f"nnU-Net model folder must match recovered folder {exp_f}, got {nnunet_folder_p.resolve()}")

    plans_path = nnunet_folder_p / "plans.json"
    dataset_path = nnunet_folder_p / "dataset.json"
    ckpt_final_path = nnunet_folder_p / f"fold_{int(nnunet_fold)}" / nnunet_checkpoint_name

    for req in (rasnet_p, nnunet_last_p, plans_path, dataset_path, ckpt_final_path):
        if not req.is_file():
            raise FileNotFoundError(f"Missing required endpoint artifact: {req}")

    hashes = dict(EXPECTED_ENDPOINT_HASHES if expected_hashes is None else expected_hashes)
    rasnet_sha = sha256_file(rasnet_p)
    nnunet_last_sha = sha256_file(nnunet_last_p)
    ckpt_final_sha = sha256_file(ckpt_final_path)
    plans_sha = sha256_file(plans_path)
    dataset_sha = sha256_file(dataset_path)

    if rasnet_sha != hashes["rasnet_last"]:
        raise ValueError(f"RASNet checkpoint SHA-256 mismatch: {rasnet_sha} != {hashes['rasnet_last']}")
    if nnunet_last_sha != hashes["nnunet_last"]:
        raise ValueError(f"nnU-Net last checkpoint SHA-256 mismatch: {nnunet_last_sha} != {hashes['nnunet_last']}")
    if ckpt_final_sha != nnunet_last_sha:
        raise ValueError(
            f"Native fold_0/{nnunet_checkpoint_name} SHA-256 ({ckpt_final_sha}) != nnunet_last.pt ({nnunet_last_sha})"
        )
    if plans_sha != hashes["nnunet_plans"]:
        raise ValueError(f"plans.json SHA-256 mismatch: {plans_sha} != {hashes['nnunet_plans']}")
    if dataset_sha != hashes["nnunet_dataset"]:
        raise ValueError(f"dataset.json SHA-256 mismatch: {dataset_sha} != {hashes['nnunet_dataset']}")

    # Strict RASNet state_dict validation
    rasnet_obj = torch.load(rasnet_p, map_location="cpu", weights_only=True)
    if not isinstance(rasnet_obj, Mapping) or len(rasnet_obj) == 0:
        raise ValueError(f"Expected non-empty state_dict mapping in {rasnet_p}")
    for k, v in rasnet_obj.items():
        if not isinstance(v, torch.Tensor) or not torch.isfinite(v).all():
            raise ValueError(f"Non-tensor or non-finite parameter in RASNet checkpoint key {k!r}")

    from Phase3_Local_Integration.rasnet_model import RASNet

    probe_net = RASNet(**model_constructor_args("rasnet"))
    probe_net.load_state_dict(rasnet_obj, strict=True)
    del probe_net

    plans_data = json.loads(plans_path.read_text(encoding="utf-8"))
    dataset_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    _check_nnunet_checkpoint_metadata(ckpt_final_path, plans_data, dataset_data, expected_fold=nnunet_fold)

    cfg_3d = plans_data.get("configurations", {}).get("3d_fullres", {})
    return {
        "checkpoint_role": CHECKPOINT_ROLE,
        "target_epoch": TARGET_EPOCH,
        "rasnet": {
            "path": str(rasnet_p.resolve()),
            "sha256": rasnet_sha,
            "role": CHECKPOINT_ROLE,
            "embedded_epoch_metadata_present": False,
            "epoch_identity_supported_by": "script_log_archive_provenance",
            "state_dict_key_count": int(len(rasnet_obj)),
            "constructor_args": model_constructor_args("rasnet"),
        },
        "nnunet": {
            "last_pt_path": str(nnunet_last_p.resolve()),
            "last_pt_sha256": nnunet_last_sha,
            "model_folder": str(nnunet_folder_p.resolve()),
            "checkpoint_final_path": str(ckpt_final_path.resolve()),
            "checkpoint_final_sha256": ckpt_final_sha,
            "plans_path": str(plans_path.resolve()),
            "plans_sha256": plans_sha,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": dataset_sha,
            "role": CHECKPOINT_ROLE,
            "embedded_epoch_metadata_present": True,
            "current_epoch": TARGET_EPOCH,
            "trainer_name": "nnUNetTrainer_200epochs",
            "configuration": "3d_fullres",
            "fold": int(nnunet_fold),
            "patch_size": cfg_3d.get("patch_size"),
            "spacing": cfg_3d.get("spacing"),
            "normalization_schemes": cfg_3d.get("normalization_schemes"),
            "architecture_class": cfg_3d.get("architecture", {}).get("network_class_name"),
        },
    }


def build_nnunet_endpoint_predictor(
    model_folder: str | Path,
    fold: int = DEFAULT_NNUNET_FOLD,
    checkpoint_name: str = DEFAULT_NNUNET_CKPT_NAME,
    device: str = "cuda",
    use_mirroring: bool = False,
):
    """Dedicated endpoint nnUNetPredictor loader enforcing checkpoint_final.pth and epoch 200."""
    folder = Path(model_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing nnU-Net trained model folder: {folder}")
    if "best" in checkpoint_name.lower() or checkpoint_name != DEFAULT_NNUNET_CKPT_NAME:
        raise ValueError(
            f"Endpoint predictor requires '{DEFAULT_NNUNET_CKPT_NAME}', got {checkpoint_name!r}"
        )
    if int(fold) != DEFAULT_NNUNET_FOLD:
        raise ValueError(f"Endpoint predictor requires fold=={DEFAULT_NNUNET_FOLD}, got {fold}")

    plans_path = folder / "plans.json"
    dataset_path = folder / "dataset.json"
    ckpt_path = folder / f"fold_{int(fold)}" / checkpoint_name
    for req in (plans_path, dataset_path, ckpt_path):
        if not req.is_file():
            raise FileNotFoundError(f"Missing required nnU-Net artifact: {req}")

    plans_data = json.loads(plans_path.read_text(encoding="utf-8"))
    dataset_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    _check_nnunet_checkpoint_metadata(ckpt_path, plans_data, dataset_data, expected_fold=fold)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    ensure_nnunet_trainer_registered()
    dev = torch.device(device)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=bool(use_mirroring),
        perform_everything_on_device=(dev.type == "cuda"),
        device=dev,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(folder),
        use_folds=(int(fold),),
        checkpoint_name=checkpoint_name,
    )
    return predictor


FLAT_STATE_KEYS = (
    "run_id",
    "protocol_version",
    "checkpoint_role",
    "case_ids",
    "splits_sha256",
    "rasnet_sha256",
    "nnunet_sha256",
    "plans_sha256",
    "dataset_sha256",
)


def _validate_flat_state_dict(existing: Mapping, expected_state: Mapping, label: str) -> None:
    for key in FLAT_STATE_KEYS:
        if key not in existing:
            raise ValueError(f"Missing required {label} field '{key}'")
        if existing[key] != expected_state[key]:
            raise ValueError(
                f"Stale/mismatched {label} field '{key}': "
                f"existing={existing[key]!r} != expected={expected_state[key]!r}"
            )


def _validate_nested_manifest_dict(existing: Mapping, expected_state: Mapping) -> None:
    for top_key in ("run_id", "checkpoint_role"):
        if top_key not in existing:
            raise ValueError(f"Missing required manifest.json field '{top_key}'")
        if existing[top_key] != expected_state[top_key]:
            raise ValueError(
                f"Stale/mismatched manifest.json field '{top_key}': "
                f"existing={existing[top_key]!r} != expected={expected_state[top_key]!r}"
            )

    split_sec = existing.get("split")
    if not isinstance(split_sec, Mapping):
        raise ValueError("Missing or invalid manifest.json section 'split'")
    if "case_ids" not in split_sec or split_sec["case_ids"] != expected_state["case_ids"]:
        raise ValueError(
            f"Stale/mismatched manifest.json field 'split.case_ids': "
            f"existing={split_sec.get('case_ids')!r} != expected={expected_state['case_ids']!r}"
        )
    if "splits_sha256" not in split_sec or split_sec["splits_sha256"] != expected_state["splits_sha256"]:
        raise ValueError(
            f"Stale/mismatched manifest.json field 'split.splits_sha256': "
            f"existing={split_sec.get('splits_sha256')!r} != expected={expected_state['splits_sha256']!r}"
        )

    ckpts_sec = existing.get("checkpoints")
    if not isinstance(ckpts_sec, Mapping):
        raise ValueError("Missing or invalid manifest.json section 'checkpoints'")

    ras_sec = ckpts_sec.get("rasnet")
    if not isinstance(ras_sec, Mapping):
        raise ValueError("Missing or invalid manifest.json section 'checkpoints.rasnet'")
    if ras_sec.get("sha256") != expected_state["rasnet_sha256"]:
        raise ValueError(f"Stale manifest rasnet_sha256: {ras_sec.get('sha256')!r}")
    if ras_sec.get("role") != expected_state["checkpoint_role"]:
        raise ValueError(f"Stale manifest checkpoints.rasnet.role: {ras_sec.get('role')!r}")

    nn_sec = ckpts_sec.get("nnunet")
    if not isinstance(nn_sec, Mapping):
        raise ValueError("Missing or invalid manifest.json section 'checkpoints.nnunet'")
    if nn_sec.get("checkpoint_final_sha256") != expected_state["nnunet_sha256"]:
        raise ValueError(f"Stale manifest nnunet_sha256: {nn_sec.get('checkpoint_final_sha256')!r}")
    if "last_pt_sha256" in nn_sec and nn_sec.get("last_pt_sha256") != expected_state["nnunet_sha256"]:
        raise ValueError(f"Stale manifest checkpoints.nnunet.last_pt_sha256: {nn_sec.get('last_pt_sha256')!r}")
    if nn_sec.get("plans_sha256") != expected_state["plans_sha256"]:
        raise ValueError(f"Stale manifest checkpoints.nnunet.plans_sha256: {nn_sec.get('plans_sha256')!r}")
    if nn_sec.get("dataset_sha256") != expected_state["dataset_sha256"]:
        raise ValueError(f"Stale manifest checkpoints.nnunet.dataset_sha256: {nn_sec.get('dataset_sha256')!r}")
    if nn_sec.get("role") != expected_state["checkpoint_role"]:
        raise ValueError(f"Stale manifest checkpoints.nnunet.role: {nn_sec.get('role')!r}")

    proto_sec = existing.get("protocol")
    if not isinstance(proto_sec, Mapping):
        raise ValueError("Missing or invalid manifest.json section 'protocol'")
    if proto_sec.get("protocol_version") != expected_state["protocol_version"]:
        raise ValueError(
            f"Stale manifest protocol.protocol_version: {proto_sec.get('protocol_version')!r}"
        )
    if proto_sec.get("checkpoint_role") != expected_state["checkpoint_role"]:
        raise ValueError(
            f"Stale manifest protocol.checkpoint_role: {proto_sec.get('checkpoint_role')!r}"
        )
    for pk in ("tta", "threshold", "postprocess", "sw_overlap", "sw_mode"):
        if pk not in proto_sec or proto_sec[pk] != FROZEN_PROTOCOL[pk]:
            raise ValueError(f"Stale manifest protocol '{pk}' mismatch: {proto_sec.get(pk)!r}")


def validate_run_state_and_reject_stale(
    out_dir: Path,
    expected_state: dict,
    case_ids: list[int],
    pairs: dict[int, tuple[Path, Path]],
    resume: bool = False,
    export_masks: bool = True,
) -> dict[str, set[int]]:
    """Reject stale/mismatched state, manifests, masks, or partial CSVs; return completed case IDs."""
    out = Path(out_dir)
    req_set = {int(c) for c in case_ids}
    done_ids: dict[str, set[int]] = {m: set() for m in MODELS}
    disk_mask_ids: dict[str, set[int]] = {m: set() for m in MODELS}

    if not out.exists():
        return done_ids

    manifest_p = out / "manifest.json"
    manifest_data: dict | None = None
    if resume:
        state_p = out / "endpoint_run_state.json"
        if state_p.is_file():
            existing_state = json.loads(state_p.read_text(encoding="utf-8"))
            if not isinstance(existing_state, Mapping):
                raise ValueError(f"Invalid endpoint_run_state.json payload in {state_p}")
            _validate_flat_state_dict(existing_state, expected_state, "endpoint_run_state.json")

        if manifest_p.is_file():
            manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
            if not isinstance(manifest_data, Mapping):
                raise ValueError(f"Invalid manifest.json payload in {manifest_p}")
            is_nested = any(
                k in manifest_data for k in ("split", "checkpoints", "protocol", "generated_at_utc", "outputs")
            )
            if is_nested or not all(k in manifest_data for k in FLAT_STATE_KEYS):
                _validate_nested_manifest_dict(manifest_data, expected_state)
            else:
                _validate_flat_state_dict(manifest_data, expected_state, "manifest.json")

    for model in MODELS:
        pred_dir = out / "predictions" / model
        if pred_dir.is_dir():
            for p in sorted(pred_dir.iterdir()):
                if not p.name.endswith(".nii.gz"):
                    raise ValueError(f"Stale unexpected artifact in predictions/{model}: {p}")
                stem = p.name[: -len(".nii.gz")]
                try:
                    cid = int(stem)
                except ValueError as exc:
                    raise ValueError(f"Stale non-case NIfTI in predictions/{model}: {p}") from exc
                if cid not in req_set:
                    raise ValueError(f"Stale prediction mask for case {cid} outside requested case_ids: {p}")
                pred_img = sitk.ReadImage(str(p))
                ref_img = sitk.ReadImage(str(pairs[cid][0]))
                lbl_img = sitk.ReadImage(str(pairs[cid][1]))
                try:
                    check_geometry(image_geometry(ref_img), image_geometry(pred_img), atol=1e-5)
                    check_geometry(image_geometry(lbl_img), image_geometry(pred_img), atol=1e-4)
                except ValueError as exc:
                    raise ValueError(
                        f"Stale prediction geometry mismatch for {model} case {cid}: {exc}"
                    ) from exc
                disk_mask_ids[model].add(cid)

        csv_p = out / f"per_case_metrics_{model}_epoch200.csv"
        if csv_p.is_file() and csv_p.stat().st_size > 0:
            df = pd.read_csv(csv_p)
            for req_col in ("case_id", "checkpoint_sha256", "checkpoint_role"):
                if req_col not in df.columns:
                    raise ValueError(f"Missing required column '{req_col}' in {csv_p}")
            if df["case_id"].isna().any() or df["case_id"].duplicated().any():
                raise ValueError(f"Duplicate or NaN case_id rows in existing {csv_p}")
            present_ids = {int(c) for c in df["case_id"]}
            extra = present_ids - req_set
            if extra:
                raise ValueError(f"Stale case IDs in {csv_p} outside requested case_ids: {sorted(extra)}")
            exp_sha = (
                expected_state["rasnet_sha256"] if model == "rasnet" else expected_state["nnunet_sha256"]
            )
            if not (df["checkpoint_sha256"] == exp_sha).all():
                raise ValueError(f"Stale checkpoint_sha256 in {csv_p}")
            if not (df["checkpoint_role"] == CHECKPOINT_ROLE).all():
                raise ValueError(f"Stale checkpoint_role in {csv_p}")
            done_ids[model] = present_ids

        orphaned = sorted(disk_mask_ids[model] - done_ids[model])
        if orphaned:
            raise ValueError(
                f"Stale/orphaned prediction mask(s) in predictions/{model} missing from {csv_p.name}: {orphaned}"
            )

    if resume and manifest_p.is_file():
        for model in MODELS:
            csv_p = out / f"per_case_metrics_{model}_epoch200.csv"
            if not csv_p.is_file() or csv_p.stat().st_size == 0:
                raise ValueError(f"Incomplete completed run: missing or empty {csv_p} while manifest.json exists")
            if done_ids[model] != req_set:
                missing = sorted(req_set - done_ids[model])
                raise ValueError(
                    f"Incomplete completed run: {csv_p.name} missing case_ids {missing} while manifest.json exists"
                )

        comb_p = out / "per_case_metrics_combined.csv"
        if not comb_p.is_file() or comb_p.stat().st_size == 0:
            raise ValueError(f"Incomplete completed run: missing or empty {comb_p} while manifest.json exists")
        comb_df = pd.read_csv(comb_p)
        for req_col in ("model", "case_id", "checkpoint_sha256", "checkpoint_role"):
            if req_col not in comb_df.columns:
                raise ValueError(f"Missing required column '{req_col}' in {comb_p}")
        if comb_df["model"].isna().any() or comb_df["case_id"].isna().any():
            raise ValueError(f"NaN model/case_id in {comb_p}")
        if comb_df.duplicated(subset=["model", "case_id"]).any():
            raise ValueError(f"Duplicate (model, case_id) rows in {comb_p}")
        expected_pairs = {(m, c) for m in MODELS for c in req_set}
        got_pairs = {(str(r["model"]), int(r["case_id"])) for _, r in comb_df.iterrows()}
        if got_pairs != expected_pairs:
            missing_pairs = sorted(expected_pairs - got_pairs)
            extra_pairs = sorted(got_pairs - expected_pairs)
            raise ValueError(
                f"Incomplete/mismatched {comb_p.name}: missing={missing_pairs}, extra={extra_pairs}"
            )
        if not (comb_df["checkpoint_role"] == CHECKPOINT_ROLE).all():
            raise ValueError(f"Stale checkpoint_role in {comb_p}")
        for m, exp_sha in (
            ("rasnet", expected_state["rasnet_sha256"]),
            ("nnunet", expected_state["nnunet_sha256"]),
        ):
            sub_m = comb_df[comb_df["model"] == m]
            if not (sub_m["checkpoint_sha256"] == exp_sha).all():
                raise ValueError(f"Stale checkpoint_sha256 for {m} in {comb_p}")

        outputs_sec = manifest_data.get("outputs") if isinstance(manifest_data, Mapping) else None
        geom_in_manifest = True
        if isinstance(outputs_sec, Mapping) and outputs_sec.get("geometry_verification_csv") is None:
            geom_in_manifest = False
        if export_masks and geom_in_manifest:
            geom_p = out / "geometry_verification.csv"
            if not geom_p.is_file() or geom_p.stat().st_size == 0:
                raise ValueError(
                    f"Incomplete completed run: missing or empty {geom_p} while manifest.json exists"
                )
            geom_df = pd.read_csv(geom_p)
            for req_col in ("model", "case_id", "geometry_ok"):
                if req_col not in geom_df.columns:
                    raise ValueError(f"Missing required column '{req_col}' in {geom_p}")
            if len(geom_df) != 2 * len(req_set):
                raise ValueError(
                    f"Incomplete {geom_p.name}: expected {2 * len(req_set)} rows, got {len(geom_df)}"
                )
            got_geom_pairs = {(str(r["model"]), int(r["case_id"])) for _, r in geom_df.iterrows()}
            if got_geom_pairs != expected_pairs or not bool(geom_df["geometry_ok"].all()):
                raise ValueError(f"Incomplete or failed geometry verification in {geom_p}")

    return done_ids


def validate_complete_results(
    df_rasnet: pd.DataFrame,
    df_nnunet: pd.DataFrame,
    case_ids: list[int],
) -> None:
    """Verify exact unique case coverage for both models with zero silent drops."""
    expected = sorted(int(c) for c in case_ids)
    for name, df in (("rasnet", df_rasnet), ("nnunet", df_nnunet)):
        if df.empty or "case_id" not in df.columns:
            raise ValueError(f"Incomplete results for {name}: empty table")
        if df["case_id"].isna().any() or df["case_id"].duplicated().any():
            raise ValueError(f"Incomplete or duplicate case_id in {name} results")
        got = sorted(int(c) for c in df["case_id"])
        if got != expected:
            missing = sorted(set(expected) - set(got))
            extra = sorted(set(got) - set(expected))
            raise ValueError(
                f"Incomplete results for {name}: missing={missing}, extra={extra}"
            )


def summarize_endpoint_metrics(
    df: pd.DataFrame,
    seed: int = 42,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Summarize per-model metrics with explicit status counts and bootstrap 95% CIs."""
    rows = []
    for model in MODELS:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        r = {
            "model": model,
            "checkpoint_role": CHECKPOINT_ROLE,
            "n_cases": int(len(sub)),
            "n_ok": int((sub["status"] == "ok").sum()),
            "n_empty_pred": int((sub["status"] == "prediction_empty").sum()),
            "n_empty_gt": int((sub["status"] == "ground_truth_empty").sum()),
            "n_both_empty": int((sub["status"] == "both_empty").sum()),
            "n_empty_skeleton": int((sub["status"] == "empty_skeleton").sum()),
        }
        for m in ALL_METRICS:
            vals = sub[m].dropna().to_numpy(dtype=float)
            r[f"{m}_n_valid"] = int(len(vals))
            if len(vals) > 0:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                med = float(np.median(vals))
                q25 = float(np.quantile(vals, 0.25))
                q75 = float(np.quantile(vals, 0.75))
                ci_lo, ci_hi = bootstrap_ci(vals, seed=seed, n_boot=n_boot, confidence=0.95)
                r[f"{m}_mean"] = mean
                r[f"{m}_std"] = std
                r[f"{m}_median"] = med
                r[f"{m}_q25"] = q25
                r[f"{m}_q75"] = q75
                r[f"{m}_ci95_lo"] = ci_lo
                r[f"{m}_ci95_hi"] = ci_hi
            else:
                for suf in ("mean", "std", "median", "q25", "q75", "ci95_lo", "ci95_hi"):
                    r[f"{m}_{suf}"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def compute_endpoint_paired_stats(
    df_rasnet: pd.DataFrame,
    df_nnunet: pd.DataFrame,
    metrics: tuple[str, ...] = ALL_METRICS,
    seed: int = 42,
    n_boot: int = 2000,
    tie_tolerance: float = 1e-12,
) -> tuple[dict, pd.DataFrame]:
    """Compute paired RASNet-minus-nnU-Net statistics across all metrics with NaN tracking."""
    if df_rasnet["case_id"].duplicated().any() or df_nnunet["case_id"].duplicated().any():
        raise ValueError("Duplicate case_id in paired stats inputs")
    if set(df_rasnet["case_id"]) != set(df_nnunet["case_id"]):
        raise ValueError("Case ID sets differ between RASNet and nnU-Net")

    n_total = int(len(df_rasnet))
    rows = []
    for metric in metrics:
        direction = METRIC_DIRECTIONS.get(metric, "higher")
        merged = (
            df_rasnet[["case_id", metric]]
            .merge(df_nnunet[["case_id", metric]], on="case_id", validate="one_to_one", suffixes=("_a", "_b"))
        )
        valid_ids = set(merged.dropna()["case_id"])
        n_excl = int(n_total - len(valid_ids))
        if not valid_ids:
            res = {
                "metric": metric,
                "direction": direction,
                "difference_definition": "rasnet_minus_nnunet",
                "n": 0,
                "n_excluded_nan": n_excl,
                "rasnet_mean": np.nan,
                "nnunet_mean": np.nan,
                "mean_difference": np.nan,
                "diff_ci95_lo": np.nan,
                "diff_ci95_hi": np.nan,
                "median_difference": np.nan,
                "q25_difference": np.nan,
                "q75_difference": np.nan,
                "iqr_difference": np.nan,
                "p": 1.0,
                "effect_size_rank_biserial": np.nan,
                "wins": 0,
                "ties": 0,
                "losses": 0,
                "tie_tolerance": float(tie_tolerance),
            }
        else:
            ta = df_rasnet[df_rasnet["case_id"].isin(valid_ids)].copy()
            tb = df_nnunet[df_nnunet["case_id"].isin(valid_ids)].copy()
            res = paired_analysis(ta, tb, metric, tie_tolerance=tie_tolerance, seed=seed, n_boot=n_boot)
            if direction == "lower":
                res["wins"], res["losses"] = res["losses"], res["wins"]
            res.update({
                "metric": metric,
                "direction": direction,
                "difference_definition": "rasnet_minus_nnunet",
                "n_excluded_nan": n_excl,
                "rasnet_mean": float(ta[metric].mean()),
                "nnunet_mean": float(tb[metric].mean()),
                "tie_tolerance": float(tie_tolerance),
            })
        rows.append(res)

    paired_df = pd.DataFrame(rows)
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_role": CHECKPOINT_ROLE,
        "comparison": "rasnet_minus_nnunet",
        "seed": int(seed),
        "n_boot": int(n_boot),
        "tie_tolerance": float(tie_tolerance),
        "wilcoxon_note": (
            "benchmark.stats.paired_analysis applies tie_tolerance to wins/ties/losses counts, "
            "while Wilcoxon/rank-biserial treats exact d != 0 as non-zero."
        ),
        "n_total_cases": n_total,
        "status_summary": {
            "rasnet": df_rasnet["status"].value_counts().to_dict(),
            "nnunet": df_nnunet["status"].value_counts().to_dict(),
        },
        "metrics": rows,
    }
    return meta, paired_df


def _load_gt_and_ref(img_path: Path, lbl_path: Path):
    ref_img = sitk.ReadImage(str(img_path))
    lbl_img = sitk.ReadImage(str(lbl_path))
    gt_bool = sitk.GetArrayFromImage(lbl_img) > 0
    spacing_zyx = tuple(float(s) for s in reversed(lbl_img.GetSpacing()))
    return ref_img, lbl_img, gt_bool, spacing_zyx


def _score_and_record_case(
    model: str,
    cid: int,
    prob_zyx: np.ndarray,
    pairs: dict[int, tuple[Path, Path]],
    ckpt_sha256: str,
    out_dir: Path,
    export_masks: bool,
) -> tuple[dict, dict | None]:
    img_p, lbl_p = pairs[cid]
    ref_img, lbl_img, gt_bool, spacing_zyx = _load_gt_and_ref(img_p, lbl_p)

    if prob_zyx.shape != gt_bool.shape:
        raise RuntimeError(f"{model} case {cid}: prob shape {prob_zyx.shape} != gt shape {gt_bool.shape}")
    if not np.all(np.isfinite(prob_zyx)):
        raise RuntimeError(f"Non-finite probability from {model} on case {cid}")
    if not np.all((prob_zyx >= -1e-6) & (prob_zyx <= 1.0 + 1e-6)):
        raise RuntimeError(
            f"Probability out of [0, 1] range from {model} on case {cid}: "
            f"min={float(prob_zyx.min())}, max={float(prob_zyx.max())}"
        )

    raw_mask = (prob_zyx >= FROZEN_PROTOCOL["threshold"]).astype(np.uint8)
    final_mask, comp_stats = prune_components(
        raw_mask,
        spacing_zyx=spacing_zyx,
        keep_top_k=None,
        connectivity=int(FROZEN_PROTOCOL["connectivity"]),
    )
    metrics = compute_metrics(final_mask.astype(bool), gt_bool, spacing_zyx)

    geom_row = None
    if export_masks:
        mask_path = out_dir / "predictions" / model / f"{cid}.nii.gz"
        export_mask_like(final_mask, str(img_p), str(mask_path))
        written = sitk.ReadImage(str(mask_path))
        ref_geom = image_geometry(ref_img)
        lbl_geom = image_geometry(lbl_img)
        mask_geom = image_geometry(written)
        check_geometry(ref_geom, mask_geom, atol=1e-5)
        check_geometry(lbl_geom, mask_geom, atol=1e-4)
        geom_row = {
            "case_id": int(cid),
            "model": model,
            "mask_path": str(mask_path),
            "image_size": json.dumps(list(ref_geom["size"])),
            "mask_size": json.dumps(list(mask_geom["size"])),
            "image_spacing": json.dumps([round(x, 6) for x in ref_geom["spacing"]]),
            "mask_spacing": json.dumps([round(x, 6) for x in mask_geom["spacing"]]),
            "image_origin": json.dumps([round(x, 6) for x in ref_geom["origin"]]),
            "mask_origin": json.dumps([round(x, 6) for x in mask_geom["origin"]]),
            "geometry_ok": True,
        }

    row = {
        "case_id": int(cid),
        "model": model,
        "checkpoint_role": CHECKPOINT_ROLE,
        "checkpoint_sha256": ckpt_sha256,
        "tta": False,
        "threshold": float(FROZEN_PROTOCOL["threshold"]),
        "postprocess": "none",
        "keep_top_k": 0,
        "metric_version": METRIC_VERSION,
        **metrics,
        **comp_stats,
    }
    return row, geom_row


def run_endpoint_evaluation(
    repo_root: Path,
    config_path: Path,
    out_dir: Path,
    rasnet_ckpt: Path | None = None,
    nnunet_last_ckpt: Path | None = None,
    nnunet_model_folder: Path | None = None,
    nnunet_fold: int = DEFAULT_NNUNET_FOLD,
    nnunet_checkpoint_name: str = DEFAULT_NNUNET_CKPT_NAME,
    case_ids: list[int] | None = None,
    smoke: bool = False,
    device: str = "cuda",
    sw_batch_size: int | None = None,
    use_amp: bool | None = None,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    export_masks: bool = True,
    resume: bool = False,
    run_id: str | None = None,
    expected_val_count: int = 94,
    expected_hashes: dict[str, str] | None = None,
    require_default_paths: bool = True,
    rasnet_model_builder=None,
    nnunet_predictor_builder=None,
    nnunet_batch_runner=None,
    nnunet_num_processes: int = 4,
) -> Path:
    """Execute the frozen Val94 epoch-200 endpoint evaluation for RASNet and nnU-Net V2."""
    import yaml

    repo = Path(repo_root).resolve()
    cfg_p = Path(config_path)
    if not cfg_p.is_absolute():
        cfg_p = (repo / cfg_p).resolve()
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))

    split_p = Path(cfg["split_file"])
    if not split_p.is_absolute():
        split_p = (repo / split_p).resolve()
    splits = json.loads(split_p.read_text(encoding="utf-8"))
    splits_sha = sha256_file(split_p)
    cfg_sha = sha256_file(cfg_p)

    selected_ids = validate_endpoint_case_ids(
        splits, requested_case_ids=case_ids, expected_val_count=expected_val_count
    )
    pairs = resolve_case_pairs(cfg["dataset_root"], selected_ids, check_headers=True)
    validate_case_files(pairs)

    rasnet_p = Path(rasnet_ckpt) if rasnet_ckpt is not None else (repo / DEFAULT_RASNET_CKPT)
    nnunet_last_p = Path(nnunet_last_ckpt) if nnunet_last_ckpt is not None else (repo / DEFAULT_NNUNET_LAST_CKPT)
    nnunet_folder_p = (
        Path(nnunet_model_folder) if nnunet_model_folder is not None else (repo / DEFAULT_NNUNET_MODEL_FOLDER)
    )
    if not rasnet_p.is_absolute():
        rasnet_p = (repo / rasnet_p).resolve()
    if not nnunet_last_p.is_absolute():
        nnunet_last_p = (repo / nnunet_last_p).resolve()
    if not nnunet_folder_p.is_absolute():
        nnunet_folder_p = (repo / nnunet_folder_p).resolve()

    prov = verify_endpoint_checkpoints(
        rasnet_ckpt=rasnet_p,
        nnunet_last_ckpt=nnunet_last_p,
        nnunet_model_folder=nnunet_folder_p,
        nnunet_fold=nnunet_fold,
        nnunet_checkpoint_name=nnunet_checkpoint_name,
        repo_root=repo,
        expected_hashes=expected_hashes,
        require_default_paths=require_default_paths,
    )

    out = validate_endpoint_out_dir(out_dir, repo, resume=resume)
    effective_run_id = run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    expected_state = {
        "run_id": effective_run_id,
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_role": CHECKPOINT_ROLE,
        "case_ids": selected_ids,
        "splits_sha256": splits_sha,
        "rasnet_sha256": prov["rasnet"]["sha256"],
        "nnunet_sha256": prov["nnunet"]["checkpoint_final_sha256"],
        "plans_sha256": prov["nnunet"]["plans_sha256"],
        "dataset_sha256": prov["nnunet"]["dataset_sha256"],
    }
    done_ids = validate_run_state_and_reject_stale(
        out, expected_state, selected_ids, pairs, resume=resume, export_masks=export_masks
    )

    out.mkdir(parents=True, exist_ok=True)
    (out / "endpoint_run_state.json").write_text(
        json.dumps(expected_state, indent=2, sort_keys=True), encoding="utf-8"
    )

    effective_sw_bs = int(sw_batch_size if sw_batch_size is not None else cfg.get("sw_batch_size", 8))
    amp_enabled = str(device).startswith("cuda") if use_amp is None else bool(use_amp)

    geom_rows: list[dict] = []
    geom_csv = out / "geometry_verification.csv"
    if resume and geom_csv.is_file() and geom_csv.stat().st_size > 0:
        geom_rows = pd.read_csv(geom_csv).to_dict("records")
    existing_geom_keys = {(str(r["model"]), int(r["case_id"])) for r in geom_rows}
    if resume and export_masks:
        for model in MODELS:
            for cid in sorted(done_ids[model]):
                mask_path = out / "predictions" / model / f"{cid}.nii.gz"
                if not mask_path.is_file():
                    raise ValueError(
                        f"Incomplete resume state: missing exported mask {mask_path} for completed {model} case {cid}"
                    )
                if (model, cid) not in existing_geom_keys:
                    ref_img = sitk.ReadImage(str(pairs[cid][0]))
                    written = sitk.ReadImage(str(mask_path))
                    ref_geom = image_geometry(ref_img)
                    mask_geom = image_geometry(written)
                    check_geometry(ref_geom, mask_geom, atol=1e-5)
                    geom_rows.append({
                        "case_id": int(cid),
                        "model": model,
                        "mask_path": str(mask_path),
                        "image_size": json.dumps(list(ref_geom["size"])),
                        "mask_size": json.dumps(list(mask_geom["size"])),
                        "image_spacing": json.dumps([round(x, 6) for x in ref_geom["spacing"]]),
                        "mask_spacing": json.dumps([round(x, 6) for x in mask_geom["spacing"]]),
                        "image_origin": json.dumps([round(x, 6) for x in ref_geom["origin"]]),
                        "mask_origin": json.dumps([round(x, 6) for x in mask_geom["origin"]]),
                        "geometry_ok": True,
                    })

    def _existing_df(model: str) -> pd.DataFrame:
        p = out / f"per_case_metrics_{model}_epoch200.csv"
        if p.is_file() and p.stat().st_size > 0:
            return pd.read_csv(p)
        return pd.DataFrame()

    # 1. RASNet epoch-200 inference
    rasnet_todo = [c for c in selected_ids if c not in done_ids["rasnet"]]
    t_rasnet_start = time.perf_counter()
    if rasnet_todo:
        r_builder = rasnet_model_builder or (lambda p, dev: build_model("rasnet", str(p), dev))
        net = r_builder(rasnet_p, device)
        tf = build_transform("rasnet")
        r_rows: list[dict] = []

        def _prep(cid: int):
            return cid, tf({"image": str(pairs[cid][0])})

        with ThreadPoolExecutor(max_workers=2) as pool:
            queue = deque(pool.submit(_prep, c) for c in rasnet_todo[:2])
            nxt = len(queue)
            while queue:
                cid, batch = queue.popleft().result()
                if nxt < len(rasnet_todo):
                    queue.append(pool.submit(_prep, rasnet_todo[nxt]))
                    nxt += 1
                modes = infer_monai_symmetric(
                    "rasnet",
                    net,
                    batch,
                    tf,
                    device,
                    roi_size=roi_size,
                    sw_batch_size=effective_sw_bs,
                    use_amp=amp_enabled,
                    tta=False,
                )
                del batch
                prob_zyx = modes[SYM_MODES[0]]
                row, g_row = _score_and_record_case(
                    "rasnet", cid, prob_zyx, pairs, prov["rasnet"]["sha256"], out, export_masks
                )
                r_rows.append(row)
                if g_row is not None:
                    geom_rows.append(g_row)

        r_csv = out / "per_case_metrics_rasnet_epoch200.csv"
        pd.DataFrame(r_rows).to_csv(
            r_csv,
            mode="a",
            header=(not r_csv.exists() or r_csv.stat().st_size == 0),
            index=False,
        )
    t_rasnet_sec = time.perf_counter() - t_rasnet_start

    # 2. nnU-Net V2 epoch-200 inference
    nnunet_todo = [c for c in selected_ids if c not in done_ids["nnunet"]]
    t_nnunet_start = time.perf_counter()
    if nnunet_todo:
        n_builder = nnunet_predictor_builder or (
            lambda folder, fold, ckpt_name, dev: build_nnunet_endpoint_predictor(
                model_folder=folder,
                fold=fold,
                checkpoint_name=ckpt_name,
                device=dev,
                use_mirroring=False,
            )
        )
        predictor = n_builder(nnunet_folder_p, nnunet_fold, nnunet_checkpoint_name, device)
        runner = nnunet_batch_runner or _run_nnunet_batch_probabilities
        n_rows: list[dict] = []
        yielded_cids: list[int] = []
        for cid, prob_1pass, _ in runner(
            predictor, nnunet_todo, pairs, need_tta=False, num_processes=nnunet_num_processes
        ):
            yielded_cids.append(int(cid))
            row, g_row = _score_and_record_case(
                "nnunet",
                int(cid),
                prob_1pass,
                pairs,
                prov["nnunet"]["checkpoint_final_sha256"],
                out,
                export_masks,
            )
            n_rows.append(row)
            if g_row is not None:
                geom_rows.append(g_row)
        if sorted(yielded_cids) != sorted(nnunet_todo):
            raise RuntimeError(
                f"nnU-Net yielded case IDs {sorted(yielded_cids)} != expected {sorted(nnunet_todo)}"
            )
        n_csv = out / "per_case_metrics_nnunet_epoch200.csv"
        pd.DataFrame(n_rows).to_csv(
            n_csv,
            mode="a",
            header=(not n_csv.exists() or n_csv.stat().st_size == 0),
            index=False,
        )
    t_nnunet_sec = time.perf_counter() - t_nnunet_start

    # 3. Completeness verification & combined outputs
    df_r = _existing_df("rasnet").sort_values("case_id").reset_index(drop=True)
    df_n = _existing_df("nnunet").sort_values("case_id").reset_index(drop=True)
    validate_complete_results(df_r, df_n, selected_ids)

    df_r.to_csv(out / "per_case_metrics_rasnet_epoch200.csv", index=False)
    df_n.to_csv(out / "per_case_metrics_nnunet_epoch200.csv", index=False)
    combined = pd.concat([df_r, df_n], ignore_index=True)
    combined.to_csv(out / "per_case_metrics_combined.csv", index=False)

    if export_masks:
        geom_df = (
            pd.DataFrame(geom_rows)
            .drop_duplicates(subset=["model", "case_id"], keep="last")
            .sort_values(["model", "case_id"])
            .reset_index(drop=True)
        )
        if len(geom_df) != 2 * len(selected_ids) or not bool(geom_df["geometry_ok"].all()):
            raise RuntimeError("Geometry verification incomplete or failed")
        geom_df.to_csv(geom_csv, index=False)

    summary_df = summarize_endpoint_metrics(
        combined,
        seed=int(FROZEN_PROTOCOL["bootstrap_seed"]),
        n_boot=int(FROZEN_PROTOCOL["bootstrap_n_boot"]),
    )
    summary_df.to_csv(out / "summary_epoch200.csv", index=False)

    paired_meta, paired_df = compute_endpoint_paired_stats(
        df_r,
        df_n,
        metrics=ALL_METRICS,
        seed=int(FROZEN_PROTOCOL["bootstrap_seed"]),
        n_boot=int(FROZEN_PROTOCOL["bootstrap_n_boot"]),
        tie_tolerance=float(FROZEN_PROTOCOL["tie_tolerance"]),
    )
    paired_df.to_csv(out / "paired_stats_epoch200.csv", index=False)
    (out / "paired_stats_epoch200.json").write_text(
        json.dumps(paired_meta, indent=2), encoding="utf-8"
    )

    script_p = repo / "scripts" / "run_v1_epoch200_endpoint_eval.py"
    endpoint_mod_p = repo / "benchmark" / "endpoint_eval.py"
    prov_json_p = repo / DEFAULT_PROVENANCE_JSON
    manifest = {
        "run_id": effective_run_id,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke": bool(smoke),
        "checkpoint_role": CHECKPOINT_ROLE,
        "evidence_separation_note": (
            "Epoch-200 endpoint comparison (final_epoch200 checkpoints, 1-pass SW overlap 0.50, "
            "Gaussian, tau=0.50, no post-processing). Strictly distinct from best-checkpoint runs "
            "(RASNet best ~ep160, nnU-Net best current_epoch=199) and Option A."
        ),
        "source": {
            "source_tree_sha256": source_tree_sha256(repo),
            "endpoint_eval_sha256": sha256_file(endpoint_mod_p) if endpoint_mod_p.is_file() else None,
            "cli_script_sha256": sha256_file(script_p) if script_p.is_file() else None,
            "provenance_verification_json": str(prov_json_p) if prov_json_p.is_file() else None,
            "provenance_verification_sha256": sha256_file(prov_json_p) if prov_json_p.is_file() else None,
        },
        "config": {
            "path": str(cfg_p),
            "sha256": cfg_sha,
            "dataset_root": str(cfg["dataset_root"]),
            "split_file": str(split_p),
            "sw_batch_size": effective_sw_bs,
        },
        "env": {**env_info(), **package_versions()},
        "git": git_info(repo),
        "split": {
            "split_name": "val",
            "splits_sha256": splits_sha,
            "total_val_cases": int(len(splits["val"])),
            "evaluated_case_count": int(len(selected_ids)),
            "case_ids": selected_ids,
        },
        "checkpoints": prov,
        "protocol": {
            **FROZEN_PROTOCOL,
            "overlap": float(SW_OVERLAP),
            "patch_size": {"rasnet": list(roi_size), "nnunet": prov["nnunet"]["patch_size"]},
            "normalization": {
                "rasnet": "ScaleIntensityRanged(a_min=-100, a_max=800, b_min=0, b_max=1, clip=True)",
                "nnunet": prov["nnunet"]["normalization_schemes"],
            },
            "spacing": {"rasnet": [0.5, 0.5, 0.5], "nnunet": prov["nnunet"]["spacing"]},
            "inversion": {
                "rasnet": "Invertd(nearest_interp=False, to_tensor=True) + np.transpose((2, 1, 0))",
                "nnunet": "nnunetv2.inference.export_prediction.convert_predicted_logits_to_segmentation_with_correct_shape",
            },
            "precision_mode": {
                "rasnet": "amp_fp16_autocast" if amp_enabled else "fp32",
                "nnunet": "amp_autocast" if str(device).startswith("cuda") else "fp32",
            },
            "sw_batch_size": {"rasnet": effective_sw_bs, "nnunet": 1},
            "rasnet": {
                "patch_size": list(roi_size),
                "normalization": "ScaleIntensityRanged(a_min=-100, a_max=800, b_min=0, b_max=1, clip=True)",
                "spacing": [0.5, 0.5, 0.5],
                "inversion": "Invertd(nearest_interp=False, to_tensor=True) + np.transpose((2, 1, 0))",
                "precision_mode": "amp_fp16_autocast" if amp_enabled else "fp32",
                "sw_batch_size": effective_sw_bs,
                "transform_recipe": list(transform_recipe("rasnet")),
                "overlap": float(SW_OVERLAP),
                "sw_mode": str(SW_MODE),
                "sigma_scale": float(SIGMA_SCALE),
            },
            "nnunet": {
                "patch_size": prov["nnunet"]["patch_size"],
                "normalization": prov["nnunet"]["normalization_schemes"],
                "spacing": prov["nnunet"]["spacing"],
                "inversion": "nnunetv2.inference.export_prediction.convert_predicted_logits_to_segmentation_with_correct_shape",
                "precision_mode": "amp_autocast" if str(device).startswith("cuda") else "fp32",
                "sw_batch_size": 1,
                "tile_step_size": 0.5,
                "use_gaussian": True,
                "use_mirroring": False,
            },
        },
        "timing_seconds": {
            "rasnet": round(t_rasnet_sec, 3),
            "nnunet": round(t_nnunet_sec, 3),
        },
        "outputs": {
            "per_case_rasnet_csv": str(out / "per_case_metrics_rasnet_epoch200.csv"),
            "per_case_nnunet_csv": str(out / "per_case_metrics_nnunet_epoch200.csv"),
            "per_case_combined_csv": str(out / "per_case_metrics_combined.csv"),
            "geometry_verification_csv": str(geom_csv) if export_masks else None,
            "summary_csv": str(out / "summary_epoch200.csv"),
            "paired_stats_csv": str(out / "paired_stats_epoch200.csv"),
            "paired_stats_json": str(out / "paired_stats_epoch200.json"),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
