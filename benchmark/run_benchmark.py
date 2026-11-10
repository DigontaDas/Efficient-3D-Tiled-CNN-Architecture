"""End-to-end fail-closed benchmark runner for the 5-model ImageCAS evaluation."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

from benchmark.adapters.monai_models import (
    build_model,
    build_random_model,
    build_transform,
    infer_probability,
    model_constructor_args,
    transform_recipe,
)
from benchmark.adapters.nnunet_adapter import (
    build_nnunet_predictor,
    export_mask_like,
    infer_nnunet_probability,
)
from benchmark.comparison import compare_with_historical
from benchmark.dataset import resolve_case_pairs
from benchmark.metrics import METRIC_VERSION, check_geometry, compute_metrics
from benchmark.postprocess import prune_components
from benchmark.preflight import (
    cache_key,
    image_geometry,
    validate_case_files,
    validate_splits,
    validate_thresholds,
)
from benchmark.stats import analyze_family, bootstrap_ci
from benchmark.thresholds import (
    DEFAULT_THRESHOLD_GRID,
    load_and_validate_thresholds,
    save_thresholds_json,
    select_validation_thresholds,
)

ALL_MODELS = ("rasnet", "segresnet", "vnet", "3dunet", "nnunet")
ALL_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "hd95",
    "asd",
    "cldice",
    "centerline_recall",
)
METRIC_DIRECTIONS = {
    "dice": "higher",
    "iou": "higher",
    "precision": "higher",
    "recall": "higher",
    "hd95": "lower",
    "asd": "lower",
    "cldice": "higher",
    "centerline_recall": "higher",
}


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing required file for SHA-256: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo_root: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_root), text=True
        ).strip()
        return {"commit": commit, "branch": branch, "dirty": bool(status)}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def package_versions() -> dict:
    import cc3d
    import monai
    import nibabel
    import scipy
    import skimage

    try:
        import importlib.metadata as im

        nnunet_ver = im.version("nnunetv2")
    except Exception:
        nnunet_ver = "unknown"
    return {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "monai": monai.__version__,
        "nnunetv2": nnunet_ver,
        "SimpleITK": sitk.__version__,
        "nibabel": nibabel.__version__,
        "scikit_image": skimage.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cc3d": getattr(cc3d, "__version__", "installed"),
    }


def _infer_monai_both_passes(
    name: str,
    net,
    image_path: str,
    device: str,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 8,
    need_tta: bool = False,
    transform=None,
    batch: dict | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Run MONAI preprocessing once and return (prob_1pass_zyx, prob_tta_zyx or None).
    Guarantees 1-pass and 4-pass TTA share the exact same loaded/preprocessed tensor.
    """
    from monai.inferers import sliding_window_inference
    from monai.transforms import Invertd

    if hasattr(net, "eval"):
        net.eval()
    if name == "vnet":
        roi_size = tuple(((int(r) + 15) // 16) * 16 for r in roi_size)

    if transform is None:
        transform = build_transform(name)
    if batch is None:
        batch = transform({"image": str(image_path)})
    x = batch["image"].unsqueeze(0).to(device)

    use_amp = str(device).startswith("cuda")

    def predict(tensor):
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = (
                net(tensor)
                if name == "3dunet"
                else sliding_window_inference(
                    tensor, roi_size, sw_batch_size, net, overlap=0.25
                )
            )
            if isinstance(out, (list, tuple)):
                out = out[0]
            probs = (
                torch.sigmoid(out)
                if name == "3dunet"
                else torch.softmax(out.float(), dim=1)[:, 1:2]
            )
        return probs.float()

    with torch.no_grad():
        p_raw = predict(x)
        p_tta = None
        if need_tta:
            acc = p_raw.clone()
            for axis in (2, 3, 4):
                acc += torch.flip(predict(torch.flip(x, (axis,))), (axis,))
            p_tta = acc / 4.0

    batch_raw = dict(batch)
    batch_raw["pred"] = p_raw.squeeze(0)
    inv_raw = Invertd(
        keys=["pred"],
        transform=transform,
        orig_keys=["image"],
        nearest_interp=False,
        to_tensor=True,
    )(batch_raw)["pred"]
    prob_1pass = np.transpose(
        inv_raw.squeeze(0).detach().cpu().numpy(), (2, 1, 0)
    ).astype(np.float32, copy=False)

    prob_tta_zyx = None
    if p_tta is not None:
        batch_tta = dict(batch)
        batch_tta["pred"] = p_tta.squeeze(0)
        inv_tta = Invertd(
            keys=["pred"],
            transform=transform,
            orig_keys=["image"],
            nearest_interp=False,
            to_tensor=True,
        )(batch_tta)["pred"]
        prob_tta_zyx = np.transpose(
            inv_tta.squeeze(0).detach().cpu().numpy(), (2, 1, 0)
        ).astype(np.float32, copy=False)

    return prob_1pass, prob_tta_zyx


def _evaluate_binary_mask(
    cid: int,
    model: str,
    arm: str,
    raw_mask: np.ndarray,
    gt_bool: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    threshold: float,
    tta_used: bool,
    keep_top_k: int | None,
) -> tuple[np.ndarray, dict]:
    if keep_top_k is not None:
        final_mask, comp_stats = prune_components(
            raw_mask, spacing_zyx=spacing_zyx, keep_top_k=keep_top_k
        )
    else:
        _, comp_stats = prune_components(
            raw_mask, spacing_zyx=spacing_zyx, keep_top_k=None
        )
        final_mask = (raw_mask > 0).astype(np.uint8)

    m = compute_metrics(final_mask.astype(bool), gt_bool, spacing_zyx)
    row = {
        "case_id": int(cid),
        "model": model,
        "arm": arm,
        "threshold": float(threshold),
        "tta": bool(tta_used),
        "keep_top_k": int(keep_top_k) if keep_top_k is not None else 0,
        **m,
        **comp_stats,
    }
    return final_mask, row


def summarize_arm(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rows = []
    for model in ALL_MODELS:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        r = {
            "model": model,
            "n_cases": int(len(sub)),
            "n_ok": int((sub["status"] == "ok").sum()),
            "n_empty_pred": int((sub["status"] == "prediction_empty").sum()),
        }
        for m in ALL_METRICS:
            vals = sub[m].dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                med = float(np.median(vals))
                q25 = float(np.quantile(vals, 0.25))
                q75 = float(np.quantile(vals, 0.75))
                ci_lo, ci_hi = bootstrap_ci(vals, seed=seed, n_boot=2000, confidence=0.95)
                r[f"{m}_mean"] = mean
                r[f"{m}_std"] = std
                r[f"{m}_median"] = med
                r[f"{m}_q25"] = q25
                r[f"{m}_q75"] = q75
                r[f"{m}_ci95_lo"] = ci_lo
                r[f"{m}_ci95_hi"] = ci_hi
                r[f"{m}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
                r[f"{m}_median_iqr"] = f"{med:.4f} [{q25:.4f}, {q75:.4f}]"
            else:
                r[f"{m}_mean"] = np.nan
                r[f"{m}_std"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def compute_arm_statistics(df: pd.DataFrame, primary_hypotheses: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = {m: df[df["model"] == m].copy() for m in df["model"].unique()}
    if "rasnet" not in tables:
        return pd.DataFrame(), pd.DataFrame()

    # For boundary/skeleton metrics that can be NaN if a prediction is empty,
    # filter paired comparisons to cases finite in both or handle cleanly
    def safe_analyze(hyps: list[dict]) -> pd.DataFrame:
        valid_hyps = [h for h in hyps if h["a"] in tables and h["b"] in tables]
        if not valid_hyps:
            return pd.DataFrame()
        # Filter per-pair finite rows if needed
        filtered_tables = {}
        records = []
        from benchmark.stats import holm_adjust, paired_analysis

        for h in valid_hyps:
            ta = tables[h["a"]]
            tb = tables[h["b"]]
            metric = h["metric"]
            merged = ta[["case_id", metric]].merge(tb[["case_id", metric]], on="case_id", suffixes=("_a", "_b"))
            finite_ids = set(merged.dropna()["case_id"])
            ta_f = ta[ta["case_id"].isin(finite_ids)]
            tb_f = tb[tb["case_id"].isin(finite_ids)]
            if len(ta_f) == 0:
                r = {
                    "n": 0,
                    "mean_difference": np.nan,
                    "p": 1.0,
                    "effect_size_rank_biserial": np.nan,
                    "wins": 0,
                    "ties": 0,
                    "losses": 0,
                    "a_mean": np.nan,
                    "b_mean": np.nan,
                    "n_excluded_nan": int(len(ta)),
                }
            else:
                r = paired_analysis(ta_f, tb_f, metric, h.get("tie_tolerance", 0.0))
                direction = h.get("direction", "higher")
                if direction == "lower":
                    r["wins"], r["losses"] = r["losses"], r["wins"]
                r["a_mean"] = float(ta_f[metric].mean())
                r["b_mean"] = float(tb_f[metric].mean())
                r["n_excluded_nan"] = int(len(ta) - len(ta_f))
            r.update(h)
            records.append(r)
        for r, p_adj in zip(records, holm_adjust([x["p"] for x in records])):
            r["holm_p"] = p_adj
        return pd.DataFrame(records)

    primary_df = safe_analyze(primary_hypotheses)
    full_hyps = [
        {"a": "rasnet", "b": b, "metric": m, "direction": METRIC_DIRECTIONS[m]}
        for b in ("nnunet", "segresnet", "vnet", "3dunet")
        if b in tables
        for m in ALL_METRICS
    ]
    full_df = safe_analyze(full_hyps)
    return primary_df, full_df


class _SyntheticNNUNetPredictor:
    """Explicit test double for nnUNetPredictor used only when synthetic_dry_run=True."""

    def __init__(self):
        self.use_mirroring = False

    def predict_single_npy_array(
        self,
        input_image,
        image_properties,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    ):
        shape = input_image.shape[1:]
        fg = np.zeros(shape, dtype=np.float32)
        z0, z1 = max(1, shape[0] // 4), max(2, (3 * shape[0]) // 4)
        y0, y1 = max(1, shape[1] // 4), max(2, (3 * shape[1]) // 4)
        x0, x1 = max(1, shape[2] // 4), max(2, (3 * shape[2]) // 4)
        fg[z0:z1, y0:y1, x0:x1] = 0.80 if self.use_mirroring else 0.70
        probs = np.stack([1.0 - fg, fg], axis=0)
        seg = (fg >= 0.5).astype(np.uint8)
        return (seg, probs) if save_or_return_probabilities else seg


def _run_nnunet_batch_probabilities(
    predictor,
    case_ids: list[int],
    pairs: dict[int, tuple[Path, Path]],
    need_tta: bool,
    num_processes: int = 4,
):
    """
    Yield (cid, prob_1pass_zyx, prob_tta_zyx_or_None) for each case_id using native
    nnUNet preprocessing_iterator_fromfiles and convert_predicted_logits_to_segmentation_with_correct_shape.
    """
    if len(case_ids) <= 3 or isinstance(predictor, _SyntheticNNUNetPredictor):
        for cid in case_ids:
            img_p, _ = pairs[cid]
            prob_1 = infer_nnunet_probability(predictor, str(img_p), tta=False)
            prob_tta = infer_nnunet_probability(predictor, str(img_p), tta=True) if need_tta else None
            yield cid, prob_1, prob_tta
        return

    from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
    )

    list_of_lists = [[str(pairs[cid][0])] for cid in case_ids]
    iterator = preprocessing_iterator_fromfiles(
        list_of_lists=list_of_lists,
        list_of_segs_from_prev_stage_files=None,
        output_filenames_truncated=None,
        plans_manager=predictor.plans_manager,
        dataset_json=predictor.dataset_json,
        configuration_manager=predictor.configuration_manager,
        num_processes=max(1, min(num_processes, len(case_ids))),
        pin_memory=True,
        verbose=False,
    )

    for cid, dct in zip(case_ids, iterator):
        data_tensor = dct["data"]
        props = dct["data_properties"]

        predictor.use_mirroring = False
        logits_1 = predictor.predict_logits_from_preprocessed_data(data_tensor).cpu()
        _, probs_1 = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits_1,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.label_manager,
            props,
            return_probabilities=True,
            num_threads_torch=8,
        )
        prob_1pass = np.asarray(probs_1[1], dtype=np.float32)

        prob_tta = None
        if need_tta:
            predictor.use_mirroring = True
            logits_tta = predictor.predict_logits_from_preprocessed_data(data_tensor).cpu()
            _, probs_tta = convert_predicted_logits_to_segmentation_with_correct_shape(
                logits_tta,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.label_manager,
                props,
                return_probabilities=True,
                num_threads_torch=8,
            )
            prob_tta = np.asarray(probs_tta[1], dtype=np.float32)

        yield cid, prob_1pass, prob_tta


def _evaluate_case_all_arms(
    cid: int,
    model: str,
    arms: list[str],
    split: str,
    smoke: bool,
    export_masks: bool,
    run_dir: Path,
    img_p: Path,
    prob_1pass: np.ndarray,
    prob_tta: np.ndarray | None,
    gt_bool: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    threshold_grid: list[float],
    keep_top_k: int,
    thresholds_data: dict | None,
) -> tuple[int, dict[str, dict], dict | None, dict[float, float] | None]:
    """
    Evaluate all requested arms (A, B, C, and diagnostic HIST_RECIPE) for a single case.
    Uses a local thread pool to compute independent binary masks' surface/skeleton metrics concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor

    grid_scores = None
    if split == "val":
        from benchmark.thresholds import _fast_dice

        grid_scores = {}
        for tau in threshold_grid:
            raw_bin = (prob_1pass >= tau).astype(np.uint8)
            cleaned, _ = prune_components(
                raw_bin, spacing_zyx=spacing_zyx, keep_top_k=keep_top_k
            )
            grid_scores[round(float(tau), 4)] = _fast_dice(cleaned, gt_bool)

    use_tta_b = model in ("rasnet", "nnunet")
    need_b_eval = "B" in arms and (use_tta_b or "A" not in arms)
    need_c_eval = "C" in arms and split == "test"
    need_hist_eval = split == "test" and not smoke and model == "rasnet" and prob_tta is not None

    raw_a = (prob_1pass >= 0.50).astype(np.uint8) if ("A" in arms or ("B" in arms and not use_tta_b)) else None
    prob_b = prob_tta if (use_tta_b and prob_tta is not None) else prob_1pass
    raw_b = (prob_b >= 0.50).astype(np.uint8) if need_b_eval else None
    tau_c = float(thresholds_data["thresholds"][model]) if need_c_eval else 0.50
    raw_c = (prob_1pass >= tau_c).astype(np.uint8) if need_c_eval else None
    raw_hist = (prob_tta >= 0.60).astype(np.uint8) if need_hist_eval else None

    # Free float32 probability volumes before running parallel EDT/skeletonize
    del prob_1pass, prob_tta, prob_b

    case_arm_rows: dict[str, dict] = {}
    hist_row: dict | None = None

    with ThreadPoolExecutor(max_workers=4) as arm_pool:
        fut_a = (
            arm_pool.submit(
                _evaluate_binary_mask,
                cid,
                model,
                "A",
                raw_a,
                gt_bool,
                spacing_zyx,
                0.50,
                False,
                None,
            )
            if "A" in arms
            else None
        )
        fut_b = (
            arm_pool.submit(
                _evaluate_binary_mask,
                cid,
                model,
                "B",
                raw_b,
                gt_bool,
                spacing_zyx,
                0.50,
                use_tta_b,
                None,
            )
            if need_b_eval
            else None
        )
        fut_c = (
            arm_pool.submit(
                _evaluate_binary_mask,
                cid,
                model,
                "C",
                raw_c,
                gt_bool,
                spacing_zyx,
                tau_c,
                False,
                keep_top_k,
            )
            if need_c_eval
            else None
        )
        fut_hist = (
            arm_pool.submit(
                _evaluate_binary_mask,
                cid,
                "rasnet",
                "HIST_RECIPE",
                raw_hist,
                gt_bool,
                spacing_zyx,
                0.60,
                True,
                2,
            )
            if need_hist_eval
            else None
        )

        mask_a, row_a = fut_a.result() if fut_a is not None else (None, None)
        if row_a is not None:
            case_arm_rows["A"] = row_a
            if smoke or export_masks:
                out_m = run_dir / "predictions" / "arm_A" / model / f"{cid}.nii.gz"
                export_mask_like(mask_a, str(img_p), str(out_m))

        if "B" in arms:
            if fut_b is not None:
                mask_b, row_b = fut_b.result()
            else:
                mask_b = mask_a
                row_b = {**row_a, "arm": "B", "tta": False}
            case_arm_rows["B"] = row_b
            if smoke or export_masks:
                out_m = run_dir / "predictions" / "arm_B" / model / f"{cid}.nii.gz"
                export_mask_like(mask_b, str(img_p), str(out_m))

        if fut_c is not None:
            mask_c, row_c = fut_c.result()
            case_arm_rows["C"] = row_c
            if smoke or export_masks:
                out_m = run_dir / "predictions" / "arm_C" / model / f"{cid}.nii.gz"
                export_mask_like(mask_c, str(img_p), str(out_m))

        if fut_hist is not None:
            _, hist_row = fut_hist.result()

    return cid, case_arm_rows, hist_row, grid_scores


def execute_benchmark(
    config_path: str | Path,
    split: str,
    arm_arg: str = "all",
    models_arg: str = "all",
    smoke: bool = False,
    run_id: str | None = None,
    device: str | None = None,
    synthetic_dry_run: bool = False,
    export_masks: bool = False,
    resume: bool = False,
) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        cfg_path = repo_root / config_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    split_path = Path(cfg["split_file"])
    if not split_path.is_file():
        split_path = repo_root / cfg["split_file"]
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    validated_parts = validate_splits(splits)
    splits_sha = sha256_file(split_path)

    if split not in ("val", "test"):
        raise ValueError(f"Unsupported split: {split}")

    models = (
        list(ALL_MODELS)
        if models_arg == "all"
        else [m.strip() for m in models_arg.split(",") if m.strip()]
    )
    for m in models:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model: {m}")

    if arm_arg == "all":
        arms = ["A"] if split == "val" else ["A", "B", "C"]
    else:
        arms = [a.strip().upper() for a in arm_arg.split(",") if a.strip()]
        for a in arms:
            if a not in ("A", "B", "C"):
                raise ValueError(f"Unknown arm: {a}")

    all_case_ids = sorted(validated_parts[split])
    case_ids = all_case_ids[:3] if smoke else all_case_ids

    dataset_root = Path(cfg["dataset_root"])
    pairs = resolve_case_pairs(str(dataset_root), case_ids, check_headers=True)
    if smoke:
        validate_case_files(pairs)

    dev = device or ("cuda" if torch.cuda.is_available() and not synthetic_dry_run else "cpu")
    sw_batch_size = int(cfg.get("sw_batch_size", 8 if dev == "cuda" else 1))
    roi_size = (32, 32, 32) if synthetic_dry_run else (96, 96, 96)
    keep_top_k = int(cfg.get("postprocess", {}).get("keep_top_k", 2))
    threshold_grid = [float(x) for x in cfg.get("threshold_grid", list(DEFAULT_THRESHOLD_GRID))]

    # Verify checkpoints and plans hashes up front (fail-closed)
    ckpt_hashes = {}
    ckpt_paths = {}
    nnunet_meta = {}
    if not synthetic_dry_run:
        for m in models:
            rel_ckpt = cfg["checkpoints"][m]
            p_ckpt = Path(rel_ckpt) if Path(rel_ckpt).is_file() else (repo_root / rel_ckpt)
            if not p_ckpt.is_file():
                raise FileNotFoundError(f"Missing best checkpoint for {m}: {p_ckpt}")
            if "best" not in p_ckpt.name.lower():
                raise ValueError(f"Refusing non-best checkpoint for {m}: {p_ckpt}")
            ckpt_paths[m] = str(p_ckpt)
            ckpt_hashes[m] = sha256_file(p_ckpt)
        if "nnunet" in models:
            nn_folder = Path(cfg["nnunet_model_folder"])
            if not nn_folder.is_dir():
                nn_folder = repo_root / cfg["nnunet_model_folder"]
            plans_p = nn_folder / "plans.json"
            dset_p = nn_folder / "dataset.json"
            nnunet_meta = {
                "model_folder": str(nn_folder),
                "fold": int(cfg.get("nnunet_fold", 0)),
                "checkpoint_name": str(cfg.get("nnunet_checkpoint_name", "checkpoint_best.pth")),
                "plans_sha256": sha256_file(plans_p),
                "dataset_sha256": sha256_file(dset_p),
            }
    else:
        for m in models:
            ckpt_paths[m] = f"synthetic_random_{m}"
            ckpt_hashes[m] = hashlib.sha256(f"synthetic_{m}".encode()).hexdigest()

    # If running test split with Arm C, validate thresholds.json BEFORE running inference
    thresholds_path = Path(cfg.get("thresholds_file", "results/thresholds.json"))
    if not thresholds_path.is_absolute():
        thresholds_path = repo_root / thresholds_path
    thresholds_data = None
    thresholds_sha = None
    if split == "test" and "C" in arms:
        thresholds_data, thresholds_sha = load_and_validate_thresholds(
            str(thresholds_path), splits, models
        )

    # Create unique run directory (never overwrite or reuse stale directory unless resume=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_smoke" if smoke else ""
    rid = run_id or f"{ts}_{split}_{''.join(arms)}{suffix}"
    out_root = Path(cfg.get("output_root", "results/benchmark_runs"))
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    run_dir = out_root / rid
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run directory already exists (refusing stale reuse): {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=resume)

    git_meta = git_info(repo_root)
    pkg_meta = package_versions()

    arm_rows: dict[str, list[dict]] = {a: [] for a in arms}
    rasnet_hist_recipe_rows: list[dict] = []
    per_case_grid_dice: dict[str, dict[int, dict[float, float]]] = {m: {} for m in models}
    need_tta_for_model = {
        m: (("B" in arms or split == "test") and m in ("rasnet", "nnunet")) for m in models
    }

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    for model in models:
        t_model = time.time()
        # Check if model already has complete per-model CSV checkpoints when resume=True
        if resume and split == "test":
            existing_arm_dfs = {}
            all_arms_ready = True
            for a in arms:
                p_m_a = run_dir / f"metrics_{model}_arm_{a}.csv"
                if p_m_a.is_file():
                    df_m_a = pd.read_csv(p_m_a)
                    if set(df_m_a["case_id"].astype(int)) == set(case_ids):
                        existing_arm_dfs[a] = df_m_a.to_dict(orient="records")
                    else:
                        all_arms_ready = False
                        break
                else:
                    all_arms_ready = False
                    break
            if all_arms_ready and model == "rasnet" and not smoke:
                p_hist_r = run_dir / "diagnostic_rasnet_historical_recipe_tau060_tta_top2.csv"
                if p_hist_r.is_file():
                    df_hr = pd.read_csv(p_hist_r)
                    if set(df_hr["case_id"].astype(int)) == set(case_ids):
                        rasnet_hist_recipe_rows.extend(df_hr.to_dict(orient="records"))
                    else:
                        all_arms_ready = False
                else:
                    all_arms_ready = False
            if all_arms_ready:
                for a in arms:
                    arm_rows[a].extend(existing_arm_dfs[a])
                print(
                    f"\n[>>>] Resuming: model '{model}' already completed ({len(case_ids)} cases loaded from checkpoint).",
                    flush=True,
                )
                continue

        print(
            f"\n[>>>] Running model '{model}' on split='{split}' ({len(case_ids)} cases, arms={arms})...",
            flush=True,
        )
        model_arm_rows: dict[str, list[dict]] = {a: [] for a in arms}
        model_hist_rows: list[dict] = []

        def _consume_eval_future(fut_eval):
            cid_done, c_arm_rows, h_row, g_scores = fut_eval.result()
            if g_scores is not None:
                per_case_grid_dice[model][cid_done] = g_scores
            for a_k, r_dict in c_arm_rows.items():
                model_arm_rows[a_k].append(r_dict)
            if h_row is not None:
                model_hist_rows.append(h_row)

        if model != "nnunet":
            net = (
                build_random_model(model)
                if synthetic_dry_run
                else build_model(model, ckpt_paths[model], dev)
            )
            tf_model = build_transform(model)

            def _load_and_prep(cid_item: int):
                im_p, lb_p = pairs[cid_item]
                lb_im = sitk.ReadImage(str(lb_p))
                gt_b = sitk.GetArrayFromImage(lb_im) > 0
                sp_zyx = tuple(float(s) for s in reversed(lb_im.GetSpacing()))
                b_dict = tf_model({"image": str(im_p)})
                return cid_item, im_p, gt_b, sp_zyx, b_dict

            with ThreadPoolExecutor(max_workers=2) as prep_pool, ThreadPoolExecutor(
                max_workers=2
            ) as eval_pool:
                q_prep = deque()
                q_eval = deque()
                next_submit = 0
                while next_submit < min(3, len(case_ids)):
                    q_prep.append(prep_pool.submit(_load_and_prep, case_ids[next_submit]))
                    next_submit += 1

                for idx in range(len(case_ids)):
                    cid, img_p, gt_bool, spacing_zyx, batch_prepped = q_prep.popleft().result()
                    if next_submit < len(case_ids):
                        q_prep.append(prep_pool.submit(_load_and_prep, case_ids[next_submit]))
                        next_submit += 1

                    prob_1pass, prob_tta = _infer_monai_both_passes(
                        name=model,
                        net=net,
                        image_path=str(img_p),
                        device=dev,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        need_tta=need_tta_for_model[model],
                        transform=tf_model,
                        batch=batch_prepped,
                    )
                    del batch_prepped
                    if prob_1pass.shape != gt_bool.shape:
                        raise RuntimeError(
                            f"Shape mismatch for case {cid} ({model}): {prob_1pass.shape} vs {gt_bool.shape}"
                        )

                    # Keep at most 2 cases evaluating in background so RAM stays bounded
                    while len(q_eval) >= 2:
                        _consume_eval_future(q_eval.popleft())

                    q_eval.append(
                        eval_pool.submit(
                            _evaluate_case_all_arms,
                            cid,
                            model,
                            arms,
                            split,
                            smoke,
                            export_masks,
                            run_dir,
                            img_p,
                            prob_1pass,
                            prob_tta,
                            gt_bool,
                            spacing_zyx,
                            threshold_grid,
                            keep_top_k,
                            thresholds_data,
                        )
                    )
                    del prob_1pass, prob_tta

                    if (idx + 1) % 25 == 0 or idx == len(case_ids) - 1:
                        print(
                            f"  [{model}] {idx + 1}/{len(case_ids)} GPU passes done ({time.time() - t_model:.1f}s)",
                            flush=True,
                        )

                while q_eval:
                    _consume_eval_future(q_eval.popleft())

            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            # nnU-Net V2 native predictor
            predictor = (
                _SyntheticNNUNetPredictor()
                if synthetic_dry_run
                else build_nnunet_predictor(
                    model_folder=nnunet_meta["model_folder"],
                    fold=nnunet_meta["fold"],
                    checkpoint_name=nnunet_meta["checkpoint_name"],
                    device=dev,
                    use_mirroring=False,
                )
            )
            with ThreadPoolExecutor(max_workers=2) as eval_pool:
                q_eval = deque()
                for idx, (cid, prob_1pass, prob_tta) in enumerate(
                    _run_nnunet_batch_probabilities(
                        predictor=predictor,
                        case_ids=case_ids,
                        pairs=pairs,
                        need_tta=need_tta_for_model["nnunet"],
                        num_processes=4,
                    )
                ):
                    img_p, lbl_p = pairs[cid]
                    lbl_img = sitk.ReadImage(str(lbl_p))
                    gt_bool = sitk.GetArrayFromImage(lbl_img) > 0
                    spacing_zyx = tuple(float(s) for s in reversed(lbl_img.GetSpacing()))

                    if prob_1pass.shape != gt_bool.shape:
                        raise RuntimeError(
                            f"Shape mismatch for case {cid} (nnunet): {prob_1pass.shape} vs {gt_bool.shape}"
                        )

                    while len(q_eval) >= 2:
                        _consume_eval_future(q_eval.popleft())

                    q_eval.append(
                        eval_pool.submit(
                            _evaluate_case_all_arms,
                            cid,
                            "nnunet",
                            arms,
                            split,
                            smoke,
                            export_masks,
                            run_dir,
                            img_p,
                            prob_1pass,
                            prob_tta,
                            gt_bool,
                            spacing_zyx,
                            threshold_grid,
                            keep_top_k,
                            thresholds_data,
                        )
                    )
                    del prob_1pass, prob_tta

                    if (idx + 1) % 25 == 0 or idx == len(case_ids) - 1:
                        print(
                            f"  [nnunet] {idx + 1}/{len(case_ids)} GPU passes done ({time.time() - t_model:.1f}s)",
                            flush=True,
                        )

                while q_eval:
                    _consume_eval_future(q_eval.popleft())

            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Flush per-model CSV checkpoints immediately after model completes
        for a in arms:
            m_rows = model_arm_rows[a]
            arm_rows[a].extend(m_rows)
            df_m_sub = pd.DataFrame(m_rows).sort_values("case_id").reset_index(drop=True)
            df_m_sub.to_csv(run_dir / f"metrics_{model}_arm_{a}.csv", index=False)
        if model_hist_rows:
            rasnet_hist_recipe_rows.extend(model_hist_rows)
            df_hr_sub = pd.DataFrame(model_hist_rows).sort_values("case_id").reset_index(drop=True)
            df_hr_sub.to_csv(
                run_dir / "diagnostic_rasnet_historical_recipe_tau060_tta_top2.csv", index=False
            )
        print(
            f"  [OK] Checkpointed model '{model}' CSVs in {time.time() - t_model:.1f}s.",
            flush=True,
        )

    # If full validation run (not smoke), freeze thresholds.json
    if split == "val" and not smoke:
        from benchmark.thresholds import select_validation_thresholds_from_scores

        provenance = {
            "run_id": rid,
            "timestamp_utc": ts,
            "git_commit": git_meta["commit"],
            "splits_sha256": splits_sha,
            "checkpoints_sha256": ckpt_hashes,
            "metric_version": METRIC_VERSION,
        }
        thresholds_data = select_validation_thresholds_from_scores(
            per_case_grid_dice=per_case_grid_dice,
            splits=splits,
            grid=threshold_grid,
            keep_top_k=keep_top_k,
            provenance=provenance,
        )
        run_thresh_path = run_dir / "thresholds.json"
        thresholds_sha = save_thresholds_json(thresholds_data, str(run_thresh_path))
        # Also write to configured canonical thresholds_file if all 5 models were evaluated
        if set(models) == set(ALL_MODELS):
            save_thresholds_json(thresholds_data, str(thresholds_path))

        # Save detailed grid score table
        grid_rows = []
        for m, gmap in thresholds_data["grid_scores"].items():
            for tau_s, score in gmap.items():
                grid_rows.append(
                    {
                        "model": m,
                        "threshold": float(tau_s),
                        "mean_val_dice": float(score),
                        "selected": float(tau_s) == float(thresholds_data["thresholds"][m]),
                    }
                )
        pd.DataFrame(grid_rows).to_csv(run_dir / "val_threshold_grid_scores.csv", index=False)

    # Save per-arm outputs, summaries, statistics, and historical comparisons
    primary_hyps = cfg.get("hypotheses", [])
    hist_tolerance = float(cfg.get("historical_tolerance", 1e-4))
    hist_diff_summary = {}

    for arm in arms:
        rows = arm_rows.get(arm, [])
        if not rows:
            continue
        df_arm = pd.DataFrame(rows).sort_values(["model", "case_id"]).reset_index(drop=True)
        # Verify completeness: every requested model and case_id must be present
        for m in models:
            sub_m = df_arm[df_arm["model"] == m]
            if set(sub_m["case_id"]) != set(case_ids):
                raise RuntimeError(f"Incomplete output for model={m}, arm={arm}")
            sub_m.to_csv(run_dir / f"metrics_{m}_arm_{arm}.csv", index=False)

        df_arm.to_csv(run_dir / f"per_case_arm_{arm}.csv", index=False)
        sum_df = summarize_arm(df_arm, seed=int(cfg.get("seed", 42)))
        sum_df.to_csv(run_dir / f"summary_arm_{arm}.csv", index=False)

        if not smoke and len(models) >= 2:
            prim_stats, full_stats = compute_arm_statistics(df_arm, primary_hyps)
            if not prim_stats.empty:
                prim_stats.to_csv(run_dir / f"paired_stats_primary_arm_{arm}.csv", index=False)
            if not full_stats.empty:
                full_stats.to_csv(run_dir / f"paired_stats_full32_arm_{arm}.csv", index=False)

        if split == "test" and not smoke and "historical_csvs" in cfg:
            hist_diff_summary[arm] = {}
            for m in models:
                rel_h = cfg["historical_csvs"].get(m)
                if not rel_h:
                    continue
                p_h = Path(rel_h) if Path(rel_h).is_file() else (repo_root / rel_h)
                if p_h.is_file():
                    hist_df = pd.read_csv(p_h)
                    new_sub = df_arm[df_arm["model"] == m]
                    diff_df, d_sum = compare_with_historical(
                        new_sub, hist_df, metrics=list(ALL_METRICS), tolerance=hist_tolerance
                    )
                    diff_df.to_csv(run_dir / f"diff_vs_historical_{m}_arm_{arm}.csv", index=False)
                    hist_diff_summary[arm][m] = d_sum

    if rasnet_hist_recipe_rows:
        df_hist_recipe = pd.DataFrame(rasnet_hist_recipe_rows).sort_values("case_id").reset_index(drop=True)
        df_hist_recipe.to_csv(
            run_dir / "diagnostic_rasnet_historical_recipe_tau060_tta_top2.csv", index=False
        )
        sum_hist_recipe = summarize_arm(df_hist_recipe, seed=int(cfg.get("seed", 42)))
        sum_hist_recipe.to_csv(
            run_dir / "summary_diagnostic_rasnet_historical_recipe_tau060_tta_top2.csv", index=False
        )
        rel_h = cfg.get("historical_csvs", {}).get("rasnet")
        if rel_h:
            p_h = Path(rel_h) if Path(rel_h).is_file() else (repo_root / rel_h)
            if p_h.is_file():
                hist_df = pd.read_csv(p_h)
                diff_df, d_sum = compare_with_historical(
                    df_hist_recipe, hist_df, metrics=list(ALL_METRICS), tolerance=hist_tolerance
                )
                diff_df.to_csv(
                    run_dir / "diff_vs_historical_rasnet_recipe_tau060_tta_top2.csv", index=False
                )
                hist_diff_summary["diagnostic_rasnet_historical_recipe_tau060_tta_top2"] = {
                    "rasnet": d_sum
                }

    if hist_diff_summary:
        (run_dir / "diff_vs_historical_summary.json").write_text(
            json.dumps(hist_diff_summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    manifest = {
        "run_id": rid,
        "timestamp_utc": ts,
        "smoke": bool(smoke),
        "synthetic_dry_run": bool(synthetic_dry_run),
        "retrospective_evaluation": True,
        "retrospective_notice": (
            "ImageCAS 851-1000 and unseen-66 were previously inspected during historical "
            "threshold/post-processing sweeps; this evaluation is retrospective."
        ),
        "split": split,
        "arms": arms,
        "models": models,
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "splits_file": str(split_path),
        "splits_sha256": splits_sha,
        "git": git_meta,
        "seed": int(cfg.get("seed", 42)),
        "device": dev,
        "sw_batch_size": sw_batch_size,
        "roi_size": list(roi_size),
        "metric_version": METRIC_VERSION,
        "package_versions": pkg_meta,
        "checkpoints": {
            m: {
                "path": ckpt_paths[m],
                "sha256": ckpt_hashes[m],
                "transform_recipe": list(transform_recipe(m)) if m != "nnunet" else ["nnUNetPredictor_3d_fullres"],
                "constructor_args": model_constructor_args(m) if m != "nnunet" else nnunet_meta,
            }
            for m in models
        },
        "nnunet_metadata": nnunet_meta,
        "thresholds_file": str(thresholds_path) if thresholds_sha else None,
        "thresholds_sha256": thresholds_sha,
        "thresholds_used": thresholds_data["thresholds"] if thresholds_data else None,
        "postprocessing": {
            "arm_A": {"threshold": 0.50, "tta": False, "keep_top_k": None},
            "arm_B": {
                "threshold": 0.50,
                "tta_policy": {
                    "rasnet": "4-pass flip TTA on spatial axes (2,3,4)",
                    "nnunet": "native nnUNetPredictor 3D mirroring on axes (0,1,2)",
                    "segresnet": "none (1-pass; no documented standard TTA)",
                    "vnet": "none (1-pass; no documented standard TTA)",
                    "3dunet": "none (1-pass; no documented standard TTA)",
                },
                "keep_top_k": None,
            },
            "arm_C": {
                "threshold_source": "validation split grid search (0.30..0.70 step 0.05)",
                "tta": False,
                "keep_top_k": keep_top_k,
                "connectivity": 26,
            },
        },
        "historical_tolerance": hist_tolerance,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[OK] Benchmark run complete -> {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="5-Model ImageCAS Benchmark Runner")
    parser.add_argument("--config", type=str, default="config.lab.yaml")
    parser.add_argument("--split", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--arm", type=str, default="all")
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--export-masks", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    execute_benchmark(
        config_path=args.config,
        split=args.split,
        arm_arg=args.arm,
        models_arg=args.models,
        smoke=args.smoke,
        run_id=args.run_id,
        device=args.device,
        synthetic_dry_run=args.synthetic_dry_run,
        export_masks=args.export_masks,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
