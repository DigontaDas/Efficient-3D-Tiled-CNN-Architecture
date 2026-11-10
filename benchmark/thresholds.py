"""Validation-only probability threshold selection and freezing."""
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark.postprocess import prune_components
from benchmark.preflight import validate_splits, validate_thresholds

DEFAULT_THRESHOLD_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def _fast_dice(pred_u8: np.ndarray, gt_bool: np.ndarray) -> float:
    p = pred_u8.astype(bool, copy=False)
    ps = int(p.sum())
    gs = int(gt_bool.sum())
    if ps + gs == 0:
        return 1.0
    inter = int((p & gt_bool).sum())
    return float(2.0 * inter / (ps + gs))


def select_validation_thresholds_from_scores(
    per_case_grid_dice: dict[str, dict[int, dict[float, float]]],
    splits: dict,
    grid: tuple[float, ...] | list[float] = DEFAULT_THRESHOLD_GRID,
    keep_top_k: int | None = 2,
    provenance: dict | None = None,
) -> dict:
    """
    Select per-model probability thresholds from pre-computed per-case validation Dice scores
    across the threshold grid. Identical selection and tie-breaking rule as select_validation_thresholds.
    """
    parts = validate_splits(splits)
    val_ids = sorted(parts["val"])
    grid_vals = [round(float(t), 4) for t in grid]
    if not grid_vals or any(t <= 0.0 or t >= 1.0 for t in grid_vals):
        raise ValueError("Invalid threshold grid")

    selected = {}
    grid_scores = {}
    for model, case_scores in per_case_grid_dice.items():
        if set(case_scores.keys()) != parts["val"]:
            raise ValueError(f"Model {model} validation scores do not match exact validation split IDs")
        scores_for_model = {}
        best_tau = None
        best_score = -1.0
        for tau in grid_vals:
            dices = [float(case_scores[cid][tau]) for cid in val_ids]
            mean_dice = float(np.mean(dices))
            scores_for_model[f"{tau:.2f}"] = mean_dice
            if best_tau is None or (mean_dice > best_score + 1e-12):
                best_score = mean_dice
                best_tau = tau
            elif abs(mean_dice - best_score) <= 1e-12:
                if (abs(tau - 0.50), tau) < (abs(best_tau - 0.50), best_tau):
                    best_tau = tau
                    best_score = mean_dice
        selected[model] = float(best_tau)
        grid_scores[model] = scores_for_model

    payload = {
        "split": "val",
        "case_ids": val_ids,
        "grid": grid_vals,
        "objective": "mean_val_dice",
        "tie_breaker": "closest_to_0.50_then_lower",
        "postprocessing": {"keep_top_k": keep_top_k, "connectivity": 26},
        "thresholds": selected,
        "grid_scores": grid_scores,
        "provenance": provenance or {},
    }
    validate_thresholds(payload, splits, list(per_case_grid_dice.keys()))
    return payload


def select_validation_thresholds(
    val_probs: dict[str, dict[int, np.ndarray]],
    val_gts: dict[int, tuple[np.ndarray, tuple[float, float, float]]],
    splits: dict,
    grid: tuple[float, ...] | list[float] = DEFAULT_THRESHOLD_GRID,
    keep_top_k: int | None = 2,
    provenance: dict | None = None,
) -> dict:
    """
    Select per-model probability thresholds on the full validation split only.
    Objective: maximize mean validation Dice after Arm C post-processing (keep_top_k).
    Tie-breaker: closest to 0.50, then lower threshold.
    """
    parts = validate_splits(splits)
    if set(val_gts.keys()) != parts["val"]:
        raise ValueError("val_gts must match the exact validation split case IDs")
    grid_vals = [round(float(t), 4) for t in grid]
    per_case_grid_dice: dict[str, dict[int, dict[float, float]]] = {}
    for model, case_map in val_probs.items():
        if set(case_map.keys()) != parts["val"]:
            raise ValueError(f"Model {model} validation probabilities do not match exact validation split IDs")
        per_case_grid_dice[model] = {}
        for cid, prob in case_map.items():
            gt_mask, spacing_zyx = val_gts[cid]
            per_case_grid_dice[model][cid] = {}
            for tau in grid_vals:
                raw_bin = (prob >= tau).astype(np.uint8)
                cleaned, _ = prune_components(raw_bin, spacing_zyx=spacing_zyx, keep_top_k=keep_top_k)
                per_case_grid_dice[model][cid][tau] = _fast_dice(cleaned, gt_mask)
    return select_validation_thresholds_from_scores(
        per_case_grid_dice=per_case_grid_dice,
        splits=splits,
        grid=grid_vals,
        keep_top_k=keep_top_k,
        provenance=provenance,
    )


def save_thresholds_json(payload: dict, output_path: str) -> str:
    """Write thresholds.json deterministically and return its SHA-256 hex digest."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    p.write_text(text, encoding="utf-8")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_and_validate_thresholds(path: str, splits: dict, models: list[str]) -> tuple[dict, str]:
    """Load thresholds.json, validate against splits and models, and return (data, sha256)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing thresholds file: {p}")
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))
    validate_thresholds(data, splits, models)
    return data, sha

