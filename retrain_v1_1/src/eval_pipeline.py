"""Symmetric Val-94 Lock, One-Shot Retrospective Test-150 Eval, 10k-Bootstrap Stats & Figures for Strategy A.

Reuses verified `surface-v1` metrics (`benchmark/metrics.py`), 26-connected Top-2 CC (`benchmark/postprocess.py`),
Wilcoxon/Holm/bootstrap statistics (`benchmark/stats.py`), crash-safe CSV helpers (`benchmark/symmetric_eval.py`),
and native nnU-Net V2 predictor (`benchmark/adapters/nnunet_adapter.py`).
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Invertd

from benchmark import symmetric_eval as se
from benchmark.adapters.nnunet_adapter import build_nnunet_predictor, infer_nnunet_probability
from benchmark.dataset import resolve_case_pairs
from benchmark.metrics import compute_metrics
from benchmark.postprocess import prune_components
from benchmark.preflight import validate_splits
from benchmark.stats import holm_adjust, paired_analysis
from retrain_v1_1.src.models import build_strategy_a_model, count_parameters
from retrain_v1_1.src.transforms import build_deterministic_preprocess

PROTOCOL_VERSION_V11 = "strategy-a-retrain-v1.1"
MODELS_V11 = (
    "rasnet_v1_1",
    "segresnet_v1_1",
    "vnet_v1_1",
    "ablation_no_attn_v1_1",
    "ablation_no_ds_v1_1",
    "nnunet_v2",
)
MODES_V11 = ("sw050_gauss_1pass", "sw050_gauss_tta8")
TAU_GRID_V11 = tuple(round(0.10 + 0.05 * i, 2) for i in range(13))  # 0.10 .. 0.70 (13 levels)
GRID_ROWS_PER_CASE_V11 = len(MODES_V11) * len(TAU_GRID_V11)
PRIMARY_RECIPE_V11 = "R4_tuned_tta8_top2"
RECIPES_V11 = (
    "R1_raw_1pass",
    "R2_raw_tta8",
    "R3_tuned_1pass_top2",
    "R4_tuned_tta8_top2",
)

FAMILIES_V11 = {
    "FA_primary_rasnet_vs_nnunet": [
        ("rasnet_v1_1", "nnunet_v2", "precision", "higher"),
        ("rasnet_v1_1", "nnunet_v2", "hd95", "lower"),
        ("rasnet_v1_1", "nnunet_v2", "asd", "lower"),
    ],
    "FB_rasnet_vs_segresnet": [
        ("rasnet_v1_1", "segresnet_v1_1", "dice", "higher"),
        ("rasnet_v1_1", "segresnet_v1_1", "precision", "higher"),
        ("rasnet_v1_1", "segresnet_v1_1", "hd95", "lower"),
        ("rasnet_v1_1", "segresnet_v1_1", "cldice", "higher"),
    ],
    "FC_rasnet_vs_vnet": [
        ("rasnet_v1_1", "vnet_v1_1", "dice", "higher"),
        ("rasnet_v1_1", "vnet_v1_1", "precision", "higher"),
        ("rasnet_v1_1", "vnet_v1_1", "hd95", "lower"),
        ("rasnet_v1_1", "vnet_v1_1", "cldice", "higher"),
    ],
    "FD_secondary_rasnet_vs_nnunet": [
        ("rasnet_v1_1", "nnunet_v2", "dice", "higher"),
        ("rasnet_v1_1", "nnunet_v2", "cldice", "higher"),
        ("rasnet_v1_1", "nnunet_v2", "recall", "higher"),
    ],
    "FE_ablations_vs_rasnet": [
        ("rasnet_v1_1", "ablation_no_attn_v1_1", "dice", "higher"),
        ("rasnet_v1_1", "ablation_no_attn_v1_1", "precision", "higher"),
        ("rasnet_v1_1", "ablation_no_attn_v1_1", "hd95", "lower"),
        ("rasnet_v1_1", "ablation_no_attn_v1_1", "cldice", "higher"),
        ("rasnet_v1_1", "ablation_no_ds_v1_1", "dice", "higher"),
        ("rasnet_v1_1", "ablation_no_ds_v1_1", "precision", "higher"),
        ("rasnet_v1_1", "ablation_no_ds_v1_1", "hd95", "lower"),
        ("rasnet_v1_1", "ablation_no_ds_v1_1", "cldice", "higher"),
    ],
}


def load_gt_zyx(label_path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(str(label_path))
    arr = (sitk.GetArrayFromImage(img) > 0).astype(bool)
    spacing_zyx = tuple(float(s) for s in reversed(img.GetSpacing()))
    return arr, spacing_zyx


def _invert_prob_to_zyx(prob_tensor_1xyz: torch.Tensor, batch: dict, transform) -> np.ndarray:
    d = dict(batch)
    d["pred"] = prob_tensor_1xyz.detach().cpu()
    inv = Invertd(
        keys=["pred"],
        transform=transform,
        orig_keys=["image"],
        nearest_interp=False,
        to_tensor=True,
    )(d)["pred"]
    arr = np.transpose(inv.squeeze(0).detach().cpu().numpy(), (2, 1, 0)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("Non-finite probability array after Invertd")
    return np.clip(arr, 0.0, 1.0)


def infer_monai_v11_1pass_and_tta8(
    model: torch.nn.Module,
    image_path: str | Path,
    transform,
    roi_size_xyz: tuple[int, int, int],
    sw_overlap: float,
    sw_mode: str,
    sw_sigma_scale: float,
    sw_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric 1-pass and 8-pass logit-averaged TTA returning native SimpleITK (Z, Y, X) probs."""
    model.eval()
    batch = transform({"image": str(image_path)})
    x = batch["image"].unsqueeze(0)  # (1, 1, X, Y, Z) on CPU; SW moves tiles to GPU
    amp_ok = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_ok and torch.cuda.is_bf16_supported()) else torch.float16
    cpu_dev = torch.device("cpu")

    def _run_logits(inp: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_ok):
            out = sliding_window_inference(
                inp,
                roi_size=roi_size_xyz,
                sw_batch_size=int(sw_batch_size),
                predictor=model,
                overlap=float(sw_overlap),
                mode=str(sw_mode),
                sigma_scale=float(sw_sigma_scale),
                sw_device=device,
                device=cpu_dev,
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
        return out.float()

    # All 8 mirroring combinations across spatial dims (2, 3, 4) = (X, Y, Z), matching nnU-Net
    combos = [()] + [c for r in (1, 2, 3) for c in itertools.combinations((2, 3, 4), r)]
    with torch.no_grad():
        logits_1 = _run_logits(x)
        prob_1 = torch.softmax(logits_1, dim=1)[0, 1:2]  # (1, X, Y, Z)

        logits_acc = logits_1.clone()
        for axes in combos[1:]:
            flipped = torch.flip(x, dims=axes)
            l_f = _run_logits(flipped)
            logits_acc += torch.flip(l_f, dims=axes)
        logits_tta8 = logits_acc / float(len(combos))
        prob_tta8 = torch.softmax(logits_tta8, dim=1)[0, 1:2]

    return (
        _invert_prob_to_zyx(prob_1, batch, transform),
        _invert_prob_to_zyx(prob_tta8, batch, transform),
    )


def score_val_grid_v11(
    prob_zyx: np.ndarray,
    gt_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    model: str,
    case_id: int,
    mode: str,
    tau_grid: tuple[float, ...] = TAU_GRID_V11,
    keep_top_k: int = 2,
    connectivity: int = 26,
) -> list[dict[str, Any]]:
    """Score one case across the widened [0.10..0.70] grid after symmetric Top-2 CC."""
    rows = []
    for tau in tau_grid:
        raw = (prob_zyx >= float(tau)).astype(np.uint8)
        cleaned, cc = prune_components(raw, spacing_zyx=spacing_zyx, keep_top_k=keep_top_k, connectivity=connectivity)
        inter = int((cleaned.astype(bool) & gt_zyx).sum())
        ps = int(cleaned.sum())
        gs = int(gt_zyx.sum())
        dice = float(2.0 * inter / (ps + gs)) if (ps + gs) > 0 else float("nan")
        rows.append(
            {
                "model": model,
                "case_id": int(case_id),
                "mode": mode,
                "tau": round(float(tau), 4),
                "dice": dice,
                "components_total": int(cc["components_total"]),
                "voxels_retained": int(cc["voxels_retained"]),
            }
        )
    return rows


def build_strategy_a_lock(
    val_grid_df: pd.DataFrame,
    splits: dict[str, list[int]],
    models: list[str] | tuple[str, ...] = MODELS_V11,
    tau_grid: tuple[float, ...] = TAU_GRID_V11,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Select per-model x per-mode tau* strictly on Val-94 (widened [0.10..0.70] grid)."""
    parts = validate_splits(splits)
    val_ids = sorted(parts["val"])
    grid_vals = [round(float(t), 4) for t in tau_grid]
    thresholds: dict[str, dict[str, float]] = {}
    curves: dict[str, dict[str, dict[str, float]]] = {}

    df = val_grid_df.copy()
    df["tau"] = pd.to_numeric(df["tau"]).round(4)
    df = df.drop_duplicates(subset=["model", "case_id", "mode", "tau"], keep="last")

    for m in models:
        thresholds[m] = {}
        curves[m] = {}
        for mode in MODES_V11:
            sub = df[(df["model"] == m) & (df["mode"] == mode)]
            if not allow_partial and set(int(c) for c in sub["case_id"].unique()) != parts["val"]:
                raise ValueError(f"Val-94 grid incomplete for ({m}, {mode})")
            best_tau = None
            best_score = -1.0
            curve: dict[str, float] = {}
            for tau in grid_vals:
                t_sub = sub[sub["tau"] == tau]
                if not allow_partial and len(t_sub) != len(val_ids):
                    raise ValueError(f"Expected {len(val_ids)} val rows for ({m}, {mode}, {tau}), got {len(t_sub)}")
                mean_d = float(np.nanmean(t_sub["dice"].to_numpy(float)))
                curve[f"{tau:.2f}"] = mean_d
                if best_tau is None or mean_d > best_score + 1e-12:
                    best_score = mean_d
                    best_tau = tau
                elif abs(mean_d - best_score) <= 1e-12:
                    if (abs(tau - 0.50), tau) < (abs(best_tau - 0.50), best_tau):
                        best_tau = tau
                        best_score = mean_d
            thresholds[m][mode] = float(best_tau)
            curves[m][mode] = curve

    return {
        "protocol_version": PROTOCOL_VERSION_V11,
        "split": "val",
        "case_ids": val_ids if not allow_partial else sorted(int(c) for c in df["case_id"].unique()),
        "models": list(models),
        "modes": list(MODES_V11),
        "grid": grid_vals,
        "objective": "mean_val_dice_after_top2_cc",
        "tie_breaker": "closest_to_0.50_then_lower",
        "postprocessing": {"keep_top_k": 2, "connectivity": 26},
        "thresholds": thresholds,
        "grid_scores": curves,
    }


def write_strategy_a_lock(path: Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Lock already exists and is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    sha = se.sha256_file(path)
    path.with_suffix(".sha256").write_text(sha + "\n", encoding="utf-8", newline="\n")
    return sha


def load_strategy_a_lock(
    path: Path,
    splits: dict[str, list[int]],
    models: list[str] | tuple[str, ...] = MODELS_V11,
    tau_grid: tuple[float, ...] = TAU_GRID_V11,
    allow_smoke: bool = False,
) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Strategy A lock: {path}")
    side = path.with_suffix(".sha256")
    if not side.is_file():
        raise ValueError(f"Lock sidecar {side.name} is missing")
    sha = se.sha256_file(path)
    if side.read_text(encoding="utf-8").strip() != sha:
        raise ValueError("Strategy A lock file SHA-256 does not match its .sha256 sidecar")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("split") != "val":
        raise ValueError("Lock must be selected on validation split")
    parts = validate_splits(splits)
    if not allow_smoke and set(int(c) for c in data.get("case_ids", [])) != parts["val"]:
        raise ValueError("Lock case_ids do not match canonical Val-94 split")
    if [round(float(t), 4) for t in data.get("grid", [])] != [round(float(t), 4) for t in tau_grid]:
        raise ValueError("Lock threshold grid does not match Strategy A widened [0.10..0.70] grid")
    for m in models:
        if m not in data.get("thresholds", {}):
            raise ValueError(f"Model {m} missing from lock")
    return data, sha


def evaluate_four_recipes_case(
    model: str,
    case_id: int,
    prob_1pass: np.ndarray,
    prob_tta8: np.ndarray,
    gt_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    lock_thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Compute surface-v1 metrics for R1..R4 on one Test-150 scan."""
    specs = [
        ("R1_raw_1pass", prob_1pass, 0.50, None),
        ("R2_raw_tta8", prob_tta8, 0.50, None),
        ("R3_tuned_1pass_top2", prob_1pass, float(lock_thresholds["sw050_gauss_1pass"]), 2),
        ("R4_tuned_tta8_top2", prob_tta8, float(lock_thresholds["sw050_gauss_tta8"]), 2),
    ]
    rows = []
    for recipe, prob, tau, top_k in specs:
        raw = (prob >= tau).astype(np.uint8)
        mask, cc = prune_components(raw, spacing_zyx=spacing_zyx, keep_top_k=top_k, connectivity=26)
        m = compute_metrics(mask.astype(bool), gt_zyx, spacing_zyx)
        rows.append(
            {
                "model": model,
                "case_id": int(case_id),
                "recipe": recipe,
                "tau": round(float(tau), 4),
                **m,
                **cc,
            }
        )
    return rows


def _analyze_family_v11(
    tables: dict[str, pd.DataFrame],
    hypotheses: list[tuple[str, str, str, str]],
    n_boot: int = 10000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run paired Wilcoxon + 10k-resample bootstrap 95% CI + Holm-Bonferroni within family."""
    results: list[dict[str, Any]] = []
    for a_name, b_name, metric, direction in hypotheses:
        ta, tb = tables[a_name], tables[b_name]
        merged = ta[["case_id", metric]].merge(
            tb[["case_id", metric]], on="case_id", validate="one_to_one", suffixes=("_a", "_b")
        )
        finite_ids = set(merged.dropna()["case_id"])
        ta_f = ta[ta["case_id"].isin(finite_ids)]
        tb_f = tb[tb["case_id"].isin(finite_ids)]
        n_excl = int(len(ta) - len(ta_f))
        if len(ta_f) == 0:
            r: dict[str, Any] = {
                "n": 0, "mean_difference": np.nan, "diff_ci95_lo": np.nan, "diff_ci95_hi": np.nan,
                "median_difference": np.nan, "q25_difference": np.nan, "q75_difference": np.nan,
                "iqr_difference": np.nan, "p": 1.0, "effect_size_rank_biserial": np.nan,
                "wins": 0, "ties": 0, "losses": 0, "a_mean": np.nan, "b_mean": np.nan,
                "n_excluded_nan": n_excl,
            }
        else:
            r = paired_analysis(ta_f, tb_f, metric, seed=int(seed), n_boot=int(n_boot))
            if direction == "lower":
                r["wins"], r["losses"] = r["losses"], r["wins"]
            r["a_mean"] = float(ta_f[metric].mean())
            r["b_mean"] = float(tb_f[metric].mean())
            r["n_excluded_nan"] = n_excl
        r.update({"a": a_name, "b": b_name, "metric": metric, "direction": direction})
        results.append(r)
    for r, hp in zip(results, holm_adjust([r["p"] for r in results])):
        r["holm_p"] = float(hp)
    return results


def paired_noninferiority_v11(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str = "dice",
    margin: float = -0.005,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    merged = df_a[["case_id", metric]].merge(
        df_b[["case_id", metric]], on="case_id", validate="one_to_one", suffixes=("_a", "_b")
    ).dropna()
    finite_ids = set(merged["case_id"])
    pa = paired_analysis(
        df_a[df_a["case_id"].isin(finite_ids)],
        df_b[df_b["case_id"].isin(finite_ids)],
        metric,
        seed=int(seed),
        n_boot=int(n_boot),
    )
    return {
        "n": int(pa["n"]),
        "mean_diff": float(pa["mean_difference"]),
        "median_diff": float(pa["median_difference"]),
        "ci95_lo": float(pa["diff_ci95_lo"]),
        "ci95_hi": float(pa["diff_ci95_hi"]),
        "margin": float(margin),
        "n_boot": int(n_boot),
        "p_wilcoxon": float(pa["p"]),
        "wins": int(pa["wins"]),
        "ties": int(pa["ties"]),
        "losses": int(pa["losses"]),
        "non_inferior": bool(pa["diff_ci95_lo"] > float(margin)),
        "superior": bool(pa["diff_ci95_lo"] > 0.0),
    }


def compute_strategy_a_stats(
    per_case_df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
    ni_margin: float = -0.005,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-model summary (with n_empty_pred & strict precision), 5 Holm families, and Dice NI."""
    df = per_case_df.drop_duplicates(subset=["model", "case_id", "recipe"], keep="last").copy()
    summ = se.summarize(df)

    fam_rows = []
    ni_rows = []
    for recipe in sorted(df["recipe"].unique()):
        role = "primary" if recipe == PRIMARY_RECIPE_V11 else "secondary"
        sub = df[df["recipe"] == recipe]
        tables = {m: g for m, g in sub.groupby("model")}
        for fam_name, hyps in FAMILIES_V11.items():
            valid_hyps = [h for h in hyps if h[0] in tables and h[1] in tables]
            if not valid_hyps:
                continue
            for r in _analyze_family_v11(tables, valid_hyps, n_boot=int(n_boot), seed=int(seed)):
                fam_rows.append({"family": fam_name, "recipe": recipe, "role": role, **r})
        if "rasnet_v1_1" in tables and "nnunet_v2" in tables:
            ni_rows.append(
                {
                    "recipe": recipe,
                    "role": role,
                    **paired_noninferiority_v11(
                        tables["rasnet_v1_1"],
                        tables["nnunet_v2"],
                        metric="dice",
                        margin=float(ni_margin),
                        n_boot=int(n_boot),
                        seed=int(seed),
                    ),
                }
            )
    return summ, pd.DataFrame(fam_rows), pd.DataFrame(ni_rows)


def write_strategy_a_report(
    out_dir: Path,
    summ: pd.DataFrame,
    fam: pd.DataFrame,
    ni: pd.DataFrame,
    timing: pd.DataFrame | None = None,
) -> Path:
    """Write STRATEGY_A_RESULTS.md with PRIMARY R4 first and explicit retrospective disclosure."""
    out_dir = Path(out_dir)

    def _fmt(row, col):
        return f"{row[f'{col}_mean']:.4f} ± {row[f'{col}_sd']:.4f}"

    def _model_table(recipe: str) -> list[str]:
        rows = [
            "| Model | τ | Dice | Precision | Precision (strict) | Recall | HD95 (mm) | ASD (mm) | clDice | Empties |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        sub = summ[summ["recipe"] == recipe].sort_values("dice_mean", ascending=False)
        for _, r in sub.iterrows():
            rows.append(
                f"| {r['model']} | {r['tau']:.2f} | {_fmt(r, 'dice')} | {_fmt(r, 'precision')} | "
                f"{r['precision_strict_mean']:.4f} | {_fmt(r, 'recall')} | {_fmt(r, 'hd95')} | "
                f"{_fmt(r, 'asd')} | {_fmt(r, 'cldice')} | {int(r['n_empty_pred'])} |"
            )
        return rows

    def _fam_table(sub: pd.DataFrame) -> list[str]:
        rows = [
            "| Recipe | Family | A vs B | Metric | Mean Diff (A−B) | 95% CI (10k boot) | p | Holm p | W/T/L | n excl. |",
            "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for _, r in sub.iterrows():
            rows.append(
                f"| {r['recipe']} | {r['family']} | {r['a']} vs {r['b']} | {r['metric']} | "
                f"{r['mean_difference']:+.4f} | [{r['diff_ci95_lo']:+.4f}, {r['diff_ci95_hi']:+.4f}] | "
                f"{r['p']:.3g} | {r['holm_p']:.3g} | {int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])} | "
                f"{int(r.get('n_excluded_nan', 0))} |"
            )
        return rows

    def _ni_table(sub: pd.DataFrame) -> list[str]:
        rows = [
            "| Recipe | Mean Diff | Median Diff | 95% CI (10k boot) | Wilcoxon p | W/T/L | Non-inferior (> −0.005) | Superior (> 0) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for _, r in sub.iterrows():
            rows.append(
                f"| {r['recipe']} | {r['mean_diff']:+.4f} | {r['median_diff']:+.4f} | "
                f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | {r['p_wilcoxon']:.3g} | "
                f"{int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])} | {bool(r['non_inferior'])} | {bool(r['superior'])} |"
            )
        return rows

    lines = [
        "# Strategy A — Upgraded 200-Epoch Retrain Results (`retrain_v1_1`)",
        "",
        "> **Retrospective Scope Notice:** ImageCAS Test-150 (`IDs 851–1000`) was previously inspected during v1 development; all Test-150 results below are reported strictly as **retrospective reassessments** under the pre-registered Amendment §7 protocol (`BENCHMARK_PREREGISTRATION.md`).",
        "",
        "All numbers are generated directly from `summary_test.csv`, `paired_stats_families.csv`, `noninferiority_rasnet_vs_nnunet_dice.csv`, and `timing_summary.csv`.",
        "",
    ]
    n_empty = int(summ["n_empty_pred"].sum())
    n_excl = int(fam["n_excluded_nan"].fillna(0).sum()) if len(fam) else 0
    if n_empty or n_excl:
        lines += [
            f"> **WARNING:** `{n_empty}` empty prediction(s) and `{n_excl}` NaN-excluded case-comparisons occurred. Report these counts with every claim.",
            "",
        ]
    lines += [
        f"## PRIMARY Result (Pre-Registered): `{PRIMARY_RECIPE_V11}`",
        "",
        *_model_table(PRIMARY_RECIPE_V11),
        "",
        "### Pre-Declared Hypothesis Families (Holm Within Family, Primary R4)",
        "",
        *_fam_table(fam[fam["role"] == "primary"]),
        "",
        "### Dice Non-Inferiority: `rasnet_v1_1` vs `nnunet_v2` (Margin −0.005)",
        "",
        *_ni_table(ni[ni["recipe"] == PRIMARY_RECIPE_V11]),
        "",
        "## Secondary Recipes (Descriptive Only)",
        "",
    ]
    for recipe in sorted(r for r in summ["recipe"].unique() if r != PRIMARY_RECIPE_V11):
        lines += [f"### `{recipe}`", "", *_model_table(recipe), ""]
    lines += [
        "### Secondary Families",
        "",
        *_fam_table(fam[fam["role"] == "secondary"]),
        "",
        "### Secondary Non-Inferiority",
        "",
        *_ni_table(ni[ni["recipe"] != PRIMARY_RECIPE_V11]),
        "",
    ]
    if timing is not None and len(timing):
        lines += [
            "## Computational Efficiency & Full-Volume Runtime (RTX 4080 SUPER, Val Cases)",
            "",
            "| Model | Params (M) | Mode | n | Median s/case [IQR] | Peak VRAM (MB) |",
            "| :--- | :---: | :--- | :---: | :---: | :---: |",
        ]
        for _, r in timing.iterrows():
            lines.append(
                f"| {r['model']} | {r['params_m']:.3f} | {r['mode']} | {int(r['n'])} | "
                f"{r['seconds_median']:.2f} [{r['seconds_q25']:.2f}, {r['seconds_q75']:.2f}] | "
                f"{r['peak_vram_mb']:.0f} |"
            )
        lines.append("")

    report_path = out_dir / "STRATEGY_A_RESULTS.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return report_path


def make_strategy_a_figures(
    summ: pd.DataFrame,
    per_case: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    """Generate publication figures in `retrain_v1_1/results/`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    saved: list[Path] = []
    r4 = per_case[per_case["recipe"] == PRIMARY_RECIPE_V11].copy()

    # 1. R4 Dice distribution across all 6 models
    order = [m for m in MODELS_V11 if m in set(r4["model"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [r4[r4["model"] == m]["dice"].dropna().to_numpy(float) for m in order]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel("Test-150 Dice (R4_tuned_tta8_top2)")
    ax.set_title("Strategy A (retrain_v1_1): Primary R4 Dice Distributions (N = 150 Retrospective)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "fig_strategyA_r4_dice_distributions.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    saved.append(p1)

    # 2. Paired Dice diff RASNet-v1.1 vs nnU-Net V2
    if {"rasnet_v1_1", "nnunet_v2"} <= set(r4["model"]):
        merged = pd.merge(
            r4[r4["model"] == "rasnet_v1_1"][["case_id", "dice"]].rename(columns={"dice": "ras"}),
            r4[r4["model"] == "nnunet_v2"][["case_id", "dice"]].rename(columns={"dice": "nnu"}),
            on="case_id",
        )
        diff = (merged["ras"] - merged["nnu"]).to_numpy(float)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(diff, bins=30, alpha=0.8, edgecolor="black")
        ax.axvline(0.0, color="black", linestyle="--", label="Zero")
        ax.axvline(-0.005, color="red", linestyle=":", label="NI margin (-0.005)")
        ax.axvline(float(np.mean(diff)), color="blue", linestyle="-", label=f"Mean ({np.mean(diff):+.4f})")
        ax.set_xlabel("Paired Dice Difference (rasnet_v1_1 − nnunet_v2)")
        ax.set_ylabel("Case Count")
        ax.set_title("Strategy A R4 Paired Dice Difference: RASNet-v1.1 vs nnU-Net V2 (N = 150)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p2 = out_dir / "fig_strategyA_paired_diff_rasnet_vs_nnunet.png"
        fig.savefig(p2, dpi=180)
        plt.close(fig)
        saved.append(p2)

    return saved


def load_model_for_eval(
    cfg: dict[str, Any],
    model_name: str,
    repo_root: Path,
    device: torch.device,
):
    """Load trained Strategy A best checkpoint or frozen v1 nnU-Net V2 predictor."""
    study_dir = repo_root / cfg["study"]["study_dir"]
    if model_name == "nnunet_v2":
        ref = cfg["study"]["nnunet_v2_ref"]
        folder = repo_root / ref["model_folder"]
        ckpt_path = folder / f"fold_{int(ref['fold'])}" / ref["checkpoint_name"]
        got_sha = se.sha256_file(ckpt_path)
        exp_sha = ref["expected_checkpoint_sha256"]
        if got_sha != exp_sha:
            raise RuntimeError(f"nnU-Net checkpoint SHA mismatch: {got_sha} != {exp_sha}")
        predictor = build_nnunet_predictor(
            model_folder=str(folder),
            fold=int(ref["fold"]),
            checkpoint_name=ref["checkpoint_name"],
            device=str(device),
            use_mirroring=False,
        )
        return predictor, {"checkpoint_path": str(ckpt_path), "checkpoint_sha256": got_sha, "n_params": 30_785_994}

    spec = next(m for m in cfg["models"] if m["name"] == model_name)
    ckpt_path = study_dir / "checkpoints" / model_name / "best_model.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint for {model_name}: {ckpt_path}")
    net = build_strategy_a_model(
        arch=spec["arch"],
        init_filters=int(spec.get("init_filters", 16)),
        dropout_prob=float(spec.get("dropout_prob", 0.1)),
        enable_attention=bool(spec.get("enable_attention", True)),
        enable_ds=bool(spec.get("enable_ds", True)),
    ).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net, {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": se.sha256_file(ckpt_path),
        "n_params": count_parameters(net),
    }


def run_eval_stage_val(
    cfg: dict[str, Any],
    repo_root: Path,
    dataset_root: str | Path,
    models: list[str],
    device_str: str = "cuda",
    smoke: int = 0,
) -> Path:
    """Run symmetric Val-94 grid sweep across [0.10..0.70] with crash-safe per-case resume."""
    study_dir = repo_root / cfg["study"]["study_dir"]
    out_dir = study_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = cfg["evaluation"]
    device = torch.device(device_str)

    fp = se.run_fingerprint(repo_root, int(eval_cfg["sw_batch_size"]), str(device))
    fp["protocol_version"] = PROTOCOL_VERSION_V11
    se.check_or_write_fingerprint(out_dir, fp)

    splits = json.loads((repo_root / cfg["study"]["splits_file"]).read_text(encoding="utf-8"))
    val_ids = sorted(int(c) for c in splits["val"])[: smoke or None]
    pairs = resolve_case_pairs(str(dataset_root), val_ids, check_headers=False)
    csv_path = out_dir / "val_grid_scores.csv"
    done = se.done_keys(csv_path, rows_per_case=GRID_ROWS_PER_CASE_V11)
    transform = build_deterministic_preprocess(cfg["preprocessing"], with_label=False, track_meta=True)
    roi_xyz = tuple(int(v) for v in eval_cfg["roi_size_xyz"])

    for m in models:
        todo = [cid for cid in val_ids if (m, cid) not in done]
        if not todo:
            print(f"[val] {m}: all {len(val_ids)} cases already done", flush=True)
            continue
        model_obj, meta = load_model_for_eval(cfg, m, repo_root, device)
        print(f"[val] {m}: running {len(todo)}/{len(val_ids)} cases (ckpt={meta['checkpoint_sha256'][:12]})", flush=True)
        t0 = time.time()
        for idx, cid in enumerate(todo, 1):
            img_p, lbl_p = pairs[cid]
            gt_zyx, sp_zyx = load_gt_zyx(lbl_p)
            if m == "nnunet_v2":
                p1 = infer_nnunet_probability(model_obj, str(img_p), tta=False)
                p8 = infer_nnunet_probability(model_obj, str(img_p), tta=True)
            else:
                p1, p8 = infer_monai_v11_1pass_and_tta8(
                    model_obj, img_p, transform, roi_xyz,
                    float(eval_cfg["sw_overlap"]), str(eval_cfg["sw_mode"]),
                    float(eval_cfg.get("sw_sigma_scale", 0.125)), int(eval_cfg["sw_batch_size"]), device,
                )
            rows = score_val_grid_v11(p1, gt_zyx, sp_zyx, m, cid, "sw050_gauss_1pass")
            rows += score_val_grid_v11(p8, gt_zyx, sp_zyx, m, cid, "sw050_gauss_tta8")
            se.append_rows(csv_path, rows)
            if idx % 10 == 0 or idx == len(todo):
                print(f"  [val/{m}] {idx}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        del model_obj
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return csv_path


def run_eval_stage_test(
    cfg: dict[str, Any],
    repo_root: Path,
    dataset_root: str | Path,
    models: list[str],
    device_str: str = "cuda",
    smoke: int = 0,
) -> Path:
    """Run one-shot retrospective Test-150 evaluation after verifying committed Val-94 lock."""
    study_dir = repo_root / cfg["study"]["study_dir"]
    out_dir = study_dir / "results"
    eval_cfg = cfg["evaluation"]
    device = torch.device(device_str)

    fp = se.run_fingerprint(repo_root, int(eval_cfg["sw_batch_size"]), str(device))
    fp["protocol_version"] = PROTOCOL_VERSION_V11
    se.check_or_write_fingerprint(out_dir, fp)

    splits = json.loads((repo_root / cfg["study"]["splits_file"]).read_text(encoding="utf-8"))
    lock_path = out_dir / "locked_recipes.json"
    lock, sha = load_strategy_a_lock(lock_path, splits, models=models, allow_smoke=bool(smoke))
    if not smoke:
        se.assert_lock_committed(repo_root, lock_path)

    test_ids = sorted(int(c) for c in splits["test"])[: smoke or None]
    pairs = resolve_case_pairs(str(dataset_root), test_ids, check_headers=False)
    csv_path = out_dir / "test_per_case.csv"
    done = se.done_keys(csv_path, rows_per_case=len(RECIPES_V11))
    transform = build_deterministic_preprocess(cfg["preprocessing"], with_label=False, track_meta=True)
    roi_xyz = tuple(int(v) for v in eval_cfg["roi_size_xyz"])

    for m in models:
        todo = [cid for cid in test_ids if (m, cid) not in done]
        if not todo:
            print(f"[test] {m}: all {len(test_ids)} cases already done", flush=True)
            continue
        model_obj, meta = load_model_for_eval(cfg, m, repo_root, device)
        print(
            f"[test] {m}: running {len(todo)}/{len(test_ids)} cases "
            f"(lock={sha[:12]}, ckpt={meta['checkpoint_sha256'][:12]})",
            flush=True,
        )
        t0 = time.time()
        for idx, cid in enumerate(todo, 1):
            img_p, lbl_p = pairs[cid]
            gt_zyx, sp_zyx = load_gt_zyx(lbl_p)
            if m == "nnunet_v2":
                p1 = infer_nnunet_probability(model_obj, str(img_p), tta=False)
                p8 = infer_nnunet_probability(model_obj, str(img_p), tta=True)
            else:
                p1, p8 = infer_monai_v11_1pass_and_tta8(
                    model_obj, img_p, transform, roi_xyz,
                    float(eval_cfg["sw_overlap"]), str(eval_cfg["sw_mode"]),
                    float(eval_cfg.get("sw_sigma_scale", 0.125)), int(eval_cfg["sw_batch_size"]), device,
                )
            rows = evaluate_four_recipes_case(m, cid, p1, p8, gt_zyx, sp_zyx, lock["thresholds"][m])
            se.append_rows(csv_path, rows)
            if idx % 15 == 0 or idx == len(todo):
                print(f"  [test/{m}] {idx}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        del model_obj
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return csv_path


def run_eval_stage_timing(
    cfg: dict[str, Any],
    repo_root: Path,
    dataset_root: str | Path,
    models: list[str],
    device_str: str = "cuda",
) -> Path:
    """Measure full-volume 1-pass and TTA-8 runtime + peak VRAM on Val cases only (never Test)."""
    study_dir = repo_root / cfg["study"]["study_dir"]
    out_dir = study_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = cfg["evaluation"]
    device = torch.device(device_str)
    n_cases = int(eval_cfg.get("timing_val_cases", 10))
    splits = json.loads((repo_root / cfg["study"]["splits_file"]).read_text(encoding="utf-8"))
    ids = sorted(int(c) for c in splits["val"])[: n_cases + 1]  # +1 warm-up
    pairs = resolve_case_pairs(str(dataset_root), ids, check_headers=False)
    transform = build_deterministic_preprocess(cfg["preprocessing"], with_label=False, track_meta=True)
    roi_xyz = tuple(int(v) for v in eval_cfg["roi_size_xyz"])

    per_case_rows = []
    for m in models:
        model_obj, meta = load_model_for_eval(cfg, m, repo_root, device)
        params_m = float(meta["n_params"]) / 1e6
        for mode in MODES_V11:
            for i, cid in enumerate(ids):
                img_p, _ = pairs[cid]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                t0 = time.perf_counter()
                if m == "nnunet_v2":
                    _ = infer_nnunet_probability(model_obj, str(img_p), tta=(mode == "sw050_gauss_tta8"))
                else:
                    if mode == "sw050_gauss_1pass":
                        batch = transform({"image": str(img_p)})
                        x = batch["image"].unsqueeze(0)
                        amp_ok = device.type == "cuda"
                        amp_dtype = torch.bfloat16 if (amp_ok and torch.cuda.is_bf16_supported()) else torch.float16
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_ok):
                            out = sliding_window_inference(
                                x, roi_xyz, int(eval_cfg["sw_batch_size"]), model_obj,
                                overlap=float(eval_cfg["sw_overlap"]), mode=str(eval_cfg["sw_mode"]),
                                sigma_scale=float(eval_cfg.get("sw_sigma_scale", 0.125)),
                                sw_device=device, device=torch.device("cpu"),
                            )
                            if isinstance(out, (tuple, list)):
                                out = out[0]
                            p1 = torch.softmax(out.float(), dim=1)[0, 1:2]
                        _ = _invert_prob_to_zyx(p1, batch, transform)
                    else:
                        _, _ = infer_monai_v11_1pass_and_tta8(
                            model_obj, img_p, transform, roi_xyz,
                            float(eval_cfg["sw_overlap"]), str(eval_cfg["sw_mode"]),
                            float(eval_cfg.get("sw_sigma_scale", 0.125)), int(eval_cfg["sw_batch_size"]), device,
                        )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                else:
                    peak_mb = 0.0
                sec = time.perf_counter() - t0
                if i == 0:
                    continue  # skip warm-up case
                per_case_rows.append(
                    {
                        "model": m,
                        "params_m": params_m,
                        "mode": mode,
                        "case_id": int(cid),
                        "seconds": float(sec),
                        "peak_vram_mb": peak_mb,
                    }
                )
        del model_obj
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df_t = pd.DataFrame(per_case_rows)
    df_t.to_csv(out_dir / "timing_per_case.csv", index=False, lineterminator="\n")
    summ_rows = []
    for (m, mode), g in df_t.groupby(["model", "mode"]):
        s = g["seconds"].to_numpy(float)
        summ_rows.append(
            {
                "model": m,
                "params_m": float(g["params_m"].iloc[0]),
                "mode": mode,
                "n": int(len(s)),
                "seconds_median": float(np.median(s)),
                "seconds_q25": float(np.percentile(s, 25)),
                "seconds_q75": float(np.percentile(s, 75)),
                "peak_vram_mb": float(g["peak_vram_mb"].max()),
            }
        )
    summ_path = out_dir / "timing_summary.csv"
    pd.DataFrame(summ_rows).to_csv(summ_path, index=False, lineterminator="\n")
    return summ_path
