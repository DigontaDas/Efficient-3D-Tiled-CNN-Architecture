"""Option A: symmetric post-hoc inference + evaluation of frozen v1 checkpoints.

Retrospective supplement. Every model gets identical sliding-window blending, identical 8-pass
mirroring TTA (logit-averaged, matching nnUNetPredictor), an identical tau grid and identical
validation-only selection. Test IDs 851-1000 are never used for selection.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "optionA-1.0"
MODELS = ("rasnet", "segresnet", "vnet", "nnunet")
MONAI_MODELS = ("rasnet", "segresnet", "vnet")
MODES = ("sw050_gauss_1pass", "sw050_gauss_tta8")
SW_OVERLAP = 0.50
SW_MODE = "gaussian"
SIGMA_SCALE = 0.125
KEEP_TOP_K = 2
CONNECTIVITY = 26
TAU_GRID = tuple(round(0.20 + 0.05 * i, 2) for i in range(11))
LOCKED_TEST_RANGE = range(851, 1001)
NI_MARGIN = -0.005

# From CHECKPOINTS.md (verified 2026-11-10). A mismatch aborts the run.
EXPECTED_CHECKPOINT_SHA256 = {
    "rasnet": "8d4c25cd12d07c40423062b27b1d36836e5fd6cecb30c20c34055eeac976d1b8",
    "segresnet": "b8ea61d95106a0bbe15e1ce9f9e5af1a38d9508e59d0639d32bc883d93c6e109",
    "vnet": "a89d71bd41c3e773c68e3f4b5e27c5a067f959154aa049eead76f1642c5c1063",
    "nnunet": "9accddf283b1c480e386f30f426b9200ca8d4aa1ecd70d5b5af0b8aad6058711",
}

PROTECTED_RESULT_DIRS = (
    "results/benchmark_runs/val_full_94",
    "results/benchmark_runs/test_retrospective_150",
    "results/rasnet_fixed_weight_optimization",
    "results/thesis_outputs",
)

GRID_ROWS_PER_CASE = len(MODES) * len(TAU_GRID)
TEST_ROWS_PER_CASE = 4


# ============================================================ provenance helpers
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Folders whose state defines the Option A result. Rasnet-v2 run logs (queue_status.json,
# training_log.csv, DONE.flag) are git-tracked and change while training runs on the lab PC;
# they must not make an Option A manifest look "dirty".
CODE_PATHS = ("benchmark", "scripts", "tests", "Phase3_Local_Integration", "config.lab.yaml")


def git_info(repo: Path, code_paths=CODE_PATHS) -> dict:
    """Commit + two dirty flags: dirty_code (only the folders above; the one that matters) and
    dirty_any (whole repo, informational). None means git was unavailable."""
    def _run(*a):
        try:
            return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    commit = _run("rev-parse", "HEAD")
    any_changes = _run("status", "--porcelain")
    code_changes = _run("status", "--porcelain", "--", *code_paths)
    return {"commit": commit if commit is not None else "unavailable",
            "dirty_code": None if code_changes is None else bool(code_changes),
            "dirty_any": None if any_changes is None else bool(any_changes),
            "code_paths": list(code_paths)}


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "monai", "nnunetv2", "SimpleITK", "numpy", "cc3d"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "missing"
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def check_out_dir(out_dir: Path, repo: Path, resume: bool) -> Path:
    out = Path(out_dir).resolve()
    for rel in PROTECTED_RESULT_DIRS:
        prot = (Path(repo) / rel).resolve()
        if out == prot or prot in out.parents:
            raise ValueError(f"Refusing to write into protected results directory: {prot}")
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"{out} is not empty; pass --resume to continue an interrupted run")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================ inference
def flip_axis_combinations(spatial_axes=(2, 3, 4)) -> list[tuple[int, ...]]:
    """Identity + every non-empty combination of spatial axes (8 for 3D), as nnUNetPredictor."""
    combos: list[tuple[int, ...]] = [()]
    for r in range(1, len(spatial_axes) + 1):
        combos.extend(itertools.combinations(spatial_axes, r))
    return combos


def infer_monai_symmetric(name, net, batch, transform, device, roi_size=(96, 96, 96),
                          sw_batch_size=8, use_amp=None, tta=True) -> dict[str, np.ndarray]:
    """Return {mode: foreground prob (Z,Y,X) on the native grid} for the symmetric modes.

    Logits are Gaussian-blended by the sliding window, mirrored-averaged over 8 flips,
    then soft-maxed: the same order of operations as nnUNetPredictor. tta=False returns only
    the 1-pass mode (used by the timing stage).
    """
    import torch
    from monai.inferers import sliding_window_inference
    from monai.transforms import Invertd

    if name not in MONAI_MODELS:
        raise ValueError(f"Not a MONAI sliding-window model: {name}")
    if name == "vnet":
        roi_size = tuple(((int(r) + 15) // 16) * 16 for r in roi_size)
    net.eval()
    x = batch["image"].unsqueeze(0).to(device)
    amp = str(device).startswith("cuda") if use_amp is None else bool(use_amp)

    def logits(t):
        with torch.amp.autocast("cuda", enabled=amp):
            out = sliding_window_inference(t, roi_size, sw_batch_size, net, overlap=SW_OVERLAP,
                                           mode=SW_MODE, sigma_scale=SIGMA_SCALE)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.float()

    combos = flip_axis_combinations()
    probs = []
    with torch.no_grad():
        identity = logits(x)
        probs.append((MODES[0], torch.softmax(identity, dim=1)[:, 1:2]))
        if tta:
            acc = identity.clone()
            for axes in combos[1:]:
                acc += torch.flip(logits(torch.flip(x, axes)), axes)
            probs.append((MODES[1], torch.softmax(acc / float(len(combos)), dim=1)[:, 1:2]))
            del acc
        del identity

    inv = Invertd(keys=["pred"], transform=transform, orig_keys=["image"],
                  nearest_interp=False, to_tensor=True)
    out: dict[str, np.ndarray] = {}
    for mode, p in probs:
        b = dict(batch)
        b["pred"] = p.squeeze(0)
        arr = inv(b)["pred"].squeeze(0).detach().cpu().numpy()
        arr = np.transpose(arr, (2, 1, 0)).astype(np.float32, copy=False)
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Non-finite probability from {name} ({mode})")
        out[mode] = arr
    return out


def iter_nnunet_modes(predictor, case_ids, pairs):
    """Yield (cid, {mode: prob_zyx}) using native nnUNetPredictor (0.5 Gaussian; mirroring = 8 passes)."""
    from benchmark.run_benchmark import _run_nnunet_batch_probabilities
    for cid, p1, p8 in _run_nnunet_batch_probabilities(predictor, list(case_ids), pairs,
                                                         need_tta=True, num_processes=4):
        if p8 is None:
            raise RuntimeError(f"nnU-Net mirroring pass missing for case {cid}")
        yield int(cid), {MODES[0]: p1, MODES[1]: p8}


def _load_gt(label_path):
    import SimpleITK as sitk
    lbl = sitk.ReadImage(str(label_path))
    gt = sitk.GetArrayFromImage(lbl) > 0
    sp = tuple(float(s) for s in reversed(lbl.GetSpacing()))
    return gt, sp


def iter_case_probabilities(model, cfg, repo: Path, case_ids, pairs, device="cuda", sw_batch_size=8):
    """Yield (cid, {mode: prob}, gt_bool, spacing_zyx) for one model; prefetches CPU preprocessing."""
    case_ids = [int(c) for c in case_ids]
    if model == "nnunet":
        from benchmark.adapters.nnunet_adapter import build_nnunet_predictor
        predictor = build_nnunet_predictor(
            model_folder=str(Path(repo) / cfg["nnunet_model_folder"]),
            fold=int(cfg["nnunet_fold"]),
            checkpoint_name=str(cfg["nnunet_checkpoint_name"]),
            device=device,
            use_mirroring=False,
        )
        for cid, modes in iter_nnunet_modes(predictor, case_ids, pairs):
            gt, sp = _load_gt(pairs[cid][1])
            _check_shapes(model, cid, modes, gt)
            yield cid, modes, gt, sp
        return

    from benchmark.adapters.monai_models import build_model, build_transform
    net = build_model(model, str(Path(repo) / cfg["checkpoints"][model]), device)
    tf = build_transform(model)

    def prep(cid):
        return cid, tf({"image": str(pairs[cid][0])})

    with ThreadPoolExecutor(max_workers=2) as pool:
        queue = deque(pool.submit(prep, c) for c in case_ids[:2])
        nxt = len(queue)
        while queue:
            cid, batch = queue.popleft().result()
            if nxt < len(case_ids):
                queue.append(pool.submit(prep, case_ids[nxt]))
                nxt += 1
            modes = infer_monai_symmetric(model, net, batch, tf, device, sw_batch_size=sw_batch_size)
            del batch
            gt, sp = _load_gt(pairs[cid][1])
            _check_shapes(model, cid, modes, gt)
            yield cid, modes, gt, sp


def _check_shapes(model, cid, modes, gt):
    for mode, p in modes.items():
        if p.shape != gt.shape:
            raise RuntimeError(f"{model} case {cid} {mode}: prob {p.shape} != label {gt.shape}")


# ============================================================ scoring
def score_tau_grid(prob, gt_bool, spacing_zyx, grid=TAU_GRID, keep_top_k=KEEP_TOP_K) -> dict[float, float]:
    from benchmark.postprocess import prune_components
    from benchmark.thresholds import _fast_dice
    out = {}
    for tau in grid:
        raw = (prob >= tau).astype(np.uint8)
        cleaned, _ = prune_components(raw, spacing_zyx=spacing_zyx, keep_top_k=keep_top_k,
                                      connectivity=CONNECTIVITY)
        out[round(float(tau), 2)] = _fast_dice(cleaned, gt_bool)
    return out


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    mode: str
    tau: float
    keep_top_k: int | None


def recipes_for(model: str, lock: dict) -> tuple[Recipe, ...]:
    t = lock["thresholds"][model]
    return (
        Recipe("R1_raw_1pass", MODES[0], 0.50, None),
        Recipe("R2_raw_tta8", MODES[1], 0.50, None),
        Recipe("R3_tuned_1pass_top2", MODES[0], float(t[MODES[0]]), KEEP_TOP_K),
        Recipe("R4_tuned_tta8_top2", MODES[1], float(t[MODES[1]]), KEEP_TOP_K),
    )


def evaluate_recipe(model, cid, recipe: Recipe, prob, gt_bool, spacing_zyx) -> dict:
    from benchmark.metrics import METRIC_VERSION, compute_metrics
    from benchmark.postprocess import prune_components
    raw = (prob >= recipe.tau).astype(np.uint8)
    mask, info = prune_components(raw, spacing_zyx=spacing_zyx, keep_top_k=recipe.keep_top_k,
                                  connectivity=CONNECTIVITY)
    metrics = compute_metrics(mask, gt_bool, spacing_zyx)
    return {"model": model, "case_id": int(cid), "recipe": recipe.recipe_id, "mode": recipe.mode,
            "tau": float(recipe.tau), "keep_top_k": int(recipe.keep_top_k or 0),
            "metric_version": METRIC_VERSION, **metrics, **info}


# ============================================================ selection & lock
def select_tau(per_case: dict[int, dict[float, float]], case_ids, grid=TAU_GRID) -> tuple[float, dict]:
    ids = sorted(int(c) for c in case_ids)
    if any(c in LOCKED_TEST_RANGE for c in ids):
        raise ValueError("Threshold selection received locked test IDs 851-1000")
    if set(per_case) != set(ids):
        raise ValueError("Per-case scores do not match the requested validation IDs")
    grid = [round(float(t), 2) for t in grid]
    for cid in ids:
        if set(round(float(t), 2) for t in per_case[cid]) != set(grid):
            raise ValueError(f"Case {cid} does not cover the full tau grid")
    means = {t: float(np.mean([per_case[c][t] for c in ids])) for t in grid}
    best = min(grid, key=lambda t: (-round(means[t], 12), round(abs(t - 0.50), 4), t))
    return best, {f"{t:.2f}": means[t] for t in grid}


def build_lock(grid_df: pd.DataFrame, splits: dict, models, allow_partial=False) -> dict:
    from benchmark.preflight import validate_splits
    parts = validate_splits(splits)
    df = grid_df.drop_duplicates(subset=["model", "case_id", "mode", "tau"], keep="last")
    present = {int(c) for c in df["case_id"].unique()}
    if present & set(LOCKED_TEST_RANGE):
        raise ValueError("Validation grid file contains locked test IDs")
    if not allow_partial and present != parts["val"]:
        raise ValueError("Lock requires the entire validation split (94 cases)")
    ids = sorted(present)
    thresholds, scores = {}, {}
    for model in models:
        thresholds[model], scores[model] = {}, {}
        for mode in MODES:
            sub = df[(df["model"] == model) & (df["mode"] == mode)]
            per_case = {int(c): {round(float(t), 2): float(d) for t, d in zip(g["tau"], g["dice"])}
                        for c, g in sub.groupby("case_id")}
            tau, means = select_tau(per_case, ids)
            thresholds[model][mode] = tau
            scores[model][mode] = means
    return {"protocol_version": PROTOCOL_VERSION, "split": "val", "smoke": bool(allow_partial),
            "case_ids": ids, "grid": list(TAU_GRID), "keep_top_k": KEEP_TOP_K,
            "connectivity": CONNECTIVITY, "objective": "mean_val_dice_after_top2cc",
            "tie_breaker": "closest_to_0.50_then_lower", "thresholds": thresholds,
            "grid_scores": scores}


def write_lock(path: Path, payload: dict) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Lock already exists and is immutable: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    sha = sha256_file(path)
    path.with_suffix(".sha256").write_text(sha, encoding="utf-8")
    return sha


def load_lock(path: Path, splits: dict, models, allow_smoke=False) -> tuple[dict, str]:
    from benchmark.preflight import validate_splits
    parts = validate_splits(splits)
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    sha = sha256_file(path)
    side = path.with_suffix(".sha256")
    if side.is_file() and side.read_text(encoding="utf-8").strip() != sha:
        raise ValueError("Lock file hash does not match its .sha256 sidecar (modified after locking)")
    if data.get("split") != "val":
        raise ValueError("Lock must be selected on the validation split")
    if [round(float(t), 2) for t in data.get("grid", [])] != list(TAU_GRID):
        raise ValueError("Lock grid differs from the pre-registered tau grid")
    if data.get("smoke") and not allow_smoke:
        raise ValueError("Smoke lock cannot be used for a final test run")
    if not data.get("smoke") and set(data["case_ids"]) != parts["val"]:
        raise ValueError("Lock does not cover the full validation split")
    for m in models:
        for mode in MODES:
            t = float(data["thresholds"][m][mode])
            if round(t, 2) not in TAU_GRID:
                raise ValueError(f"Locked tau {t} for {m}/{mode} is off-grid")
    return data, sha


# ============================================================ CSV resume helpers
def done_keys(csv_path: Path, rows_per_case: int, key_cols=("model", "case_id")) -> set:
    p = Path(csv_path)
    if not p.is_file() or p.stat().st_size == 0:
        return set()
    df = pd.read_csv(p)
    counts = df.groupby(list(key_cols)).size()
    return {(str(m), int(c)) for (m, c), n in counts.items() if n >= rows_per_case}


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    p = Path(csv_path)
    pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)


# ============================================================ statistics
FAMILIES = {
    "FA_primary_rasnet_vs_nnunet": [
        {"a": "rasnet", "b": "nnunet", "metric": "precision", "direction": "higher"},
        {"a": "rasnet", "b": "nnunet", "metric": "hd95", "direction": "lower"},
        {"a": "rasnet", "b": "nnunet", "metric": "asd", "direction": "lower"},
    ],
    "FB_rasnet_vs_segresnet": [
        {"a": "rasnet", "b": "segresnet", "metric": m, "direction": d}
        for m, d in (("dice", "higher"), ("precision", "higher"), ("hd95", "lower"), ("cldice", "higher"))
    ],
    "FC_rasnet_vs_vnet": [
        {"a": "rasnet", "b": "vnet", "metric": m, "direction": d}
        for m, d in (("dice", "higher"), ("precision", "higher"), ("hd95", "lower"), ("cldice", "higher"))
    ],
    "FD_secondary_rasnet_vs_nnunet": [
        {"a": "rasnet", "b": "nnunet", "metric": m, "direction": "higher"}
        for m in ("dice", "cldice", "recall")
    ],
    "EX_baselines": [
        {"a": a, "b": b, "metric": "dice", "direction": "higher"}
        for a, b in (("nnunet", "segresnet"), ("nnunet", "vnet"), ("segresnet", "vnet"))
    ],
}
SUMMARY_METRICS = ("dice", "iou", "precision", "recall", "hd95", "asd", "cldice",
                   "centerline_recall", "components_total")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, recipe), g in df.groupby(["model", "recipe"]):
        r = {"model": model, "recipe": recipe, "n": int(len(g)),
             "n_ok": int((g["status"] == "ok").sum()), "tau": float(g["tau"].iloc[0])}
        for m in SUMMARY_METRICS:
            v = g[m].dropna().to_numpy(float)
            r[f"{m}_mean"] = float(v.mean()) if len(v) else np.nan
            r[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            r[f"{m}_median"] = float(np.median(v)) if len(v) else np.nan
            r[f"{m}_q25"] = float(np.quantile(v, 0.25)) if len(v) else np.nan
            r[f"{m}_q75"] = float(np.quantile(v, 0.75)) if len(v) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def run_families(df: pd.DataFrame, recipe: str) -> pd.DataFrame:
    from benchmark.stats import analyze_family
    sub = df[df["recipe"] == recipe]
    tables = {m: g.drop_duplicates("case_id").reset_index(drop=True) for m, g in sub.groupby("model")}
    out = []
    for fam, hyps in FAMILIES.items():
        hyps = [h for h in hyps if h["a"] in tables and h["b"] in tables]
        if not hyps:
            continue
        for r in analyze_family(tables, hyps, allow_nan_filtering=True):
            out.append({"family": fam, "recipe": recipe, **r})
    return pd.DataFrame(out)


def paired_noninferiority(a: pd.DataFrame, b: pd.DataFrame, metric="dice", margin=NI_MARGIN) -> dict:
    from benchmark.stats import bootstrap_ci
    pair = a[["case_id", metric]].merge(b[["case_id", metric]], on="case_id",
                                        validate="one_to_one", suffixes=("_a", "_b")).dropna()
    d = pair[f"{metric}_a"].to_numpy(float) - pair[f"{metric}_b"].to_numpy(float)
    lo, hi = bootstrap_ci(d, seed=42, n_boot=2000, confidence=0.95)
    return {"metric": metric, "n": int(len(d)), "mean_diff": float(d.mean()), "ci95_lo": lo,
            "ci95_hi": hi, "margin": float(margin), "non_inferior": bool(lo > margin),
            "superior": bool(lo > 0.0)}


# ============================================================ timing (efficiency evidence)
def _time_calls(run, model: str, case_ids, device: str, warmup: int = 1) -> list[dict]:
    """Time run(cid, tta) end-to-end for 1-pass and TTA-8 separately (same definition for every model).

    The first `warmup` cases are run once untimed (cuDNN autotune, allocator). CUDA is synchronised
    around every call; peak allocated VRAM is reset per mode.
    """
    import time
    cuda = str(device).startswith("cuda")
    if cuda:
        import torch
    ids = [int(c) for c in case_ids]
    if len(ids) <= warmup:
        raise ValueError("Need more cases than warm-up cases")
    for cid in ids[:warmup]:
        run(cid, False)
    rows = []
    for tta in (False, True):
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        mode_rows = []
        for cid in ids[warmup:]:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run(cid, tta)
            if cuda:
                torch.cuda.synchronize()
            mode_rows.append({"model": model, "case_id": cid, "mode": MODES[1] if tta else MODES[0],
                              "seconds": time.perf_counter() - t0})
        peak = float(torch.cuda.max_memory_allocated() / 2 ** 20) if cuda else float("nan")
        for r in mode_rows:
            r["peak_vram_mb"] = peak
        rows.extend(mode_rows)
    return rows


def time_model(model, cfg, repo: Path, case_ids, pairs, device="cuda", sw_batch_size=8, warmup=1):
    """Per-case wall-clock: preprocessing + inference + native-grid inversion, as a user would run it."""
    if model == "nnunet":
        from benchmark.adapters.nnunet_adapter import build_nnunet_predictor, infer_nnunet_probability
        predictor = build_nnunet_predictor(model_folder=str(Path(repo) / cfg["nnunet_model_folder"]),
                                           fold=int(cfg["nnunet_fold"]),
                                           checkpoint_name=str(cfg["nnunet_checkpoint_name"]),
                                           device=device, use_mirroring=False)

        def run(cid, tta):
            infer_nnunet_probability(predictor, str(pairs[cid][0]), tta=tta)
    else:
        from benchmark.adapters.monai_models import build_model, build_transform
        net = build_model(model, str(Path(repo) / cfg["checkpoints"][model]), device)
        tf = build_transform(model)

        def run(cid, tta):
            batch = tf({"image": str(pairs[cid][0])})
            infer_monai_symmetric(model, net, batch, tf, device, sw_batch_size=sw_batch_size, tta=tta)
    return _time_calls(run, model, case_ids, device, warmup)


def summarize_timing(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "mode"])
    return pd.DataFrame({
        "n": g["seconds"].size(),
        "seconds_median": g["seconds"].median(),
        "seconds_q25": g["seconds"].quantile(0.25),
        "seconds_q75": g["seconds"].quantile(0.75),
        "peak_vram_mb": g["peak_vram_mb"].max(),
    }).reset_index()


# ============================================================ figures (thesis-ready, static)
# Fixed colour per model (never by rank). Validated with the dataviz palette checker
# (light surface): CVD/normal-vision separation PASS; aqua/yellow are < 3:1 contrast, so every
# mark is also identified by a text label (direct labels / axis labels), never colour alone.
MODEL_LABELS = {"rasnet": "RASNet", "nnunet": "nnU-Net V2", "segresnet": "SegResNet", "vnet": "V-Net"}
MODEL_COLORS = {"rasnet": "#2a78d6", "nnunet": "#eb6834", "segresnet": "#1baf7a", "vnet": "#eda100"}
_INK, _INK2, _GRID = "#1f1f1d", "#5f5e5a", "#e6e5e0"


def _style(plt):
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8a86", "axes.labelcolor": _INK2, "xtick.color": _INK2,
        "ytick.color": _INK2, "axes.titlecolor": _INK, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.bbox": "tight",
    })


def _save(fig, out_dir: Path, name: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        p = Path(out_dir) / f"{name}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(str(p))
    return paths


def make_figures(summary: pd.DataFrame, per_case: pd.DataFrame, efficiency: pd.DataFrame,
                 out_dir: Path, primary_recipe: str = "R4_tuned_tta8_top2", ni_margin: float = NI_MARGIN):
    """Three figures, all computed from the result CSVs:
    1) Dice vs parameters (accuracy-efficiency, primary recipe)
    2) per-case Dice distributions, one panel per recipe
    3) paired per-case Dice difference RASNet - nnU-Net (primary recipe) with the NI margin
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    out_dir = Path(out_dir)
    written = []
    models = [m for m in MODELS if m in set(per_case["model"])]

    # 1 -- accuracy vs size
    s = summary[summary["recipe"] == primary_recipe].set_index("model")
    e = efficiency.set_index("model_key")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    xs = {m: float(e.loc[m, "Parameters_M"]) for m in models}
    pos, side = {}, {}
    for m in models:                                  # dodge models of near-identical size (RASNet/SegResNet)
        near = [o for o in models if abs(np.log(xs[o] / xs[m])) < 0.1]
        k, n = near.index(m), len(near)
        pos[m] = xs[m] * (1.0 + 0.10 * (k - (n - 1) / 2)) if n > 1 else xs[m]
        side[m] = -1 if (n > 1 and k < (n - 1) / 2) else 1
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for m in models:
        x, y = pos[m], float(s.loc[m, "dice_mean"])
        ax.vlines(x, float(s.loc[m, "dice_q25"]), float(s.loc[m, "dice_q75"]),
                  color=MODEL_COLORS[m], lw=2, alpha=0.45)
        ax.scatter([x], [y], s=64, color=MODEL_COLORS[m], edgecolor="white", linewidth=2, zorder=3,
                   label=MODEL_LABELS[m])
        ax.annotate(f"{MODEL_LABELS[m]}  {y:.3f}\n{float(e.loc[m, 'GFLOPs_per_Patch']):.0f} GFLOPs/patch",
                    (x, y), xytext=(8 * side[m], 3), textcoords="offset points", fontsize=8, color=_INK,
                    ha="left" if side[m] > 0 else "right")
    ax.set_xscale("log")
    ax.set_xlim(1.5, 120)
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Test Dice (mean; bar = IQR)")
    ax.grid(True, axis="y", color=_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Accuracy vs model size — {primary_recipe}", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    written += _save(fig, out_dir, "fig_optionA_dice_vs_params")
    plt.close(fig)

    # 2 -- per-case Dice distributions (small multiples, shared y)
    recipes = sorted(per_case["recipe"].unique())
    fig, axes = plt.subplots(1, len(recipes), figsize=(2.3 * len(recipes), 3.4), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], recipes):
        data = [per_case[(per_case["recipe"] == r) & (per_case["model"] == m)]["dice"].dropna().to_numpy()
                for m in models]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops={"color": _INK, "lw": 1.2},
                        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.5})
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(MODEL_COLORS[m])
            patch.set_alpha(0.45)
            patch.set_edgecolor(MODEL_COLORS[m])
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=35, ha="right")
        ax.set_title(r.replace("_", " "), fontsize=9, loc="left")
        ax.grid(True, axis="y", color=_GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Per-case Dice (Test-150)")
    written += _save(fig, out_dir, "fig_optionA_dice_distributions")
    plt.close(fig)

    # 3 -- paired difference RASNet - nnU-Net (single series)
    sub = per_case[per_case["recipe"] == primary_recipe]
    a = sub[sub["model"] == "rasnet"][["case_id", "dice"]]
    b = sub[sub["model"] == "nnunet"][["case_id", "dice"]]
    if len(a) and len(b):
        pair = a.merge(b, on="case_id", suffixes=("_r", "_n")).dropna()
        d = np.sort((pair["dice_r"] - pair["dice_n"]).to_numpy())
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.scatter(np.arange(len(d)), d, s=10, color=MODEL_COLORS["rasnet"], zorder=3)
        ax.axhline(0, color=_INK2, lw=0.8)
        ax.axhline(ni_margin, color=_INK2, lw=0.8, ls="--")
        ax.text(0.98, 0.04, f"dashed line: non-inferiority margin {ni_margin:+.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=_INK2)
        ax.annotate(f"mean {d.mean():+.4f}  ({int((d > 0).sum())} cases RASNet higher / "
                    f"{int((d < 0).sum())} lower)", (0.02, 0.95), xycoords="axes fraction",
                    fontsize=8, color=_INK, va="top")
        ax.set_xlabel("Test cases, sorted by difference")
        ax.set_ylabel("Dice: RASNet − nnU-Net V2")
        ax.grid(True, axis="y", color=_GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"Per-case paired difference — {primary_recipe}", loc="left", fontsize=10)
        written += _save(fig, out_dir, "fig_optionA_paired_diff_rasnet_vs_nnunet")
        plt.close(fig)
    return written
