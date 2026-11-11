"""
Stage 4A: Stratified N=20 Validation Learning Curve Evaluation (`val20_curves`).
Evaluates epoch_{50,100,150,200} across all 20 Stage 3 runs (0.50 Gauss, 1-Pass, tau=0.50)
on raw .nii.gz + Invertd onto native SimpleITK (Z, Y, X) grids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_fair_v2
from study_registry import (
    atomic_write_csv,
    atomic_write_json,
    build_stage3_eval_manifest,
    resolve_run_epoch_checkpoint,
)

METRICS_COLS = ("dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall")


def _plot_convergence(df_sum: pd.DataFrame, out_dir: Path) -> None:
    if df_sum.empty:
        return
    fam_grp = df_sum.groupby(["family", "epoch"], as_index=False).agg(
        dice_mean=("dice_mean", "mean"),
        dice_std=("dice_mean", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
        hd95_mean=("hd95_mean", "mean"),
        hd95_std=("hd95_mean", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
        precision_mean=("precision_mean", "mean"),
        cldice_mean=("cldice_mean", "mean"),
    )
    atomic_write_csv(fam_grp, out_dir / "val20_curves_by_family.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for fam, grp in fam_grp.groupby("family"):
        grp = grp.sort_values("epoch")
        ep = grp["epoch"].to_numpy()
        axes[0].errorbar(ep, grp["dice_mean"], yerr=grp["dice_std"], marker="o", capsize=3, label=fam)
        axes[1].errorbar(ep, grp["hd95_mean"], yerr=grp["hd95_std"], marker="o", capsize=3, label=fam)

    axes[0].set_title("Val20 Trajectory: Dice")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Dice")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Val20 Trajectory: HD95 (mm)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean HD95 (mm)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_val20_convergence_curves.{ext}", dpi=180)
    plt.close(fig)


def run_stage4a_val20_curves(
    config_path: str | Path,
    run_names_filter: list[str] | None = None,
    epochs_override: list[int] | None = None,
    case_ids_override: list[int] | None = None,
    precomputed_probs_by_key: dict[tuple[str, int], dict[int, tuple[np.ndarray, np.ndarray | None]]] | None = None,
    pairs_override: dict[int, tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    out_dir = root_dir / "results" / "val20_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    val20_cids = [
        int(c)
        for c in (
            case_ids_override
            if case_ids_override is not None
            else json.loads(Path(cfg["study"]["val20_ids_file"]).read_text(encoding="utf-8"))["case_ids"]
        )
    ]
    epochs_list = [int(e) for e in (epochs_override or cfg["training_budget"]["epoch_snapshots"])]
    jobs = build_stage3_eval_manifest(cfg)
    if run_names_filter:
        allow = set(run_names_filter)
        jobs = [j for j in jobs if j["run_name"] in allow]

    rows: list[dict[str, Any]] = []
    for job in jobs:
        rname = job["run_name"]
        for ep in epochs_list:
            csv_p = out_dir / f"per_case_{rname}_ep{ep}.csv"
            pre_probs = precomputed_probs_by_key.get((rname, ep)) if precomputed_probs_by_key else None
            if pre_probs is None:
                ckpt_or_dir, nn_name = resolve_run_epoch_checkpoint(cfg, job, ep)
                if job["kind"] == "monai" and not ckpt_or_dir.is_file():
                    raise FileNotFoundError(f"Missing Stage 3 epoch snapshot for {rname}: {ckpt_or_dir}")
                if job["kind"] == "nnunet" and not (ckpt_or_dir / f"fold_{cfg['nnunet_v2']['fold']}" / str(nn_name)).is_file():
                    raise FileNotFoundError(f"Missing Stage 3 nnU-Net epoch snapshot for {rname}: {ckpt_or_dir} / {nn_name}")
            else:
                ckpt_or_dir, nn_name = root_dir / "checkpoints" / rname / f"epoch_{ep}.pth", f"checkpoint_epoch_{ep}.pth"

            if job["kind"] == "monai":
                df = eval_fair_v2.evaluate_monai_checkpoint_to_csv(
                    config_path=config_path,
                    checkpoint_path=ckpt_or_dir,
                    arch=job["arch"],
                    init_filters=job["init_filters"],
                    gate_mode=job.get("gate_mode", "conv_bias"),
                    enable_ds=job.get("enable_ds", True),
                    case_ids=val20_cids,
                    output_csv=csv_p,
                    tau=0.50,
                    need_tta=False,
                    postprocess_mode="top_k",
                    keep_top_k=None,
                    resume=True,
                    pairs_override=pairs_override,
                    precomputed_probs=pre_probs,
                )
            else:
                df = eval_fair_v2.evaluate_nnunet_checkpoint_to_csv(
                    config_path=config_path,
                    model_folder=ckpt_or_dir,
                    checkpoint_name=str(nn_name),
                    case_ids=val20_cids,
                    output_csv=csv_p,
                    tau=0.50,
                    need_tta=False,
                    postprocess_mode="top_k",
                    keep_top_k=None,
                    resume=True,
                    pairs_override=pairs_override,
                    precomputed_probs=pre_probs,
                )

            rec: dict[str, Any] = {
                "run_name": rname,
                "family": job["family"],
                "kind": job["kind"],
                "arch": job["arch"],
                "init_filters": int(job["init_filters"]),
                "seed": int(job["seed"]),
                "epoch": int(ep),
                "n_cases": int(len(df)),
            }
            for m in METRICS_COLS:
                vals = df[m].dropna().to_numpy(dtype=float) if m in df else np.array([], dtype=float)
                rec[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
                rec[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append(rec)

    df_sum = pd.DataFrame(rows)
    atomic_write_csv(df_sum, out_dir / "val20_curves_summary.csv")
    _plot_convergence(df_sum, out_dir)
    manifest = {
        "n_runs": len(jobs),
        "epochs": epochs_list,
        "n_val20_cases": len(val20_cids),
        "summary_csv": str(out_dir / "val20_curves_summary.csv"),
    }
    atomic_write_json(manifest, out_dir / "val20_curves_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4A Val20 Convergence Curve Evaluator")
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, nargs="*", default=None)
    args = parser.parse_args()
    res = run_stage4a_val20_curves(args.config, run_names_filter=args.runs, epochs_override=args.epochs)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
