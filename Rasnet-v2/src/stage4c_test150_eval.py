"""
Stage 4C: Retrospective Final Test Evaluation (`Test N = 150`, ImageCAS IDs 851-1000).
Requires frozen Stage 4B thresholds_v2.json (fail-closed), runs a single GPU pass per
model across Arms A, B (4-pass MONAI / 8-pass nnU-Net TTA), C-Top2, and C-Seeded
onto native SimpleITK (Z, Y, X) physical grids using surface-v1 metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_pairs import load_validated_splits
import eval_fair_v2
from eval_fair_v2 import ArmEvalSpec, bootstrap_ci
from study_registry import (
    atomic_write_csv,
    atomic_write_json,
    build_stage3_eval_manifest,
    resolve_run_best_checkpoint,
    sha256_file,
)

METRICS_COLS = ("dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall")


def _validate_thresholds_v2(
    thresholds_path: Path,
    splits: dict[str, Any],
    jobs: list[dict[str, Any]],
    scope: str = "run",
    skip_val_split_check: bool = False,
) -> tuple[dict[str, Any], str]:
    if not thresholds_path.is_file():
        raise FileNotFoundError(f"Missing frozen thresholds_v2.json: {thresholds_path}. Run Stage 4B first.")
    raw = thresholds_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))
    if data.get("split") != "val":
        raise ValueError(f"thresholds_v2.json split must be 'val', got {data.get('split')!r}")
    if not skip_val_split_check and set(int(c) for c in data.get("case_ids", [])) != set(int(c) for c in splits["val"]):
        raise ValueError("thresholds_v2.json case_ids do not match canonical validation split")
    bucket = "per_run" if scope == "run" else "per_family"
    key_field = "run_name" if scope == "run" else "family"
    for job in jobs:
        k = job[key_field]
        if k not in data.get(bucket, {}):
            raise KeyError(f"thresholds_v2.json[{bucket}] missing entry for {k!r}")
    return data, sha


def _summarize_df_with_ci(df: pd.DataFrame, meta: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    rec = dict(meta)
    rec["n_cases"] = int(len(df))
    rec["n_ok"] = int((df["status"] == "ok").sum()) if "status" in df else int(len(df))
    rec["n_empty_pred"] = int((df["status"] == "prediction_empty").sum()) if "status" in df else 0
    for m in METRICS_COLS:
        vals = df[m].dropna().to_numpy(dtype=float) if m in df else np.array([], dtype=float)
        if len(vals) > 0:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            med = float(np.median(vals))
            q25 = float(np.quantile(vals, 0.25))
            q75 = float(np.quantile(vals, 0.75))
            ci_lo, ci_hi = bootstrap_ci(vals, seed=seed, n_boot=2000, confidence=0.95)
            rec[f"{m}_mean"] = mean
            rec[f"{m}_std"] = std
            rec[f"{m}_median"] = med
            rec[f"{m}_q25"] = q25
            rec[f"{m}_q75"] = q75
            rec[f"{m}_ci95_lo"] = ci_lo
            rec[f"{m}_ci95_hi"] = ci_hi
            rec[f"{m}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
        else:
            rec[f"{m}_mean"] = float("nan")
            rec[f"{m}_std"] = float("nan")
            rec[f"{m}_median"] = float("nan")
            rec[f"{m}_q25"] = float("nan")
            rec[f"{m}_q75"] = float("nan")
            rec[f"{m}_ci95_lo"] = float("nan")
            rec[f"{m}_ci95_hi"] = float("nan")
            rec[f"{m}_mean_std"] = "nan"
    for pp in ("components_total", "components_retained"):
        vals = df[pp].dropna().to_numpy(dtype=float) if pp in df else np.array([], dtype=float)
        rec[f"{pp}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
    return rec


def run_stage4c_test150(
    config_path: str | Path,
    thresholds_json: str | Path | None = None,
    threshold_scope: str = "run",
    run_names_filter: list[str] | None = None,
    case_ids_override: list[int] | None = None,
    precomputed_probs_by_run: dict[str, dict[int, tuple[np.ndarray, np.ndarray | None]]] | None = None,
    pairs_override: dict[int, tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    out_dir = root_dir / "results" / "test_retrospective_150"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_validated_splits(cfg["study"]["splits_file"])
    test_ids = sorted(int(c) for c in (case_ids_override if case_ids_override is not None else splits["test"]))

    jobs = build_stage3_eval_manifest(cfg)
    if run_names_filter:
        allow = set(run_names_filter)
        jobs = [j for j in jobs if j["run_name"] in allow]

    thresh_p = Path(thresholds_json) if thresholds_json else (root_dir / "results" / "val94_final" / "thresholds_v2.json")
    thresh_data, thresh_sha = _validate_thresholds_v2(
        thresh_p,
        splits,
        jobs,
        scope=threshold_scope,
        skip_val_split_check=(case_ids_override is not None),
    )

    summary_rows: list[dict[str, Any]] = []
    family_dfs: dict[tuple[str, str], list[pd.DataFrame]] = {}
    ckpt_manifest: dict[str, str] = {}

    for job in jobs:
        rname = job["run_name"]
        fam = job["family"]
        t_entry = thresh_data["per_run"][rname] if threshold_scope == "run" else thresh_data["per_family"][fam]
        tau_t2 = float(t_entry["top2"]["tau"])
        s_cfg = t_entry["seeded"]
        tau_sd = float(s_cfg["tau"])
        ts_sd = float(s_cfg["tau_seed"])
        msv_sd = int(s_cfg["min_seed_voxels"])
        mvol_sd = float(s_cfg["min_comp_vol_mm3"])

        pre_probs = precomputed_probs_by_run.get(rname) if precomputed_probs_by_run else None
        if pre_probs is None:
            ckpt_or_dir, nn_name = resolve_run_best_checkpoint(cfg, job)
            if job["kind"] == "monai" and not ckpt_or_dir.is_file():
                raise FileNotFoundError(f"Missing best checkpoint for {rname}: {ckpt_or_dir}")
            if job["kind"] == "nnunet" and not (ckpt_or_dir / f"fold_{cfg['nnunet_v2']['fold']}" / str(nn_name)).is_file():
                raise FileNotFoundError(f"Missing nnU-Net best checkpoint for {rname}: {ckpt_or_dir} / {nn_name}")
            ckpt_manifest[rname] = (
                sha256_file(ckpt_or_dir)
                if job["kind"] == "monai"
                else sha256_file(ckpt_or_dir / f"fold_{cfg['nnunet_v2']['fold']}" / str(nn_name))
            )
        else:
            ckpt_or_dir, nn_name = root_dir / "checkpoints" / rname / "best_model.pth", "checkpoint_best.pth"
            ckpt_manifest[rname] = "synthetic_precomputed"

        arm_specs = [
            ArmEvalSpec(
                arm_name="A",
                output_csv=out_dir / f"per_case_{rname}_arm_A.csv",
                tau=0.50,
                need_tta=False,
                postprocess_mode="top_k",
                keep_top_k=None,
            ),
            ArmEvalSpec(
                arm_name="B",
                output_csv=out_dir / f"per_case_{rname}_arm_B.csv",
                tau=0.50,
                need_tta=True,
                postprocess_mode="top_k",
                keep_top_k=None,
            ),
            ArmEvalSpec(
                arm_name="C_top2",
                output_csv=out_dir / f"per_case_{rname}_arm_C_top2.csv",
                tau=tau_t2,
                need_tta=False,
                postprocess_mode="top_k",
                keep_top_k=2,
            ),
            ArmEvalSpec(
                arm_name="C_seeded",
                output_csv=out_dir / f"per_case_{rname}_arm_C_seeded.csv",
                tau=tau_sd,
                need_tta=False,
                postprocess_mode="seeded",
                keep_top_k=None,
                tau_seed=ts_sd,
                min_seed_voxels=msv_sd,
                min_comp_vol_mm3=mvol_sd,
            ),
        ]

        if job["kind"] == "monai":
            arm_dfs = eval_fair_v2.evaluate_monai_multi_arm_to_csvs(
                config_path=config_path,
                checkpoint_path=ckpt_or_dir,
                arch=job["arch"],
                init_filters=job["init_filters"],
                gate_mode=job.get("gate_mode", "conv_bias"),
                enable_ds=job.get("enable_ds", True),
                case_ids=test_ids,
                arm_specs=arm_specs,
                resume=True,
                pairs_override=pairs_override,
                precomputed_probs=pre_probs,
            )
        else:
            arm_dfs = eval_fair_v2.evaluate_nnunet_multi_arm_to_csvs(
                config_path=config_path,
                model_folder=ckpt_or_dir,
                checkpoint_name=str(nn_name),
                case_ids=test_ids,
                arm_specs=arm_specs,
                resume=True,
                pairs_override=pairs_override,
                precomputed_probs=pre_probs,
            )

        for arm_k, df_arm in arm_dfs.items():
            summary_rows.append(_summarize_df_with_ci(df_arm, {
                "run_name": rname,
                "family": fam,
                "kind": job["kind"],
                "arch": job["arch"],
                "init_filters": int(job["init_filters"]),
                "seed": int(job["seed"]),
                "arm": arm_k,
            }))
            df_tag = df_arm.copy()
            df_tag["seed"] = int(job["seed"])
            df_tag["run_name"] = rname
            family_dfs.setdefault((fam, arm_k), []).append(df_tag)

    df_all = pd.DataFrame(summary_rows)
    atomic_write_csv(df_all, out_dir / "summary_all_runs.csv")

    fam_rows: list[dict[str, Any]] = []
    for (fam, arm_k), dfs in sorted(family_dfs.items()):
        pooled = pd.concat(dfs, ignore_index=True)
        # Seed-averaged case table (N=150 mean across seeds per case)
        seed_avg = pooled.groupby("case_id", as_index=False)[[m for m in METRICS_COLS] + ["components_total", "components_retained"]].mean()
        seed_avg["status"] = "ok"
        rec_avg = _summarize_df_with_ci(seed_avg, {"family": fam, "arm": arm_k, "aggregation": "seed_averaged", "n_seeds": len(dfs)})
        # Also record across-seed SD of per-seed means
        per_seed_means = [float(d["dice"].mean()) for d in dfs if "dice" in d]
        rec_avg["dice_across_seed_std"] = float(np.std(per_seed_means, ddof=1)) if len(per_seed_means) > 1 else 0.0
        fam_rows.append(rec_avg)

        rec_pool = _summarize_df_with_ci(pooled, {"family": fam, "arm": arm_k, "aggregation": "seed_pooled", "n_seeds": len(dfs)})
        rec_pool["dice_across_seed_std"] = rec_avg["dice_across_seed_std"]
        fam_rows.append(rec_pool)

    atomic_write_csv(pd.DataFrame(fam_rows), out_dir / "summary_by_family_multiseed.csv")

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort_status": "RETROSPECTIVE_FINAL_RUN_IDs_851_1000",
        "note": "ImageCAS IDs 851-1000 were inspected in v1/Phase H and are reported strictly as a retrospective final evaluation, never untouched single-shot.",
        "n_test_cases": len(test_ids),
        "threshold_scope": threshold_scope,
        "thresholds_v2_path": str(thresh_p),
        "thresholds_v2_sha256": thresh_sha,
        "checkpoint_sha256s": ckpt_manifest,
        "summary_all_runs_csv": str(out_dir / "summary_all_runs.csv"),
        "summary_by_family_multiseed_csv": str(out_dir / "summary_by_family_multiseed.csv"),
    }
    atomic_write_json(manifest, out_dir / "RETROSPECTIVE_TEST150_MANIFEST.json")
    print(f"[Stage 4C] Completed Test N={len(test_ids)} evaluation across {len(jobs)} runs x 4 arms")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4C Retrospective Test N=150 Multi-Arm Evaluator")
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument("--thresholds-json", default=None)
    parser.add_argument("--threshold-scope", choices=["run", "family"], default="run")
    parser.add_argument("--runs", nargs="*", default=None)
    args = parser.parse_args()
    run_stage4c_test150(
        args.config,
        thresholds_json=args.thresholds_json,
        threshold_scope=args.threshold_scope,
        run_names_filter=args.runs,
    )


if __name__ == "__main__":
    main()
