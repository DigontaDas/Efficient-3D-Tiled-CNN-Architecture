"""
Stage 4B: Full Validation N=94 Evaluation, 11x12 Arm C Threshold/Seeded-Filter Sweep,
and Frozen Validation Thresholds Lock (`results/val94_final/thresholds_v2.json`).

Executes a single GPU inference pass per model checkpoint and reuses cached float32
native-(Z,Y,X) probabilities for the 143-config sweep and final Arm C surface-v1 scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
from eval_fair_v2 import ArmEvalSpec
from study_registry import (
    atomic_write_csv,
    atomic_write_json,
    build_stage3_eval_manifest,
    resolve_run_best_checkpoint,
    sha256_file,
)

METRICS_COLS = ("dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall")


def _top2_key(tau: float) -> str:
    return f"{round(float(tau), 4):.4f}"


def _seeded_key(tau: float, ts: float, msv: int, mvol: float) -> str:
    return f"{round(float(tau), 4):.4f}|{round(float(ts), 4):.4f}|{int(msv)}|{round(float(mvol), 4):.4f}"


def _get_top2_val(entry: dict[str, Any], tau: float) -> float | None:
    t2 = entry.get("top2", {}) if isinstance(entry, dict) else {}
    k4 = _top2_key(tau)
    if k4 in t2:
        return float(t2[k4])
    k2 = f"{tau:.2f}"
    if k2 in t2:
        return float(t2[k2])
    return None


def _get_seeded_val(entry: dict[str, Any], tau: float, ts: float, msv: int, mvol: float) -> float | None:
    sd = entry.get("seeded", {}) if isinstance(entry, dict) else {}
    k4 = _seeded_key(tau, ts, msv, mvol)
    if k4 in sd:
        return float(sd[k4])
    k2 = f"{tau:.2f}|{ts:.2f}|{msv}|{mvol:.1f}"
    if k2 in sd:
        return float(sd[k2])
    return None


def _has_complete_sweep_entry(
    entry: dict[str, Any] | None,
    tau_grid: list[float],
    ts_vals: list[float],
    msv_vals: list[int],
    mvol_vals: list[float],
) -> bool:
    if not isinstance(entry, dict):
        return False
    for tau in tau_grid:
        if _get_top2_val(entry, tau) is None:
            return False
        for ts in ts_vals:
            for msv in msv_vals:
                for mvol in mvol_vals:
                    if _get_seeded_val(entry, tau, ts, msv, mvol) is None:
                        return False
    return True


def _pick_best_top2(mean_scores: dict[float, float]) -> tuple[float, float]:
    best_tau = None
    best_score = -1.0
    for tau in sorted(mean_scores.keys()):
        sc = float(mean_scores[tau])
        if best_tau is None or sc > best_score + 1e-12:
            best_tau, best_score = float(tau), sc
        elif abs(sc - best_score) <= 1e-12:
            if (abs(tau - 0.50), tau) < (abs(best_tau - 0.50), best_tau):
                best_tau, best_score = float(tau), sc
    assert best_tau is not None
    return best_tau, best_score


def _pick_best_seeded(mean_scores: dict[tuple[float, float, int, float], float]) -> tuple[tuple[float, float, int, float], float]:
    best_key = None
    best_score = -1.0
    for key in sorted(mean_scores.keys()):
        sc = float(mean_scores[key])
        tau, ts, msv, mvol = key
        tie_k = (abs(tau - 0.50), tau, abs(ts - 0.55), msv, mvol)
        if best_key is None or sc > best_score + 1e-12:
            best_key, best_score = key, sc
        elif abs(sc - best_score) <= 1e-12:
            bt, bts, bmsv, bmvol = best_key
            if tie_k < (abs(bt - 0.50), bt, abs(bts - 0.55), bmsv, bmvol):
                best_key, best_score = key, sc
    assert best_key is not None
    return best_key, best_score


def _summarize_val_df(df: pd.DataFrame, run_name: str, family: str, arm: str, seed: int) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "run_name": run_name,
        "family": family,
        "arm": arm,
        "seed": int(seed),
        "n_cases": int(len(df)),
        "n_ok": int((df["status"] == "ok").sum()) if "status" in df else int(len(df)),
    }
    for m in METRICS_COLS:
        vals = df[m].dropna().to_numpy(dtype=float) if m in df else np.array([], dtype=float)
        rec[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
        rec[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    for pp in ("components_total", "components_retained"):
        vals = df[pp].dropna().to_numpy(dtype=float) if pp in df else np.array([], dtype=float)
        rec[f"{pp}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
    return rec


def run_stage4b_val94_lock(
    config_path: str | Path,
    run_names_filter: list[str] | None = None,
    case_ids_override: list[int] | None = None,
    precomputed_probs_by_run: dict[str, dict[int, tuple[np.ndarray, np.ndarray | None]]] | None = None,
    pairs_override: dict[int, tuple[Path, Path]] | None = None,
    keep_tmp_probs: bool = False,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    out_dir = root_dir / "results" / "val94_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_validated_splits(cfg["study"]["splits_file"])
    val_ids = sorted(int(c) for c in (case_ids_override if case_ids_override is not None else splits["val"]))

    ie = cfg["inference_eval"]
    tau_grid = [round(float(t), 4) for t in ie["tau_grid"]]
    sfg = ie["seeded_filter_grid"]
    ts_vals = [round(float(v), 4) for v in sfg["tau_seed"]]
    msv_vals = [int(v) for v in sfg["min_seed_voxels"]]
    mvol_vals = [float(v) for v in sfg["min_comp_vol_mm3"]]

    jobs = build_stage3_eval_manifest(cfg)
    if run_names_filter:
        allow = set(run_names_filter)
        jobs = [j for j in jobs if j["run_name"] in allow]

    top2_sweep_rows: list[dict[str, Any]] = []
    seeded_sweep_rows: list[dict[str, Any]] = []
    val_summary_rows: list[dict[str, Any]] = []
    per_run_lock: dict[str, Any] = {}
    family_top2_accum: dict[str, dict[float, list[float]]] = {}
    family_seeded_accum: dict[str, dict[tuple[float, float, int, float], list[float]]] = {}

    for job in jobs:
        rname = job["run_name"]
        fam = job["family"]
        pre_probs = precomputed_probs_by_run.get(rname) if precomputed_probs_by_run else None
        if pre_probs is None:
            ckpt_or_dir, nn_name = resolve_run_best_checkpoint(cfg, job)
            if job["kind"] == "monai" and not ckpt_or_dir.is_file():
                raise FileNotFoundError(f"Missing best checkpoint for {rname}: {ckpt_or_dir}")
            if job["kind"] == "nnunet" and not (ckpt_or_dir / f"fold_{cfg['nnunet_v2']['fold']}" / str(nn_name)).is_file():
                raise FileNotFoundError(f"Missing nnU-Net best checkpoint for {rname}: {ckpt_or_dir} / {nn_name}")
            ckpt_sha = (
                sha256_file(ckpt_or_dir)
                if job["kind"] == "monai"
                else sha256_file(ckpt_or_dir / f"fold_{cfg['nnunet_v2']['fold']}" / str(nn_name))
            )
        else:
            ckpt_or_dir, nn_name = root_dir / "checkpoints" / rname / "best_model.pth", "checkpoint_best.pth"
            ckpt_sha = "synthetic_precomputed"

        csv_a = out_dir / f"per_case_{rname}_arm_A.csv"
        csv_c_top2 = out_dir / f"per_case_{rname}_arm_C_top2.csv"
        csv_c_seed = out_dir / f"per_case_{rname}_arm_C_seeded.csv"
        sweep_cache_p = out_dir / f"_sweep_cache_{rname}.json"
        tmp_prob_dir = None if pre_probs is not None else (out_dir / "_tmp_probs" / rname)

        sweep_cache: dict[str, dict[str, Any]] = {}
        if sweep_cache_p.is_file() and sweep_cache_p.stat().st_size > 0:
            try:
                sweep_cache = json.loads(sweep_cache_p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                sweep_cache = {}

        def _on_case_probs(cid_item: int, p1: np.ndarray, _pt: np.ndarray | None, gt_b: np.ndarray, sp_zyx: tuple[float, float, float]):
            key = str(int(cid_item))
            if _has_complete_sweep_entry(sweep_cache.get(key), tau_grid, ts_vals, msv_vals, mvol_vals):
                return
            t2, sd = eval_fair_v2.sweep_case_arm_c_grid(
                p1, gt_b, sp_zyx, tau_grid, ts_vals, msv_vals, mvol_vals
            )
            sweep_cache[key] = {
                "top2": {_top2_key(k): float(v) for k, v in t2.items()},
                "seeded": {
                    _seeded_key(k[0], k[1], k[2], k[3]): float(v)
                    for k, v in sd.items()
                },
            }
            atomic_write_json(sweep_cache, sweep_cache_p)

        arm_a_spec = [
            ArmEvalSpec(
                arm_name="A",
                output_csv=csv_a,
                tau=0.50,
                need_tta=False,
                postprocess_mode="top_k",
                keep_top_k=None,
            )
        ]

        val_id_set = set(val_ids)
        need_sweep = any(
            not _has_complete_sweep_entry(sweep_cache.get(str(cid)), tau_grid, ts_vals, msv_vals, mvol_vals)
            for cid in val_ids
        )
        _, done_a_ids = eval_fair_v2._load_existing_csv_rows(csv_a, resume=True, valid_cids=val_id_set)
        need_arm_a = not (val_id_set <= done_a_ids)
        if need_sweep or need_arm_a:
            if job["kind"] == "monai":
                res_a = eval_fair_v2.evaluate_monai_multi_arm_to_csvs(
                    config_path=config_path,
                    checkpoint_path=ckpt_or_dir,
                    arch=job["arch"],
                    init_filters=job["init_filters"],
                    gate_mode=job.get("gate_mode", "conv_bias"),
                    enable_ds=job.get("enable_ds", True),
                    case_ids=val_ids,
                    arm_specs=arm_a_spec,
                    resume=True,
                    pairs_override=pairs_override,
                    precomputed_probs=pre_probs,
                    prob_cache_dir=tmp_prob_dir,
                    on_case_probs=_on_case_probs if need_sweep else None,
                )
            else:
                res_a = eval_fair_v2.evaluate_nnunet_multi_arm_to_csvs(
                    config_path=config_path,
                    model_folder=ckpt_or_dir,
                    checkpoint_name=str(nn_name),
                    case_ids=val_ids,
                    arm_specs=arm_a_spec,
                    resume=True,
                    pairs_override=pairs_override,
                    precomputed_probs=pre_probs,
                    prob_cache_dir=tmp_prob_dir,
                    on_case_probs=_on_case_probs if need_sweep else None,
                )
            df_a = res_a["A"]
        else:
            df_a = pd.read_csv(csv_a)
            df_a = df_a[df_a["case_id"].isin(val_id_set)].drop_duplicates(subset=["case_id"], keep="last").sort_values("case_id").reset_index(drop=True)

        # Aggregate Top-2 & Seeded sweep means strictly over Val cases
        top2_means: dict[float, float] = {}
        for tau in tau_grid:
            vals = []
            for cid in val_ids:
                v = _get_top2_val(sweep_cache[str(cid)], tau)
                if v is None:
                    raise KeyError(f"Missing top2 sweep score for case {cid}, tau={tau}")
                vals.append(v)
            m_dice = float(np.mean(vals))
            top2_means[tau] = m_dice
            top2_sweep_rows.append({"run_name": rname, "family": fam, "seed": int(job["seed"]), "tau": tau, "val_mean_dice": m_dice})
            family_top2_accum.setdefault(fam, {}).setdefault(tau, []).extend(vals)

        seeded_means: dict[tuple[float, float, int, float], float] = {}
        for tau in tau_grid:
            for ts in ts_vals:
                for msv in msv_vals:
                    for mvol in mvol_vals:
                        vals = []
                        for cid in val_ids:
                            v = _get_seeded_val(sweep_cache[str(cid)], tau, ts, msv, mvol)
                            if v is None:
                                raise KeyError(f"Missing seeded sweep score for case {cid}, params={(tau, ts, msv, mvol)}")
                            vals.append(v)
                        m_dice = float(np.mean(vals))
                        s_key = (tau, ts, msv, mvol)
                        seeded_means[s_key] = m_dice
                        seeded_sweep_rows.append({
                            "run_name": rname,
                            "family": fam,
                            "seed": int(job["seed"]),
                            "tau": tau,
                            "tau_seed": ts,
                            "min_seed_voxels": msv,
                            "min_comp_vol_mm3": mvol,
                            "val_mean_dice": m_dice,
                        })
                        family_seeded_accum.setdefault(fam, {}).setdefault(s_key, []).extend(vals)

        best_tau_t2, best_dice_t2 = _pick_best_top2(top2_means)
        (b_tau_s, b_ts, b_msv, b_mvol), best_dice_seed = _pick_best_seeded(seeded_means)

        per_run_lock[rname] = {
            "family": fam,
            "kind": job["kind"],
            "arch": job["arch"],
            "init_filters": int(job["init_filters"]),
            "seed": int(job["seed"]),
            "ckpt_sha256": ckpt_sha,
            "top2": {"tau": float(best_tau_t2), "keep_top_k": 2, "val_mean_dice": float(best_dice_t2)},
            "seeded": {
                "tau": float(b_tau_s),
                "tau_seed": float(b_ts),
                "min_seed_voxels": int(b_msv),
                "min_comp_vol_mm3": float(b_mvol),
                "val_mean_dice": float(best_dice_seed),
            },
        }

        # Reuse cached probability volumes for final Arm C-Top2 & Arm C-Seeded surface-v1 scoring
        arm_c_specs = [
            ArmEvalSpec(
                arm_name="C_top2",
                output_csv=csv_c_top2,
                tau=float(best_tau_t2),
                need_tta=False,
                postprocess_mode="top_k",
                keep_top_k=2,
            ),
            ArmEvalSpec(
                arm_name="C_seeded",
                output_csv=csv_c_seed,
                tau=float(b_tau_s),
                need_tta=False,
                postprocess_mode="seeded",
                keep_top_k=None,
                tau_seed=float(b_ts),
                min_seed_voxels=int(b_msv),
                min_comp_vol_mm3=float(b_mvol),
            ),
        ]
        if job["kind"] == "monai":
            res_c = eval_fair_v2.evaluate_monai_multi_arm_to_csvs(
                config_path=config_path,
                checkpoint_path=ckpt_or_dir,
                arch=job["arch"],
                init_filters=job["init_filters"],
                gate_mode=job.get("gate_mode", "conv_bias"),
                enable_ds=job.get("enable_ds", True),
                case_ids=val_ids,
                arm_specs=arm_c_specs,
                resume=True,
                pairs_override=pairs_override,
                precomputed_probs=pre_probs,
                prob_cache_dir=tmp_prob_dir,
            )
        else:
            res_c = eval_fair_v2.evaluate_nnunet_multi_arm_to_csvs(
                config_path=config_path,
                model_folder=ckpt_or_dir,
                checkpoint_name=str(nn_name),
                case_ids=val_ids,
                arm_specs=arm_c_specs,
                resume=True,
                pairs_override=pairs_override,
                precomputed_probs=pre_probs,
                prob_cache_dir=tmp_prob_dir,
            )

        val_summary_rows.append(_summarize_val_df(df_a, rname, fam, "A", int(job["seed"])))
        val_summary_rows.append(_summarize_val_df(res_c["C_top2"], rname, fam, "C_top2", int(job["seed"])))
        val_summary_rows.append(_summarize_val_df(res_c["C_seeded"], rname, fam, "C_seeded", int(job["seed"])))

        # Remove temporary float32 probabilities once all Val94 CSVs for this run are flushed
        if tmp_prob_dir is not None and tmp_prob_dir.exists() and not keep_tmp_probs:
            shutil.rmtree(tmp_prob_dir, ignore_errors=True)

    # Also compute family-pooled consensus thresholds across seeds
    per_family_lock: dict[str, Any] = {}
    for fam in sorted(family_top2_accum.keys()):
        f_t2 = {t: float(np.mean(vals)) for t, vals in family_top2_accum[fam].items()}
        f_sd = {k: float(np.mean(vals)) for k, vals in family_seeded_accum[fam].items()}
        ft_tau, ft_dice = _pick_best_top2(f_t2)
        (fs_tau, fs_ts, fs_msv, fs_mvol), fs_dice = _pick_best_seeded(f_sd)
        per_family_lock[fam] = {
            "top2": {"tau": float(ft_tau), "keep_top_k": 2, "val_pooled_mean_dice": float(ft_dice)},
            "seeded": {
                "tau": float(fs_tau),
                "tau_seed": float(fs_ts),
                "min_seed_voxels": int(fs_msv),
                "min_comp_vol_mm3": float(fs_mvol),
                "val_pooled_mean_dice": float(fs_dice),
            },
        }

    atomic_write_csv(pd.DataFrame(top2_sweep_rows), out_dir / "val94_grid_sweep_top2.csv")
    atomic_write_csv(pd.DataFrame(seeded_sweep_rows), out_dir / "val94_grid_sweep_seeded.csv")
    atomic_write_csv(pd.DataFrame(val_summary_rows), out_dir / "val94_summary.csv")

    thresholds_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": "val",
        "case_ids": val_ids,
        "tau_grid": tau_grid,
        "seeded_filter_grid": {
            "tau_seed": ts_vals,
            "min_seed_voxels": msv_vals,
            "min_comp_vol_mm3": mvol_vals,
        },
        "objective": "mean_val_dice",
        "tie_breaker": "closest_to_0.50_then_lower",
        "per_run": per_run_lock,
        "per_family": per_family_lock,
    }
    thresh_path = out_dir / "thresholds_v2.json"
    atomic_write_json(thresholds_payload, thresh_path, sort_keys=True)
    thresholds_payload["sha256"] = hashlib.sha256(thresh_path.read_bytes()).hexdigest()
    print(f"[Stage 4B] Frozen thresholds_v2.json (sha256={thresholds_payload['sha256'][:12]}...) for {len(per_run_lock)} runs")
    return thresholds_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B Val N=94 Threshold & Seeded-Filter Lock")
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--keep-tmp-probs", action="store_true")
    args = parser.parse_args()
    run_stage4b_val94_lock(args.config, run_names_filter=args.runs, keep_tmp_probs=args.keep_tmp_probs)


if __name__ == "__main__":
    main()
