"""
Stage 2A & 2B Validation-Only (Val N=94) Pilot Evaluation and Pre-Registered Design Lock.
Strictly evaluates on splits_final.json["val"] (zero access to IDs 851-1000).
"""
from __future__ import annotations

import argparse
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
from study_registry import (
    REQUIRED_LR_KEYS,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    build_stage2a_jobs,
    build_stage2b_candidates,
    df_to_markdown,
    lr_tag,
    sha256_file,
    validate_locked_lrs,
)

METRICS_COLS = ("dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall")


def _summarize_df(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_cases": int(len(df)),
        "n_ok": int((df["status"] == "ok").sum()) if "status" in df else int(len(df)),
    }
    for m in METRICS_COLS:
        vals = df[m].dropna().to_numpy(dtype=float) if m in df else np.array([], dtype=float)
        out[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
        out[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    for pp in ("components_total", "components_retained"):
        vals = df[pp].dropna().to_numpy(dtype=float) if pp in df else np.array([], dtype=float)
        out[f"{pp}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
    return out


LOCKED_TEST_IDS = set(range(851, 1001))


def guard_val_ids(ids) -> list[int]:
    ids = [int(c) for c in ids]
    bad = sorted(set(ids) & LOCKED_TEST_IDS)
    if bad:
        raise ValueError(f"Locked test IDs 851-1000 passed to a validation-only stage: {bad[:5]}")
    return ids


def gate_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Gate inputs where an empty prediction can never help a candidate."""
    status = df["status"] if "status" in df else pd.Series(["ok"] * len(df))
    n_empty = int((status == "prediction_empty").sum())
    prec_strict = float(df["precision"].fillna(0.0).mean()) if len(df) else float("nan")
    comps = float(df["components_total"].mean()) if "components_total" in df and len(df) else float("nan")
    return {"n_empty_pred": n_empty, "precision_mean_strict": prec_strict, "components_total_mean": comps}


def candidate_qualifies(g: dict, seg_components: float) -> bool:
    return bool(g["n_empty_pred"] == 0
                and g["precision_mean_strict"] >= 0.8000
                and g["components_total_mean"] <= float(seg_components) + 1e-9)


def _resolve_pilot_ckpt(root_dir: Path, run_name: str, pilot_epochs: int, ckpt_kind: str) -> Path:
    ckpt_dir = root_dir / "checkpoints" / run_name
    if ckpt_kind == "best":
        p = ckpt_dir / "best_model.pth"
    else:
        p = ckpt_dir / f"epoch_{int(pilot_epochs)}.pth"
        if not p.is_file():
            p = ckpt_dir / "best_model.pth"
    if not p.is_file():
        raise FileNotFoundError(f"Missing pilot checkpoint for {run_name}: {p}")
    return p


def evaluate_stage2a_pilots(
    config_path: str | Path,
    ckpt_kind: str = "epoch_50",
    case_ids_override: list[int] | None = None,
    precomputed_probs_by_run: dict[str, dict[int, tuple[np.ndarray, np.ndarray | None]]] | None = None,
    pairs_override: dict[int, tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    out_dir = root_dir / "results" / "pilots_val94"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_validated_splits(cfg["study"]["splits_file"])
    val_ids = guard_val_ids(case_ids_override or sorted(splits["val"]))
    pilot_ep = int(cfg["training_budget"]["pilot_epochs"])
    jobs = build_stage2a_jobs(cfg)

    rows: list[dict[str, Any]] = []
    for job in jobs:
        rname = job["run_name"]
        csv_p = out_dir / f"per_case_{rname}.csv"
        pre_probs = precomputed_probs_by_run.get(rname) if precomputed_probs_by_run else None
        if pre_probs is not None:
            ckpt_p = root_dir / "checkpoints" / rname / "synthetic.pth"
            ckpt_sha = "synthetic_precomputed"
        else:
            ckpt_p = _resolve_pilot_ckpt(root_dir, rname, pilot_ep, ckpt_kind)
            ckpt_sha = sha256_file(ckpt_p)

        df = eval_fair_v2.evaluate_monai_checkpoint_to_csv(
            config_path=config_path,
            checkpoint_path=ckpt_p,
            arch=job["arch"],
            init_filters=job["init_filters"],
            gate_mode=job["gate_mode"],
            enable_ds=job["enable_ds"],
            case_ids=val_ids,
            output_csv=csv_p,
            tau=0.50,
            need_tta=False,
            postprocess_mode="top_k",
            keep_top_k=None,
            resume=True,
            pairs_override=pairs_override,
            precomputed_probs=pre_probs,
        )
        summ = _summarize_df(df)
        rows.append({
            "run_name": rname,
            "lr_key": job["lr_key"],
            "arch": job["arch"],
            "init_filters": int(job["init_filters"]),
            "lr": float(job["lr"]),
            "ckpt_path": str(ckpt_p),
            "ckpt_sha256": ckpt_sha,
            "per_case_csv": str(csv_p),
            **summ,
        })

    df_sum = pd.DataFrame(rows)
    atomic_write_csv(df_sum, out_dir / "stage2a_summary.csv")

    locked_lrs: dict[str, float] = {}
    details: dict[str, Any] = {}
    for lr_key in REQUIRED_LR_KEYS:
        sub = df_sum[df_sum["lr_key"] == lr_key].copy()
        if sub.empty:
            raise RuntimeError(f"No Stage 2A rows found for required lr_key={lr_key!r}")
        # Tie-breaker: highest val94 dice, then closest to 3e-4, then lower LR
        sub["tie_dist"] = (sub["lr"] - 3e-4).abs()
        sub = sub.sort_values(by=["dice_mean", "tie_dist", "lr"], ascending=[False, True, True]).reset_index(drop=True)
        win = sub.iloc[0]
        locked_lrs[lr_key] = float(win["lr"])
        details[lr_key] = {
            "lr": float(win["lr"]),
            "winning_run_name": str(win["run_name"]),
            "val94_dice": float(win["dice_mean"]),
            "val94_precision": float(win["precision_mean"]),
            "val94_hd95": float(win["hd95_mean"]),
            "val94_components_total": float(win["components_total_mean"]),
            "ckpt_sha256": str(win["ckpt_sha256"]),
        }

    validate_locked_lrs(locked_lrs)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": "val",
        "n_cases": len(val_ids),
        "ckpt_kind": ckpt_kind,
        "locked_lrs": locked_lrs,
        "details": details,
    }
    atomic_write_json(payload, out_dir / "locked_lrs.json")
    print(f"[Stage 2A] Locked LRs: {json.dumps(locked_lrs)}")
    return payload


def evaluate_stage2b_pilots(
    config_path: str | Path,
    locked_lrs_path: str | Path | None = None,
    ckpt_kind: str = "epoch_50",
    case_ids_override: list[int] | None = None,
    precomputed_probs_by_run: dict[str, dict[int, tuple[np.ndarray, np.ndarray | None]]] | None = None,
    pairs_override: dict[int, tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    out_dir = root_dir / "results" / "pilots_val94"
    out_dir.mkdir(parents=True, exist_ok=True)

    lrs_p = Path(locked_lrs_path) if locked_lrs_path else (out_dir / "locked_lrs.json")
    if not lrs_p.is_file():
        raise FileNotFoundError(f"Missing locked_lrs.json: {lrs_p}")
    lrs_obj = json.loads(lrs_p.read_text(encoding="utf-8"))
    locked_lrs = validate_locked_lrs(lrs_obj["locked_lrs"] if "locked_lrs" in lrs_obj else lrs_obj)

    splits = load_validated_splits(cfg["study"]["splits_file"])
    val_ids = guard_val_ids(case_ids_override or sorted(splits["val"]))
    pilot_ep = int(cfg["training_budget"]["pilot_epochs"])

    # Reference matched-width SegResNet-v2 fragmentation baselines from Stage 2A
    seg_ref_csvs: dict[int, Path] = {
        16: out_dir / f"per_case_pilot2a_segresnet_f16_lr{lr_tag(locked_lrs['segresnet_f16'])}_seed42.csv",
        32: out_dir / f"per_case_pilot2a_segresnet_f32_lr{lr_tag(locked_lrs['segresnet_f32'])}_seed42.csv",
    }
    seg_ref_stats: dict[int, dict[str, Any]] = {}
    val_id_set = set(val_ids)
    for width, spath in seg_ref_csvs.items():
        if not spath.is_file() or spath.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty Stage 2A SegResNet F={width} eval CSV: {spath}")
        df_ref = pd.read_csv(spath)
        if df_ref.empty or "case_id" not in df_ref.columns:
            raise RuntimeError(f"Empty/corrupt Stage 2A SegResNet F={width} eval CSV: {spath}")
        df_ref = df_ref[df_ref["case_id"].isin(val_id_set)].drop_duplicates(subset=["case_id"], keep="last")
        if set(df_ref["case_id"].astype(int).tolist()) != val_id_set:
            raise RuntimeError(f"Stage 2A SegResNet F={width} CSV missing val_ids: {spath}")
        seg_ref_stats[width] = _summarize_df(df_ref)

    candidates = build_stage2b_candidates(cfg, locked_lrs)
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        rname = cand["run_name"]
        dedup_src = cand.get("dedup_source_run")
        own_done = (root_dir / "checkpoints" / rname / "DONE.flag").is_file()
        resolved_run = rname if (own_done or not dedup_src or (precomputed_probs_by_run and rname in precomputed_probs_by_run)) else str(dedup_src)
        csv_p = out_dir / f"per_case_{resolved_run}.csv"

        df_cached = None
        if csv_p.is_file() and csv_p.stat().st_size > 0:
            try:
                tmp_df = pd.read_csv(csv_p)
                if not tmp_df.empty and "case_id" in tmp_df.columns:
                    tmp_df = tmp_df[tmp_df["case_id"].notna()].copy()
                    tmp_df["case_id"] = tmp_df["case_id"].astype(int)
                    if val_id_set <= set(tmp_df["case_id"].tolist()):
                        df_cached = tmp_df[tmp_df["case_id"].isin(val_id_set)].drop_duplicates(subset=["case_id"], keep="last")
            except pd.errors.EmptyDataError:
                df_cached = None

        if df_cached is not None:
            df = df_cached
            try:
                ckpt_p = _resolve_pilot_ckpt(root_dir, resolved_run, pilot_ep, ckpt_kind)
                ckpt_sha = sha256_file(ckpt_p)
            except FileNotFoundError:
                ckpt_sha = "cached_csv"
        else:
            pre_probs = None
            if precomputed_probs_by_run:
                pre_probs = precomputed_probs_by_run.get(resolved_run) or precomputed_probs_by_run.get(rname)
            if pre_probs is not None:
                ckpt_p = root_dir / "checkpoints" / resolved_run / "synthetic.pth"
                ckpt_sha = "synthetic_precomputed"
            else:
                ckpt_p = _resolve_pilot_ckpt(root_dir, resolved_run, pilot_ep, ckpt_kind)
                ckpt_sha = sha256_file(ckpt_p)
            df = eval_fair_v2.evaluate_monai_checkpoint_to_csv(
                config_path=config_path,
                checkpoint_path=ckpt_p,
                arch=cand["arch"],
                init_filters=cand["init_filters"],
                gate_mode=cand["gate_mode"],
                enable_ds=cand["enable_ds"],
                case_ids=val_ids,
                output_csv=csv_p,
                tau=0.50,
                need_tta=False,
                postprocess_mode="top_k",
                keep_top_k=None,
                resume=True,
                pairs_override=pairs_override,
                precomputed_probs=pre_probs,
            )

        summ = _summarize_df(df)
        w = int(cand["init_filters"])
        seg_comp = float(seg_ref_stats[w]["components_total_mean"])
        seg_dice = float(seg_ref_stats[w]["dice_mean"])
        seg_prec = float(seg_ref_stats[w]["precision_mean"])

        g = gate_metrics(df)
        pass_prec = bool(g["precision_mean_strict"] >= 0.8000)
        pass_frag = bool(g["components_total_mean"] <= seg_comp + 1e-9)
        qualifies = candidate_qualifies(g, seg_comp)

        rows.append({
            "candidate_name": rname,
            "resolved_run_name": resolved_run,
            "dedup_from_stage2a": resolved_run != rname,
            "init_filters": w,
            "lr": float(cand["lr"]),
            "gate_mode": cand["gate_mode"],
            "w_cldice": float(cand["w_cldice"]),
            "segresnet_ref_dice": seg_dice,
            "segresnet_ref_precision": seg_prec,
            "segresnet_ref_components_total": seg_comp,
            "pass_precision_ge_080": pass_prec,
            "pass_frag_le_segresnet": pass_frag,
            "qualifies": qualifies,
            "ckpt_sha256": ckpt_sha,
            **summ,
            **g,
        })

    df_2b = pd.DataFrame(rows)
    atomic_write_csv(df_2b, out_dir / "stage2b_summary.csv")

    qual_df = df_2b[df_2b["qualifies"]].copy()
    if not qual_df.empty:
        winner = qual_df.sort_values(by=["dice_mean", "precision_mean", "init_filters"], ascending=[False, False, True]).iloc[0]
        all_gates_passed = True
    else:
        winner = df_2b.sort_values(by=["dice_mean", "precision_mean", "init_filters"], ascending=[False, False, True]).iloc[0]
        all_gates_passed = False

    prereg_path = root_dir / "PREREGISTRATION_V2.md"
    decision = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": "val",
        "n_cases": len(val_ids),
        "all_gates_passed": all_gates_passed,
        "locked_design": {
            "candidate_name": str(winner["candidate_name"]),
            "resolved_run_name": str(winner["resolved_run_name"]),
            "init_filters": int(winner["init_filters"]),
            "gate_mode": str(winner["gate_mode"]),
            "w_cldice": float(winner["w_cldice"]),
            "val94_dice": float(winner["dice_mean"]),
            "val94_precision": float(winner["precision_mean"]),
            "val94_hd95": float(winner["hd95_mean"]),
            "val94_components_total": float(winner["components_total_mean"]),
            "segresnet_ref_components_total": float(winner["segresnet_ref_components_total"]),
            "pass_precision_ge_080": bool(winner["pass_precision_ge_080"]),
            "pass_frag_le_segresnet": bool(winner["pass_frag_le_segresnet"]),
        },
        "locked_lrs_sha256": sha256_file(lrs_p),
        "preregistration_sha256": sha256_file(prereg_path) if prereg_path.is_file() else "missing",
        "stage2b_candidates": rows,
    }
    atomic_write_json(decision, out_dir / "pilot_lock_decision.json")

    s2a_csv = out_dir / "stage2a_summary.csv"
    df_2a = pd.read_csv(s2a_csv) if (s2a_csv.is_file() and s2a_csv.stat().st_size > 0) else pd.DataFrame()
    md_lines = [
        "# Stage 2 Validation-Only Pilot Lock Decision (`Val N = 94`)",
        "",
        f"- **Timestamp**: `{decision['timestamp']}`",
        f"- **Validation Cases Evaluated**: `N = {len(val_ids)}` (`IDs 851–1000` strictly untouched)",
        f"- **Locked LRs (`locked_lrs.json`, SHA-256 `{decision['locked_lrs_sha256'][:12]}...`)**: `{json.dumps(locked_lrs)}`",
        f"- **Gate Status (`Val Precision >= 0.8000` & `<=` matched `SegResNet-v2` component fragmentation)**: **`{'PASS' if all_gates_passed else 'FALLBACK (NO CANDIDATE PASSED BOTH GATES)'}`**",
        f"- **Locked Lead Design**: `F*={winner['init_filters']}`, `gate_mode={winner['gate_mode']}`, `w_cldice={winner['w_cldice']}` "
        f"(Val Dice `{winner['dice_mean']:.4f}`, Precision `{winner['precision_mean']:.4f}`, Components `{winner['components_total_mean']:.2f}` vs SegResNet `{winner['segresnet_ref_components_total']:.2f}`)",
        "",
        "## Stage 2B Design Sweep (`Val N = 94`, 0.50 Gauss, 1-Pass, τ = 0.50)",
        "",
        df_to_markdown(df_2b[[
            "candidate_name", "init_filters", "gate_mode", "w_cldice",
            "dice_mean", "precision_mean", "recall_mean", "hd95_mean", "cldice_mean",
            "components_total_mean", "segresnet_ref_components_total",
            "pass_precision_ge_080", "pass_frag_le_segresnet", "qualifies",
        ]]),
        "",
    ]
    if not df_2a.empty:
        md_lines.extend([
            "## Stage 2A Learning-Rate Sweep (`Val N = 94`)",
            "",
            df_to_markdown(df_2a[["run_name", "lr_key", "arch", "init_filters", "lr", "dice_mean", "precision_mean", "hd95_mean", "components_total_mean"]]),
            "",
        ])
    atomic_write_text("\n".join(md_lines), out_dir / "pilot_lock_decision.md")
    print(f"[Stage 2B] Locked Design: {json.dumps(decision['locked_design'])}")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2A/2B Val N=94 Pilot Evaluation & Lock")
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument("--phase", required=True, choices=["stage2a", "stage2b"])
    parser.add_argument("--ckpt-kind", default="epoch_50", choices=["epoch_50", "best"])
    parser.add_argument("--locked-lrs-json", default=None)
    args = parser.parse_args()
    if args.phase == "stage2a":
        evaluate_stage2a_pilots(args.config, ckpt_kind=args.ckpt_kind)
    else:
        evaluate_stage2b_pilots(args.config, locked_lrs_path=args.locked_lrs_json, ckpt_kind=args.ckpt_kind)


if __name__ == "__main__":
    main()
