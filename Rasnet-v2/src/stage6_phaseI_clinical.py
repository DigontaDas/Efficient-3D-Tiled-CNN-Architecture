"""
Stage 6: Phase I Prospective Clinical Cohort Confirmatory Evaluator (`stage6_phaseI_clinical.py`).
Strictly single-shot evaluation using frozen Stage 5 MANIFEST_V2_LOCKED.json and
frozen Val N=94 thresholds_v2.json on manually doctor-annotated primary clinical CT scans
(zero post-arrival tuning).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_pairs import check_geometry
import eval_fair_v2
from eval_fair_v2 import ArmEvalSpec, bootstrap_ci
from study_registry import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    df_to_markdown,
    sha256_file,
)

METRICS_COLS = ("dice", "iou", "precision", "recall", "hd95", "asd", "cldice", "centerline_recall")
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


def _read_header_geom(path: Path) -> dict[str, Any]:
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return {
        "size": r.GetSize(),
        "origin": r.GetOrigin(),
        "spacing": r.GetSpacing(),
        "direction": r.GetDirection(),
    }


def discover_clinical_pairs(
    cohort_dir: str | Path | None = None,
    manifest_csv: str | Path | None = None,
    check_headers: bool = True,
) -> tuple[dict[int, tuple[Path, Path]], dict[int, str]]:
    """
    Discover (image_path, doctor_gt_label_path) pairs from either an explicit CSV manifest
    (`case_id,image_path,label_path`) or a directory layout (`*.img.nii.gz`/`*.label.nii.gz`,
    `imagesTr`/`labelsTr`, or `images`/`labels`). Verifies physical header geometry alignment.
    """
    pairs: dict[int, tuple[Path, Path]] = {}
    case_labels: dict[int, str] = {}

    if manifest_csv is not None:
        df = pd.read_csv(manifest_csv)
        for req in ("case_id", "image_path", "label_path"):
            if req not in df.columns:
                raise ValueError(f"Manifest CSV missing required column {req!r}")
        seen_raw_ids: set[str] = set()
        for idx, row in df.iterrows():
            raw_id = str(row["case_id"])
            if raw_id in seen_raw_ids:
                raise ValueError(f"Duplicate case_id {raw_id!r} in manifest CSV: {manifest_csv}")
            seen_raw_ids.add(raw_id)
            try:
                cid = int(row["case_id"])
            except Exception:
                cid = int(idx) + 1
            while cid in pairs:
                cid += 100000
            im_p, lb_p = Path(str(row["image_path"])), Path(str(row["label_path"]))
            if not im_p.is_file() or not lb_p.is_file():
                raise FileNotFoundError(f"Missing clinical pair for {raw_id}: {im_p}, {lb_p}")
            if check_headers:
                check_geometry(_read_header_geom(im_p), _read_header_geom(lb_p))
            pairs[cid] = (im_p, lb_p)
            case_labels[cid] = raw_id
        if not pairs:
            raise ValueError(f"No clinical pairs in manifest: {manifest_csv}")
        return pairs, case_labels

    if cohort_dir is None:
        raise ValueError("Either --cohort-dir or --manifest-csv must be provided")
    cdir = Path(cohort_dir)
    if not cdir.is_dir():
        raise FileNotFoundError(f"Cohort directory does not exist: {cdir}")

    # Layout 1: ImageCAS-style {stem}.img.nii.gz & {stem}.label.nii.gz
    img_files = sorted(cdir.rglob("*.img.nii.gz"))
    if img_files:
        for idx, im_p in enumerate(img_files, 1):
            stem = im_p.name[: -len(".img.nii.gz")]
            lb_p = im_p.with_name(f"{stem}.label.nii.gz")
            if not lb_p.is_file():
                raise FileNotFoundError(f"Missing label for {im_p}: {lb_p}")
            m = re.search(r"(\d+)$", stem)
            cid = int(m.group(1)) if m else idx
            while cid in pairs:
                cid += 100000
            if check_headers:
                check_geometry(_read_header_geom(im_p), _read_header_geom(lb_p))
            pairs[cid] = (im_p, lb_p)
            case_labels[cid] = stem
        return pairs, case_labels

    # Layout 2: imagesTr/labelsTr or images/labels
    for im_sub, lb_sub in (("imagesTr", "labelsTr"), ("images", "labels")):
        id_dir, ld_dir = cdir / im_sub, cdir / lb_sub
        if id_dir.is_dir() and ld_dir.is_dir():
            for idx, im_p in enumerate(sorted(id_dir.glob("*.nii.gz")), 1):
                stem = im_p.name[: -len(".nii.gz")]
                base = stem[:-5] if stem.endswith("_0000") else stem
                lb_p = ld_dir / f"{base}.nii.gz"
                if not lb_p.is_file():
                    raise FileNotFoundError(f"Missing label for {im_p}: {lb_p}")
                m = re.search(r"(\d+)$", base)
                cid = int(m.group(1)) if m else idx
                while cid in pairs:
                    cid += 100000
                if check_headers:
                    check_geometry(_read_header_geom(im_p), _read_header_geom(lb_p))
                pairs[cid] = (im_p, lb_p)
                case_labels[cid] = base
            if pairs:
                return pairs, case_labels

    raise FileNotFoundError(f"No recognized .nii.gz image/label pairs found inside {cdir}")


def run_stage6_phaseI(
    config_path: str | Path,
    cohort_dir: str | Path | None = None,
    manifest_csv: str | Path | None = None,
    manifest_locked_path: str | Path | None = None,
    verify_hashes: bool = True,
    resume: bool = True,
    precomputed_probs_by_role: dict[str, dict[int, tuple[np.ndarray, np.ndarray | None]]] | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    out_dir = root_dir / "results" / "phaseI_primary_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    complete_flag = out_dir / "PHASE_I_COMPLETE.flag"
    if complete_flag.is_file():
        raise RuntimeError(
            f"Phase I single-shot evaluation is already complete ({complete_flag}). "
            "Refusing to re-evaluate or tune on prospective cohort."
        )

    lock_p = Path(manifest_locked_path) if manifest_locked_path else (root_dir / "results" / "stats_and_efficiency" / "MANIFEST_V2_LOCKED.json")
    if not lock_p.is_file():
        raise FileNotFoundError(f"Missing MANIFEST_V2_LOCKED.json: {lock_p}. Complete Stage 5 first.")
    manifest = json.loads(lock_p.read_text(encoding="utf-8"))

    thresh_p = Path(manifest["thresholds_v2_path"])
    if not thresh_p.is_file():
        raise FileNotFoundError(f"Missing frozen thresholds_v2.json: {thresh_p}")
    thresh_sha_actual = sha256_file(thresh_p)
    if verify_hashes and thresh_sha_actual != manifest["thresholds_v2_sha256"]:
        raise RuntimeError(
            f"thresholds_v2.json SHA-256 mismatch: expected {manifest['thresholds_v2_sha256']}, got {thresh_sha_actual}"
        )
    thresh_data = json.loads(thresh_p.read_text(encoding="utf-8"))

    pairs, case_labels = discover_clinical_pairs(cohort_dir=cohort_dir, manifest_csv=manifest_csv, check_headers=True)
    case_ids = sorted(pairs.keys())
    atomic_write_json(
        {"n_cases": len(case_ids), "cases": {str(c): {"label": case_labels[c], "image": str(pairs[c][0]), "gt": str(pairs[c][1])} for c in case_ids}},
        out_dir / "phaseI_cohort_manifest.json",
    )

    roles = ("rasnet_v2", "segresnet_v2", "nnunet_v2")
    role_arm_dfs: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for role in roles:
        role_info = manifest["locked_models"][role]
        model_meta = role_info["canonical"]
        rname = model_meta["run_name"]
        fam = role_info["family"]

        if rname in thresh_data.get("per_run", {}):
            t_entry = thresh_data["per_run"][rname]
        elif fam in thresh_data.get("per_family", {}):
            t_entry = thresh_data["per_family"][fam]
        else:
            raise KeyError(f"No frozen thresholds entry for role={role} (run={rname}, family={fam})")

        tau_t2 = float(t_entry["top2"]["tau"])
        s_cfg = t_entry["seeded"]
        pre_probs = precomputed_probs_by_role.get(role) if precomputed_probs_by_role else None

        ckpt_or_dir = Path(model_meta["checkpoint_or_model_dir"])
        nn_ckpt_name = model_meta.get("nnunet_checkpoint_name")
        if pre_probs is None and verify_hashes:
            ckpt_file = ckpt_or_dir if model_meta["kind"] == "monai" else (ckpt_or_dir / f"fold_{cfg['nnunet_v2']['fold']}" / str(nn_ckpt_name))
            actual_sha = sha256_file(ckpt_file)
            if actual_sha != model_meta["checkpoint_sha256"]:
                raise RuntimeError(
                    f"Checkpoint SHA-256 mismatch for {role} ({rname}): expected {model_meta['checkpoint_sha256']}, got {actual_sha}"
                )

        arm_specs = [
            ArmEvalSpec(arm_name="A", output_csv=out_dir / f"per_case_{role}_arm_A.csv", tau=0.50, need_tta=False, postprocess_mode="top_k", keep_top_k=None),
            ArmEvalSpec(arm_name="B", output_csv=out_dir / f"per_case_{role}_arm_B.csv", tau=0.50, need_tta=True, postprocess_mode="top_k", keep_top_k=None),
            ArmEvalSpec(arm_name="C_top2", output_csv=out_dir / f"per_case_{role}_arm_C_top2.csv", tau=tau_t2, need_tta=False, postprocess_mode="top_k", keep_top_k=2),
            ArmEvalSpec(
                arm_name="C_seeded",
                output_csv=out_dir / f"per_case_{role}_arm_C_seeded.csv",
                tau=float(s_cfg["tau"]),
                need_tta=False,
                postprocess_mode="seeded",
                keep_top_k=None,
                tau_seed=float(s_cfg["tau_seed"]),
                min_seed_voxels=int(s_cfg["min_seed_voxels"]),
                min_comp_vol_mm3=float(s_cfg["min_comp_vol_mm3"]),
            ),
        ]

        if model_meta["kind"] == "monai":
            arm_dfs = eval_fair_v2.evaluate_monai_multi_arm_to_csvs(
                config_path=config_path,
                checkpoint_path=ckpt_or_dir,
                arch=model_meta["arch"],
                init_filters=int(model_meta["init_filters"]),
                gate_mode=model_meta.get("gate_mode", "conv_bias"),
                enable_ds=bool(model_meta.get("enable_ds", True)),
                case_ids=case_ids,
                arm_specs=arm_specs,
                resume=resume,
                pairs_override=pairs,
                precomputed_probs=pre_probs,
            )
        else:
            arm_dfs = eval_fair_v2.evaluate_nnunet_multi_arm_to_csvs(
                config_path=config_path,
                model_folder=ckpt_or_dir,
                checkpoint_name=str(nn_ckpt_name),
                case_ids=case_ids,
                arm_specs=arm_specs,
                resume=resume,
                pairs_override=pairs,
                precomputed_probs=pre_probs,
            )

        for arm_k, df_arm in arm_dfs.items():
            role_arm_dfs[(role, arm_k)] = df_arm
            rec: dict[str, Any] = {"role": role, "run_name": rname, "family": fam, "arm": arm_k, "n_cases": int(len(df_arm))}
            for m in METRICS_COLS:
                vals = df_arm[m].dropna().to_numpy(dtype=float) if m in df_arm else np.array([], dtype=float)
                if len(vals):
                    ci_lo, ci_hi = bootstrap_ci(vals, seed=42, n_boot=2000, confidence=0.95)
                    rec[f"{m}_mean"] = float(np.mean(vals))
                    rec[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    rec[f"{m}_ci95_lo"] = ci_lo
                    rec[f"{m}_ci95_hi"] = ci_hi
                else:
                    rec[f"{m}_mean"] = float("nan")
                    rec[f"{m}_std"] = float("nan")
                    rec[f"{m}_ci95_lo"] = float("nan")
                    rec[f"{m}_ci95_hi"] = float("nan")
            summary_rows.append(rec)

    df_sum = pd.DataFrame(summary_rows)
    atomic_write_csv(df_sum, out_dir / "phaseI_summary.csv")

    # Prospective paired statistical tests (Family 1 K=4 vs SegResNet-v2; Family 2 K=3 + NI vs nnU-Net V2)
    stat_rows: list[dict[str, Any]] = []
    for arm_k in ("A", "B", "C_top2", "C_seeded"):
        df_ras = role_arm_dfs[("rasnet_v2", arm_k)]
        df_seg = role_arm_dfs[("segresnet_v2", arm_k)]
        df_nn = role_arm_dfs[("nnunet_v2", arm_k)]

        f1_res = []
        for m in ("dice", "cldice", "precision", "hd95"):
            st = eval_fair_v2.paired_with_noninferiority(df_ras, df_seg, metric=m, allow_nan_filtering=True, direction=METRIC_DIRECTIONS[m])
            st.update({"family_id": "PhaseI_Family1_K4", "contrast": "rasnet_v2_vs_segresnet_v2", "arm": arm_k, "metric": m})
            f1_res.append(st)
        for r, hp in zip(f1_res, eval_fair_v2.holm_adjust([x["p_wilcoxon"] for x in f1_res])):
            r["p_holm"] = float(hp)
            stat_rows.append(r)

        f2_res = []
        for m in ("precision", "hd95", "cldice"):
            st = eval_fair_v2.paired_with_noninferiority(df_ras, df_nn, metric=m, allow_nan_filtering=True, direction=METRIC_DIRECTIONS[m])
            st.update({"family_id": "PhaseI_Family2_K3", "contrast": "rasnet_v2_vs_nnunet_v2", "arm": arm_k, "metric": m})
            f2_res.append(st)
        for r, hp in zip(f2_res, eval_fair_v2.holm_adjust([x["p_wilcoxon"] for x in f2_res])):
            r["p_holm"] = float(hp)
            stat_rows.append(r)

        ni_st = eval_fair_v2.paired_with_noninferiority(df_ras, df_nn, metric="dice", ni_margin=-0.0050, allow_nan_filtering=True, direction="higher")
        ni_st.update({"family_id": "PhaseI_NonInferiority_Dice", "contrast": "rasnet_v2_vs_nnunet_v2", "arm": arm_k, "metric": "dice", "p_holm": float(ni_st["p_wilcoxon"])})
        stat_rows.append(ni_st)

    df_stats = pd.DataFrame(stat_rows)
    atomic_write_csv(df_stats, out_dir / "phaseI_paired_stats.csv")

    report_md = [
        "# Phase I Prospective Clinical Confirmatory Report (Single-Shot)",
        "",
        f"- **Evaluated**: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
        f"- **Prospective Cases**: `N = {len(case_ids)}` (manually annotated by physicians; zero model-assisted draft masks)",
        f"- **Locked Manifest**: `{lock_p}` (`thresholds_v2.json` SHA-256 `{thresh_sha_actual[:12]}...`)",
        "",
        "## Summary Metrics Across Locked Models & Arms",
        "",
        df_to_markdown(df_sum[["role", "arm", "n_cases", "dice_mean", "precision_mean", "recall_mean", "hd95_mean", "asd_mean", "cldice_mean", "centerline_recall_mean"]]),
        "",
        "## Prospective Paired Hypothesis Tests (`Family 1 K=4` & `Family 2 K=3 + Δ_NI = -0.0050`)",
        "",
        df_to_markdown(df_stats[["family_id", "contrast", "arm", "metric", "n", "mean_diff", "ci95_lo", "ci95_hi", "p_wilcoxon", "p_holm"]]),
        "",
    ]
    atomic_write_text("\n".join(report_md), out_dir / "PHASE_I_PROSPECTIVE_REPORT.md")

    flag_payload = {
        "status": "complete_single_shot",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_cases": len(case_ids),
        "thresholds_v2_sha256": thresh_sha_actual,
        "manifest_locked_sha256": hashlib.sha256(lock_p.read_bytes()).hexdigest(),
    }
    atomic_write_json(flag_payload, complete_flag)
    print(f"[Stage 6 Phase I] Completed prospective single-shot evaluation over N={len(case_ids)} clinical cases")
    return flag_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 Phase I Prospective Clinical Confirmatory CLI")
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument("--cohort-dir", default=None, help="Directory with doctor-annotated clinical .nii.gz scans")
    parser.add_argument("--manifest-csv", default=None, help="Optional CSV with case_id,image_path,label_path columns")
    parser.add_argument("--manifest-locked", default=None, help="Path to Stage 5 MANIFEST_V2_LOCKED.json")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_stage6_phaseI(
        config_path=args.config,
        cohort_dir=args.cohort_dir,
        manifest_csv=args.manifest_csv,
        manifest_locked_path=args.manifest_locked,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
