"""Synthetic unit tests for V1 epoch-200 report, paired case CSV, and figure generation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.generate_v1_epoch200_report_and_figures as rep


def _sample_per_case_df(model: str, dices: list[float], statuses: list[str] | None = None) -> pd.DataFrame:
    case_ids = [14, 45, 47, 51]
    if statuses is None:
        statuses = ["ok"] * len(case_ids)
    rows = []
    for cid, d, st in zip(case_ids, dices, statuses):
        rows.append({
            "case_id": cid,
            "model": model,
            "checkpoint_role": "final_epoch200",
            "checkpoint_sha256": "a" * 64 if model == "rasnet" else "b" * 64,
            "tta": False,
            "threshold": 0.5,
            "postprocess": "none",
            "keep_top_k": 0,
            "metric_version": "surface-v1",
            "dice": d,
            "iou": d / (2.0 - d) if d > 0 else 0.0,
            "precision": min(1.0, d + 0.02) if st == "ok" else np.nan,
            "recall": max(0.0, d - 0.02) if st == "ok" else 0.0,
            "hd95": 10.0 + (1.0 - d) * 20.0 if st == "ok" else np.nan,
            "asd": 1.5 + (1.0 - d) * 4.0 if st == "ok" else np.nan,
            "cldice": min(1.0, d + 0.05) if st == "ok" else 0.0,
            "centerline_recall": min(1.0, d + 0.08) if st == "ok" else 0.0,
            "status": st,
        })
    return pd.DataFrame(rows)


def test_build_paired_case_level_table_rejects_mismatch_and_computes_diffs():
    dr = _sample_per_case_df("rasnet", [0.80, 0.75, 0.70, 0.0], ["ok", "ok", "ok", "prediction_empty"])
    dn = _sample_per_case_df("nnunet", [0.78, 0.75 + 5e-13, 0.74, 0.65], ["ok", "ok", "ok", "ok"])

    paired = rep.build_paired_case_level_table(dr, dn, tie_tolerance=1e-12)
    assert list(paired["case_id"]) == [14, 45, 47, 51]
    assert paired.loc[paired["case_id"] == 14, "dice_diff_rasnet_minus_nnunet"].iloc[0] == pytest.approx(0.02)
    assert paired.loc[paired["case_id"] == 14, "dice_outcome"].iloc[0] == "rasnet_win"
    assert paired.loc[paired["case_id"] == 45, "dice_outcome"].iloc[0] == "tie"
    assert paired.loc[paired["case_id"] == 47, "dice_outcome"].iloc[0] == "nnunet_win"
    assert paired.loc[paired["case_id"] == 51, "rasnet_status"].iloc[0] == "prediction_empty"
    assert np.isnan(paired.loc[paired["case_id"] == 51, "precision_diff_rasnet_minus_nnunet"].iloc[0])

    # Reject mismatched IDs
    dn_bad = dn.copy()
    dn_bad.loc[0, "case_id"] = 999
    with pytest.raises(ValueError, match="Case ID sets differ"):
        rep.build_paired_case_level_table(dr, dn_bad)

    # Reject duplicates
    dr_dup = pd.concat([dr, dr.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        rep.build_paired_case_level_table(dr_dup, dn)


def test_separate_panel_reference_table_keeps_recipes_distinct():
    summary_ep200 = pd.DataFrame([
        {
            "model": "rasnet", "checkpoint_role": "final_epoch200", "n_cases": 4, "n_ok": 4,
            "n_empty_pred": 0, "dice_mean": 0.779, "dice_std": 0.05, "precision_mean": 0.76,
            "precision_std": 0.08, "recall_mean": 0.80, "recall_std": 0.05, "hd95_mean": 17.3,
            "hd95_std": 15.0, "asd_mean": 2.6, "asd_std": 1.8, "cldice_mean": 0.84,
            "cldice_std": 0.06, "centerline_recall_mean": 0.90, "centerline_recall_std": 0.05,
        },
        {
            "model": "nnunet", "checkpoint_role": "final_epoch200", "n_cases": 4, "n_ok": 4,
            "n_empty_pred": 0, "dice_mean": 0.777, "dice_std": 0.05, "precision_mean": 0.74,
            "precision_std": 0.08, "recall_mean": 0.82, "recall_std": 0.05, "hd95_mean": 18.7,
            "hd95_std": 16.0, "asd_mean": 2.7, "asd_std": 2.0, "cldice_mean": 0.83,
            "cldice_std": 0.06, "centerline_recall_mean": 0.91, "centerline_recall_std": 0.05,
        },
    ])
    ref_rows = [
        {
            "panel": "B_existing_best_reference",
            "comparison_group": "Val94 Arm A (best; 0.25 const vs 0.50 gauss, no CC)",
            "model": "rasnet",
            "checkpoint_role": "best",
            "epoch_record": "epoch 160 (rasnet_200ep_log.csv)",
            "cohort": "Val94 (N=94)",
            "sw_recipe": "1-pass, 0.25 constant SW, tau=0.50",
            "postprocess": "none",
            "n_cases": 94,
            "dice_mean": 0.7789,
            "dice_std": 0.0521,
            "precision_mean": 0.8165,
            "recall_mean": 0.7483,
            "hd95_mean": 13.07,
            "asd_mean": 2.066,
            "cldice_mean": 0.8480,
            "centerline_recall_mean": 0.8617,
        }
    ]
    tbl = rep.build_separated_comparison_table(summary_ep200, ref_rows)
    assert "overall_rank" not in tbl.columns
    assert set(tbl["panel"]) == {"A_epoch200_endpoint", "B_existing_best_reference"}
    assert (tbl[tbl["panel"] == "A_epoch200_endpoint"]["checkpoint_role"] == "final_epoch200").all()
    assert (tbl[tbl["panel"] == "B_existing_best_reference"]["checkpoint_role"] == "best").all()

    with pytest.raises(ValueError, match="Panel B"):
        rep.build_separated_comparison_table(summary_ep200, [{"panel": "A_epoch200_endpoint", "model": "rasnet"}])


def test_generate_report_and_figures_end_to_end_and_protected_guard(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "ep200_out"
    out_dir.mkdir(parents=True)

    dr = _sample_per_case_df("rasnet", [0.80, 0.75, 0.72, 0.78])
    dn = _sample_per_case_df("nnunet", [0.78, 0.76, 0.70, 0.81])
    pd.concat([dr, dn], ignore_index=True).to_csv(out_dir / "per_case_metrics_combined.csv", index=False)
    dr.to_csv(out_dir / "per_case_metrics_rasnet_epoch200.csv", index=False)
    dn.to_csv(out_dir / "per_case_metrics_nnunet_epoch200.csv", index=False)

    from benchmark.endpoint_eval import compute_endpoint_paired_stats, summarize_endpoint_metrics

    comb = pd.concat([dr, dn], ignore_index=True)
    summarize_endpoint_metrics(comb, seed=42, n_boot=200).to_csv(out_dir / "summary_epoch200.csv", index=False)
    meta, paired_df = compute_endpoint_paired_stats(dr, dn, seed=42, n_boot=200)
    paired_df.to_csv(out_dir / "paired_stats_epoch200.csv", index=False)
    (out_dir / "paired_stats_epoch200.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "geometry_verification.csv").write_text(
        "case_id,model,geometry_ok\n14,rasnet,True\n14,nnunet,True\n", encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "synthetic_test",
            "checkpoint_role": "final_epoch200",
            "git": {"commit": "test", "dirty_code": False},
            "source": {"source_tree_sha256": "abc"},
            "split": {"total_val_cases": 4, "evaluated_case_count": 4, "splits_sha256": "def"},
            "checkpoints": {
                "rasnet": {"path": "rasnet_last.pth", "sha256": "a" * 64, "embedded_epoch_metadata_present": False},
                "nnunet": {"last_pt_path": "nnunet_last.pt", "last_pt_sha256": "b" * 64, "current_epoch": 200, "plans_sha256": "c" * 64, "dataset_sha256": "d" * 64},
            },
            "protocol": {"threshold": 0.5, "sw_overlap": 0.5, "sw_mode": "gaussian", "bootstrap_seed": 42, "bootstrap_n_boot": 200},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="protected"):
        rep.validate_report_out_dir(repo_root / "results" / "thesis_outputs", repo_root)

    artifacts = rep.generate_outputs(out_dir=out_dir, repo_root=repo_root)
    for key in (
        "paired_case_csv",
        "comparison_table_csv",
        "comparison_table_md",
        "figure_png",
        "figure_svg",
        "thesis_text_md",
        "report_md",
    ):
        p = Path(artifacts[key])
        assert p.is_file() and p.stat().st_size > 0, f"Missing/empty output {key}: {p}"

    svg_text = Path(artifacts["figure_svg"]).read_text(encoding="utf-8")
    assert "Epoch 50" not in svg_text
    assert "Epoch 100" not in svg_text
    assert "Epoch 150" not in svg_text
