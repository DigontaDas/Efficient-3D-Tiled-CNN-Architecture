"""Tests-first suite for V1 reporting corrections, curve audit, stats recomputation, and provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.generate_v1_epoch200_report_and_figures as rep_orig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_ep200_inputs():
    summary_ep200 = pd.DataFrame([
        {
            "model": "rasnet", "checkpoint_role": "final_epoch200", "n_cases": 94, "n_ok": 94,
            "n_empty_pred": 0, "dice_mean": 0.7792, "dice_std": 0.0569, "dice_median": 0.7921,
            "precision_mean": 0.7586, "recall_mean": 0.8065, "hd95_mean": 17.32,
            "asd_mean": 2.619, "cldice_mean": 0.8413, "centerline_recall_mean": 0.9068,
        },
        {
            "model": "nnunet", "checkpoint_role": "final_epoch200", "n_cases": 94, "n_ok": 94,
            "n_empty_pred": 0, "dice_mean": 0.7767, "dice_std": 0.0562, "dice_median": 0.7824,
            "precision_mean": 0.7428, "recall_mean": 0.8203, "hd95_mean": 18.67,
            "asd_mean": 2.748, "cldice_mean": 0.8347, "centerline_recall_mean": 0.9142,
        },
    ])
    paired_json = {
        "seed": 42,
        "n_boot": 2000,
        "tie_tolerance": 1e-12,
        "status_summary": {"rasnet": {"ok": 94}, "nnunet": {"ok": 94}},
        "metrics": [
            {"metric": "dice", "n": 94, "mean_difference": 0.00247, "diff_ci95_lo": -0.00275, "diff_ci95_hi": 0.00755, "median_difference": 0.00323, "p": 0.2401, "wins": 54, "ties": 0, "losses": 40},
            {"metric": "iou", "n": 94, "mean_difference": 0.00337, "diff_ci95_lo": -0.00359, "diff_ci95_hi": 0.01028, "median_difference": 0.00415, "p": 0.2478, "wins": 54, "ties": 0, "losses": 40},
            {"metric": "precision", "n": 94, "mean_difference": 0.01577, "diff_ci95_lo": 0.00724, "diff_ci95_hi": 0.02454, "median_difference": 0.01715, "p": 1.068e-4, "wins": 66, "ties": 0, "losses": 28},
            {"metric": "recall", "n": 94, "mean_difference": -0.01379, "diff_ci95_lo": -0.01749, "diff_ci95_hi": -0.00980, "median_difference": -0.01558, "p": 4.849e-10, "wins": 16, "ties": 0, "losses": 78},
            {"metric": "hd95", "n": 94, "mean_difference": -1.349, "diff_ci95_lo": -3.826, "diff_ci95_hi": 0.947, "median_difference": -0.262, "p": 0.6742, "wins": 49, "ties": 0, "losses": 45},
            {"metric": "asd", "n": 94, "mean_difference": -0.128, "diff_ci95_lo": -0.362, "diff_ci95_hi": 0.110, "median_difference": -0.036, "p": 0.3372, "wins": 49, "ties": 0, "losses": 45},
            {"metric": "cldice", "n": 94, "mean_difference": 0.00666, "diff_ci95_lo": -0.00059, "diff_ci95_hi": 0.01353, "median_difference": 0.00937, "p": 0.0352, "wins": 57, "ties": 0, "losses": 37},
            {"metric": "centerline_recall", "n": 94, "mean_difference": -0.00741, "diff_ci95_lo": -0.01236, "diff_ci95_hi": -0.00249, "median_difference": -0.00375, "p": 0.0092, "wins": 37, "ties": 2, "losses": 55},
        ],
    }
    manifest = {
        "run_id": "test_run",
        "generated_at_utc": "2026-11-17T06:16:55Z",
        "checkpoint_role": "final_epoch200",
        "split": {"evaluated_case_count": 94, "splits_sha256": "abc"},
        "timing_seconds": {"rasnet": 0.0, "nnunet": 1579.378},
    }
    return summary_ep200, paired_json, manifest


def test_epoch200_report_narrative_fixes_r4_r5_r6():
    summary_ep200, paired_json, manifest = _synthetic_ep200_inputs()
    text = rep_orig.render_thesis_curve_and_endpoint_text(summary_ep200, paired_json, [], manifest)

    # R4: prohibited causal claim removed; exact non-causal phrasing present
    assert "artifact of patch sampling" not in text.lower()
    assert (
        "The logged metrics use different sampling, aggregation and spatial evaluation protocols "
        "and therefore cannot establish comparative full-volume performance."
    ) in text

    # R4: all 5 model validation protocols/cadences listed
    for token in ("12", "20", "13", "21", "200", "128", "96x160x160"):
        assert token in text, f"Missing protocol detail token: {token}"
    assert "SegResNet" in text and "V-Net" in text and "3D U-Net" in text

    # R5: all 8 metric p-values explicitly tagged exploratory and unadjusted; Wilcoxon vs bootstrap CI explained
    low = text.lower()
    assert "exploratory" in low
    assert "unadjusted" in low
    assert "wilcoxon" in low and "bootstrap" in low and "different quantities" in low

    # R6: timing=0.0s reflects skipped completed work on --resume, not inference latency
    geom_df = pd.DataFrame({"geometry_ok": [True, True]})
    report_md = rep_orig.render_endpoint_report(
        Path("dummy"), manifest, summary_ep200, paired_json, geom_df, "TABLE"
    )
    assert "0.0s" in report_md
    assert "resume" in report_md.lower() and "skipped" in report_md.lower()
    assert "not" in report_md.lower() and "latency" in report_md.lower()


def test_val94_endpoint_recomputation_and_posthoc_holm8():
    import scripts.generate_v1_reporting_corrections as rc

    ep200_dir = REPO_ROOT / "results" / "v1_epoch200_comparison_20261117_041500Z"
    splits_path = REPO_ROOT / "Phase3_Local_Integration" / "splits_final.json"
    res = rc.recompute_val94_endpoint_stats(ep200_dir=ep200_dir, splits_path=splits_path)

    assert res["n_matched_cases"] == 94
    assert res["n_empty_pred"] == 0
    assert res["n_excluded_nan"] == 0
    assert res["geometry_ok_count"] == 188
    assert res["geometry_total_count"] == 188

    summary_df = res["summary_df"]
    r_dice = float(summary_df.loc[summary_df["model"] == "rasnet", "dice_mean"].iloc[0])
    n_dice = float(summary_df.loc[summary_df["model"] == "nnunet", "dice_mean"].iloc[0])
    assert r_dice == pytest.approx(0.7791706, abs=1e-5)
    assert n_dice == pytest.approx(0.7766974, abs=1e-5)

    paired_df = res["paired_df"]
    dice_row = paired_df.loc[paired_df["metric"] == "dice"].iloc[0]
    assert float(dice_row["mean_difference"]) == pytest.approx(0.0024732, abs=1e-5)
    assert float(dice_row["diff_ci95_lo"]) == pytest.approx(-0.0027466, abs=1e-5)
    assert float(dice_row["diff_ci95_hi"]) == pytest.approx(0.0075452, abs=1e-5)
    assert (int(dice_row["wins"]), int(dice_row["ties"]), int(dice_row["losses"])) == (54, 0, 40)
    assert float(dice_row["p_unadjusted"]) == pytest.approx(0.240136, abs=1e-5)
    assert (paired_df["p_value_status"] == "exploratory_unadjusted").all()

    holm_df = res["posthoc_holm8_df"]
    assert len(holm_df) == 8
    assert (holm_df["family_label"] == "POST-HOC EXPLORATORY — NOT PREREGISTERED (8-metric Val94 endpoint)").all()
    by_m = holm_df.set_index("metric")
    assert float(by_m.loc["cldice", "holm8_p"]) == pytest.approx(0.175993, abs=1e-4)
    assert float(by_m.loc["centerline_recall", "holm8_p"]) == pytest.approx(0.055122, abs=1e-4)
    assert float(by_m.loc["precision", "holm8_p"]) == pytest.approx(7.4749e-4, abs=1e-6)
    assert float(by_m.loc["recall", "holm8_p"]) == pytest.approx(3.8789e-9, abs=1e-11)


def test_training_log_parsing_all_5_models():
    import scripts.generate_v1_reporting_corrections as rc

    logs_dir = REPO_ROOT / "Q1_Publication_Package" / "matched_200ep_benchmark" / "logs"
    nnunet_last = REPO_ROOT / "recovered_checkpoints" / "v1_epoch200_20261117_041500Z" / "nnunet_last.pt"
    parsed = rc.parse_all_training_logs(logs_dir=logs_dir, nnunet_last_pt_path=nnunet_last)

    assert parsed["rasnet"]["n_val_evals"] == 12
    assert parsed["segresnet"]["n_val_evals"] == 20
    assert parsed["vnet"]["n_val_evals"] == 13
    assert parsed["3dunet"]["n_val_evals"] == 21
    assert parsed["nnunet"]["n_val_evals"] == 200

    assert "DiceFocalLoss" in parsed["segresnet"]["loss_label"]
    assert "DiceFocalLoss" in parsed["vnet"]["loss_label"]
    assert "StenosisAwareLoss" in parsed["rasnet"]["loss_label"]
    assert "DiceCELoss" in parsed["3dunet"]["loss_label"]
    assert "DC_and_CE_loss" in parsed["nnunet"]["loss_label"]

    assert parsed["nnunet"]["raw_pseudo_dice_peak_epoch"] == 189
    assert parsed["nnunet"]["raw_pseudo_dice_peak_value"] == pytest.approx(0.8376, abs=1e-4)
    assert parsed["nnunet"]["ema_best_epoch"] == 199
    assert parsed["nnunet"]["ema_best_value"] == pytest.approx(0.816239, abs=1e-4)


def test_regimes_efficiency_and_protected_guard():
    import scripts.generate_v1_reporting_corrections as rc

    with pytest.raises(ValueError, match="protected"):
        rc.validate_correction_out_dir(REPO_ROOT / "results" / "thesis_outputs", REPO_ROOT)

    regimes_df = rc.build_four_regimes_table(REPO_ROOT)
    r4_row = regimes_df[regimes_df["regime_id"] == "D_optionA_R4"].iloc[0]
    assert int(r4_row["n_models"]) == 4
    assert float(r4_row["rasnet_dice_mean"]) == pytest.approx(0.8039, abs=1e-4)
    assert float(r4_row["nnunet_dice_mean"]) == pytest.approx(0.8059, abs=1e-4)
    assert bool(r4_row["non_inferior_at_margin_neg0005"]) is False

    eff_df = rc.build_efficiency_and_timing_table(REPO_ROOT)
    assert set(eff_df["model_key"]) == {"rasnet", "segresnet", "3dunet", "nnunet", "vnet"}
    by_k = eff_df.set_index("model_key")
    assert int(by_k.loc["rasnet", "params_exact"]) == 4706772
    assert int(by_k.loc["segresnet", "params_exact"]) == 4700914
    assert int(by_k.loc["3dunet", "params_exact"]) == 4805534
    assert int(by_k.loc["nnunet", "params_exact"]) == 30785994
    assert int(by_k.loc["vnet", "params_exact"]) == 45599452
    assert float(by_k.loc["rasnet", "optionA_1pass_median_sec"]) == pytest.approx(5.3959, abs=1e-3)
    assert float(by_k.loc["nnunet", "optionA_1pass_median_sec"]) == pytest.approx(29.8432, abs=1e-3)
    assert "excluded" in str(by_k.loc["rasnet", "invalid_timing_note"]).lower()
    assert "0.0s" in str(by_k.loc["rasnet", "invalid_timing_note"])
    assert "1.85" in str(by_k.loc["rasnet", "invalid_timing_note"])
