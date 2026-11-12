"""Amendment A1 contract tests (CPU, synthetic). Must FAIL before the patches, PASS after."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

V2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_DIR.parent))
sys.path.insert(0, str(V2_DIR / "src"))

import eval_fair_v2  # noqa: E402
import run_queue  # noqa: E402
import stage2_pilot_eval  # noqa: E402
from study_registry import build_stage2a_jobs, build_stage3_jobs  # noqa: E402

LOCKED = {"segresnet_f16": 1e-4, "segresnet_f32": 1e-4, "rasnet_v2_f16": 1e-4,
          "rasnet_v2_f32": 3e-4, "vnet": 3e-4, "unet3d": 1e-4}


def _cfg():
    return yaml.safe_load((V2_DIR / "configs" / "fair_v2.yaml").read_text(encoding="utf-8"))


# D1 -----------------------------------------------------------------------------
@pytest.mark.parametrize("w_cl", [0.0, 0.2])
def test_stage3_loss_matched_across_all_v2loss_jobs(w_cl):
    lock = {"all_gates_passed": True, "locked_design": {"gate_mode": "conv_bias", "w_cldice": w_cl}}
    jobs = build_stage3_jobs(_cfg(), LOCKED, pilot_lock=lock)
    v2 = [j for j in jobs if j.get("kind") == "monai" and j.get("loss_name") == "rasnet_v2"]
    assert len(v2) >= 10
    assert {round(float(j["w_cldice"]), 6) for j in v2} == {w_cl}
    base = [j for j in jobs if j.get("loss_name") == "base_dice_focal"]
    assert base and all("w_cldice" not in j or j["w_cldice"] is None for j in base)


# D6 -----------------------------------------------------------------------------
def test_ds_ablation_has_all_three_seeds():
    lock = {"all_gates_passed": True, "locked_design": {"gate_mode": "conv_bias", "w_cldice": 0.2}}
    jobs = build_stage3_jobs(_cfg(), LOCKED, pilot_lock=lock)
    nods = [j for j in jobs if j.get("family") == "rasnet_fixed_ag_nods_f16"]
    assert sorted(j["seed"] for j in nods) == sorted(int(s) for s in _cfg()["study"]["seeds"])


# D5 -----------------------------------------------------------------------------
def test_f32_lr_grid_matches_f16():
    jobs = build_stage2a_jobs(_cfg())
    for arch in ("segresnet", "rasnet_v2"):
        f16 = sorted(j["lr"] for j in jobs if j["lr_key"] == f"{arch}_f16")
        f32 = sorted(j["lr"] for j in jobs if j["lr_key"] == f"{arch}_f32")
        assert f16 == f32 == [1e-4, 3e-4, 1e-3]
    assert len(jobs) == 18
    names = [j["run_name"] for j in jobs]
    # the paused Job 15 / Job 16 must resume before the two new pilots
    assert names[-2:] == ["pilot2a_segresnet_f32_lr1e-4_seed42", "pilot2a_rasnet_v2_f32_lr1e-4_seed42"]
    assert names.index("pilot2a_rasnet_v2_f32_lr3e-4_seed42") < len(names) - 2


# D2 -----------------------------------------------------------------------------
def test_stage3_refuses_when_gates_failed():
    with pytest.raises(RuntimeError, match="gates"):
        run_queue.assert_stage3_allowed({"all_gates_passed": False}, allow_gate_failure=False)
    run_queue.assert_stage3_allowed({"all_gates_passed": True}, allow_gate_failure=False)
    run_queue.assert_stage3_allowed({"all_gates_passed": False}, allow_gate_failure=True)


# D3 -----------------------------------------------------------------------------
def test_gate_metrics_count_empty_predictions_against_candidate():
    df = pd.DataFrame({
        "case_id": [1, 2, 3, 4],
        "status": ["ok", "ok", "ok", "prediction_empty"],
        "precision": [0.9, 0.9, 0.9, float("nan")],
        "components_total": [3, 3, 3, 0],
    })
    g = stage2_pilot_eval.gate_metrics(df)
    assert g["n_empty_pred"] == 1
    assert g["precision_mean_strict"] == pytest.approx(0.675)
    assert stage2_pilot_eval.candidate_qualifies(g, seg_components=5.0) is False
    ok = stage2_pilot_eval.gate_metrics(df.iloc[:3])
    assert stage2_pilot_eval.candidate_qualifies(ok, seg_components=5.0) is True


# D4 -----------------------------------------------------------------------------
def test_mirror_average_logits_is_8_pass_and_equivariant():
    calls = []

    def pointwise(t):                      # flip-equivariant "network"
        calls.append(1)
        return torch.cat([t * 0.5, -t * 0.5], dim=1)

    x = torch.randn(1, 1, 6, 7, 8)
    lg_1, lg_8 = eval_fair_v2.mirror_average_logits(pointwise, x)
    assert len(calls) == 8
    torch.testing.assert_close(lg_1, lg_8, atol=1e-6, rtol=0)

    conv = torch.nn.Conv3d(1, 2, 3, padding=1)
    torch.manual_seed(0)
    with torch.no_grad():
        a1, a8 = eval_fair_v2.mirror_average_logits(conv, x)
    assert (a1 - a8).abs().max() > 1e-4


# D7 -----------------------------------------------------------------------------
def test_override_ids_reject_locked_test_range():
    with pytest.raises(ValueError, match="851"):
        stage2_pilot_eval.guard_val_ids([10, 851])
    assert stage2_pilot_eval.guard_val_ids([10, 11]) == [10, 11]


# D6 (Stage 5) ----------------------------------------------------------------------
def _write_family_csvs(tmp_path, seeds):
    rng = np.random.default_rng(0)
    jobs = []
    for fam in ("rasnet_v2_f16", "rasnet_fixed_ag_nods_f16"):
        for s in seeds:
            rn = f"{fam}_seed{s}"
            jobs.append({"run_name": rn, "family": fam, "seed": s})
            pd.DataFrame({
                "case_id": list(range(851, 871)),
                "dice": rng.uniform(0.70, 0.80, 20), "cldice": rng.uniform(0.70, 0.80, 20),
                "precision": rng.uniform(0.70, 0.80, 20), "hd95": rng.uniform(5.0, 10.0, 20),
            }).to_csv(tmp_path / f"per_case_{rn}_arm_A.csv", index=False)
    return jobs


def test_stage5_ds_contrast_uses_all_three_seeds(tmp_path):
    import stage5_stats_and_efficiency as s5
    jobs = _write_family_csvs(tmp_path, (42, 123, 456))
    out = s5._run_family1(s5._load_family_tables(tmp_path, jobs, arms=("A",)), ("A",))
    ds = out[out["family_id"] == "Ablation_DS_3seed_K4"]
    assert len(ds) == 8                                    # 4 metrics x 2 aggregations
    assert set(ds["aggregation"]) == {"seed_averaged", "seed_pooled"}


def test_stage5_ds_contrast_absent_with_single_seed(tmp_path):
    import stage5_stats_and_efficiency as s5
    jobs = _write_family_csvs(tmp_path, (42,))
    out = s5._run_family1(s5._load_family_tables(tmp_path, jobs, arms=("A",)), ("A",))
    assert "Ablation_DS_3seed_K4" not in set(out.get("family_id", pd.Series(dtype=str)))


# D9 (diagnostic helper) -------------------------------------------------------------
def test_batchstats_wrapper_keeps_bn_training_and_dropout_off():
    import diag_vnet_bn
    inner = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 3, padding=1), torch.nn.BatchNorm3d(2),
                                torch.nn.Dropout3d(0.5))
    w = diag_vnet_bn.BatchStatsBN(inner)
    w.eval()
    assert inner[1].training is True and inner[2].training is False
