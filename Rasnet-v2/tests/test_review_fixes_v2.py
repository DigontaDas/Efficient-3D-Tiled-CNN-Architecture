"""Code-review fixes R7-R8 for Rasnet-v2 (2026-11-13). CPU, synthetic. FAIL before, PASS after."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

V2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_DIR.parent))
sys.path.insert(0, str(V2_DIR / "src"))

import stage5_stats_and_efficiency as s5  # noqa: E402

GATE_DEPENDENT = {"Abl_Gate_ConvBias_vs_OrigGN", "Abl_DS_Full_vs_NoDS", "Abl_DS_Full_vs_NoDS_3seed"}


def _tables(tmp_path):
    rng = np.random.default_rng(0)
    jobs = []
    fams = {"rasnet_v2_f16": (42, 123, 456), "rasnet_fixed_ag_nods_f16": (42, 123, 456),
            "rasnet_orig_ag_f16": (42,), "segresnet_f16_v2loss": (42, 123, 456), "segresnet_f16_baseloss": (42,)}
    for fam, seeds in fams.items():
        for s in seeds:
            rn = f"{fam}_seed{s}"
            jobs.append({"run_name": rn, "family": fam, "seed": s})
            pd.DataFrame({"case_id": list(range(851, 871)),
                          "dice": rng.uniform(0.7, 0.8, 20), "cldice": rng.uniform(0.7, 0.8, 20),
                          "precision": rng.uniform(0.7, 0.8, 20), "hd95": rng.uniform(5.0, 10.0, 20)}
                         ).to_csv(tmp_path / f"per_case_{rn}_arm_A.csv", index=False)
    return s5._load_family_tables(tmp_path, jobs, arms=("A",))


@pytest.mark.parametrize("gate, valid", [("conv_bias", True), ("groupnorm_orig", False), (None, True)])
def test_gate_dependent_ablations_are_flagged(tmp_path, gate, valid):
    out = s5._run_family1(_tables(tmp_path), ("A",), lead_gate_mode=gate)
    assert {"lead_gate_mode", "ablation_label_valid"} <= set(out.columns)
    dep = out[out["contrast"].isin(GATE_DEPENDENT)]
    assert len(dep) and set(dep["ablation_label_valid"]) == {valid}
    other = out[~out["contrast"].isin(GATE_DEPENDENT)]
    assert len(other) and set(other["ablation_label_valid"]) == {True}   # loss / AG / Family 1 rows unaffected
