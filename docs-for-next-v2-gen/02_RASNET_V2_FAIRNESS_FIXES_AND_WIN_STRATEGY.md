# 02 — Rasnet-v2: Fairness Fixes Before Stage 3, and the Legitimate Strategy to Beat the Baselines

**Target machine:** lab PC `C:\Thesis_RASNET\Rasnet-v2`, RTX 4080 SUPER.
**Applies to:** `Rasnet-v2/src/*.py`, `Rasnet-v2/tests/*.py`, `Rasnet-v2/PREREGISTRATION_V2.md`.
**Current state** (from your side-thread transcript, not verifiable on this machine):

- Stage 2A is paused at 14/16 jobs.
- Job 15 (`pilot2a_rasnet_v2_f32_lr3e-4_seed42`) has resumable state saved at epoch 30, step 200.
- `STATE_HANDOFF.md` holds the resume instructions.

**Patch validation (2026-11-12, scratch copy on the H: machine, CPU only):**

- Every patch in §3 was applied mechanically, exactly as written, to a copy of `Rasnet-v2/`.
- `tests/test_amendment_a1.py`: **fails before** the patches (10 fail; the 1 that passes is a negative check that is trivially true before patching), **11 / 11 pass after**. The patched Stage 2A queue keeps jobs 1–16 in their existing positions (Job 15 is still `pilot2a_rasnet_v2_f32_lr3e-4_seed42`); the two new pilots are jobs 17–18. This includes the Stage 5 3-seed DS contrast and the D9 diagnostic helper (`src/diag_vnet_bn.py`).
- Full v2 suite after patching (including the new file): 40 passed, 3 failed. All 3 failures are environment-only on that machine:
  - `fair_v2.yaml` hard-codes `C:\Thesis_RASNET`
  - `nnunetv2` is not installed
  - MONAI 1.5.1 vs 1.6.1 `SoftclDiceLoss` signature
- The end-to-end synthetic Stage 2→6 test passed with the new 22-run count.
- Re-run the full suite on the lab PC (LAB-CHECK).

All patches below must land **before Stage 2A's LR lock and before Stage 3**. Nothing in Stage 3 has been trained yet, so fixing them now costs no completed work.

---

## 0. The goal, stated so it survives an examiner

You want RASNet / RASNet-v2 to beat nnU-Net, 3D U-Net, SegResNet and V-Net, so that people adopt it as the lighter, better model. That goal is only worth something if the comparison is one nobody can attack.

If Stage 3 runs with the defects in §1, a reviewer can dismiss any RASNet win as "different loss", "un-gated design choice" or "extra tuning". The fixes below make a win **credible**. They do not guarantee one. No honest protocol can, and you should not want one that does.

What *is* in your favour, legitimately:

1. **Efficiency is a guaranteed, objective win.**
   - RASNet-v2 F16 has 4.71 M parameters; nnU-Net has 30.79 M (6.5×) and V-Net 45.6 M (9.7×).
   - Stage 5 already profiles parameters, GFLOPs, latency and peak VRAM on the 4080 SUPER (`stage5_stats_and_efficiency.py:225-370`).
2. **Precision and HD95 are where RASNet has historically led.** v1 Arm A precision was 0.8032 vs 0.7391. Family 2 already tests precision, HD95 and clDice superiority.
3. **The pre-registered Dice non-inferiority test** (margin −0.005, `PREREGISTRATION_V2.md:53`) is the right primary Dice claim against a 6.5× larger model.
   - "Non-inferior at 15 % of the size" is a publishable headline.
   - "Better on Dice" only happens if the data say so.
4. **Two capacity tiers (F16 and F32)** let you show a Pareto front: accuracy versus compute.

---

## 1. Defects found in the code review (file:line, confirmed by reading the source)

| # | Severity | Defect | Evidence |
| :--- | :--- | :--- | :--- |
| D1 | Critical | Stage 3 loss can differ between RASNet and its baselines | `study_registry.py:228` (`lead_w_cl` from 2B lock) is applied only to the RASNet full jobs (`:321`, `:337`). The SegResNet, V-Net, U-Net and ablation jobs carry no `w_cldice` and fall back to 0.2 (`train_fair_v2.py:222`). If 2B locks `cl000`, Family 1 compares different losses. |
| D2 | Critical | Stage 2B gates are not binding | If no candidate qualifies, the best-Dice candidate is still locked (`stage2_pilot_eval.py:294-296`). Stage 3 launches without checking `all_gates_passed` (`run_queue.py:377-383`). |
| D3 | Critical | Gates can be passed by degenerate candidates | Empty predictions give NaN precision (`eval_fair_v2.py:85`), and NaNs are dropped from the mean (`stage2_pilot_eval.py:43`). Empty masks give `components_total=0`, which helps the fragmentation gate. |
| D4 | Important | TTA asymmetry: MONAI models get 4 passes (identity + 3 single flips, `eval_fair_v2.py:607-610`), nnU-Net gets 8 | nnU-Net averages logits over all 2³ mirror combinations (MIC-DKFZ/nnUNet `predict_from_raw_data.py`, `_internal_maybe_mirror_and_predict`). The pre-registration fixes this asymmetry in Arm B (`PREREGISTRATION_V2.md:46`). |
| D5 | Important | The F32 LR grid lacks 1e-4 | `study_registry.py:119-120`. The F16 val20 logs rank 1e-4 best for SegResNet. If F16 locks at 1e-4, the F32 lock sits at the grid edge for both SegResNet and RASNet. |
| D6 | Important | The deep-supervision ablation has one seed | `s3_rasnet_fixed_ag_f16_nods_v2loss_seed42` (`study_registry.py:292-305`). "Gates vs DS" cannot be separated with n = 1 seed. |
| D7 | Minor | `case_ids_override` has no locked-ID guard | `stage2_pilot_eval.py:78,182`, `stage4b_val94_lock.py:152` |
| D8 | Note | Val20 ⊂ Val-94 and `best_model.pth` is chosen on val20 | Val-94 numbers are slightly optimistic for MONAI models, and more so for nnU-Net (checkpoint chosen on all 94). Test is unaffected. State it in the limitations section. |
| D9 | Note | V-Net is the only model with BatchNorm and runs at micro-batch 2 | Its lr = 1e-4 collapse (val20 0.1483, from your transcript) may be a BN statistics issue, not a V-Net weakness. Report it; do not exploit it. |
| D10 | Not a bug | The "+0.049 RASNet-v2 F16 over SegResNet F16" pilot gap | Single seed, 20 cases, before each model's own LR lock, scored on the resampled grid. **Never cite it in the thesis.** |

Resume after the pause is sound. Data order is seeded per `(seed, global_idx)` (`train_fair_v2.py:119-153`) and the RNG is restored. The result is not bit-identical (cuDNN), but the difference is at noise level.

---

## 2. Pre-registration amendment (commit BEFORE applying the code fixes)

Append this to `Rasnet-v2/PREREGISTRATION_V2.md`. Commit it with the timestamp, and note that no Stage 2A LR lock, Stage 2B, Stage 3 or Test-150 result exists at the time of the amendment.

```markdown
## Amendment A1 (dated YYYY-MM-DD, before Stage 2A LR lock, before any Stage 3 run)
1. Matched loss (D1): every Stage 3 job trained with loss `rasnet_v2` uses the clDice weight
   locked in Stage 2B (`w_cldice*`), for RASNet-v2, SegResNet-v2, V-Net-v2, UNet3D-v2 and the
   RASNet ablations alike. `s3_segresnet_f16_baseloss` keeps `base_dice_focal` (loss control).
2. Binding gates (D2, D3): a Stage 2B candidate qualifies only if (a) zero empty predictions on
   Val-94, (b) strict precision (empty prediction counted as 0) >= 0.80, and (c) mean components
   <= matched-width SegResNet-v2. If no candidate qualifies, Stage 3 does NOT start automatically;
   a written amendment is required before `--allow-gate-failure` may be used.
3. Symmetric TTA (D4): Arm B uses 8-pass mirroring (identity + all 7 flip-axis combinations,
   logit-averaged before softmax) for every MONAI model, matching nnU-Net V2's native mirroring.
4. LR grid (D5): F32 LR grids for SegResNet and RASNet-v2 are {1e-4, 3e-4, 1e-3}, identical to F16.
5. DS ablation seeds (D6): `rasnet_fixed_ag_nods_f16` is trained with all three seeds {42, 123, 456}.
Rationale: each change removes an asymmetry; none was chosen after observing any validation or
test outcome of the affected comparisons.
```

> **CLAUDE.md conflict check.** The rule says "Do not change … losses, hyperparameters, epoch counts". Items 1, 4 and 5 change *which* jobs get which loss weight, LR grid and seed count. They change no loss formula, no architecture and no epoch count. They need **your explicit sign-off** as a dated amendment, which is what the block above is. If you do not sign off on an item, skip it and list the corresponding defect as a limitation.

---

## 3. Fixes — tests first, then code

Add the tests to `Rasnet-v2/tests/test_amendment_a1.py` and run them. They **must fail** against the current code; keep that output. Then apply the patches and re-run.

### 3.1 Tests (`Rasnet-v2/tests/test_amendment_a1.py`)

```python
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
```

Also update the existing assertion in `tests/test_v2_pipeline_cpu.py::test_run_queue_preflight_dedup_and_stage3_manifest` (lines ~291-294), because the grid now has 18 jobs:

```python
    assert len(s2a) == 18
    assert sum(1 for j in s2a if j["lr_key"] == "segresnet_f32") == 3
    assert sum(1 for j in s2a if j["lr_key"] == "rasnet_v2_f32") == 3
```

Run on the lab PC from `C:\Thesis_RASNET` (repo root). This must fail before the patches; keep the output:

```powershell
$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"; Set-Location C:\Thesis_RASNET
& $PY -m pytest Rasnet-v2/tests/test_amendment_a1.py -q
```

### 3.2 Patch D1 + D6 — `src/study_registry.py`, inside `build_stage3_jobs`

(a) Replace the single DS-ablation dict (`"family": "rasnet_fixed_ag_nods_f16"`, lines ~292-305) with a seed loop. Append it right after the `rasnet_orig_ag_f16` dict inside the same `jobs.extend([...])`, or as a separate loop:

```python
    for s in seeds:
        jobs.append({
            "kind": "monai",
            "stage": "stage3",
            "family": "rasnet_fixed_ag_nods_f16",
            "run_name": f"s3_rasnet_fixed_ag_f16_nods_v2loss_seed{s}",
            "arch": "rasnet_v2",
            "init_filters": 16,
            "lr": lrs["rasnet_v2_f16"],
            "seed": s,
            "epochs": ep,
            "gate_mode": "conv_bias",
            "enable_ds": False,
            "loss_name": "rasnet_v2",
        })
```

The seed-42 run name is unchanged (`s3_rasnet_fixed_ag_f16_nods_v2loss_seed42`), so nothing already queued is renamed.

Stage 5 currently tests the ablations on **seed 42 only** (`ablation_contrasts_seed42`, `stage5_stats_and_efficiency.py:106-159`). Keep that block, which is pre-registered.

Add the 3-seed DS contrast as a **separate secondary family** with its own Holm correction; never merge it into Family 1's K = 4. In `_run_family1`, inside `for arm in arms:` and directly after the seed-42 ablation loop (after line 159, before `return pd.DataFrame(rows)`), insert:

```python
        # Amendment A1.5 — deep-supervision isolation with all 3 seeds (secondary family, own Holm).
        ds_a, ds_b = "rasnet_v2_f16", "rasnet_fixed_ag_nods_f16"
        if (ds_a, arm) in fam_tables and (ds_b, arm) in fam_tables:
            seeds_a = set(fam_tables[(ds_a, arm)]["per_seed"])
            seeds_b = set(fam_tables[(ds_b, arm)]["per_seed"])
            if len(seeds_a & seeds_b) >= 3:
                for agg in ("seed_averaged", "seed_pooled"):
                    df_a = fam_tables[(ds_a, arm)][agg]
                    df_b = fam_tables[(ds_b, arm)][agg]
                    if set(df_a["case_id"]) != set(df_b["case_id"]):
                        raise RuntimeError(f"DS 3-seed case_id/seed mismatch ({arm}, {agg})")
                    fam_res = []
                    for m in k4_metrics:
                        st = eval_fair_v2.paired_with_noninferiority(
                            df_a, df_b, metric=m, allow_nan_filtering=True, direction=METRIC_DIRECTIONS[m]
                        )
                        st.update({"family_id": "Ablation_DS_3seed_K4", "contrast": "Abl_DS_Full_vs_NoDS_3seed",
                                   "arm": arm, "aggregation": agg, "metric": m, "direction": METRIC_DIRECTIONS[m]})
                        fam_res.append(st)
                    holm_ps = eval_fair_v2.holm_adjust([r["p_wilcoxon"] for r in fam_res])
                    for r, hp in zip(fam_res, holm_ps):
                        r["p_holm"] = float(hp)
                        r["significant_holm_005"] = bool(hp < 0.05)
                        rows.append(r)
```

> **Interpretation caveat — decide before Stage 3.** The full RASNet-v2 runs use the gate mode locked in Stage 2B (`lead_gate`), while the no-DS runs are fixed to `conv_bias` (`study_registry.py:292-305`).
> - If 2B locks `conv_bias`, this contrast isolates deep supervision cleanly.
> - If 2B locks `groupnorm_orig`, it measures gate mode **and** DS together. Report it as such; do not call it a pure DS effect.
>
> The existing seed-42 ablation has the same property.

The existing test `tests/test_v2_pipeline_cpu.py` asserts 20 Stage 3 runs (lines 315, 395, 407). Change all three to **22** (20 + 2 extra no-DS seeds).

The existing test `tests/test_v2_pipeline_cpu.py` asserts 20 Stage 3 runs (lines 315, 395, 407). Change all three to **22** (20 + 2 extra no-DS seeds).

(b) Immediately before the final `return jobs` of `build_stage3_jobs` (line ~381):

```python
    # Amendment A1.1 — one loss for every architecture trained with the v2 loss (matched Family 1).
    for j in jobs:
        if j.get("kind") == "monai" and j.get("loss_name") == "rasnet_v2":
            j["w_cldice"] = lead_w_cl
```

Also update the docstring count ("Exact 20-run Stage 3 matrix" becomes 22 runs) and any test that asserts 20.

### 3.3 Patch D5 — `src/study_registry.py`, end of `build_stage2a_jobs`

**Leave the existing F32 loop at lines 117-135 untouched**, including `for lr in (3e-4, 1e-3):`. Instead, insert this block immediately before that function's `return jobs` (line ~136):

```python
    # Amendment A1.4 — the F32 LR grid gains 1e-4 so it matches the F16 grid {1e-4, 3e-4, 1e-3}.
    # Appended LAST on purpose: run_queue executes jobs in list order, so the paused
    # Job 15 (rasnet_v2_f32 lr3e-4, mid_epoch_state.pt) and Job 16 resume first.
    # Job order never changes any result (each job is seeded per (seed, global_idx)).
    for arch in ("segresnet", "rasnet_v2"):
        lr = 1e-4
        jobs.append({
            "kind": "monai",
            "stage": "stage2a",
            "lr_key": f"{arch}_f32",
            "run_name": f"pilot2a_{arch}_f32_lr{lr_tag(lr)}_seed42",
            "arch": arch,
            "init_filters": 32,
            "lr": float(lr),
            "seed": 42,
            "epochs": ep,
            "gate_mode": "conv_bias",
            "enable_ds": True,
            "loss_name": "rasnet_v2",
            "w_cldice": float(cfg["loss"]["rasnet_v2"]["w_cldice"]),
        })
```

The queue skips jobs that have `DONE.flag`, so this only **adds** two 50-epoch pilots, at the end of the queue:

- `pilot2a_segresnet_f32_lr1e-4_seed42`
- `pilot2a_rasnet_v2_f32_lr1e-4_seed42`

Resume order on the lab PC: Job 15 (`pilot2a_rasnet_v2_f32_lr3e-4_seed42`, resumes from `mid_epoch_state.pt`), Job 16 (`…_f32_lr1e-3_seed42`), then the two new pilots.

Job 15's resume check (`config_sha256`, `train_fair_v2.py:319-340`) hashes only that job's own settings. Those are unchanged, so the resume is accepted.

### 3.4 Patch D2 — `src/run_queue.py`

Add the function near the other helpers:

```python
def assert_stage3_allowed(pilot_lock: dict, allow_gate_failure: bool) -> None:
    """Amendment A1.2: Stage 3 starts only if a Stage 2B candidate passed every pre-registered gate."""
    if bool(pilot_lock.get("all_gates_passed", False)):
        return
    if allow_gate_failure:
        print("[WARN] Stage 2B gates FAILED; proceeding under an explicit written amendment.", flush=True)
        return
    raise RuntimeError(
        "Stage 2B: no RASNet-v2 candidate passed the pre-registered gates "
        "(zero empty predictions, strict precision >= 0.80, fragmentation <= SegResNet). "
        "Write a dated amendment, then re-run with --allow-gate-failure."
    )
```

In `main()`, add the CLI flag next to the other `add_argument` calls (~line 330):

```python
    parser.add_argument("--allow-gate-failure", action="store_true",
                        help="Only after a dated written amendment when Stage 2B gates failed")
```

In the `elif args.stage == "stage3":` branch (line ~377), right after `pilot_lock = json.loads(...)`:

```python
        assert_stage3_allowed(pilot_lock, allow_gate_failure=args.allow_gate_failure)
```

### 3.5 Patch D3 + D7 — `src/stage2_pilot_eval.py`

Add these helpers below `_summarize_df`:

```python
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
```

At lines 78 and 182, wrap the IDs:

```python
    val_ids = guard_val_ids(case_ids_override or sorted(splits["val"]))
```

In the Stage 2B loop (lines ~266-268), replace the three `pass_*`/`qualifies` lines with:

```python
        g = gate_metrics(df)
        pass_prec = bool(g["precision_mean_strict"] >= 0.8000)
        pass_frag = bool(g["components_total_mean"] <= seg_comp + 1e-9)
        qualifies = candidate_qualifies(g, seg_comp)
```

Then add `**g,` to the `rows.append({...})` dict, so `n_empty_pred` and `precision_mean_strict` are written to `stage2b_summary.csv` and `pilot_lock_decision.json`.

Apply the same `guard_val_ids` call in `stage4b_val94_lock.py:152`:

```python
    val_ids = sorted(guard_val_ids(case_ids_override if case_ids_override is not None else splits["val"]))
```

Import it there with `from stage2_pilot_eval import guard_val_ids`.

### 3.6 Patch D4 — `src/eval_fair_v2.py` (symmetric 8-pass TTA)

Add a module-level helper (it needs `import itertools` at the top):

```python
def mirror_average_logits(logits_fn, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (identity logits, 8-pass mirror-averaged logits).

    Identity + all 7 non-empty combinations of the 3 spatial axes; logits averaged before softmax.
    This is exactly nnUNetPredictor._internal_maybe_mirror_and_predict.
    """
    axes_all = (2, 3, 4)
    combos = [c for r in range(1, 4) for c in itertools.combinations(axes_all, r)]
    identity = logits_fn(x).float()
    acc = identity.clone()
    for axes in combos:
        acc += torch.flip(logits_fn(torch.flip(x, axes)).float(), axes)
    return identity, acc / float(len(combos) + 1)
```

In `infer_monai_native_zyx` (lines 587-610), replace `_pred` and the `with torch.no_grad():` block with:

```python
    def _logits(tensor_cpu: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_ok):
            out = sliding_window_inference(
                tensor_cpu,
                roi_size=roi_size_xyz,
                sw_batch_size=int(sw_batch_size),
                predictor=net,
                overlap=float(sw_overlap),
                mode=str(sw_mode),
                sw_device=sw_dev,
                device=cpu_dev,
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
        return out.float()

    with torch.no_grad():
        if need_tta:
            lg_1, lg_8 = mirror_average_logits(_logits, x_cpu)
            p_raw = torch.softmax(lg_1, dim=1)[:, 1:2]
            p_tta = torch.softmax(lg_8, dim=1)[:, 1:2]
            del lg_1, lg_8
        else:
            p_raw = torch.softmax(_logits(x_cpu), dim=1)[:, 1:2]
            p_tta = None
```

Notes:

- **Performance on the 4080 SUPER:** the logits stay on the CPU (`device=cpu_dev`), exactly as now, so peak VRAM is unchanged. TTA time is about 8× the 1-pass, against 4× before. Arm B runs once per checkpoint in Stage 4C, so this is acceptable.
- **Parity:** the 1-pass path is numerically unchanged. Softmax of the same blended logits is the same as before, so Gate 4 parity (`STAGE1_GATES_REPORT.json`) still holds. Re-run `tests/test_gate1_arch_and_loss.py::test_surface_v1_and_postprocess_exact_parity_with_round1` to confirm.

### 3.7 After the patches

First create `Rasnet-v2/src/diag_vnet_bn.py` from §4: the D9 test imports it. Then insert the Stage 5 DS block from §3.2. Then run all three:

```powershell
$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"; Set-Location C:\Thesis_RASNET
& $PY -m pytest Rasnet-v2/tests/test_amendment_a1.py -q
```

```powershell
& $PY -m pytest Rasnet-v2/tests -q
```

```powershell
& $PY Rasnet-v2/src/run_stage1_gates.py
```

All must pass. Then commit code and amendment together:

```powershell
git add Rasnet-v2/PREREGISTRATION_V2.md Rasnet-v2/src Rasnet-v2/tests; git commit -m "Rasnet-v2 Amendment A1: matched loss, binding gates, 8-pass TTA, F32 LR grid, DS-ablation seeds"
```

Do not push without your explicit decision (CLAUDE.md).

---

## 4. D9 — V-Net BatchNorm diagnostic (optional, diagnostic only)

The point is to make sure V-Net's lr = 1e-4 collapse is not presented as "V-Net is weak".

The script below evaluates a Stage 2A V-Net checkpoint on the first N val20 cases (validation only) twice:

1. with normal `eval()` BatchNorm, which uses running statistics;
2. with BatchNorm using per-batch statistics and dropout off.

It writes one CSV under `Rasnet-v2/results/diagnostics/`. **It is never used for any selection.** The model copy used for batch statistics is a deep copy, so the original weights and running statistics are untouched.

Create `Rasnet-v2/src/diag_vnet_bn.py`:

```python
"""D9 diagnostic: is a V-Net pilot collapse a BatchNorm running-statistics artefact?

Validation (val20) cases only; writes Rasnet-v2/results/diagnostics/vnet_bn_diag_<run>.csv.
Never used for LR, design, threshold or checkpoint selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))

from augmentations import build_raw_eval_transform  # noqa: E402
from dataset_pairs import resolve_case_pairs  # noqa: E402
from eval_fair_v2 import build_monai_invert_op, compute_metrics, infer_monai_native_zyx  # noqa: E402
from rasnet_v2_model import build_monai_model  # noqa: E402

_DROPOUTS = (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout)


class BatchStatsBN(torch.nn.Module):
    """Wrapper whose eval() keeps BatchNorm on batch statistics while dropout stays off."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)

    def train(self, mode: bool = True):
        self.training = mode
        self.inner.train(True)
        for m in self.inner.modules():
            if isinstance(m, _DROPOUTS):
                m.eval()
        return self


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(V2 / "configs" / "fair_v2.yaml"))
    ap.add_argument("--run", default="pilot2a_vnet_f16_lr1e-4_seed42")
    ap.add_argument("--ckpt", default="epoch_50.pth")
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ids = json.loads(Path(cfg["study"]["val20_ids_file"]).read_text(encoding="utf-8"))["case_ids"][: args.n_cases]
    if any(851 <= int(c) <= 1000 for c in ids):
        raise ValueError("Locked test IDs in diagnostic case list")
    pairs = resolve_case_pairs(cfg["study"]["dataset_root"], [int(c) for c in ids], check_headers=False)

    dev = torch.device(args.device)
    net = build_monai_model(arch="vnet", init_filters=16, gate_mode="conv_bias", enable_ds=True).to(dev)
    ckpt = Path(cfg["study"]["root_dir"]) / "checkpoints" / args.run / args.ckpt
    obj = torch.load(ckpt, map_location=dev, weights_only=True)
    net.load_state_dict(obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj,
                        strict=True)
    net.eval()
    bs_net = BatchStatsBN(copy.deepcopy(net)).to(dev)

    tf = build_raw_eval_transform(pixdim_xyz=tuple(cfg["preprocessing"]["pixdim_xyz"]),
                                  body_crop_hu=float(cfg["preprocessing"]["body_crop_hu"]))
    inv = build_monai_invert_op(tf)
    ie = cfg["inference_eval"]
    rows = []
    for cid, (img_p, lbl_p) in pairs.items():
        lbl = sitk.ReadImage(str(lbl_p))
        gt = sitk.GetArrayFromImage(lbl) > 0
        sp = tuple(float(s) for s in reversed(lbl.GetSpacing()))
        for mode, model in (("bn_running_stats", net), ("bn_batch_stats", bs_net)):
            prob, _ = infer_monai_native_zyx(model, img_p, tf, roi_size_xyz=tuple(ie["roi_size_xyz"]),
                                             sw_overlap=float(ie["sw_overlap"]), sw_mode=str(ie["sw_mode"]),
                                             sw_batch_size=int(ie["sw_batch_size"]), need_tta=False,
                                             device=str(dev), inv_op=inv)
            m = compute_metrics((prob >= 0.5).astype(np.uint8), gt, sp)
            rows.append({"run": args.run, "ckpt": args.ckpt, "case_id": int(cid), "bn_mode": mode,
                         "dice": m["dice"], "precision": m["precision"], "status": m["status"],
                         "fg_prob_max": float(prob.max())})
            print(rows[-1], flush=True)

    out = V2 / "results" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"vnet_bn_diag_{args.run}.csv", index=False)
    print(df.groupby("bn_mode")[["dice", "precision", "fg_prob_max"]].mean())


if __name__ == "__main__":
    main()
```

Run it on the lab PC after Stage 2A, when the GPU is free:

```powershell
& $PY Rasnet-v2/src/diag_vnet_bn.py --run pilot2a_vnet_f16_lr1e-4_seed42 --n-cases 3
```

**Interpretation:**

- **`bn_batch_stats` Dice ≫ `bn_running_stats` Dice:** the collapse is a BatchNorm running-statistics artefact (micro-batch 2). Report V-Net at its Val-94-locked LR as-is, and state the artefact in the limitations. Do **not** change V-Net's architecture or add V-Net-only fixes.
- **Both low:** genuine training failure at that LR. The LR lock already handles it by choosing another LR.

---

## 5. Where RASNet-v2 can legitimately win — and how to present it

| Claim | Where it comes from (pre-registered) | Strength |
| :--- | :--- | :--- |
| 6.5× fewer parameters than nnU-Net, 9.7× fewer than V-Net; GFLOPs, latency and VRAM on the 4080 SUPER | Stage 5 `_profile_model_efficiency` | Certain, objective |
| RASNet-v2 vs SegResNet-v2 at identical width, loss and protocol (3 seeds) | Family 1 (K = 4) | The causal architecture claim. Strongest scientific contribution if significant. |
| Precision / HD95 / clDice vs nnU-Net | Family 2 (K = 3) | Historically RASNet's strong side |
| Dice non-inferior to nnU-Net (margin −0.005) | Family 2 NI test | Realistic headline vs a 6.5× larger model |
| Accuracy–efficiency Pareto front (F16 and F32 tiers) | Stage 5 `_plot_pareto` | Visual, persuasive, honest |
| Generalisation to the prospective clinical cohort | Stage 6 / Phase I | Strongest external evidence if it holds |

**Things that would invalidate a win — do not do them:**

- Train RASNet longer, or with more seeds or LRs, than the baselines.
- Re-run until a seed wins.
- Drop hard cases.
- Look at Test-150 before Stage 4C.
- Add a post-processor only to RASNet.
- Report the better of F16 and F32 per metric.
- Cite pilot numbers as results.

**If you later want a "RASNet-v3" with real architecture improvements**, it is a *new* study. CLAUDE.md freezes the current architectures. It needs its own pre-registration, validation-only design selection, and the same baselines re-run under the same protocol. Do not mix it into Rasnet-v2.

---

## 6. Release checklist (so people can actually adopt the model)

Adoption needs more than a better number. When the thesis is locked:

1. **Public repo:**
   - `rasnet_v2_model.py`
   - a minimal `predict.py` (NIfTI in, NIfTI out, using the exact `eval_fair_v2.infer_monai_native_zyx` path)
   - `requirements.txt` pinned to the lab environment
2. **Weights** on a model hub, with SHA-256 matching `CHECKPOINTS.md`, and a model card listing:
   - training data (ImageCAS split)
   - intended use (research only, not clinical)
   - metrics with CIs, efficiency numbers, known failure cases (e.g. ostial disconnections, test cases 0930 / 0978 / 0866 from Phase H)
3. **Licence** and citation file. ImageCAS licence terms must allow weight redistribution; check this before uploading.
4. **One-command reproducibility:** `run_queue.py --stage stage4c` on released weights reproduces the paper table.
