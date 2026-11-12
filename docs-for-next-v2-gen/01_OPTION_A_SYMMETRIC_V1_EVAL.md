# 01 — Option A: Symmetric Post-Hoc Evaluation of the Frozen v1 Checkpoints

**Target machine:** lab PC `C:\Thesis_RASNET`, RTX 4080 SUPER 16 GB, env `C:\Thesis_RASNET_env\bench` (Python 3.12, torch 2.x + cu126, MONAI 1.6.1, nnunetv2 2.8.1).
**Status of this document:** protocol + ready-to-paste code. Every number in the thesis must come from the files this run produces (see `CLAUDE.md`).

**Code validation (2026-11-12, done on a scratch copy of the repo on the H: machine, CPU only, MONAI 1.5.1, no data):**

- The three code blocks below were extracted verbatim and compiled.
- `tests/test_optionA_symmetric_eval.py`: **18 / 18 pass**. This includes 8-flip TTA correctness on a synthetic NIfTI, nnU-Net mirroring via the synthetic predictor, lock immutability, the protected-folder guard, the timing helper, figure writing, and the code-scoped `git_info` dirty flag. Whole `tests/` suite: **51 / 51** (33 existing + 18 new).
- The `stats` stage, the generated `OPTION_A_RESULTS.md` and the three figures were exercised on fabricated per-case rows. The figures were rendered and inspected for label collisions and fixed. Those rows were deleted and are not results.
- **Not yet verified:** GPU inference on real data and parity with locked numbers. That is exactly what the `parity` stage checks on the lab PC. Treat it as a LAB-CHECK.

---

## 0. Read this first — what Option A can and cannot prove

**What it does.** It gives the four sliding-window v1 models the exact same inference, test-time augmentation (TTA) and threshold-tuning budget. The models are RASNet, SegResNet, V-Net and nnU-Net V2. Each model is then evaluated once on the locked test set.

**Why it is needed.** Your current results contain two asymmetries a reviewer will find:

1. **Inference mismatch.** In Arms A/B/C, the MONAI models used 25 % overlap with constant blending, while nnU-Net used 50 % overlap with Gaussian blending. TTA also differed: RASNet got 4 passes, nnU-Net got 8, and SegResNet and V-Net got none (`benchmark/run_benchmark.py:395`).
2. **Tuning-budget mismatch in Phase H.** RASNet alone was searched over 152 recipes, while the baselines got only a 9-value threshold sweep.

Option A removes both asymmetries.

**What it cannot fix.** Training differences are baked into the weights:

| | MONAI models (incl. RASNet) | nnU-Net |
| :--- | :--- | :--- |
| Spacing | 0.5³ mm | 0.35 × 0.35 × 0.5 mm |
| Patch size | 96³ | 96 × 160 × 160 |
| Normalisation | `[-100, 800] → [0, 1]` | CT normalisation |
| Augmentation | flips only | full nnU-Net augmentation |
| Parameters | 4.7 M | 30.8 M |

Option A is therefore a **supplementary, retrospective inference-parity analysis**. The test IDs 851–1000 have already been looked at in Arms A/B/C and Phase H, so say so in the thesis. The causal "is the RASNet design better" question is answered by Rasnet-v2 (see `02_RASNET_V2_FAIRNESS_FIXES_AND_WIN_STRATEGY.md`).

**Honest expectation, based on existing files:**

- On validation, nnU-Net single-pass Arm C is already at 0.8075 Dice (`CURRENT_EVIDENCE_INDEX.md` §2).
- RASNet's best Phase H validation result, after 152 recipes, is 0.8023 (`PHASE_H_OPTIMIZATION_REPORT.md` §2).
- With 8-pass TTA nnU-Net will also improve. SegResNet and V-Net will also gain from Gaussian blending.
- So **do not expect RASNet to beat nnU-Net on Dice in Option A.** It may stay ahead on precision. It is certainly ahead on efficiency: 4.71 M vs 30.79 M parameters and 123 vs 1,119 GFLOPs per patch (Table 4).
- §7 gives the pre-written thesis sentence for every possible outcome. That is how the thesis stays safe whatever the numbers say.

> I will not design the protocol to guarantee a RASNet win. A protocol tuned until one model wins is exactly what examiners reject, and it would destroy the value of every other result you have. What this protocol does is give RASNet the **best fair chance**: identical TTA, an identical and widened threshold grid that removes the 0.30 clamp that hurt RASNet, and Gaussian blending that Phase H showed RASNet benefits from. It also pre-declares efficiency-aware endpoints that RASNet can legitimately win.

---

## 1. Pre-registered protocol (commit this section before any GPU run)

| Item | Value (identical for every model) |
| :--- | :--- |
| Models | `rasnet` (`rasnet_best.pth`), `segresnet` (`segresnet_best.pt`), `vnet` (`vnet_best.pt`), `nnunet` (`fold_0/checkpoint_best.pth`). The 3D U-Net is **excluded**: it runs single-shot on a 128³ resize, so sliding window does not apply. Report its existing Table 2 rows with that caveat. |
| Preprocessing | Each model keeps the preprocessing it was trained with. This is unavoidable with frozen weights. MONAI models: `benchmark/adapters/monai_models.py:33-41`. nnU-Net: its native plans. |
| Patch size | Each model's trained patch: 96³ for MONAI, 96 × 160 × 160 for nnU-Net. |
| Sliding window | overlap 0.50, Gaussian weighting, σ = 0.125 × patch. This matches nnU-Net's `compute_gaussian(sigma_scale=1/8)`, verified in MIC-DKFZ/nnUNet `predict_from_raw_data.py`. |
| Mode `sw050_gauss_1pass` | One pass. |
| Mode `sw050_gauss_tta8` | 8-pass mirroring: identity plus all 7 non-empty combinations of the 3 spatial flips, **logits averaged, then softmax**. This is exactly what `nnUNetPredictor._internal_maybe_mirror_and_predict` does (all `itertools.combinations` of the mirror axes; raw logits summed, divided by the pass count). For nnU-Net the native predictor with `use_mirroring=True` is used. **Implementation difference to disclose:** nnU-Net mirrors each tile inside its sliding window, while the MONAI path flips the whole volume and then runs the sliding window. Every mirrored prediction is still a Gaussian-blended average of the same patch predictions, but tile borders fall at different positions, so the results are numerically close rather than bit-identical. State this in the Methods. |
| AMP | fp16 autocast on CUDA for MONAI (same as the original benchmark). nnU-Net uses its own autocast. |
| Threshold grid | τ ∈ {0.20, 0.25, …, 0.70} (11 values). The lower bound was widened from 0.30 to remove the boundary clamp. |
| Post-processing | 26-connected **Top-2 CC** for tuned recipes, none for raw recipes. No hysteresis and no Top-3, to keep the forking paths small and the same for everyone. |
| τ selection | Per model × mode, on **Val-94 only**: maximise mean Dice after Top-2 CC. Tie-break: closest to 0.50, then lower (same as `benchmark/thresholds.py`). |
| Test recipes | Four per model (below). Evaluated **once** on Test-150. |
| Primary recipe | **R4** (each model at its best symmetric pipeline). R1–R3 are secondary. |
| Metrics | `surface-v1` (`benchmark/metrics.py`): Dice, IoU, precision, recall, HD95, ASD, clDice, centreline recall, plus component counts. |
| Statistics | Two-sided Wilcoxon signed-rank, Holm within family, 2,000-bootstrap 95 % CI (`benchmark/stats.py`). |

Test recipes:

| ID | Mode | τ | CC |
| :--- | :--- | :--- | :--- |
| R1 | 1-pass | 0.50 | none |
| R2 | TTA-8 | 0.50 | none |
| R3 | 1-pass | τ\* locked for 1-pass | Top-2 |
| R4 | TTA-8 | τ\* locked for TTA-8 | Top-2 |

**Pre-declared hypothesis families.** Holm is applied within each family, evaluated on R4.

| Family | Comparison | Endpoints |
| :--- | :--- | :--- |
| F-A primary | RASNet vs nnU-Net | precision ↑, HD95 ↓, ASD ↓. This is the same K = 3 family as `config.lab.yaml → hypotheses`. |
| F-B | RASNet vs SegResNet (matched 4.7 M backbone) | Dice ↑, precision ↑, HD95 ↓, clDice ↑ |
| F-C | RASNet vs V-Net | Dice ↑, precision ↑, HD95 ↓, clDice ↑ |
| F-D secondary | RASNet vs nnU-Net | Dice, clDice, recall |
| NI | RASNet vs nnU-Net Dice non-inferiority | Lower bound of the 95 % bootstrap CI of the paired difference > **−0.005** (same margin as `Rasnet-v2/PREREGISTRATION_V2.md:53`) |
| Exploratory | Baseline vs baseline | Not used for claims |

**Freeze procedure:**

1. Paste this section into `BENCHMARK_PREREGISTRATION.md` as "Amendment: Option A".
2. Commit it, and record that commit hash in the run manifests.

---

## 2. Files to create (new files only; no existing code is modified)

| New file | Purpose |
| :--- | :--- |
| `tests/test_optionA_symmetric_eval.py` | Written **first**; must fail before the module exists (CLAUDE.md rule) |
| `benchmark/symmetric_eval.py` | Library: 8-flip TTA, τ grid, lock, recipes, stats |
| `scripts/run_optionA_symmetric_v1_eval.py` | CLI with stages `parity`, `val`, `lock`, `test`, `stats` |

Output goes **only** to `results/optionA_v1_symmetric_sw050/`. The library refuses to write into `val_full_94/`, `test_retrospective_150/`, `rasnet_fixed_weight_optimization/` or `thesis_outputs/`.

---

## 3. Step 1 — tests first (`tests/test_optionA_symmetric_eval.py`)

```python
"""Option A symmetric-evaluation contract tests (synthetic, CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark import symmetric_eval as se

REPO = Path(__file__).resolve().parents[1]


def _splits():
    return {"train": list(range(1, 11)), "val": [11, 12, 13], "test": [851, 852]}


# ---------------------------------------------------------------- protocol constants
def test_flip_combinations_are_eight_unique_with_identity_first():
    combos = se.flip_axis_combinations()
    assert combos[0] == ()
    assert len(combos) == 8 and len(set(combos)) == 8


def test_protocol_constants_are_the_preregistered_ones():
    assert se.TAU_GRID == (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    assert se.SW_OVERLAP == 0.50 and se.SW_MODE == "gaussian" and se.SIGMA_SCALE == 0.125
    assert se.MODES == ("sw050_gauss_1pass", "sw050_gauss_tta8")
    assert se.MODELS == ("rasnet", "segresnet", "vnet", "nnunet")


def test_every_model_gets_the_same_recipe_structure():
    lock = {"thresholds": {m: {se.MODES[0]: 0.35, se.MODES[1]: 0.40} for m in se.MODELS}}
    shapes = {m: [(r.recipe_id, r.mode, r.keep_top_k) for r in se.recipes_for(m, lock)] for m in se.MODELS}
    assert len({tuple(v) for v in shapes.values()}) == 1
    assert [r.recipe_id for r in se.recipes_for("rasnet", lock)] == [
        "R1_raw_1pass", "R2_raw_tta8", "R3_tuned_1pass_top2", "R4_tuned_tta8_top2"]


# ---------------------------------------------------------------- threshold selection
def test_select_tau_max_mean_then_closest_to_half_then_lower():
    flat = {cid: {t: 0.5 for t in se.TAU_GRID} for cid in (11, 12)}
    assert se.select_tau(flat, [11, 12])[0] == 0.50          # all tied -> 0.50
    peak = {cid: dict(v) for cid, v in flat.items()}
    peak[11][0.35] = 0.9
    assert se.select_tau(peak, [11, 12])[0] == 0.35          # clear max wins
    tie = {cid: {t: 0.1 for t in se.TAU_GRID} for cid in (11, 12)}
    for cid in tie:
        tie[cid][0.45] = tie[cid][0.55] = 0.8
    assert se.select_tau(tie, [11, 12])[0] == 0.45           # equal distance -> lower


def test_select_tau_rejects_locked_test_ids():
    scores = {851: {t: 0.5 for t in se.TAU_GRID}}
    with pytest.raises(ValueError, match="locked test"):
        se.select_tau(scores, [851])


def test_select_tau_rejects_incomplete_grid():
    scores = {11: {t: 0.5 for t in se.TAU_GRID[:-1]}}
    with pytest.raises(ValueError, match="grid"):
        se.select_tau(scores, [11])


def _grid_df(case_ids, models=se.MODELS):
    rows = []
    for m in models:
        for mode in se.MODES:
            for cid in case_ids:
                for t in se.TAU_GRID:
                    rows.append({"model": m, "case_id": cid, "mode": mode, "tau": t,
                                 "dice": 0.8 - abs(t - 0.40)})
    return pd.DataFrame(rows)


def test_build_lock_requires_full_validation_coverage():
    with pytest.raises(ValueError, match="validation"):
        se.build_lock(_grid_df([11, 12]), _splits(), se.MODELS)
    lock = se.build_lock(_grid_df([11, 12, 13]), _splits(), se.MODELS)
    assert lock["split"] == "val" and lock["case_ids"] == [11, 12, 13]
    assert all(lock["thresholds"][m][mode] == 0.40 for m in se.MODELS for mode in se.MODES)


def test_lock_write_is_immutable_and_load_detects_tampering(tmp_path):
    lock = se.build_lock(_grid_df([11, 12, 13]), _splits(), se.MODELS)
    p = tmp_path / "locked_recipes.json"
    sha = se.write_lock(p, lock)
    with pytest.raises(FileExistsError):
        se.write_lock(p, lock)
    data, sha2 = se.load_lock(p, _splits(), se.MODELS)
    assert sha == sha2
    bad = json.loads(p.read_text())
    bad["grid"] = [0.3, 0.5]
    p2 = tmp_path / "tampered.json"
    p2.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="grid"):
        se.load_lock(p2, _splits(), se.MODELS)


# ---------------------------------------------------------------- output guard
def test_out_dir_guard_protects_locked_results(tmp_path):
    for protected in ("results/benchmark_runs/test_retrospective_150",
                      "results/benchmark_runs/val_full_94",
                      "results/rasnet_fixed_weight_optimization",
                      "results/thesis_outputs"):
        with pytest.raises(ValueError, match="protected"):
            se.check_out_dir(REPO / protected, REPO, resume=False)
    d = tmp_path / "out"
    se.check_out_dir(d, REPO, resume=False)
    (d / "x.csv").write_text("a")
    with pytest.raises(FileExistsError):
        se.check_out_dir(d, REPO, resume=False)
    se.check_out_dir(d, REPO, resume=True)


# ---------------------------------------------------------------- scoring & recipes
def test_score_tau_grid_and_evaluate_recipe_on_perfect_prediction():
    gt = np.zeros((20, 20, 20), bool)
    gt[5:15, 5:15, 5:15] = True
    prob = gt.astype(np.float32) * 0.9
    scores = se.score_tau_grid(prob, gt, (0.5, 0.5, 0.5))
    assert set(scores) == set(se.TAU_GRID) and all(v == 1.0 for v in scores.values())
    r = se.Recipe("R1_raw_1pass", se.MODES[0], 0.50, None)
    row = se.evaluate_recipe("rasnet", 11, r, prob, gt, (0.5, 0.5, 0.5))
    assert row["dice"] == 1.0 and row["recipe"] == "R1_raw_1pass" and row["keep_top_k"] == 0


def test_noninferiority_decision():
    a = pd.DataFrame({"case_id": [1, 2, 3, 4], "dice": [0.81, 0.82, 0.80, 0.83]})
    b = pd.DataFrame({"case_id": [1, 2, 3, 4], "dice": [0.80, 0.81, 0.79, 0.82]})
    res = se.paired_noninferiority(a, b, "dice", margin=-0.005)
    assert res["non_inferior"] and res["superior"]
    res2 = se.paired_noninferiority(b, a, "dice", margin=-0.005)
    assert not res2["non_inferior"]


# ---------------------------------------------------------------- TTA correctness (MONAI path)
def _write_volume(tmp_path, cid=11, shape=(40, 40, 40)):
    sitk = pytest.importorskip("SimpleITK")
    rng = np.random.default_rng(0)
    arr = rng.uniform(100, 700, shape).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 0.5))
    # SimpleITK treats "*.img.nii.gz" as Analyze (.hdr/.img); write a temp NIfTI name then rename,
    # exactly like benchmark.adapters.nnunet_adapter.export_mask_like does.
    p = tmp_path / f"{cid}.img.nii.gz"
    tmp = tmp_path / f"{cid}_img_tmp.nii.gz"
    sitk.WriteImage(img, str(tmp))
    tmp.replace(p)
    lbl = sitk.GetImageFromArray((arr > 600).astype(np.uint8))
    lbl.CopyInformation(img)
    lp = tmp_path / f"{cid}.label.nii.gz"
    sitk.WriteImage(lbl, str(lp))
    return p, lp, arr.shape


def _monai_out(tmp_path, net):
    torch = pytest.importorskip("torch")
    from benchmark.adapters.monai_models import build_transform
    p, _, shape = _write_volume(tmp_path)
    tf = build_transform("rasnet")
    batch = tf({"image": str(p)})
    with torch.no_grad():
        out = se.infer_monai_symmetric("rasnet", net.eval(), batch, tf, "cpu",
                                       roi_size=(32, 32, 32), sw_batch_size=2)
    return out, shape


def test_tta8_equals_1pass_for_flip_equivariant_net(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    out, shape = _monai_out(tmp_path, torch.nn.Conv3d(1, 2, kernel_size=1))
    p1, p8 = out[se.MODES[0]], out[se.MODES[1]]
    assert p1.shape == shape and p8.shape == shape
    assert np.all((p1 >= 0) & (p1 <= 1))
    np.testing.assert_allclose(p1, p8, atol=1e-5)


def test_tta8_actually_flips_for_asymmetric_net(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    out, _ = _monai_out(tmp_path, torch.nn.Conv3d(1, 2, kernel_size=3, padding=1))
    assert np.max(np.abs(out[se.MODES[0]] - out[se.MODES[1]])) > 1e-4


# ---------------------------------------------------------------- nnU-Net path requests mirroring
def test_nnunet_wrapper_returns_1pass_and_mirrored(tmp_path):
    pytest.importorskip("torch")
    from benchmark.run_benchmark import _SyntheticNNUNetPredictor
    p, lp, _ = _write_volume(tmp_path, cid=11)
    got = list(se.iter_nnunet_modes(_SyntheticNNUNetPredictor(), [11], {11: (p, lp)}))
    assert len(got) == 1
    cid, modes = got[0]
    assert cid == 11
    assert float(modes[se.MODES[0]].max()) == pytest.approx(0.70)
    assert float(modes[se.MODES[1]].max()) == pytest.approx(0.80)   # mirroring was enabled


# ---------------------------------------------------------------- timing & figures
def test_monai_tta_false_returns_only_1pass(tmp_path):
    torch = pytest.importorskip("torch")
    from benchmark.adapters.monai_models import build_transform
    p, _, _ = _write_volume(tmp_path)
    tf = build_transform("rasnet")
    out = se.infer_monai_symmetric("rasnet", torch.nn.Conv3d(1, 2, 1).eval(), tf({"image": str(p)}), tf,
                                   "cpu", roi_size=(32, 32, 32), sw_batch_size=2, tta=False)
    assert list(out) == [se.MODES[0]]


def test_time_calls_excludes_warmup_and_times_both_modes():
    calls = []
    rows = se._time_calls(lambda cid, tta: calls.append((cid, tta)), "rasnet", [11, 12, 13], "cpu", warmup=1)
    assert calls[0] == (11, False)                                   # warm-up, untimed
    assert sorted({r["case_id"] for r in rows}) == [12, 13]
    assert sorted({r["mode"] for r in rows}) == sorted(se.MODES)
    s = se.summarize_timing(pd.DataFrame(rows))
    assert set(s["mode"]) == set(se.MODES) and (s["n"] == 2).all()
    with pytest.raises(ValueError):
        se._time_calls(lambda c, t: None, "rasnet", [11], "cpu", warmup=1)


def test_make_figures_writes_png_and_pdf(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    rows = []
    for m in se.MODELS:
        for cid in range(851, 871):
            for r in ("R1_raw_1pass", "R4_tuned_tta8_top2"):
                rows.append({"model": m, "case_id": cid, "recipe": r, "tau": 0.4, "status": "ok",
                             **{k: float(rng.uniform(0.7, 0.85)) for k in
                                ("dice", "iou", "precision", "recall", "cldice", "centerline_recall")},
                             "hd95": 8.0, "asd": 1.5, "components_total": 2})
    per_case = pd.DataFrame(rows)
    eff = pd.DataFrame({"model_key": list(se.MODELS), "Parameters_M": [4.707, 4.701, 45.599, 30.786],
                        "GFLOPs_per_Patch": [123.39, 122.56, 617.23, 1119.34]})
    files = se.make_figures(se.summarize(per_case), per_case, eff, tmp_path)
    assert len(files) == 6 and all(Path(f).is_file() and Path(f).stat().st_size > 0 for f in files)


# ---------------------------------------------------------------- provenance: code-scoped dirty flag
def test_git_info_ignores_run_logs_but_flags_code_changes(tmp_path):
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("git not available")

    def git(*a):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=tmp_path,
                       check=True, capture_output=True)

    (tmp_path / "benchmark").mkdir()
    (tmp_path / "benchmark" / "m.py").write_text("x = 1\n")
    (tmp_path / "Rasnet-v2" / "logs").mkdir(parents=True)
    (tmp_path / "Rasnet-v2" / "logs" / "queue_status.json").write_text("{}\n")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    assert se.git_info(tmp_path)["dirty_code"] is False

    (tmp_path / "Rasnet-v2" / "logs" / "queue_status.json").write_text('{"done": 15}\n')
    info = se.git_info(tmp_path)
    assert info["dirty_code"] is False and info["dirty_any"] is True     # training logs don't count

    (tmp_path / "benchmark" / "m.py").write_text("x = 2\n")
    assert se.git_info(tmp_path)["dirty_code"] is True                   # code changes do

    assert se.git_info(tmp_path / "no_such_dir")["dirty_code"] is None   # git unavailable -> None, not True
```

Run on the lab PC from the repo root. The module does not exist yet, so this **must fail** with `ModuleNotFoundError: benchmark.symmetric_eval`. Keep that output for the record:

```powershell
$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"; Set-Location C:\Thesis_RASNET
& $PY -m pytest tests/test_optionA_symmetric_eval.py -q
```

---

## 4. Step 2 — library (`benchmark/symmetric_eval.py`)

```python
"""Option A: symmetric post-hoc inference + evaluation of frozen v1 checkpoints.

Retrospective supplement. Every model gets identical sliding-window blending, identical 8-pass
mirroring TTA (logit-averaged, matching nnUNetPredictor), an identical tau grid and identical
validation-only selection. Test IDs 851-1000 are never used for selection.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "optionA-1.0"
MODELS = ("rasnet", "segresnet", "vnet", "nnunet")
MONAI_MODELS = ("rasnet", "segresnet", "vnet")
MODES = ("sw050_gauss_1pass", "sw050_gauss_tta8")
SW_OVERLAP = 0.50
SW_MODE = "gaussian"
SIGMA_SCALE = 0.125
KEEP_TOP_K = 2
CONNECTIVITY = 26
TAU_GRID = tuple(round(0.20 + 0.05 * i, 2) for i in range(11))
LOCKED_TEST_RANGE = range(851, 1001)
NI_MARGIN = -0.005

# From CHECKPOINTS.md (verified 2026-11-10). A mismatch aborts the run.
EXPECTED_CHECKPOINT_SHA256 = {
    "rasnet": "8d4c25cd12d07c40423062b27b1d36836e5fd6cecb30c20c34055eeac976d1b8",
    "segresnet": "b8ea61d95106a0bbe15e1ce9f9e5af1a38d9508e59d0639d32bc883d93c6e109",
    "vnet": "a89d71bd41c3e773c68e3f4b5e27c5a067f959154aa049eead76f1642c5c1063",
    "nnunet": "9accddf283b1c480e386f30f426b9200ca8d4aa1ecd70d5b5af0b8aad6058711",
}

PROTECTED_RESULT_DIRS = (
    "results/benchmark_runs/val_full_94",
    "results/benchmark_runs/test_retrospective_150",
    "results/rasnet_fixed_weight_optimization",
    "results/thesis_outputs",
)

GRID_ROWS_PER_CASE = len(MODES) * len(TAU_GRID)
TEST_ROWS_PER_CASE = 4


# ============================================================ provenance helpers
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Folders whose state defines the Option A result. Rasnet-v2 run logs (queue_status.json,
# training_log.csv, DONE.flag) are git-tracked and change while training runs on the lab PC;
# they must not make an Option A manifest look "dirty".
CODE_PATHS = ("benchmark", "scripts", "tests", "Phase3_Local_Integration", "config.lab.yaml")


def git_info(repo: Path, code_paths=CODE_PATHS) -> dict:
    """Commit + two dirty flags: dirty_code (only the folders above; the one that matters) and
    dirty_any (whole repo, informational). None means git was unavailable."""
    def _run(*a):
        try:
            return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    commit = _run("rev-parse", "HEAD")
    any_changes = _run("status", "--porcelain")
    code_changes = _run("status", "--porcelain", "--", *code_paths)
    return {"commit": commit if commit is not None else "unavailable",
            "dirty_code": None if code_changes is None else bool(code_changes),
            "dirty_any": None if any_changes is None else bool(any_changes),
            "code_paths": list(code_paths)}


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "monai", "nnunetv2", "SimpleITK", "numpy", "cc3d"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "missing"
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def check_out_dir(out_dir: Path, repo: Path, resume: bool) -> Path:
    out = Path(out_dir).resolve()
    for rel in PROTECTED_RESULT_DIRS:
        prot = (Path(repo) / rel).resolve()
        if out == prot or prot in out.parents:
            raise ValueError(f"Refusing to write into protected results directory: {prot}")
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"{out} is not empty; pass --resume to continue an interrupted run")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================ inference
def flip_axis_combinations(spatial_axes=(2, 3, 4)) -> list[tuple[int, ...]]:
    """Identity + every non-empty combination of spatial axes (8 for 3D), as nnUNetPredictor."""
    combos: list[tuple[int, ...]] = [()]
    for r in range(1, len(spatial_axes) + 1):
        combos.extend(itertools.combinations(spatial_axes, r))
    return combos


def infer_monai_symmetric(name, net, batch, transform, device, roi_size=(96, 96, 96),
                          sw_batch_size=8, use_amp=None, tta=True) -> dict[str, np.ndarray]:
    """Return {mode: foreground prob (Z,Y,X) on the native grid} for the symmetric modes.

    Logits are Gaussian-blended by the sliding window, mirrored-averaged over 8 flips,
    then soft-maxed: the same order of operations as nnUNetPredictor. tta=False returns only
    the 1-pass mode (used by the timing stage).
    """
    import torch
    from monai.inferers import sliding_window_inference
    from monai.transforms import Invertd

    if name not in MONAI_MODELS:
        raise ValueError(f"Not a MONAI sliding-window model: {name}")
    if name == "vnet":
        roi_size = tuple(((int(r) + 15) // 16) * 16 for r in roi_size)
    net.eval()
    x = batch["image"].unsqueeze(0).to(device)
    amp = str(device).startswith("cuda") if use_amp is None else bool(use_amp)

    def logits(t):
        with torch.amp.autocast("cuda", enabled=amp):
            out = sliding_window_inference(t, roi_size, sw_batch_size, net, overlap=SW_OVERLAP,
                                           mode=SW_MODE, sigma_scale=SIGMA_SCALE)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.float()

    combos = flip_axis_combinations()
    probs = []
    with torch.no_grad():
        identity = logits(x)
        probs.append((MODES[0], torch.softmax(identity, dim=1)[:, 1:2]))
        if tta:
            acc = identity.clone()
            for axes in combos[1:]:
                acc += torch.flip(logits(torch.flip(x, axes)), axes)
            probs.append((MODES[1], torch.softmax(acc / float(len(combos)), dim=1)[:, 1:2]))
            del acc
        del identity

    inv = Invertd(keys=["pred"], transform=transform, orig_keys=["image"],
                  nearest_interp=False, to_tensor=True)
    out: dict[str, np.ndarray] = {}
    for mode, p in probs:
        b = dict(batch)
        b["pred"] = p.squeeze(0)
        arr = inv(b)["pred"].squeeze(0).detach().cpu().numpy()
        arr = np.transpose(arr, (2, 1, 0)).astype(np.float32, copy=False)
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Non-finite probability from {name} ({mode})")
        out[mode] = arr
    return out


def iter_nnunet_modes(predictor, case_ids, pairs):
    """Yield (cid, {mode: prob_zyx}) using native nnUNetPredictor (0.5 Gaussian; mirroring = 8 passes)."""
    from benchmark.run_benchmark import _run_nnunet_batch_probabilities
    for cid, p1, p8 in _run_nnunet_batch_probabilities(predictor, list(case_ids), pairs,
                                                         need_tta=True, num_processes=4):
        if p8 is None:
            raise RuntimeError(f"nnU-Net mirroring pass missing for case {cid}")
        yield int(cid), {MODES[0]: p1, MODES[1]: p8}


def _load_gt(label_path):
    import SimpleITK as sitk
    lbl = sitk.ReadImage(str(label_path))
    gt = sitk.GetArrayFromImage(lbl) > 0
    sp = tuple(float(s) for s in reversed(lbl.GetSpacing()))
    return gt, sp


def iter_case_probabilities(model, cfg, repo: Path, case_ids, pairs, device="cuda", sw_batch_size=8):
    """Yield (cid, {mode: prob}, gt_bool, spacing_zyx) for one model; prefetches CPU preprocessing."""
    case_ids = [int(c) for c in case_ids]
    if model == "nnunet":
        from benchmark.adapters.nnunet_adapter import build_nnunet_predictor
        predictor = build_nnunet_predictor(
            model_folder=str(Path(repo) / cfg["nnunet_model_folder"]),
            fold=int(cfg["nnunet_fold"]),
            checkpoint_name=str(cfg["nnunet_checkpoint_name"]),
            device=device,
            use_mirroring=False,
        )
        for cid, modes in iter_nnunet_modes(predictor, case_ids, pairs):
            gt, sp = _load_gt(pairs[cid][1])
            _check_shapes(model, cid, modes, gt)
            yield cid, modes, gt, sp
        return

    from benchmark.adapters.monai_models import build_model, build_transform
    net = build_model(model, str(Path(repo) / cfg["checkpoints"][model]), device)
    tf = build_transform(model)

    def prep(cid):
        return cid, tf({"image": str(pairs[cid][0])})

    with ThreadPoolExecutor(max_workers=2) as pool:
        queue = deque(pool.submit(prep, c) for c in case_ids[:2])
        nxt = len(queue)
        while queue:
            cid, batch = queue.popleft().result()
            if nxt < len(case_ids):
                queue.append(pool.submit(prep, case_ids[nxt]))
                nxt += 1
            modes = infer_monai_symmetric(model, net, batch, tf, device, sw_batch_size=sw_batch_size)
            del batch
            gt, sp = _load_gt(pairs[cid][1])
            _check_shapes(model, cid, modes, gt)
            yield cid, modes, gt, sp


def _check_shapes(model, cid, modes, gt):
    for mode, p in modes.items():
        if p.shape != gt.shape:
            raise RuntimeError(f"{model} case {cid} {mode}: prob {p.shape} != label {gt.shape}")


# ============================================================ scoring
def score_tau_grid(prob, gt_bool, spacing_zyx, grid=TAU_GRID, keep_top_k=KEEP_TOP_K) -> dict[float, float]:
    from benchmark.postprocess import prune_components
    from benchmark.thresholds import _fast_dice
    out = {}
    for tau in grid:
        raw = (prob >= tau).astype(np.uint8)
        cleaned, _ = prune_components(raw, spacing_zyx=spacing_zyx, keep_top_k=keep_top_k,
                                      connectivity=CONNECTIVITY)
        out[round(float(tau), 2)] = _fast_dice(cleaned, gt_bool)
    return out


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    mode: str
    tau: float
    keep_top_k: int | None


def recipes_for(model: str, lock: dict) -> tuple[Recipe, ...]:
    t = lock["thresholds"][model]
    return (
        Recipe("R1_raw_1pass", MODES[0], 0.50, None),
        Recipe("R2_raw_tta8", MODES[1], 0.50, None),
        Recipe("R3_tuned_1pass_top2", MODES[0], float(t[MODES[0]]), KEEP_TOP_K),
        Recipe("R4_tuned_tta8_top2", MODES[1], float(t[MODES[1]]), KEEP_TOP_K),
    )


def evaluate_recipe(model, cid, recipe: Recipe, prob, gt_bool, spacing_zyx) -> dict:
    from benchmark.metrics import METRIC_VERSION, compute_metrics
    from benchmark.postprocess import prune_components
    raw = (prob >= recipe.tau).astype(np.uint8)
    mask, info = prune_components(raw, spacing_zyx=spacing_zyx, keep_top_k=recipe.keep_top_k,
                                  connectivity=CONNECTIVITY)
    metrics = compute_metrics(mask, gt_bool, spacing_zyx)
    return {"model": model, "case_id": int(cid), "recipe": recipe.recipe_id, "mode": recipe.mode,
            "tau": float(recipe.tau), "keep_top_k": int(recipe.keep_top_k or 0),
            "metric_version": METRIC_VERSION, **metrics, **info}


# ============================================================ selection & lock
def select_tau(per_case: dict[int, dict[float, float]], case_ids, grid=TAU_GRID) -> tuple[float, dict]:
    ids = sorted(int(c) for c in case_ids)
    if any(c in LOCKED_TEST_RANGE for c in ids):
        raise ValueError("Threshold selection received locked test IDs 851-1000")
    if set(per_case) != set(ids):
        raise ValueError("Per-case scores do not match the requested validation IDs")
    grid = [round(float(t), 2) for t in grid]
    for cid in ids:
        if set(round(float(t), 2) for t in per_case[cid]) != set(grid):
            raise ValueError(f"Case {cid} does not cover the full tau grid")
    means = {t: float(np.mean([per_case[c][t] for c in ids])) for t in grid}
    best = min(grid, key=lambda t: (-round(means[t], 12), round(abs(t - 0.50), 4), t))
    return best, {f"{t:.2f}": means[t] for t in grid}


def build_lock(grid_df: pd.DataFrame, splits: dict, models, allow_partial=False) -> dict:
    from benchmark.preflight import validate_splits
    parts = validate_splits(splits)
    df = grid_df.drop_duplicates(subset=["model", "case_id", "mode", "tau"], keep="last")
    present = {int(c) for c in df["case_id"].unique()}
    if present & set(LOCKED_TEST_RANGE):
        raise ValueError("Validation grid file contains locked test IDs")
    if not allow_partial and present != parts["val"]:
        raise ValueError("Lock requires the entire validation split (94 cases)")
    ids = sorted(present)
    thresholds, scores = {}, {}
    for model in models:
        thresholds[model], scores[model] = {}, {}
        for mode in MODES:
            sub = df[(df["model"] == model) & (df["mode"] == mode)]
            per_case = {int(c): {round(float(t), 2): float(d) for t, d in zip(g["tau"], g["dice"])}
                        for c, g in sub.groupby("case_id")}
            tau, means = select_tau(per_case, ids)
            thresholds[model][mode] = tau
            scores[model][mode] = means
    return {"protocol_version": PROTOCOL_VERSION, "split": "val", "smoke": bool(allow_partial),
            "case_ids": ids, "grid": list(TAU_GRID), "keep_top_k": KEEP_TOP_K,
            "connectivity": CONNECTIVITY, "objective": "mean_val_dice_after_top2cc",
            "tie_breaker": "closest_to_0.50_then_lower", "thresholds": thresholds,
            "grid_scores": scores}


def write_lock(path: Path, payload: dict) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Lock already exists and is immutable: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    sha = sha256_file(path)
    path.with_suffix(".sha256").write_text(sha, encoding="utf-8")
    return sha


def load_lock(path: Path, splits: dict, models, allow_smoke=False) -> tuple[dict, str]:
    from benchmark.preflight import validate_splits
    parts = validate_splits(splits)
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    sha = sha256_file(path)
    side = path.with_suffix(".sha256")
    if side.is_file() and side.read_text(encoding="utf-8").strip() != sha:
        raise ValueError("Lock file hash does not match its .sha256 sidecar (modified after locking)")
    if data.get("split") != "val":
        raise ValueError("Lock must be selected on the validation split")
    if [round(float(t), 2) for t in data.get("grid", [])] != list(TAU_GRID):
        raise ValueError("Lock grid differs from the pre-registered tau grid")
    if data.get("smoke") and not allow_smoke:
        raise ValueError("Smoke lock cannot be used for a final test run")
    if not data.get("smoke") and set(data["case_ids"]) != parts["val"]:
        raise ValueError("Lock does not cover the full validation split")
    for m in models:
        for mode in MODES:
            t = float(data["thresholds"][m][mode])
            if round(t, 2) not in TAU_GRID:
                raise ValueError(f"Locked tau {t} for {m}/{mode} is off-grid")
    return data, sha


# ============================================================ CSV resume helpers
def done_keys(csv_path: Path, rows_per_case: int, key_cols=("model", "case_id")) -> set:
    p = Path(csv_path)
    if not p.is_file() or p.stat().st_size == 0:
        return set()
    df = pd.read_csv(p)
    counts = df.groupby(list(key_cols)).size()
    return {(str(m), int(c)) for (m, c), n in counts.items() if n >= rows_per_case}


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    p = Path(csv_path)
    pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)


# ============================================================ statistics
FAMILIES = {
    "FA_primary_rasnet_vs_nnunet": [
        {"a": "rasnet", "b": "nnunet", "metric": "precision", "direction": "higher"},
        {"a": "rasnet", "b": "nnunet", "metric": "hd95", "direction": "lower"},
        {"a": "rasnet", "b": "nnunet", "metric": "asd", "direction": "lower"},
    ],
    "FB_rasnet_vs_segresnet": [
        {"a": "rasnet", "b": "segresnet", "metric": m, "direction": d}
        for m, d in (("dice", "higher"), ("precision", "higher"), ("hd95", "lower"), ("cldice", "higher"))
    ],
    "FC_rasnet_vs_vnet": [
        {"a": "rasnet", "b": "vnet", "metric": m, "direction": d}
        for m, d in (("dice", "higher"), ("precision", "higher"), ("hd95", "lower"), ("cldice", "higher"))
    ],
    "FD_secondary_rasnet_vs_nnunet": [
        {"a": "rasnet", "b": "nnunet", "metric": m, "direction": "higher"}
        for m in ("dice", "cldice", "recall")
    ],
    "EX_baselines": [
        {"a": a, "b": b, "metric": "dice", "direction": "higher"}
        for a, b in (("nnunet", "segresnet"), ("nnunet", "vnet"), ("segresnet", "vnet"))
    ],
}
SUMMARY_METRICS = ("dice", "iou", "precision", "recall", "hd95", "asd", "cldice",
                   "centerline_recall", "components_total")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, recipe), g in df.groupby(["model", "recipe"]):
        r = {"model": model, "recipe": recipe, "n": int(len(g)),
             "n_ok": int((g["status"] == "ok").sum()), "tau": float(g["tau"].iloc[0])}
        for m in SUMMARY_METRICS:
            v = g[m].dropna().to_numpy(float)
            r[f"{m}_mean"] = float(v.mean()) if len(v) else np.nan
            r[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            r[f"{m}_median"] = float(np.median(v)) if len(v) else np.nan
            r[f"{m}_q25"] = float(np.quantile(v, 0.25)) if len(v) else np.nan
            r[f"{m}_q75"] = float(np.quantile(v, 0.75)) if len(v) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def run_families(df: pd.DataFrame, recipe: str) -> pd.DataFrame:
    from benchmark.stats import analyze_family
    sub = df[df["recipe"] == recipe]
    tables = {m: g.drop_duplicates("case_id").reset_index(drop=True) for m, g in sub.groupby("model")}
    out = []
    for fam, hyps in FAMILIES.items():
        hyps = [h for h in hyps if h["a"] in tables and h["b"] in tables]
        if not hyps:
            continue
        for r in analyze_family(tables, hyps, allow_nan_filtering=True):
            out.append({"family": fam, "recipe": recipe, **r})
    return pd.DataFrame(out)


def paired_noninferiority(a: pd.DataFrame, b: pd.DataFrame, metric="dice", margin=NI_MARGIN) -> dict:
    from benchmark.stats import bootstrap_ci
    pair = a[["case_id", metric]].merge(b[["case_id", metric]], on="case_id",
                                        validate="one_to_one", suffixes=("_a", "_b")).dropna()
    d = pair[f"{metric}_a"].to_numpy(float) - pair[f"{metric}_b"].to_numpy(float)
    lo, hi = bootstrap_ci(d, seed=42, n_boot=2000, confidence=0.95)
    return {"metric": metric, "n": int(len(d)), "mean_diff": float(d.mean()), "ci95_lo": lo,
            "ci95_hi": hi, "margin": float(margin), "non_inferior": bool(lo > margin),
            "superior": bool(lo > 0.0)}


# ============================================================ timing (efficiency evidence)
def _time_calls(run, model: str, case_ids, device: str, warmup: int = 1) -> list[dict]:
    """Time run(cid, tta) end-to-end for 1-pass and TTA-8 separately (same definition for every model).

    The first `warmup` cases are run once untimed (cuDNN autotune, allocator). CUDA is synchronised
    around every call; peak allocated VRAM is reset per mode.
    """
    import time
    cuda = str(device).startswith("cuda")
    if cuda:
        import torch
    ids = [int(c) for c in case_ids]
    if len(ids) <= warmup:
        raise ValueError("Need more cases than warm-up cases")
    for cid in ids[:warmup]:
        run(cid, False)
    rows = []
    for tta in (False, True):
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        mode_rows = []
        for cid in ids[warmup:]:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run(cid, tta)
            if cuda:
                torch.cuda.synchronize()
            mode_rows.append({"model": model, "case_id": cid, "mode": MODES[1] if tta else MODES[0],
                              "seconds": time.perf_counter() - t0})
        peak = float(torch.cuda.max_memory_allocated() / 2 ** 20) if cuda else float("nan")
        for r in mode_rows:
            r["peak_vram_mb"] = peak
        rows.extend(mode_rows)
    return rows


def time_model(model, cfg, repo: Path, case_ids, pairs, device="cuda", sw_batch_size=8, warmup=1):
    """Per-case wall-clock: preprocessing + inference + native-grid inversion, as a user would run it."""
    if model == "nnunet":
        from benchmark.adapters.nnunet_adapter import build_nnunet_predictor, infer_nnunet_probability
        predictor = build_nnunet_predictor(model_folder=str(Path(repo) / cfg["nnunet_model_folder"]),
                                           fold=int(cfg["nnunet_fold"]),
                                           checkpoint_name=str(cfg["nnunet_checkpoint_name"]),
                                           device=device, use_mirroring=False)

        def run(cid, tta):
            infer_nnunet_probability(predictor, str(pairs[cid][0]), tta=tta)
    else:
        from benchmark.adapters.monai_models import build_model, build_transform
        net = build_model(model, str(Path(repo) / cfg["checkpoints"][model]), device)
        tf = build_transform(model)

        def run(cid, tta):
            batch = tf({"image": str(pairs[cid][0])})
            infer_monai_symmetric(model, net, batch, tf, device, sw_batch_size=sw_batch_size, tta=tta)
    return _time_calls(run, model, case_ids, device, warmup)


def summarize_timing(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "mode"])
    return pd.DataFrame({
        "n": g["seconds"].size(),
        "seconds_median": g["seconds"].median(),
        "seconds_q25": g["seconds"].quantile(0.25),
        "seconds_q75": g["seconds"].quantile(0.75),
        "peak_vram_mb": g["peak_vram_mb"].max(),
    }).reset_index()


# ============================================================ figures (thesis-ready, static)
# Fixed colour per model (never by rank). Validated with the dataviz palette checker
# (light surface): CVD/normal-vision separation PASS; aqua/yellow are < 3:1 contrast, so every
# mark is also identified by a text label (direct labels / axis labels), never colour alone.
MODEL_LABELS = {"rasnet": "RASNet", "nnunet": "nnU-Net V2", "segresnet": "SegResNet", "vnet": "V-Net"}
MODEL_COLORS = {"rasnet": "#2a78d6", "nnunet": "#eb6834", "segresnet": "#1baf7a", "vnet": "#eda100"}
_INK, _INK2, _GRID = "#1f1f1d", "#5f5e5a", "#e6e5e0"


def _style(plt):
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8a86", "axes.labelcolor": _INK2, "xtick.color": _INK2,
        "ytick.color": _INK2, "axes.titlecolor": _INK, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.bbox": "tight",
    })


def _save(fig, out_dir: Path, name: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        p = Path(out_dir) / f"{name}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(str(p))
    return paths


def make_figures(summary: pd.DataFrame, per_case: pd.DataFrame, efficiency: pd.DataFrame,
                 out_dir: Path, primary_recipe: str = "R4_tuned_tta8_top2", ni_margin: float = NI_MARGIN):
    """Three figures, all computed from the result CSVs:
    1) Dice vs parameters (accuracy-efficiency, primary recipe)
    2) per-case Dice distributions, one panel per recipe
    3) paired per-case Dice difference RASNet - nnU-Net (primary recipe) with the NI margin
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    out_dir = Path(out_dir)
    written = []
    models = [m for m in MODELS if m in set(per_case["model"])]

    # 1 -- accuracy vs size
    s = summary[summary["recipe"] == primary_recipe].set_index("model")
    e = efficiency.set_index("model_key")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    xs = {m: float(e.loc[m, "Parameters_M"]) for m in models}
    pos, side = {}, {}
    for m in models:                                  # dodge models of near-identical size (RASNet/SegResNet)
        near = [o for o in models if abs(np.log(xs[o] / xs[m])) < 0.1]
        k, n = near.index(m), len(near)
        pos[m] = xs[m] * (1.0 + 0.10 * (k - (n - 1) / 2)) if n > 1 else xs[m]
        side[m] = -1 if (n > 1 and k < (n - 1) / 2) else 1
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for m in models:
        x, y = pos[m], float(s.loc[m, "dice_mean"])
        ax.vlines(x, float(s.loc[m, "dice_q25"]), float(s.loc[m, "dice_q75"]),
                  color=MODEL_COLORS[m], lw=2, alpha=0.45)
        ax.scatter([x], [y], s=64, color=MODEL_COLORS[m], edgecolor="white", linewidth=2, zorder=3,
                   label=MODEL_LABELS[m])
        ax.annotate(f"{MODEL_LABELS[m]}  {y:.3f}\n{float(e.loc[m, 'GFLOPs_per_Patch']):.0f} GFLOPs/patch",
                    (x, y), xytext=(8 * side[m], 3), textcoords="offset points", fontsize=8, color=_INK,
                    ha="left" if side[m] > 0 else "right")
    ax.set_xscale("log")
    ax.set_xlim(1.5, 120)
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Test Dice (mean; bar = IQR)")
    ax.grid(True, axis="y", color=_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Accuracy vs model size — {primary_recipe}", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    written += _save(fig, out_dir, "fig_optionA_dice_vs_params")
    plt.close(fig)

    # 2 -- per-case Dice distributions (small multiples, shared y)
    recipes = sorted(per_case["recipe"].unique())
    fig, axes = plt.subplots(1, len(recipes), figsize=(2.3 * len(recipes), 3.4), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], recipes):
        data = [per_case[(per_case["recipe"] == r) & (per_case["model"] == m)]["dice"].dropna().to_numpy()
                for m in models]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops={"color": _INK, "lw": 1.2},
                        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.5})
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(MODEL_COLORS[m])
            patch.set_alpha(0.45)
            patch.set_edgecolor(MODEL_COLORS[m])
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=35, ha="right")
        ax.set_title(r.replace("_", " "), fontsize=9, loc="left")
        ax.grid(True, axis="y", color=_GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Per-case Dice (Test-150)")
    written += _save(fig, out_dir, "fig_optionA_dice_distributions")
    plt.close(fig)

    # 3 -- paired difference RASNet - nnU-Net (single series)
    sub = per_case[per_case["recipe"] == primary_recipe]
    a = sub[sub["model"] == "rasnet"][["case_id", "dice"]]
    b = sub[sub["model"] == "nnunet"][["case_id", "dice"]]
    if len(a) and len(b):
        pair = a.merge(b, on="case_id", suffixes=("_r", "_n")).dropna()
        d = np.sort((pair["dice_r"] - pair["dice_n"]).to_numpy())
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.scatter(np.arange(len(d)), d, s=10, color=MODEL_COLORS["rasnet"], zorder=3)
        ax.axhline(0, color=_INK2, lw=0.8)
        ax.axhline(ni_margin, color=_INK2, lw=0.8, ls="--")
        ax.text(0.98, 0.04, f"dashed line: non-inferiority margin {ni_margin:+.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=_INK2)
        ax.annotate(f"mean {d.mean():+.4f}  ({int((d > 0).sum())} cases RASNet higher / "
                    f"{int((d < 0).sum())} lower)", (0.02, 0.95), xycoords="axes fraction",
                    fontsize=8, color=_INK, va="top")
        ax.set_xlabel("Test cases, sorted by difference")
        ax.set_ylabel("Dice: RASNet − nnU-Net V2")
        ax.grid(True, axis="y", color=_GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"Per-case paired difference — {primary_recipe}", loc="left", fontsize=10)
        written += _save(fig, out_dir, "fig_optionA_paired_diff_rasnet_vs_nnunet")
        plt.close(fig)
    return written
```

---

## 5. Step 3 — CLI (`scripts/run_optionA_symmetric_v1_eval.py`)

```python
#!/usr/bin/env python
"""Option A runner — stages: parity -> val -> lock -> test -> stats (run in this order).

Example (lab PC, repo root):
  python scripts/run_optionA_symmetric_v1_eval.py --stage parity
  python scripts/run_optionA_symmetric_v1_eval.py --stage val  --resume
  python scripts/run_optionA_symmetric_v1_eval.py --stage lock --resume
  python scripts/run_optionA_symmetric_v1_eval.py --stage test --resume
  python scripts/run_optionA_symmetric_v1_eval.py --stage stats --resume
  python scripts/run_optionA_symmetric_v1_eval.py --stage timing     # efficiency evidence (Val cases)
  python scripts/run_optionA_symmetric_v1_eval.py --stage stats      # re-run to add the timing table
  python scripts/run_optionA_symmetric_v1_eval.py --stage figures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from benchmark import symmetric_eval as se  # noqa: E402
from benchmark.dataset import resolve_case_pairs  # noqa: E402
from benchmark.preflight import validate_splits  # noqa: E402

OUT_DEFAULT = ROOT / "results" / "optionA_v1_symmetric_sw050"


def load_cfg_and_splits():
    cfg_path = ROOT / "config.lab.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    sp_path = ROOT / cfg["split_file"]
    raw = sp_path.read_bytes()
    splits = json.loads(raw.decode("utf-8"))
    validate_splits(splits)
    return cfg, splits, {"config_sha256": se.sha256_file(cfg_path),
                         "splits_sha256": hashlib.sha256(raw).hexdigest()}


def verify_checkpoints(cfg, models):
    out = {}
    for m in models:
        p = ROOT / cfg["checkpoints"][m]
        if not p.is_file():
            raise FileNotFoundError(p)
        sha = se.sha256_file(p)
        if sha != se.EXPECTED_CHECKPOINT_SHA256[m]:
            raise RuntimeError(f"{m} checkpoint SHA-256 {sha} != CHECKPOINTS.md {se.EXPECTED_CHECKPOINT_SHA256[m]}")
        out[m] = {"path": str(p.relative_to(ROOT)), "sha256": sha}
    return out


def write_manifest(out_dir, stage, args, prov, extra=None):
    src = [ROOT / "benchmark" / f for f in sorted(p.name for p in (ROOT / "benchmark").glob("*.py"))]
    src += sorted((ROOT / "benchmark" / "adapters").glob("*.py")) + [Path(__file__).resolve()]
    payload = {
        "protocol_version": se.PROTOCOL_VERSION, "stage": stage,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git": se.git_info(ROOT), "env": se.env_info(), **prov,
        "source_sha256": {str(p.relative_to(ROOT)): se.sha256_file(p) for p in src},
        "protocol": {"sw_overlap": se.SW_OVERLAP, "sw_mode": se.SW_MODE, "sigma_scale": se.SIGMA_SCALE,
                     "modes": list(se.MODES), "tau_grid": list(se.TAU_GRID), "keep_top_k": se.KEEP_TOP_K},
        **(extra or {}),
    }
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    (out_dir / f"manifest_{stage}_{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bounded_pipeline(gen, fn, max_inflight=2, workers=2):
    """Overlap GPU inference (producer) with CPU metric work (consumer) with bounded RAM."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        q = deque()
        for item in gen:
            while len(q) >= max_inflight:
                yield q.popleft().result()
            q.append(pool.submit(fn, *item))
        while q:
            yield q.popleft().result()


# -------------------------------------------------------------------------------- stages
def stage_parity(args, cfg, splits, out_dir):
    """Prove the new code path reproduces locked numbers before trusting it."""
    import torch
    from benchmark.adapters.monai_models import build_model, build_transform, infer_monai_both_passes
    from benchmark.thresholds import _fast_dice

    ids = sorted(splits["val"])[: args.parity_cases]
    pairs = resolve_case_pairs(cfg["dataset_root"], ids, check_headers=False)
    ref_a = pd.read_csv(ROOT / "results/benchmark_runs/val_full_94/per_case_arm_A.csv")
    ref_h = pd.read_csv(ROOT / "results/rasnet_fixed_weight_optimization/val_94_selected_per_case.csv")
    tol = float(cfg.get("historical_tolerance", 1e-4))
    rows = []

    def ref(df, **kw):
        sub = df
        for k, v in kw.items():
            sub = sub[sub[k] == v]
        if len(sub) != 1:
            raise RuntimeError(f"Reference row not unique for {kw}")
        return float(sub["dice"].iloc[0])

    for model in [m for m in args.models if m in se.MONAI_MODELS]:
        net = build_model(model, str(ROOT / cfg["checkpoints"][model]), args.device)
        tf = build_transform(model)
        for cid in ids:
            batch = tf({"image": str(pairs[cid][0])})
            gt, sp = se._load_gt(pairs[cid][1])
            p_old, _ = infer_monai_both_passes(model, net, str(pairs[cid][0]), args.device,
                                               sw_batch_size=8, need_tta=False, transform=tf, batch=batch,
                                               overlap=0.25, sw_mode="constant")
            rows.append({"model": model, "case_id": cid, "check": "armA_sw025_const_tau050",
                         "new": _fast_dice((p_old >= 0.5).astype(np.uint8), gt),
                         "ref": ref(ref_a, model=model, case_id=cid)})
            if model == "rasnet":
                modes = se.infer_monai_symmetric(model, net, batch, tf, args.device, sw_batch_size=8)
                rows.append({"model": model, "case_id": cid, "check": "phaseH_sw050_gauss_1pass_tau050",
                             "new": _fast_dice((modes[se.MODES[0]] >= 0.5).astype(np.uint8), gt),
                             "ref": ref(ref_h, candidate_id="sw050_gauss_1pass__single_tau0.50_nocc",
                                        case_id=cid)})
        del net
        torch.cuda.empty_cache()

    if "nnunet" in args.models:
        for cid, modes, gt, _ in se.iter_case_probabilities("nnunet", cfg, ROOT, ids, pairs, args.device):
            rows.append({"model": "nnunet", "case_id": cid, "check": "armA_native_1pass_tau050",
                         "new": _fast_dice((modes[se.MODES[0]] >= 0.5).astype(np.uint8), gt),
                         "ref": ref(ref_a, model="nnunet", case_id=cid)})

    df = pd.DataFrame(rows)
    df["abs_diff"] = (df["new"] - df["ref"]).abs()
    df.to_csv(out_dir / "parity_check.csv", index=False)
    worst = float(df["abs_diff"].max())
    print(df.groupby(["model", "check"])["abs_diff"].max().to_string())
    if worst > tol:
        raise SystemExit(f"PARITY FAILED: max |dDice| = {worst:.2e} > {tol:.0e}. Do not continue.")
    print(f"[parity] OK: max |dDice| = {worst:.2e} <= {tol:.0e}")


def stage_val(args, cfg, splits, out_dir):
    ids = sorted(splits["val"])[: args.smoke or None]
    pairs = resolve_case_pairs(cfg["dataset_root"], ids, check_headers=False)
    csv = out_dir / "val_grid_scores.csv"
    for model in args.models:
        done = se.done_keys(csv, se.GRID_ROWS_PER_CASE)
        todo = [c for c in ids if (model, c) not in done]
        print(f"[val] {model}: {len(todo)} cases to run", flush=True)
        if not todo:
            continue
        t0 = time.time()

        def score(cid, modes, gt, sp, _model=model):
            rows = []
            for mode, prob in modes.items():
                for tau, d in se.score_tau_grid(prob, gt, sp).items():
                    rows.append({"model": _model, "case_id": cid, "mode": mode, "tau": tau, "dice": d})
            return rows

        gen = se.iter_case_probabilities(model, cfg, ROOT, todo, pairs, args.device, args.sw_batch_size)
        for i, rows in enumerate(_bounded_pipeline(gen, score), 1):
            se.append_rows(csv, rows)
            if i % 10 == 0 or i == len(todo):
                print(f"  [{model}] {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


def stage_lock(args, cfg, splits, out_dir):
    df = pd.read_csv(out_dir / "val_grid_scores.csv")
    payload = se.build_lock(df, splits, args.models, allow_partial=bool(args.smoke))
    sha = se.write_lock(out_dir / "locked_recipes.json", payload)
    print(json.dumps(payload["thresholds"], indent=2))
    print(f"[lock] locked_recipes.json sha256={sha}")
    return {"locked_recipes_sha256": sha}


def stage_test(args, cfg, splits, out_dir):
    lock, sha = se.load_lock(out_dir / "locked_recipes.json", splits, args.models, allow_smoke=bool(args.smoke))
    ids = sorted(splits["test"])[: args.smoke or None]
    pairs = resolve_case_pairs(cfg["dataset_root"], ids, check_headers=False)
    csv = out_dir / "test_per_case.csv"
    for model in args.models:
        recipes = se.recipes_for(model, lock)
        done = se.done_keys(csv, se.TEST_ROWS_PER_CASE)
        todo = [c for c in ids if (model, c) not in done]
        print(f"[test] {model}: {len(todo)} cases | recipes={[(r.recipe_id, r.tau) for r in recipes]}", flush=True)
        if not todo:
            continue
        t0 = time.time()

        def evaluate(cid, modes, gt, sp, _model=model, _recipes=recipes):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [pool.submit(se.evaluate_recipe, _model, cid, r, modes[r.mode], gt, sp) for r in _recipes]
                return [f.result() for f in futs]

        gen = se.iter_case_probabilities(model, cfg, ROOT, todo, pairs, args.device, args.sw_batch_size)
        for i, rows in enumerate(_bounded_pipeline(gen, evaluate), 1):
            se.append_rows(csv, rows)
            if i % 10 == 0 or i == len(todo):
                print(f"  [{model}] {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return {"locked_recipes_sha256": sha}


def stage_stats(args, cfg, splits, out_dir):
    df = pd.read_csv(out_dir / "test_per_case.csv").drop_duplicates(
        subset=["model", "case_id", "recipe"], keep="last")
    expected = len(splits["test"]) if not args.smoke else args.smoke
    counts = df.groupby(["model", "recipe"]).size()
    if not args.smoke and (counts != expected).any():
        raise SystemExit(f"Incomplete test results:\n{counts}")
    summ = se.summarize(df)
    summ.to_csv(out_dir / "summary_test.csv", index=False)
    fam = pd.concat([se.run_families(df, r) for r in sorted(df["recipe"].unique())], ignore_index=True)
    fam.to_csv(out_dir / "paired_stats_families.csv", index=False)
    ni = []
    for r in sorted(df["recipe"].unique()):
        sub = df[df["recipe"] == r]
        a, b = sub[sub["model"] == "rasnet"], sub[sub["model"] == "nnunet"]
        if len(a) and len(b):
            ni.append({"recipe": r, **se.paired_noninferiority(a, b, "dice")})
    pd.DataFrame(ni).to_csv(out_dir / "noninferiority_rasnet_vs_nnunet_dice.csv", index=False)
    t_path = out_dir / "timing_summary.csv"
    write_report(out_dir, summ, fam, pd.DataFrame(ni), pd.read_csv(t_path) if t_path.is_file() else None)


def stage_timing(args, cfg, splits, out_dir):
    """Validation cases only; never test. Run with nothing else on the GPU."""
    ids = sorted(splits["val"])[: args.timing_cases + 1]          # +1 warm-up case
    pairs = resolve_case_pairs(cfg["dataset_root"], ids, check_headers=False)
    rows = []
    for model in args.models:
        print(f"[timing] {model}", flush=True)
        rows += se.time_model(model, cfg, ROOT, ids, pairs, args.device, args.sw_batch_size, warmup=1)
        import torch
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "timing_val.csv", index=False)
    summ = se.summarize_timing(df)
    summ.to_csv(out_dir / "timing_summary.csv", index=False)
    print(summ.to_string(index=False))


def stage_figures(args, cfg, splits, out_dir):
    summ = pd.read_csv(out_dir / "summary_test.csv")
    per_case = pd.read_csv(out_dir / "test_per_case.csv").drop_duplicates(
        subset=["model", "case_id", "recipe"], keep="last")
    eff = pd.read_csv(ROOT / "results/thesis_outputs/Table4_computational_efficiency_rtx4080super.csv")
    for p in se.make_figures(summ, per_case, eff, out_dir):
        print("[figures]", p)


def write_report(out_dir, summ, fam, ni, timing=None):
    """Markdown generated only from the CSVs above (every number traceable to a file)."""
    def fmt(r, m):
        return f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_sd']:.4f}"
    lines = ["# Option A results (generated; do not hand-edit)", "",
             "Source files: `summary_test.csv`, `paired_stats_families.csv`, "
             "`noninferiority_rasnet_vs_nnunet_dice.csv`.", ""]
    for recipe in sorted(summ["recipe"].unique()):
        lines += [f"## {recipe}", "", "| Model | τ | Dice | Precision | Recall | HD95 (mm) | ASD (mm) | clDice |",
                  "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
        for _, r in summ[summ["recipe"] == recipe].sort_values("dice_mean", ascending=False).iterrows():
            lines.append(f"| {r['model']} | {r['tau']:.2f} | {fmt(r, 'dice')} | {fmt(r, 'precision')} | "
                         f"{fmt(r, 'recall')} | {fmt(r, 'hd95')} | {fmt(r, 'asd')} | {fmt(r, 'cldice')} |")
        lines.append("")
    lines += ["## Pre-declared families (Holm within family)", "",
              "| Recipe | Family | a vs b | Metric | Mean diff (a−b) | 95% CI | p | Holm p | W/T/L |",
              "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |"]
    for _, r in fam.iterrows():
        lines.append(f"| {r['recipe']} | {r['family']} | {r['a']} vs {r['b']} | {r['metric']} | "
                     f"{r['mean_difference']:+.4f} | [{r['diff_ci95_lo']:+.4f}, {r['diff_ci95_hi']:+.4f}] | "
                     f"{r['p']:.3g} | {r['holm_p']:.3g} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines += ["", "## Dice non-inferiority RASNet vs nnU-Net (margin −0.005)", "",
              "| Recipe | Mean diff | 95% CI | Non-inferior | Superior |", "| :--- | :---: | :---: | :---: | :---: |"]
    for _, r in ni.iterrows():
        lines.append(f"| {r['recipe']} | {r['mean_diff']:+.4f} | [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | "
                     f"{r['non_inferior']} | {r['superior']} |")
    if timing is not None and len(timing):
        lines += ["", "## End-to-end inference time per case (Val cases, RTX 4080 SUPER; `timing_summary.csv`)", "",
                  "| Model | Mode | n | Median s/case [IQR] | Peak allocated VRAM (MB) |",
                  "| :--- | :--- | :---: | :---: | :---: |"]
        for _, r in timing.iterrows():
            lines.append(f"| {r['model']} | {r['mode']} | {int(r['n'])} | {r['seconds_median']:.1f} "
                         f"[{r['seconds_q25']:.1f}, {r['seconds_q75']:.1f}] | {r['peak_vram_mb']:.0f} |")
    (out_dir / "OPTION_A_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["parity", "val", "lock", "test", "stats", "timing", "figures"])
    ap.add_argument("--timing-cases", type=int, default=10, help="Val cases timed per model (+1 warm-up)")
    ap.add_argument("--models", nargs="+", default=list(se.MODELS), choices=list(se.MODELS))
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="Use first N cases; writes a SMOKE lock (never final)")
    ap.add_argument("--parity-cases", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sw-batch-size", type=int, default=8)
    args = ap.parse_args()

    if args.smoke and args.out == OUT_DEFAULT:
        args.out = OUT_DEFAULT.with_name(OUT_DEFAULT.name + "_SMOKE")
    if args.stage in ("lock", "test", "stats", "figures", "timing") and args.models != list(se.MODELS):
        raise SystemExit("lock/test/stats/figures/timing must cover all four models (symmetry).")

    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Parity must reproduce locked numbers -> no autotuned (possibly different) conv algorithms.
    torch.backends.cudnn.benchmark = args.stage != "parity"

    cfg, splits, prov = load_cfg_and_splits()
    out_dir = se.check_out_dir(args.out, ROOT, resume=args.resume or args.stage != "parity")
    prov["checkpoints"] = verify_checkpoints(cfg, args.models)
    extra = {"parity": stage_parity, "val": stage_val, "lock": stage_lock, "test": stage_test,
             "stats": stage_stats, "timing": stage_timing, "figures": stage_figures}[args.stage](
        args, cfg, splits, out_dir)
    write_manifest(out_dir, args.stage, args, prov, extra)


if __name__ == "__main__":
    main()
```

> **Note on `check_out_dir` and `parity`:** parity runs first into an empty folder. Later stages reuse the folder, so `resume` is implied for them. A fresh run never silently overwrites: the lock is immutable, and CSV rows are appended and de-duplicated by key.

---

## 6. 4080 SUPER run notes

Everything below is for the lab PC. On the home PC only the synthetic tests run.

**Prepare the machine (once per session):**

```powershell
$PY = "C:\Thesis_RASNET_env\bench\Scripts\python.exe"; Set-Location C:\Thesis_RASNET
powercfg /setactive SCHEME_MIN
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
```

- `SCHEME_MIN` is Windows' built-in High-performance plan. It stops CPU throttling during long runs.
- The `nvidia-smi` check must show no other process using the GPU: no Rasnet-v2 queue, no other session's job.
- Close browsers and other GPU apps. Disable sleep for the duration.

**Speed settings already built into the code:**

| Setting | Why |
| :--- | :--- |
| AMP fp16 autocast for MONAI | Same as the locked benchmark |
| TF32 matmul/cuDNN on | Faster convolutions on the 4080 SUPER |
| `cudnn.benchmark` on (off only for `parity`) | Picks the fastest conv algorithm after the first case |
| CPU preprocessing prefetched 2 cases ahead | GPU does not wait for NIfTI loading/resampling |
| Metrics on a 2-worker thread pool, bounded at 2 cases in flight | GPU inference overlaps CPU metrics without RAM blow-up |
| No probability caches written | Avoids ~0.3 GB/volume disk I/O; CSVs stay a few MB |
| 8 TTA passes run sequentially | Peak VRAM ≈ 1-pass plus one logits accumulator |

**Tune `--sw-batch-size` once, before the `val` stage, then freeze it:**

1. Run `parity` with the default 8. It must pass.
2. Time 2 smoke cases at each candidate size:

   ```powershell
   foreach ($b in 4, 8, 16) { & $PY scripts/run_optionA_symmetric_v1_eval.py --stage val --smoke 2 --models rasnet --sw-batch-size $b --out results/optionA_tune_bs$b }
   ```

3. Pick the fastest size whose printed seconds per case are stable. A 3–5× jump means WDDM paging, so that size is too large.
4. Delete the `results/optionA_tune_bs*` folders.
5. Use the chosen size for **both** `val` and `test`, and write it in the lab notebook. The manifest also records it.

Batch size changes only speed, not the math, up to floating-point noise. Keeping it fixed across val and test keeps the lock and the test evaluation numerically consistent.

**Other notes:**

- **Run time:** measure, don't guess. Step 3.3 of the runbook extrapolates from the smoke run. nnU-Net will be slowest per case: 96 × 160 × 160 patches with 8 mirrors per tile.
- **Efficiency evidence:** run `--stage timing` with nothing else on the GPU. It times 10 validation cases per model plus 1 warm-up, end to end (preprocessing + inference + inversion back to the native grid), for 1-pass and TTA-8 separately, and records peak VRAM. The numbers go into `timing_summary.csv` and the report. Never time while another job runs.
- **nnU-Net preprocessing:** keep `num_processes=4` for nnU-Net's preprocessing workers, as in the original benchmark.
- **One GPU job at a time.** Never run Option A alongside Rasnet-v2 training.

---

## 7. Pre-written thesis wording for every outcome (choose after `stats`, never before)

| Outcome on R4 (from `OPTION_A_RESULTS.md`) | What you may claim |
| :--- | :--- |
| RASNet Dice ≥ nnU-Net and NI superior, F-A significant | "Under identical inference, RASNet (4.7 M) matched or exceeded nnU-Net V2 (30.8 M) on Dice and significantly on [metrics], at 6.5× fewer parameters." |
| RASNet NI = True, superior = False | "RASNet was non-inferior to nnU-Net V2 in Dice (margin 0.005) with 6.5× fewer parameters and 9× fewer GFLOPs per patch; precision [significantly higher / comparable]." |
| RASNet NI = False | "nnU-Net V2 retained a Dice advantage of X (95 % CI …) under symmetric inference. RASNet delivered [precision/HD95 result] at 15 % of nnU-Net's parameters, placing it on the accuracy–efficiency Pareto front (Table 4). Training-protocol differences (spacing, patch, augmentation) are addressed in the controlled Rasnet-v2 study." |
| F-B (vs SegResNet) significant | "Attention gating + deep supervision improved Dice/precision over the identical-size SegResNet backbone under identical inference." This is your strongest architecture claim from v1. |
| F-B not significant | Report it plainly. The architecture claim then rests on Rasnet-v2 Family 1. |

Every row is a valid, defensible thesis result. Choosing the row *after* seeing the numbers is fine because the rows were written before the run.
