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
