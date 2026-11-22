"""Code-review fixes R1-R10 for Option A (2026-11-13). Synthetic, CPU only.

Must FAIL before the fixes in docs-for-next-v2-gen/04_CODE_REVIEW_FIXES.md are applied, PASS after.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark import symmetric_eval as se

REPO = Path(__file__).resolve().parents[1]


def _splits():
    return {"train": list(range(1, 11)), "val": [11, 12, 13], "test": [851, 852]}


def _git(cwd, *a):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=cwd,
                   check=True, capture_output=True)


def _per_case(empty_model=None, empty_case=None):
    rng = np.random.default_rng(0)
    rows = []
    for m in se.MODELS:
        for cid in (851, 852, 853, 854):
            for r in ("R1_raw_1pass", se.PRIMARY_RECIPE):
                row = {"model": m, "case_id": cid, "recipe": r, "tau": 0.4, "status": "ok",
                       **{k: float(rng.uniform(0.7, 0.85)) for k in
                          ("dice", "iou", "precision", "recall", "cldice", "centerline_recall")},
                       "hd95": float(rng.uniform(5, 9)), "asd": 1.5, "components_total": 2}
                if m == empty_model and cid == empty_case:
                    row.update(status="prediction_empty", dice=0.0, iou=0.0, recall=0.0,
                               precision=np.nan, hd95=np.nan, asd=np.nan, cldice=np.nan,
                               centerline_recall=np.nan, components_total=0)
                rows.append(row)
    return pd.DataFrame(rows)


# R1 -- empty predictions are counted, never silently helpful ----------------------------
def test_summary_counts_empties_and_reports_strict_precision():
    s = se.summarize(_per_case("nnunet", 851)).set_index(["model", "recipe"])
    row = s.loc[("nnunet", se.PRIMARY_RECIPE)]
    assert row["n_empty_pred"] == 1
    assert row["precision_strict_mean"] < row["precision_mean"]          # empty counted as 0
    assert s.loc[("rasnet", se.PRIMARY_RECIPE), "n_empty_pred"] == 0


def test_families_report_excluded_cases():
    fam = se.run_families(_per_case("nnunet", 851), se.PRIMARY_RECIPE)
    hd = fam[(fam["family"] == "FA_primary_rasnet_vs_nnunet") & (fam["metric"] == "hd95")].iloc[0]
    assert int(hd["n_excluded_nan"]) == 1


# R2 -- run fingerprint ---------------------------------------------------------------------
def test_code_digest_ignores_line_endings_and_tracks_code(tmp_path):
    (tmp_path / "benchmark").mkdir()
    f = tmp_path / "benchmark" / "a.py"
    f.write_bytes(b"x = 1\n")
    d1 = se.code_digest(tmp_path, ["benchmark"])
    f.write_bytes(b"x = 1\r\n")                                         # lab checkout (CRLF)
    assert se.code_digest(tmp_path, ["benchmark"]) == d1
    f.write_bytes(b"x = 2\n")
    assert se.code_digest(tmp_path, ["benchmark"]) != d1


def test_fingerprint_written_once_then_enforced(tmp_path):
    fp = {"protocol_version": se.PROTOCOL_VERSION, "code_digest": "abc", "sw_batch_size": 8,
          "device": "cuda", "torch": "x", "monai": "y"}
    se.check_or_write_fingerprint(tmp_path, fp)
    assert json.loads((tmp_path / se.FINGERPRINT_FILE).read_text())["sw_batch_size"] == 8
    se.check_or_write_fingerprint(tmp_path, dict(fp))                  # identical -> OK
    with pytest.raises(RuntimeError, match="sw_batch_size"):
        se.check_or_write_fingerprint(tmp_path, dict(fp, sw_batch_size=4))
    with pytest.raises(RuntimeError, match="code_digest"):
        se.check_or_write_fingerprint(tmp_path, dict(fp, code_digest="changed"))


# R3 -- lock must be committed and unmodified before test inference ---------------------------
def test_assert_lock_committed(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    _git(tmp_path, "init", "-q")
    (tmp_path / "README").write_text("x\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    lock = tmp_path / "out" / "locked_recipes.json"
    lock.parent.mkdir()
    lock.write_text('{"a": 1}')
    lock.with_suffix(".sha256").write_text("deadbeef")
    with pytest.raises(RuntimeError, match="not committed"):
        se.assert_lock_committed(tmp_path, lock)                       # untracked
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "lock")
    se.assert_lock_committed(tmp_path, lock)                           # committed -> OK
    lock.write_text('{"a": 2}')
    with pytest.raises(RuntimeError, match="modified"):
        se.assert_lock_committed(tmp_path, lock)


# R4 -- primary recipe is marked ------------------------------------------------------------
def test_families_carry_role_and_report_puts_primary_first(tmp_path):
    df = _per_case()
    fam = pd.concat([se.run_families(df, r) for r in sorted(df["recipe"].unique())], ignore_index=True)
    assert set(fam.loc[fam["recipe"] == se.PRIMARY_RECIPE, "role"]) == {"primary"}
    assert set(fam.loc[fam["recipe"] != se.PRIMARY_RECIPE, "role"]) == {"secondary"}
    cli = _load_cli()
    ni = pd.DataFrame([{"recipe": r, **se.paired_noninferiority(
        df[(df.recipe == r) & (df.model == "rasnet")], df[(df.recipe == r) & (df.model == "nnunet")])}
        for r in sorted(df["recipe"].unique())])
    cli.write_report(tmp_path, se.summarize(df), fam, ni)
    text = (tmp_path / "OPTION_A_RESULTS.md").read_text(encoding="utf-8")
    assert text.index("PRIMARY") < text.index("Secondary")
    assert "n excl." in text


# R5 -- crash-safe CSV appends ------------------------------------------------------------
def test_truncated_csv_tail_is_repaired(tmp_path):
    p = tmp_path / "rows.csv"
    se.append_rows(p, [{"model": "rasnet", "case_id": 11, "v": 1.0}])
    with open(p, "a", encoding="utf-8") as f:
        f.write("rasnet,12,0.")                                         # crash mid-line
    df = se.read_rows(p)
    assert list(df["case_id"]) == [11]
    assert p.read_text(encoding="utf-8").endswith("\n")
    se.append_rows(p, [{"model": "rasnet", "case_id": 12, "v": 2.0}])
    assert list(se.read_rows(p)["case_id"]) == [11, 12]


# R6 -- missing sidecar is an error ---------------------------------------------------------
def test_load_lock_requires_sidecar(tmp_path):
    rows = [{"model": m, "case_id": c, "mode": mode, "tau": t, "dice": 0.8 - abs(t - 0.40)}
            for m in se.MODELS for mode in se.MODES for c in (11, 12, 13) for t in se.TAU_GRID]
    lock = se.build_lock(pd.DataFrame(rows), _splits(), se.MODELS)
    p = tmp_path / "locked_recipes.json"
    se.write_lock(p, lock)
    p.with_suffix(".sha256").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        se.load_lock(p, _splits(), se.MODELS)


# R9 -- the duplicated selection rule must agree with v1 thresholds.py -----------------------
def test_select_tau_agrees_with_v1_threshold_rule():
    from benchmark.thresholds import select_validation_thresholds_from_scores
    rng = np.random.default_rng(3)
    for _ in range(20):
        per_case = {c: {t: float(np.round(rng.uniform(0.6, 0.9), 2)) for t in se.TAU_GRID} for c in (11, 12, 13)}
        ours, _ = se.select_tau(per_case, [11, 12, 13])
        v1 = select_validation_thresholds_from_scores({"m": per_case}, _splits(), grid=se.TAU_GRID,
                                                      keep_top_k=2)["thresholds"]["m"]
        assert ours == pytest.approx(v1)


# R10 -- parity uses PyTorch numerics defaults ------------------------------------------------
def test_configure_torch_keeps_locked_numerics_for_parity():
    torch = pytest.importorskip("torch")
    cli = _load_cli()
    before = torch.backends.cuda.matmul.allow_tf32
    cli.configure_torch("parity")
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cuda.matmul.allow_tf32 == before            # untouched
    cli.configure_torch("val")
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cuda.matmul.allow_tf32 == before


def _load_cli():
    spec = importlib.util.spec_from_file_location("optionA_cli", REPO / "scripts" / "run_optionA_symmetric_v1_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# End-to-end CLI flow on synthetic data (val -> lock -> commit check -> test -> stats -> figures) -------
def test_cli_flow_end_to_end_synthetic(tmp_path, monkeypatch):
    sitk = pytest.importorskip("SimpleITK")
    pytest.importorskip("matplotlib")
    if shutil.which("git") is None:
        pytest.skip("git not available")
    cli = _load_cli()
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    data.mkdir()
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    monkeypatch.setattr(cli, "ROOT", repo)

    splits = {"train": [1, 2], "val": [11, 12, 13], "test": [851, 852, 853]}
    rng = np.random.default_rng(0)
    for cid in splits["val"] + splits["test"]:
        lbl = np.zeros((24, 24, 24), np.uint8)
        lbl[6:18, 6:18, 6:18] = 1
        img = sitk.GetImageFromArray(lbl.astype(np.float32) * 300)
        img.SetSpacing((0.5, 0.5, 0.5))
        tmp = data / f"{cid}_tmp.nii.gz"
        sitk.WriteImage(img, str(tmp))
        tmp.replace(data / f"{cid}.img.nii.gz")
        li = sitk.GetImageFromArray(lbl)
        li.CopyInformation(img)
        sitk.WriteImage(li, str(data / f"{cid}.label.nii.gz"))
    cfg = {"dataset_root": str(data)}

    def fake_iter(model, cfg_, repo_, case_ids, pairs, device="cuda", sw_batch_size=8):
        for cid in case_ids:
            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(pairs[cid][1]))) > 0
            base = gt.astype(np.float32) * 0.7 + 0.1
            modes = {m: np.clip(base + rng.normal(0, 0.05, gt.shape), 0, 1).astype(np.float32) for m in se.MODES}
            yield cid, modes, gt, (0.5, 0.5, 0.5)

    monkeypatch.setattr(se, "iter_case_probabilities", fake_iter)

    class A:
        smoke = 0
        models = list(se.MODELS)
        device = "cpu"
        sw_batch_size = 8

    out = repo / "results" / "optA"
    out.mkdir(parents=True)
    cli.stage_val(A, cfg, splits, out)
    cli.stage_lock(A, cfg, splits, out)
    with pytest.raises(RuntimeError, match="not committed"):
        cli.stage_test(A, cfg, splits, out)                         # lock not committed yet
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "lock")
    A.sw_batch_size = 4
    with pytest.raises(RuntimeError, match="sw_batch_size"):
        cli.stage_test(A, cfg, splits, out)                         # settings changed -> refused
    A.sw_batch_size = 8
    cli.stage_test(A, cfg, splits, out)
    cli.stage_stats(A, cfg, splits, out)
    report = (out / "OPTION_A_RESULTS.md").read_text(encoding="utf-8")
    assert "PRIMARY result (pre-registered): R4_tuned_tta8_top2" in report
    assert len(se.read_rows(out / "test_per_case.csv")) == 3 * 4 * 4
    eff = pd.DataFrame({"model_key": list(se.MODELS), "Parameters_M": [4.707, 4.701, 45.599, 30.786],
                        "GFLOPs_per_Patch": [123.39, 122.56, 617.23, 1119.34]})
    files = se.make_figures(pd.read_csv(out / "summary_test.csv"), se.read_rows(out / "test_per_case.csv"), eff, out)
    assert len(files) == 6
