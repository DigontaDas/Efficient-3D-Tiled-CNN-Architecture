"""Tests-first unit tests for Stage 2 provenance & edge-case fixes in benchmark/."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import shutil
import subprocess
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark import symmetric_eval as se
from benchmark.thresholds import _fast_dice

REPO = Path(__file__).resolve().parents[1]


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_env_info_resolves_nnunetv2_and_cc3d_via_metadata():
    info = se.env_info()
    # Both packages are installed in the lab bench env (no __version__ attr, resolved via importlib.metadata)
    for mod, dist in (("cc3d", "connected-components-3d"), ("nnunetv2", "nnunetv2")):
        try:
            expected = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        assert info[mod] == expected
        assert info[mod] != "missing"


def test_env_info_metadata_fallback_when_dunder_version_missing(monkeypatch):
    """Fallback to importlib.metadata.version must work even when a module omits __version__."""
    real_import = __import__
    fake_cc3d = types.ModuleType("cc3d")
    fake_nnunet = types.ModuleType("nnunetv2")

    def fake_import(name, *args, **kwargs):
        if name == "cc3d":
            return fake_cc3d
        if name == "nnunetv2":
            return fake_nnunet
        return real_import(name, *args, **kwargs)

    real_version = importlib.metadata.version

    def fake_version(dist):
        if dist == "connected-components-3d":
            return "9.8.7-test"
        if dist == "nnunetv2":
            return "6.5.4-test"
        return real_version(dist)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    info = se.env_info()
    assert info["cc3d"] == "9.8.7-test"
    assert info["nnunetv2"] == "6.5.4-test"


def test_done_keys_deduplicates_test_recipes_before_counting(tmp_path: Path):
    p = tmp_path / "test_per_case.csv"
    # 4 duplicate rows for a single recipe must NOT count as 4 completed recipes
    dup_rows = [{"model": "rasnet", "case_id": 851, "recipe": "R1_raw_1pass", "dice": 0.8}] * 4
    # 4 distinct recipes (plus an extra duplicate) MUST count as completed
    full_rows = [
        {"model": "rasnet", "case_id": 852, "recipe": r, "dice": 0.8}
        for r in ("R1_raw_1pass", "R2_raw_tta8", "R3_tuned_1pass_top2", "R4_tuned_tta8_top2", "R1_raw_1pass")
    ]
    pd.DataFrame(dup_rows + full_rows).to_csv(p, index=False)

    done = se.done_keys(p, rows_per_case=4)
    assert ("rasnet", 851) not in done
    assert ("rasnet", 852) in done


def test_done_keys_deduplicates_val_grid_before_counting(tmp_path: Path):
    p = tmp_path / "val_grid_scores.csv"
    # 22 duplicates of a single (mode, tau) must NOT count as 22 grid evaluations
    dup_rows = [{"model": "rasnet", "case_id": 11, "mode": se.MODES[0], "tau": 0.50, "dice": 0.8}] * 22
    # All 22 distinct (mode, tau) combinations MUST count as completed
    full_rows = [
        {"model": "rasnet", "case_id": 12, "mode": mode, "tau": tau, "dice": 0.8}
        for mode in se.MODES
        for tau in se.TAU_GRID
    ]
    pd.DataFrame(dup_rows + full_rows).to_csv(p, index=False)

    done = se.done_keys(p, rows_per_case=se.GRID_ROWS_PER_CASE)
    assert ("rasnet", 11) not in done
    assert ("rasnet", 12) in done


def test_done_keys_deduplicates_mode_only_and_float_tau_drift(tmp_path: Path):
    # mode-only CSV (e.g., timing_val.csv): 2 retries of the same mode must NOT count as 2 modes
    p_mode = tmp_path / "timing_val.csv"
    pd.DataFrame([
        {"model": "rasnet", "case_id": 11, "mode": se.MODES[0], "seconds": 5.0},
        {"model": "rasnet", "case_id": 11, "mode": se.MODES[0], "seconds": 5.1},
        {"model": "rasnet", "case_id": 12, "mode": se.MODES[0], "seconds": 5.0},
        {"model": "rasnet", "case_id": 12, "mode": se.MODES[1], "seconds": 22.0},
    ]).to_csv(p_mode, index=False)
    done_mode = se.done_keys(p_mode, rows_per_case=2)
    assert ("rasnet", 11) not in done_mode
    assert ("rasnet", 12) in done_mode

    # Float representation drift on tau (0.30000000000000004 vs 0.30) must collapse to 1 entry
    p_tau = tmp_path / "val_float_drift.csv"
    p_tau.write_text(
        "model,case_id,mode,tau,dice\n"
        f"rasnet,11,{se.MODES[0]},0.30000000000000004,0.8\n"
        f"rasnet,11,{se.MODES[0]},0.30,0.81\n",
        encoding="utf-8",
        newline="\n",
    )
    assert ("rasnet", 11) not in se.done_keys(p_tau, rows_per_case=2)


def test_fast_dice_returns_nan_on_both_empty():
    empty_pred = np.zeros((4, 4, 4), dtype=np.uint8)
    empty_gt = np.zeros((4, 4, 4), dtype=bool)
    val = _fast_dice(empty_pred, empty_gt)
    assert np.isnan(val)

    # Sanity check: non-empty identical masks still return 1.0
    nonempty_pred = np.zeros((4, 4, 4), dtype=np.uint8)
    nonempty_pred[1:3, 1:3, 1:3] = 1
    assert _fast_dice(nonempty_pred, nonempty_pred.astype(bool)) == pytest.approx(1.0)


def test_write_lock_and_gitattributes_cross_os_lf_roundtrip(tmp_path: Path):
    """A new lock written on Windows must emit LF and pass load_lock() after commit + fresh clone."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    r1 = tmp_path / "repo_orig"
    r1.mkdir()
    _git(r1, "init", "-q")
    shutil.copy(REPO / ".gitattributes", r1 / ".gitattributes")

    splits = {"train": list(range(1, 11)), "val": [11, 12, 13], "test": [851, 852]}
    rows = [
        {"model": m, "case_id": c, "mode": mode, "tau": t, "dice": 0.8 - abs(t - 0.40)}
        for m in se.MODELS
        for mode in se.MODES
        for c in (11, 12, 13)
        for t in se.TAU_GRID
    ]
    lock = se.build_lock(pd.DataFrame(rows), splits, se.MODELS)
    out_dir = r1 / "results" / "new_optionA_run"
    out_dir.mkdir(parents=True)
    lock_path = out_dir / "locked_recipes.json"
    sha_written = se.write_lock(lock_path, lock)

    assert b"\r\n" not in lock_path.read_bytes()
    assert b"\r\n" not in lock_path.with_suffix(".sha256").read_bytes()

    _git(r1, "add", "-A")
    _git(r1, "commit", "-q", "-m", "add new lock")

    r2 = tmp_path / "repo_clone"
    _git(tmp_path, "clone", "-q", str(r1), str(r2))
    cloned_lock = r2 / "results" / "new_optionA_run" / "locked_recipes.json"
    loaded, sha_loaded = se.load_lock(cloned_lock, splits, se.MODELS)
    assert sha_loaded == sha_written
    assert loaded["thresholds"] == lock["thresholds"]


def test_gitattributes_preserves_frozen_optionA_lock_and_digests_on_fresh_clone(tmp_path: Path):
    """Frozen pre-.gitattributes CRLF-hashed artifacts must keep their recorded SHA-256 on fresh clones."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    rel_files = [
        "results/optionA_v1_symmetric_sw050/locked_recipes.json",
        "results/optionA_v1_symmetric_sw050/locked_recipes.sha256",
        "results/thresholds.json",
        "Phase3_Local_Integration/splits_final.json",
    ]
    for rel in rel_files:
        if not (REPO / rel).is_file():
            pytest.skip(f"missing {rel}")

    r1 = tmp_path / "repo_orig"
    r1.mkdir()
    _git(r1, "init", "-q")
    shutil.copy(REPO / ".gitattributes", r1 / ".gitattributes")
    expected_shas = {}
    for rel in rel_files:
        dst = r1 / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dst)
        expected_shas[rel] = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()

    _git(r1, "add", "-A")
    _git(r1, "commit", "-q", "-m", "frozen artifacts")

    r2 = tmp_path / "repo_clone"
    _git(tmp_path, "clone", "-q", str(r1), str(r2))
    for rel, exp in expected_shas.items():
        got = hashlib.sha256((r2 / rel).read_bytes()).hexdigest()
        assert got == exp, f"{rel} digest changed on fresh clone: {got} != {exp}"

    sidecar = (r2 / "results/optionA_v1_symmetric_sw050/locked_recipes.sha256").read_text(encoding="utf-8").strip()
    assert hashlib.sha256((r2 / "results/optionA_v1_symmetric_sw050/locked_recipes.json").read_bytes()).hexdigest() == sidecar


def test_fingerprint_manifest_and_report_write_lf(tmp_path: Path):
    """Option A JSON/MD writers must emit LF even on Windows so hashes match .gitattributes."""
    fp = {"protocol_version": se.PROTOCOL_VERSION, "code_digest": "abc", "sw_batch_size": 8,
          "device": "cpu", "torch": "x", "monai": "y"}
    se.check_or_write_fingerprint(tmp_path, fp)
    assert b"\r\n" not in (tmp_path / se.FINGERPRINT_FILE).read_bytes()

    spec = importlib.util.spec_from_file_location("optionA_cli_lf", REPO / "scripts" / "run_optionA_symmetric_v1_eval.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    class _Args:
        stage = "stats"
        models = list(se.MODELS)
        device = "cpu"
        sw_batch_size = 8
        smoke = 0

    cli.write_manifest(tmp_path, "stats", _Args(), {"splits_sha256": "0"})
    manifests = list(tmp_path.glob("manifest_stats_*.json"))
    assert len(manifests) == 1
    assert b"\r\n" not in manifests[0].read_bytes()

    df_summ = pd.DataFrame([{
        "model": "rasnet", "recipe": se.PRIMARY_RECIPE, "tau": 0.4,
        "dice_mean": 0.8, "dice_sd": 0.05, "precision_mean": 0.82, "precision_sd": 0.04,
        "precision_strict_mean": 0.82, "recall_mean": 0.78, "recall_sd": 0.05,
        "hd95_mean": 8.5, "hd95_sd": 2.0, "asd_mean": 1.5, "asd_sd": 0.3,
        "cldice_mean": 0.87, "cldice_sd": 0.04, "n_empty_pred": 0,
    }])
    df_fam = pd.DataFrame(columns=["family", "recipe", "role", "a", "b", "metric", "mean_difference",
                                   "diff_ci95_lo", "diff_ci95_hi", "p", "holm_p", "wins", "ties", "losses", "n_excluded_nan"])
    df_ni = pd.DataFrame([{"recipe": se.PRIMARY_RECIPE, "mean_diff": -0.002, "ci95_lo": -0.008, "ci95_hi": 0.003,
                           "non_inferior": False, "superior": False}])
    cli.write_report(tmp_path, df_summ, df_fam, df_ni)
    assert b"\r\n" not in (tmp_path / "OPTION_A_RESULTS.md").read_bytes()
