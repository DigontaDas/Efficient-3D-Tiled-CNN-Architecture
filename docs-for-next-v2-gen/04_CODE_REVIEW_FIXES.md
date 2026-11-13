# 04 — Code Review Fixes (2026-11-13)

**What this is:** the result of a high-effort code review of everything committed since `302cbc0`:

- Option A (`aecf60e`): `benchmark/symmetric_eval.py`, `scripts/run_optionA_symmetric_v1_eval.py`;
- Amendment A1 (`79e0458`): 6 Rasnet-v2 source files + `diag_vnet_bn.py`.

It contains every fix as **exact, tested code**: new test files plus deterministic patch scripts, each embedded byte for byte from the copy where it was validated. Apply on the **home PC** (§4), commit, push; the lab PC only verifies (§6).

**None of the fixes change the science.** Architectures, losses, thresholds, grids, seeds, the TTA definition and the statistics are unchanged; Amendment A1's science items are untouched. The fixes add guards (fingerprint, lock-commit check, sidecar), make empty predictions visible, make CSV writes crash-safe, and label confounded ablations. Option A has produced no results yet, so applying them now changes no number.

---

## 1. Findings and decisions

| # | Severity | File | Finding | Decision |
| :--- | :--- | :--- | :--- | :--- |
| R1 | High | `benchmark/symmetric_eval.py` (`summarize`, `run_families`) | A case where a model predicts nothing (precision/HD95/ASD = NaN) is silently dropped from paired tests and means; the report never shows it. Breaks CLAUDE.md §6: *"Count empties explicitly; never let NaN-dropping help a candidate."* | **Fix:** `n_empty_pred` and `precision_strict_mean` (empty = 0) per model × recipe; the report shows both plus `n excl.` per test, with a warning banner if any exist. The pre-registered statistics stay the same as v1 (`allow_nan_filtering=True`), so nothing is silently re-defined. |
| R2 | High | `scripts/run_optionA_symmetric_v1_eval.py` | Every stage after parity resumes automatically, but nothing guarantees that val, lock and test rows in one folder came from the same code and `--sw-batch-size`. | **Fix:** `run_fingerprint.json` (protocol version, LF-normalised code digest, `sw_batch_size`, device, torch/MONAI versions). Written by the first val run, enforced by val, test and timing, required by lock. |
| R3 | High | CLI `stage_test` | The protocol requires the Val-94 lock to be committed before test inference (manual E7, CLAUDE.md §3.3), but the code never checked it. | **Fix:** `assert_lock_committed()`: the lock and its sidecar must be tracked and unmodified in git, or test inference refuses to start. Smoke runs are exempt. |
| R4 | Medium | CLI `write_report` | All four recipes were reported with equal weight, inviting post-hoc choice of the best recipe. | **Fix:** families get `role` = `primary` (R4) or `secondary`; the report leads with **"PRIMARY result (pre-registered): R4"** and labels R1–R3 "descriptive; not for confirmatory claims". |
| R5 | Medium | `append_rows` / `done_keys` | A crash or power loss mid-append leaves a truncated last line, and resume then fails or misreads the CSV. | **Fix:** one write + `fsync` per case; `repair_csv_tail()` drops a partial last line before every read or append; all result reads go through `read_rows()`. |
| R6 | Medium | `load_lock` | The tamper check was skipped when the `.sha256` sidecar was missing. | **Fix:** a missing sidecar is an error. One existing test is updated so its tampered copy carries a valid sidecar and still reaches the grid check. |
| R7 | Low | `Rasnet-v2/src/stage2_pilot_eval.py` | After A1 the gate uses strict precision, but `pilot_lock_decision.json` recorded only the NaN-dropping precision. | **Fix:** the decision also records `val94_precision_strict` and `val94_n_empty_pred`. Asserted in the existing end-to-end test. |
| R8 | Medium | `Rasnet-v2/src/stage5_stats_and_efficiency.py` | `Abl_Gate_ConvBias_vs_OrigGN`, `Abl_DS_Full_vs_NoDS` and the new 3-seed DS contrast assume the Stage 2B lead gate is `conv_bias`. If 2B locks `groupnorm_orig` they measure gate and DS together (the gate contrast becomes nearly meaningless), yet the CSV still labels them as pure effects. | **Fix:** every Family 1 / ablation row gets `lead_gate_mode` and `ablation_label_valid`; gate-dependent rows are `False` when the lead gate is not `conv_bias`. |
| R9 | Low | `select_tau` / `score_tau_grid` | They duplicate the selection rule in `benchmark/thresholds.py`. | **No refactor:** `thresholds.py` belongs to the locked v1 pipeline and requires the exact Val-94 split. Instead, a **guard test** proves both rules pick the same τ on random grids, so any future drift fails the test. |
| R10 | Low | CLI `main` | TF32 matmul was enabled for every stage, including parity; the locked runs used PyTorch defaults (TF32 matmul off). | **Fix:** `configure_torch(stage)` leaves numerics at PyTorch defaults and only toggles `cudnn.benchmark` (speed only), off for parity. |

**Checked and dropped:**

- *Stale 4-pass TTA probabilities reused from a cache.* Stage 4B's cache is 1-pass only, per run, in a temp folder deleted afterwards (`stage4b_val94_lock.py:197,391`).
- *Other TTA implementations missed by A1.* All Arm B paths (Stage 4C, Stage 6) go through `eval_fair_v2.infer_monai_native_zyx`.
- *`weights_only=True` failing on `epoch_*.pth`.* Snapshots contain only tensors and primitives (`train_fair_v2.py:593-606`).

---

## 2. Validation (home PC, 2026-11-13; CPU only, Python 3.14, MONAI 1.5.1)

| Check | Before fixes | After fixes |
| :--- | :--- | :--- |
| `tests/test_optionA_review_fixes.py` (11 tests, incl. full synthetic CLI flow: val → lock → commit check → test → stats → figures) | 10 failed, 1 passed (the R9 guard, which documents an already-true equivalence) | **11 / 11 pass** |
| Whole `tests/` suite (v1 + Option A + review) | — | **62 / 62 pass** (51 existing + 11 new) |
| `Rasnet-v2/tests/test_review_fixes_v2.py` (3 tests) | 3 failed | **3 / 3 pass** |
| Whole `Rasnet-v2/tests` suite (copy with the hard-coded `C:\Thesis_RASNET` test root pointed at the copy) | — | **43 passed**, 3 environment-only failures as before (lab paths, no `nnunetv2`, MONAI 1.5.1). On the lab PC all **46** must pass. |

The end-to-end test also proves:

- test inference is refused while the lock is uncommitted;
- test inference is refused after `--sw-batch-size` changes;
- the report leads with the primary R4 section.

---

## 3. Files touched

| File | Change |
| :--- | :--- |
| `tests/test_optionA_review_fixes.py` | **new** (§5.1) |
| `Rasnet-v2/tests/test_review_fixes_v2.py` | **new** (§5.2) |
| `benchmark/symmetric_eval.py` | patched by `patch_lib.py` (§5.3) |
| `scripts/run_optionA_symmetric_v1_eval.py` | patched by `patch_cli.py` (§5.4) |
| `tests/test_optionA_symmetric_eval.py` | one line added (R6), by `patch_test.py` (§5.5) |
| `Rasnet-v2/src/stage2_pilot_eval.py`, `Rasnet-v2/src/stage5_stats_and_efficiency.py`, `Rasnet-v2/tests/test_v2_pipeline_cpu.py` | patched by `patch_v2.py` (§5.6) |
| `docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md` | its 3 code blocks re-synced to the patched files (§5.7), so the manual's §5 D1 byte-identity check stays valid |

Every patch script checks that each anchor occurs exactly once and stops without writing if any anchor is missing.

---

## 4. How to apply (home PC, tests first)

Run from `H:\Thesis_Trainings` (Git Bash):

1. Save the five scripts in §5.3–§5.7 to a scratch folder (outside the repo): `patch_lib.py`, `patch_cli.py`, `patch_test.py`, `patch_v2.py`, `sync_doc01.py`.
2. **Tests first:**
   1. Create the two new test files from §5.1 and §5.2 exactly.
   2. Run them and save the failing output:

      ```bash
      python -m pytest tests/test_optionA_review_fixes.py -q -p no:cacheprovider > results/audits/review_fixes_tests_FAILING.txt 2>&1
      ```

      ```bash
      cd Rasnet-v2 && python -m pytest tests/test_review_fixes_v2.py -q -p no:cacheprovider > logs/review_fixes_v2_tests_FAILING.txt 2>&1; cd ..
      ```

3. **Apply the patches**, in this order:
   1. `python <scratch>/patch_lib.py`
   2. `python <scratch>/patch_cli.py`
   3. `python <scratch>/patch_test.py`
   4. `python <scratch>/patch_v2.py`
4. **Run all tests and save the passing output:**

   ```bash
   python -m pytest tests -q -p no:cacheprovider > results/audits/review_fixes_tests_PASSING.txt 2>&1
   ```

   ```bash
   cd Rasnet-v2 && python -m pytest tests/test_review_fixes_v2.py tests/test_amendment_a1.py -q -p no:cacheprovider > logs/review_fixes_v2_tests_PASSING.txt 2>&1; cd ..
   ```

   Expected: `62 passed`, and `14 passed` (3 review + 11 A1).
5. **Re-sync guide 01's code blocks:** `python <scratch>/sync_doc01.py`. It then checks byte identity and must print `OK` three times.
6. **Update the expected counts in the lab manual** (`00_LAB_PC_OPERATOR_MANUAL.md`):
   - §5 D2 and the checklist: `51 passed` → `62 passed` (33 + 18 + 11).
   - §5 D3 and the checklist: Rasnet-v2 `43` → `46`.
   - §0: reference the new commit.
7. **Commit** (no attribution trailer, per the user's preference), then push when ready.

---

## 5. The code

### 5.1 `tests/test_optionA_review_fixes.py` (new)

```python
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
```

### 5.2 `Rasnet-v2/tests/test_review_fixes_v2.py` (new)

```python
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
```

### 5.3 `patch_lib.py` — R1, R2, R3, R5, R6 (and R4's `role` column) in `benchmark/symmetric_eval.py`

```python
"""Apply code-review fixes R1-R6 to benchmark/symmetric_eval.py (run from repo root)."""
import pathlib
import sys

p = pathlib.Path("benchmark/symmetric_eval.py")
t = p.read_text(encoding="utf-8").replace("\r\n", "\n")


def rep(a, b):
    global t
    if t.count(a) != 1:
        sys.exit(f"anchor not unique/found: {a[:70]!r}")
    t = t.replace(a, b)


rep("import json\nimport platform\n", "import json\nimport os\nimport platform\n")

rep('NI_MARGIN = -0.005\n',
    'NI_MARGIN = -0.005\n'
    'PRIMARY_RECIPE = "R4_tuned_tta8_top2"   # pre-registered primary (BENCHMARK_PREREGISTRATION.md §6)\n'
    'FINGERPRINT_FILE = "run_fingerprint.json"\n')

NEW_PROVENANCE = r'''# Files whose content defines the numbers an Option A run produces. The hash is LF-normalised so the
# home PC (LF) and the lab PC checkout (CRLF) give the same digest.
FINGERPRINT_PATHS = ("benchmark", "scripts/run_optionA_symmetric_v1_eval.py",
                     "Phase3_Local_Integration/rasnet_model.py", "config.lab.yaml",
                     "Phase3_Local_Integration/splits_final.json")


def code_digest(repo: Path, paths=FINGERPRINT_PATHS) -> str:
    """sha256 over (relative path, LF-normalised content) of every .py/.yaml/.json under `paths`."""
    repo = Path(repo)
    files = []
    for rel in paths:
        p = repo / rel
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(q for q in p.rglob("*")
                         if q.is_file() and q.suffix in (".py", ".yaml", ".json") and "__pycache__" not in q.parts)
    h = hashlib.sha256()
    for f in sorted(set(files)):
        h.update(f.relative_to(repo).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()


def run_fingerprint(repo: Path, sw_batch_size: int, device: str) -> dict:
    env = env_info()
    return {"protocol_version": PROTOCOL_VERSION, "code_digest": code_digest(repo),
            "sw_batch_size": int(sw_batch_size), "device": str(device),
            "torch": env.get("torch"), "monai": env.get("monai")}


def check_or_write_fingerprint(out_dir: Path, fp: dict) -> dict:
    """The first probability-producing stage writes the fingerprint; every later one must match it.

    Guarantees that val, lock and test rows in one folder come from identical code and settings.
    """
    p = Path(out_dir) / FINGERPRINT_FILE
    if not p.is_file():
        p.write_text(json.dumps(fp, indent=2, sort_keys=True), encoding="utf-8")
        return fp
    saved = json.loads(p.read_text(encoding="utf-8"))
    diff = {k: (saved.get(k), fp.get(k)) for k in sorted(set(saved) | set(fp)) if saved.get(k) != fp.get(k)}
    if diff:
        raise RuntimeError(
            f"Run fingerprint mismatch in {out_dir} (saved -> now): {diff}. Results in one folder must come "
            "from identical code and settings: rerun with the original settings, or start a new --out folder "
            "and record why.")
    return saved


def assert_lock_committed(repo: Path, lock_path: Path) -> None:
    """The Val-94 lock and its sidecar must be tracked and unmodified in git before any test inference."""
    repo, lock_path = Path(repo), Path(lock_path)
    for f in (lock_path, lock_path.with_suffix(".sha256")):
        try:
            rel = f.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            raise RuntimeError(f"Lock file {f} is outside the repository and cannot be committed.")
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel], cwd=repo,
                                 capture_output=True, text=True)
        if tracked.returncode != 0:
            raise RuntimeError(f"{rel} is not committed. Commit the Val-94 lock (manual step E7) "
                               "before test inference.")
        status = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=repo,
                                capture_output=True, text=True, check=True).stdout.strip()
        if status:
            raise RuntimeError(f"{rel} was modified after it was committed: {status}")


def env_info() -> dict:'''
rep("def env_info() -> dict:", NEW_PROVENANCE)

rep('''    side = path.with_suffix(".sha256")
    if side.is_file() and side.read_text(encoding="utf-8").strip() != sha:''',
    '''    side = path.with_suffix(".sha256")
    if not side.is_file():
        raise ValueError(f"Lock sidecar {side.name} is missing; the lock's integrity cannot be checked.")
    if side.read_text(encoding="utf-8").strip() != sha:''')

OLD_CSV = '''def done_keys(csv_path: Path, rows_per_case: int, key_cols=("model", "case_id")) -> set:
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
    pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)'''
NEW_CSV = r'''def repair_csv_tail(csv_path: Path) -> None:
    """Drop a partially written last line (crash or power loss mid-append). Every append ends in a newline."""
    p = Path(csv_path)
    if not p.is_file() or p.stat().st_size == 0:
        return
    data = p.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    with open(p, "r+b") as f:
        f.truncate(cut + 1 if cut >= 0 else 0)
        f.flush()
        os.fsync(f.fileno())


def read_rows(csv_path: Path) -> pd.DataFrame:
    """Read an append-only results CSV after repairing a truncated tail."""
    p = Path(csv_path)
    repair_csv_tail(p)
    if not p.is_file() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def done_keys(csv_path: Path, rows_per_case: int, key_cols=("model", "case_id")) -> set:
    df = read_rows(csv_path)
    if df.empty:
        return set()
    counts = df.groupby(list(key_cols)).size()
    return {(str(m), int(c)) for (m, c), n in counts.items() if n >= rows_per_case}


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    """Append one case's rows in a single write + fsync (crash-safe together with repair_csv_tail)."""
    if not rows:
        return
    p = Path(csv_path)
    repair_csv_tail(p)
    header = not p.is_file() or p.stat().st_size == 0
    text = pd.DataFrame(rows).to_csv(index=False, header=header, lineterminator="\n")
    with open(p, "a", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())'''
rep(OLD_CSV, NEW_CSV)

rep('''        r = {"model": model, "recipe": recipe, "n": int(len(g)),
             "n_ok": int((g["status"] == "ok").sum()), "tau": float(g["tau"].iloc[0])}''',
    '''        empty = g["status"] == "prediction_empty"
        r = {"model": model, "recipe": recipe, "n": int(len(g)),
             "n_ok": int((g["status"] == "ok").sum()), "n_empty_pred": int(empty.sum()),
             "tau": float(g["tau"].iloc[0]),
             # precision with an empty prediction counted as 0: NaN-dropping must never help a model
             "precision_strict_mean": float(g["precision"].where(~empty, 0.0).mean())}''')

rep('''            out.append({"family": fam, "recipe": recipe, **r})''',
    '''            out.append({"family": fam, "recipe": recipe,
                        "role": "primary" if recipe == PRIMARY_RECIPE else "secondary", **r})''')

p.write_text(t, encoding="utf-8", newline="\n")
print("symmetric_eval.py patched")
```

### 5.4 `patch_cli.py` — R2, R3, R4, R5, R10 in `scripts/run_optionA_symmetric_v1_eval.py`

```python
"""Apply code-review fixes R1-R5, R10 to scripts/run_optionA_symmetric_v1_eval.py (run from repo root)."""
import pathlib
import sys

p = pathlib.Path("scripts/run_optionA_symmetric_v1_eval.py")
t = p.read_text(encoding="utf-8").replace("\r\n", "\n")


def rep(a, b):
    global t
    if t.count(a) != 1:
        sys.exit(f"anchor not unique/found: {a[:70]!r}")
    t = t.replace(a, b)


# R10 -- numerics: parity keeps PyTorch defaults (what the locked runs used); no global TF32 matmul.
rep('''def _bounded_pipeline(gen, fn, max_inflight=2, workers=2):''',
    '''def configure_torch(stage: str) -> None:
    """Locked v1 runs used PyTorch defaults (TF32 matmul off, cuDNN TF32 on, cudnn.benchmark off).
    Keep those numerics everywhere; only enable cuDNN autotuning (speed only) outside parity."""
    import torch
    torch.backends.cudnn.benchmark = stage != "parity"


def _fingerprint(args, out_dir):
    return se.check_or_write_fingerprint(out_dir, se.run_fingerprint(ROOT, args.sw_batch_size, args.device))


def _bounded_pipeline(gen, fn, max_inflight=2, workers=2):''')

rep('''    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Parity must reproduce locked numbers -> no autotuned (possibly different) conv algorithms.
    torch.backends.cudnn.benchmark = args.stage != "parity"
''', '''    configure_torch(args.stage)
''')

# R2 -- fingerprint on every probability-producing stage
rep('''def stage_val(args, cfg, splits, out_dir):
    ids = sorted(splits["val"])[: args.smoke or None]''',
    '''def stage_val(args, cfg, splits, out_dir):
    fp = _fingerprint(args, out_dir)
    ids = sorted(splits["val"])[: args.smoke or None]''')
rep('''                print(f"  [{model}] {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


def stage_lock(args, cfg, splits, out_dir):
    df = pd.read_csv(out_dir / "val_grid_scores.csv")''',
    '''                print(f"  [{model}] {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return {"run_fingerprint": fp}


def stage_lock(args, cfg, splits, out_dir):
    if not (out_dir / se.FINGERPRINT_FILE).is_file():
        raise SystemExit("No run_fingerprint.json: run the val stage in this folder first.")
    df = se.read_rows(out_dir / "val_grid_scores.csv")''')

# R2 + R3 -- test stage: same fingerprint, lock committed and unmodified
rep('''def stage_test(args, cfg, splits, out_dir):
    lock, sha = se.load_lock(out_dir / "locked_recipes.json", splits, args.models, allow_smoke=bool(args.smoke))''',
    '''def stage_test(args, cfg, splits, out_dir):
    fp = _fingerprint(args, out_dir)
    lock_path = out_dir / "locked_recipes.json"
    lock, sha = se.load_lock(lock_path, splits, args.models, allow_smoke=bool(args.smoke))
    if not args.smoke:
        se.assert_lock_committed(ROOT, lock_path)''')
rep('''                print(f"  [{model}] {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return {"locked_recipes_sha256": sha}''',
    '''                print(f"  [{model}] {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return {"locked_recipes_sha256": sha, "run_fingerprint": fp}''')

# R5 -- crash-safe reads
rep('''    df = pd.read_csv(out_dir / "test_per_case.csv").drop_duplicates(
        subset=["model", "case_id", "recipe"], keep="last")
    expected''', '''    df = se.read_rows(out_dir / "test_per_case.csv").drop_duplicates(
        subset=["model", "case_id", "recipe"], keep="last")
    expected''')
rep('''    per_case = pd.read_csv(out_dir / "test_per_case.csv").drop_duplicates(''',
    '''    per_case = se.read_rows(out_dir / "test_per_case.csv").drop_duplicates(''')

# R2 -- timing must use the same settings as the evaluated run
rep('''    """Validation cases only; never test. Run with nothing else on the GPU."""
    ids =''', '''    """Validation cases only; never test. Run with nothing else on the GPU."""
    _fingerprint(args, out_dir)
    ids =''')

# R1 + R4 -- report: primary first, empties and exclusions visible
start = t.index("def write_report(out_dir, summ, fam, ni, timing=None):")
end = t.index("def main():")
NEW_REPORT = r'''def write_report(out_dir, summ, fam, ni, timing=None):
    """Markdown generated only from the CSVs (every number traceable to a file).

    The pre-registered primary recipe (R4) comes first; R1-R3 are secondary and descriptive.
    Empty predictions and NaN-excluded cases are always shown.
    """
    def fmt(r, m):
        return f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_sd']:.4f}"

    def model_table(recipe):
        out = [f"| Model | τ | Dice | Precision | Precision (empty=0) | Recall | HD95 (mm) | ASD (mm) | clDice | n empty |",
               "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
        for _, r in summ[summ["recipe"] == recipe].sort_values("dice_mean", ascending=False).iterrows():
            out.append(f"| {r['model']} | {r['tau']:.2f} | {fmt(r, 'dice')} | {fmt(r, 'precision')} | "
                       f"{r['precision_strict_mean']:.4f} | {fmt(r, 'recall')} | {fmt(r, 'hd95')} | "
                       f"{fmt(r, 'asd')} | {fmt(r, 'cldice')} | {int(r['n_empty_pred'])} |")
        return out

    def fam_table(rows):
        out = ["| Family | a vs b | Metric | Mean diff (a−b) | 95% CI | p | Holm p | W/T/L | n excl. |",
               "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
        for _, r in rows.iterrows():
            out.append(f"| {r['family']} | {r['a']} vs {r['b']} | {r['metric']} | {r['mean_difference']:+.4f} | "
                       f"[{r['diff_ci95_lo']:+.4f}, {r['diff_ci95_hi']:+.4f}] | {r['p']:.3g} | {r['holm_p']:.3g} | "
                       f"{r['wins']}/{r['ties']}/{r['losses']} | {int(r['n_excluded_nan'])} |")
        return out

    def ni_table(rows):
        out = ["| Recipe | Mean diff | 95% CI | Non-inferior | Superior |", "| :--- | :---: | :---: | :---: | :---: |"]
        for _, r in rows.iterrows():
            out.append(f"| {r['recipe']} | {r['mean_diff']:+.4f} | [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | "
                       f"{r['non_inferior']} | {r['superior']} |")
        return out

    primary = se.PRIMARY_RECIPE
    lines = ["# Option A results (generated; do not hand-edit)", "",
             "Source files: `summary_test.csv`, `paired_stats_families.csv`, "
             "`noninferiority_rasnet_vs_nnunet_dice.csv`, `timing_summary.csv`.", ""]
    n_empty = int(summ["n_empty_pred"].sum())
    n_excl = int(fam["n_excluded_nan"].fillna(0).sum()) if len(fam) else 0
    if n_empty or n_excl:
        lines += [f"> **WARNING:** {n_empty} empty prediction(s) across models/recipes; {n_excl} case-comparison(s) "
                  "were excluded from paired tests because a metric was NaN. Report these counts with every "
                  "claim (CLAUDE.md §6).", ""]
    lines += [f"## PRIMARY result (pre-registered): {primary}", "",
              "Use only this section for confirmatory claims; choose the wording from doc 01 §7.", ""]
    lines += model_table(primary) + [""]
    lines += ["### Pre-declared families (Holm within family)", ""] + fam_table(fam[fam["role"] == "primary"]) + [""]
    lines += ["### Dice non-inferiority RASNet vs nnU-Net (margin −0.005)", ""] + ni_table(ni[ni["recipe"] == primary]) + [""]
    lines += ["## Secondary recipes (descriptive; not for confirmatory claims)", ""]
    for recipe in sorted(r for r in summ["recipe"].unique() if r != primary):
        lines += [f"### {recipe}", ""] + model_table(recipe) + [""]
    lines += ["### Secondary families", ""] + fam_table(fam[fam["role"] == "secondary"]) + [""]
    lines += ["### Secondary non-inferiority", ""] + ni_table(ni[ni["recipe"] != primary]) + [""]
    if timing is not None and len(timing):
        lines += ["## End-to-end inference time per case (Val cases, RTX 4080 SUPER; `timing_summary.csv`)", "",
                  "| Model | Mode | n | Median s/case [IQR] | Peak allocated VRAM (MB) |",
                  "| :--- | :--- | :---: | :---: | :---: |"]
        for _, r in timing.iterrows():
            lines.append(f"| {r['model']} | {r['mode']} | {int(r['n'])} | {r['seconds_median']:.1f} "
                         f"[{r['seconds_q25']:.1f}, {r['seconds_q75']:.1f}] | {r['peak_vram_mb']:.0f} |")
    (Path(out_dir) / "OPTION_A_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


'''
t = t[:start] + NEW_REPORT + t[end:]

p.write_text(t, encoding="utf-8", newline="\n")
print("run_optionA_symmetric_v1_eval.py patched")
```

### 5.5 `patch_test.py` — R6 adjustment in `tests/test_optionA_symmetric_eval.py`

```python
"""Apply the R6 adjustment to tests/test_optionA_symmetric_eval.py (run from repo root)."""
import pathlib
import sys

p = pathlib.Path("tests/test_optionA_symmetric_eval.py")
t = p.read_text(encoding="utf-8").replace("\r\n", "\n")
a = '''    p2.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="grid"):'''
b = '''    p2.write_text(json.dumps(bad))
    p2.with_suffix(".sha256").write_text(se.sha256_file(p2))   # valid sidecar -> the grid check is reached
    with pytest.raises(ValueError, match="grid"):'''
if t.count(a) != 1:
    sys.exit("anchor not unique/found in tests/test_optionA_symmetric_eval.py")
p.write_text(t.replace(a, b), encoding="utf-8", newline="\n")
print("patched tests/test_optionA_symmetric_eval.py")
```

### 5.6 `patch_v2.py` — R7, R8 in Rasnet-v2

```python
"""Apply code-review fixes R7-R8 to Rasnet-v2 (run from repo root)."""
import pathlib
import sys


def patch(path, pairs):
    p = pathlib.Path(path)
    t = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    for a, b in pairs:
        if t.count(a) != 1:
            sys.exit(f"{path}: anchor not unique/found: {a[:70]!r}")
        t = t.replace(a, b)
    p.write_text(t, encoding="utf-8", newline="\n")
    print("patched", path)


# R7 -- the Stage 2B decision records the same (strict) precision the gate used
patch("Rasnet-v2/src/stage2_pilot_eval.py", [(
    '''            "val94_precision": float(winner["precision_mean"]),
            "val94_hd95": float(winner["hd95_mean"]),''',
    '''            "val94_precision": float(winner["precision_mean"]),
            "val94_precision_strict": float(winner["precision_mean_strict"]),
            "val94_n_empty_pred": int(winner["n_empty_pred"]),
            "val94_hd95": float(winner["hd95_mean"]),''')])

# R8 -- ablation rows whose label assumes a conv_bias lead gate are flagged when Stage 2B locked another gate
patch("Rasnet-v2/src/stage5_stats_and_efficiency.py", [
    ('''def _run_family1(
    fam_tables: dict[tuple[str, str], dict[str, Any]],
    arms: tuple[str, ...],
) -> pd.DataFrame:''',
     '''def _run_family1(
    fam_tables: dict[tuple[str, str], dict[str, Any]],
    arms: tuple[str, ...],
    lead_gate_mode: str | None = None,
) -> pd.DataFrame:'''),
    ('''                        r["significant_holm_005"] = bool(hp < 0.05)
                        rows.append(r)

    return pd.DataFrame(rows)''',
     '''                        r["significant_holm_005"] = bool(hp < 0.05)
                        rows.append(r)

    df = pd.DataFrame(rows)
    if not df.empty:
        # These contrasts compare against rasnet_v2_f16, whose gate is the Stage 2B lead gate. Their labels
        # ("gate effect", "DS effect") are only valid when that lead gate is conv_bias, the gate of the
        # fixed ablation runs; otherwise they measure gate mode and DS together.
        gate_dependent = {"Abl_Gate_ConvBias_vs_OrigGN", "Abl_DS_Full_vs_NoDS", "Abl_DS_Full_vs_NoDS_3seed"}
        df["lead_gate_mode"] = lead_gate_mode if lead_gate_mode is not None else "unknown"
        confounded = lead_gate_mode not in (None, "conv_bias")
        df["ablation_label_valid"] = ~(df["contrast"].isin(gate_dependent) & confounded)
    return df'''),
    ('''    df_f1 = _run_family1(fam_tables, arms_present)''',
     '''    _plp = pilot_dir / "pilot_lock_decision.json"
    lead_gate_mode = (json.loads(_plp.read_text(encoding="utf-8")).get("locked_design", {}).get("gate_mode")
                      if _plp.is_file() else None)
    df_f1 = _run_family1(fam_tables, arms_present, lead_gate_mode=lead_gate_mode)'''),
])

# R7 test hook in the existing end-to-end test
patch("Rasnet-v2/tests/test_v2_pipeline_cpu.py", [(
    '''    assert dec["all_gates_passed"] is True
    assert (v2_root / "results" / "pilots_val94" / "pilot_lock_decision.md").is_file()''',
    '''    assert dec["all_gates_passed"] is True
    assert {"val94_precision_strict", "val94_n_empty_pred"} <= set(dec["locked_design"])
    assert (v2_root / "results" / "pilots_val94" / "pilot_lock_decision.md").is_file()''')])
```

### 5.7 `sync_doc01.py` — keep guide 01 byte-identical to the patched code

```python
"""Re-sync the 3 code blocks of guide 01 with the patched files, then verify byte identity (run from repo root)."""
import pathlib
import re
import sys

FENCE = "`" * 3                      # built at runtime so this script can itself be embedded in a .md
BLOCK = re.compile(re.escape(FENCE) + r"python\n(.*?)" + re.escape(FENCE), re.S)

doc_p = pathlib.Path("docs-for-next-v2-gen/01_OPTION_A_SYMMETRIC_V1_EVAL.md")
files = ["tests/test_optionA_symmetric_eval.py", "benchmark/symmetric_eval.py",
         "scripts/run_optionA_symmetric_v1_eval.py"]
doc = doc_p.read_text(encoding="utf-8").replace("\r\n", "\n")
spans = [m.span(1) for m in BLOCK.finditer(doc)]
if len(spans) != 3:
    sys.exit(f"Expected 3 python blocks in guide 01, found {len(spans)}")
codes = [pathlib.Path(f).read_text(encoding="utf-8").replace("\r\n", "\n") for f in files]
for code, f in zip(codes, files):
    if FENCE in code:
        sys.exit(f"{f} contains a code fence; cannot embed")
for (s, e), code in sorted(zip(spans, codes), key=lambda x: -x[0][0]):   # replace from the end
    doc = doc[:s] + code + doc[e:]
doc_p.write_text(doc, encoding="utf-8", newline="\n")
for f, code, block in zip(files, codes, BLOCK.findall(doc)):
    print("OK      " if code == block else "DIFFERS ", f)
```

---

## 6. Lab PC impact

- **Nothing new to do beyond the existing prompts.** Prompt 1 re-runs the suites, so the expected counts simply become 62 and 46.
- **New, intended refusals** you may see during Option A:
  - **`Run fingerprint mismatch`:** the code or `--sw-batch-size` changed inside one results folder. Re-run with the original settings, or start a new `--out` folder and record why.
  - **`… is not committed. Commit the Val-94 lock (manual step E7)`:** the agent tried test inference before committing the lock. Commit the lock, then re-run E8.
  - **`Lock sidecar … is missing`:** someone deleted `locked_recipes.sha256`. **STOP** and investigate; never re-create it by hand.
- **The report** now opens with the **PRIMARY (R4)** section and a warning banner if any model produced an empty prediction. Thesis wording (doc 01 §7) is chosen **only** from the primary section.
- **Rasnet-v2 Stage 5:** read the `ablation_label_valid` column in `results/stats_and_efficiency/family1_causal_stats.csv` before writing any ablation sentence. If it is `False`, describe the contrast as "gate mode + deep supervision together", never as a pure DS or gate effect.
