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
