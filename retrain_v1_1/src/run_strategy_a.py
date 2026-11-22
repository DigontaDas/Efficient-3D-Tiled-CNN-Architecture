"""Strategy A (`retrain_v1_1`) sequential single-GPU pipeline orchestrator.

Usage:
    python -m retrain_v1_1.src.run_strategy_a --stage cache
    python -m retrain_v1_1.src.run_strategy_a --stage train
    python -m retrain_v1_1.src.run_strategy_a --stage record_checkpoints
    python -m retrain_v1_1.src.run_strategy_a --stage val
    python -m retrain_v1_1.src.run_strategy_a --stage lock
    # commit retrain_v1_1/results/locked_recipes.json + .sha256 before test
    python -m retrain_v1_1.src.run_strategy_a --stage test
    python -m retrain_v1_1.src.run_strategy_a --stage timing
    python -m retrain_v1_1.src.run_strategy_a --stage stats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark import symmetric_eval as se
from benchmark.preflight import validate_splits
from retrain_v1_1.src.cache import build_transform_hashed_cache
from retrain_v1_1.src.eval_pipeline import (
    MODELS_V11,
    build_strategy_a_lock,
    compute_strategy_a_stats,
    make_strategy_a_figures,
    run_eval_stage_test,
    run_eval_stage_timing,
    run_eval_stage_val,
    write_strategy_a_lock,
    write_strategy_a_report,
)
from retrain_v1_1.src.train import atomic_write_json, train_model


def resolve_dataset_root(cfg: dict[str, Any], repo_root: Path) -> str:
    env_root = os.environ.get("IMAGECAS_DATA_ROOT", "").strip()
    if env_root:
        return env_root
    lab_cfg_path = repo_root / cfg["study"]["dataset_root_config"]
    lab_cfg = yaml.safe_load(lab_cfg_path.read_text(encoding="utf-8"))
    root = lab_cfg.get("dataset_root")
    if not root:
        raise RuntimeError(f"dataset_root missing in {lab_cfg_path}")
    return str(root)


def record_checkpoint_hashes(cfg: dict[str, Any], repo_root: Path) -> Path:
    study_dir = repo_root / cfg["study"]["study_dir"]
    out_dir = study_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in cfg["models"]:
        if not spec.get("train", False):
            continue
        name = spec["name"]
        cdir = study_dir / "checkpoints" / name
        ldir = study_dir / "logs" / name
        manifest = json.loads((ldir / "train_manifest.json").read_text(encoding="utf-8"))
        best_p = cdir / "best_model.pth"
        final_p = cdir / "final_epoch200.pth"
        records.append(
            {
                "model": name,
                "n_params": int(manifest["n_params"]),
                "best_epoch": int(manifest["best_epoch"]),
                "best_val94_dice": float(manifest["best_val_dice"]),
                "best_model_path": str(best_p.relative_to(repo_root).as_posix()),
                "best_size_bytes": int(best_p.stat().st_size),
                "best_sha256": se.sha256_file(best_p),
                "final_model_path": str(final_p.relative_to(repo_root).as_posix()),
                "final_size_bytes": int(final_p.stat().st_size),
                "final_sha256": se.sha256_file(final_p),
            }
        )
    out_path = out_dir / "checkpoint_hashes.json"
    atomic_write_json({"recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": records}, out_path)
    return out_path


def write_stage_manifest(out_dir: Path, stage: str, args, extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "stage": stage,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {k: str(v) for k, v in vars(args).items()},
        "git": se.git_info(REPO_ROOT, code_paths=("retrain_v1_1/src", "retrain_v1_1/configs", "benchmark")),
        "env": se.env_info(),
        **(extra or {}),
    }
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    p = out_dir / f"manifest_{stage}_{stamp}.json"
    atomic_write_json(payload, p)
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Strategy A (retrain_v1_1) Orchestrator")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "retrain_v1_1" / "configs" / "strategy_a.yaml",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("cache", "train", "record_checkpoints", "val", "lock", "test", "timing", "stats"),
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    study_dir = REPO_ROOT / cfg["study"]["study_dir"]
    out_dir = study_dir / "results"
    log_root = study_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    ginfo = se.git_info(REPO_ROOT, code_paths=("retrain_v1_1/src", "retrain_v1_1/configs", "benchmark"))
    if ginfo["dirty_code"] and not args.allow_dirty and not args.smoke and args.stage in ("train", "val", "lock", "test"):
        raise SystemExit("Refusing to run with git.dirty_code=True. Commit first (or pass --allow-dirty for debugging).")

    dataset_root = resolve_dataset_root(cfg, REPO_ROOT)
    splits_raw = json.loads((REPO_ROOT / cfg["study"]["splits_file"]).read_text(encoding="utf-8"))
    parts = validate_splits(splits_raw)
    train_ids = sorted(parts["train"])
    val_ids = sorted(parts["val"])

    if args.stage == "cache":
        summary = build_transform_hashed_cache(cfg, REPO_ROOT, dataset_root, max_workers=args.cache_workers)
        write_stage_manifest(out_dir, "cache", args, {"cache_summary": summary})
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    if args.stage == "train":
        trainable = [m for m in cfg["models"] if m.get("train", False)]
        if args.models:
            sel = set(args.models)
            trainable = [m for m in trainable if m["name"] in sel]
        q_status = log_root / "queue_status.json"
        for spec in trainable:
            name = spec["name"]
            train_model(
                cfg=cfg,
                model_spec=spec,
                cache_root=study_dir / "cache",
                ckpt_dir=study_dir / "checkpoints" / name,
                log_dir=log_root / name,
                train_ids=train_ids,
                val_ids=val_ids,
                device_str=args.device,
                queue_status_path=q_status,
            )
        write_stage_manifest(out_dir, "train", args, {"trained_models": [m["name"] for m in trainable]})
        return 0

    if args.stage == "record_checkpoints":
        p = record_checkpoint_hashes(cfg, REPO_ROOT)
        print(f"Recorded checkpoint hashes -> {p}", flush=True)
        return 0

    eval_models = list(args.models) if args.models else list(MODELS_V11)

    if args.stage == "val":
        csv_p = run_eval_stage_val(cfg, REPO_ROOT, dataset_root, eval_models, device_str=args.device, smoke=args.smoke)
        write_stage_manifest(out_dir, "val", args, {"val_grid_csv": str(csv_p)})
        return 0

    if args.stage == "lock":
        val_df = se.read_rows(out_dir / "val_grid_scores.csv")
        payload = build_strategy_a_lock(val_df, splits_raw, models=eval_models, allow_partial=bool(args.smoke))
        sha = write_strategy_a_lock(out_dir / "locked_recipes.json", payload)
        write_stage_manifest(out_dir, "lock", args, {"locked_recipes_sha256": sha})
        print(json.dumps(payload["thresholds"], indent=2), flush=True)
        print(f"LOCK SHA-256: {sha}", flush=True)
        return 0

    if args.stage == "test":
        csv_p = run_eval_stage_test(cfg, REPO_ROOT, dataset_root, eval_models, device_str=args.device, smoke=args.smoke)
        write_stage_manifest(out_dir, "test", args, {"test_per_case_csv": str(csv_p)})
        return 0

    if args.stage == "timing":
        t_p = run_eval_stage_timing(cfg, REPO_ROOT, dataset_root, eval_models, device_str=args.device)
        write_stage_manifest(out_dir, "timing", args, {"timing_summary_csv": str(t_p)})
        return 0

    if args.stage == "stats":
        per_case = se.read_rows(out_dir / "test_per_case.csv").drop_duplicates(
            subset=["model", "case_id", "recipe"], keep="last"
        )
        eval_cfg = cfg["evaluation"]
        summ, fam, ni = compute_strategy_a_stats(
            per_case,
            n_boot=int(eval_cfg.get("bootstrap_resamples", 10000)),
            seed=int(eval_cfg.get("bootstrap_seed", 42)),
            ni_margin=float(eval_cfg.get("ni_margin", -0.005)),
        )
        summ.to_csv(out_dir / "summary_test.csv", index=False, lineterminator="\n")
        fam.to_csv(out_dir / "paired_stats_families.csv", index=False, lineterminator="\n")
        ni.to_csv(out_dir / "noninferiority_rasnet_vs_nnunet_dice.csv", index=False, lineterminator="\n")
        timing_p = out_dir / "timing_summary.csv"
        timing_df = pd.read_csv(timing_p) if timing_p.is_file() else None
        rep_p = write_strategy_a_report(out_dir, summ, fam, ni, timing=timing_df)
        figs = make_strategy_a_figures(summ, per_case, out_dir)
        write_stage_manifest(out_dir, "stats", args, {"report": str(rep_p), "figures": [str(f) for f in figs]})
        print(f"Wrote report -> {rep_p} and {len(figs)} figure(s).", flush=True)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
