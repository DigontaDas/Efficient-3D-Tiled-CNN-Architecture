"""
Resilient Stage & Multi-Seed Job Queue Orchestrator (`run_queue.py`).
Checks disk headroom (> 100 GB free), 784-case cache_fp16 preflight, DONE.flag markers,
Stage 2B deduplication, CUDA cleanup between jobs, and Stages 2–5 execution.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_builder import build_cache, is_case_cached
from dataset_pairs import load_validated_splits
from study_registry import (
    atomic_write_json,
    build_stage2a_jobs,
    build_stage2b_jobs as _registry_build_stage2b_jobs,
    build_stage3_eval_manifest,
    build_stage3_jobs,
)
from train_fair_v2 import run_training
from train_nnunet_v2_ref import run_nnunet_v2_seed


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def disk_inventory(root_dir: Path) -> dict[str, float]:
    usage = shutil.disk_usage(root_dir)
    gb = 1024.0 ** 3
    return {
        "free_gb": round(usage.free / gb, 2),
        "cache_fp16_gb": round(_dir_size_bytes(root_dir / "cache_fp16") / gb, 2),
        "nnunet_preprocessed_gb": round(_dir_size_bytes(root_dir / "nnUNet_work" / "nnUNet_preprocessed") / gb, 2),
        "checkpoints_gb": round(_dir_size_bytes(root_dir / "checkpoints") / gb, 2),
        "nnunet_results_gb": round(_dir_size_bytes(root_dir / "nnUNet_work" / "nnUNet_results") / gb, 2),
        "val94_tmp_probs_gb": round(_dir_size_bytes(root_dir / "results" / "val94_final" / "_tmp_probs") / gb, 2),
    }


def check_disk_headroom(path: Path, min_free_gb: float = 25.0) -> float:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024.0 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(f"Low disk headroom on {path}: {free_gb:.1f} GB < {min_free_gb:.1f} GB required")
    return free_gb


def _all_jobs_done(root_dir: Path, jobs: list[dict[str, Any]]) -> bool:
    return bool(jobs) and all((root_dir / "checkpoints" / j["run_name"] / "DONE.flag").is_file() for j in jobs)


def cleanup_unused_caches(cfg: dict[str, Any], aggressive_low_disk_gb: float = 60.0) -> dict[str, Any]:
    """
    Safely delete intermediate caches strictly AFTER every consumer stage carries DONE.flag:
      1. Completed-run transient `mid_epoch_state.pt` (keeps `latest_state.pt`, `best_model.pth`, `epoch_*.pth`).
      2. Stale Stage 4B `_tmp_probs/<run>/` only when all 3 Val94 CSVs exist for `<run>`.
      3. `cache_fp16/` (~154 GB) strictly after Stage 2A, Stage 2B (`pilot_lock_decision.json`), and ALL 17 Stage 3 MONAI runs have DONE.flag.
      4. `nnUNet_preprocessed/` bulk blobs (~119 GB, preserving JSON plans/splits) and `nnUNet_raw/` hardlinks strictly after ALL 3 Stage 3 nnU-Net seeds have DONE.flag.
      5. Stage 2 pilot `latest_state.pt` only if `pilot_lock_decision.json` exists AND free disk < `aggressive_low_disk_gb`.
    """
    root_dir = Path(cfg["study"]["root_dir"])
    removed_paths: list[str] = []
    reclaimed_bytes = 0

    # 1. Remove mid_epoch_state.pt for already-completed runs
    ckpt_root = root_dir / "checkpoints"
    if ckpt_root.is_dir():
        for run_dir in ckpt_root.iterdir():
            if run_dir.is_dir() and (run_dir / "DONE.flag").is_file():
                mid_p = run_dir / "mid_epoch_state.pt"
                if mid_p.is_file():
                    sz = mid_p.stat().st_size
                    mid_p.unlink()
                    reclaimed_bytes += sz
                    removed_paths.append(str(mid_p))

    # 2. Remove orphaned Stage 4B _tmp_probs/<run> only if all 3 CSVs for that run exist
    val94_dir = root_dir / "results" / "val94_final"
    tmp_probs_root = val94_dir / "_tmp_probs"
    if tmp_probs_root.is_dir():
        for rdir in tmp_probs_root.iterdir():
            if not rdir.is_dir():
                continue
            rname = rdir.name
            done_csvs = all(
                (val94_dir / f"per_case_{rname}_arm_{arm}.csv").is_file()
                and (val94_dir / f"per_case_{rname}_arm_{arm}.csv").stat().st_size > 0
                for arm in ("A", "C_top2", "C_seeded")
            )
            if done_csvs:
                sz = _dir_size_bytes(rdir)
                shutil.rmtree(rdir, ignore_errors=True)
                reclaimed_bytes += sz
                removed_paths.append(str(rdir))
        if not any(tmp_probs_root.iterdir()):
            tmp_probs_root.rmdir()

    # Check Stage 2 & Stage 3 completion gates before removing large training caches
    s2a_done = _all_jobs_done(root_dir, build_stage2a_jobs(cfg))
    pilot_lock_p = root_dir / "results" / "pilots_val94" / "pilot_lock_decision.json"
    s2b_locked = pilot_lock_p.is_file() and pilot_lock_p.stat().st_size > 0
    s3_manifest = build_stage3_eval_manifest(cfg)
    s3_monai_done = _all_jobs_done(root_dir, [j for j in s3_manifest if j["kind"] == "monai"])
    s3_nnunet_done = _all_jobs_done(root_dir, [j for j in s3_manifest if j["kind"] == "nnunet"])

    # 3. Clean cache_fp16/ strictly after all Stage 2 + Stage 3 MONAI runs are DONE
    cache_dir = root_dir / "cache_fp16"
    if s2a_done and s2b_locked and s3_monai_done and cache_dir.is_dir():
        sz = _dir_size_bytes(cache_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)
        reclaimed_bytes += sz
        removed_paths.append(str(cache_dir))

    # 4. Clean nnUNet_preprocessed bulk blobs & nnUNet_raw hardlinks strictly after all 3 nnU-Net seeds are DONE
    if s3_nnunet_done:
        nn_raw = root_dir / "nnUNet_work" / "nnUNet_raw"
        if nn_raw.is_dir():
            shutil.rmtree(nn_raw, ignore_errors=True)
            removed_paths.append(str(nn_raw))

        dset_name = str(cfg.get("nnunet_v2", {}).get("dataset_name", "Dataset501_CoronarySeg"))
        prep_dset = root_dir / "nnUNet_work" / "nnUNet_preprocessed" / dset_name
        if prep_dset.is_dir():
            for child in list(prep_dset.iterdir()):
                # Preserve lightweight JSON metadata (nnUNetPlans.json, dataset.json, splits_final.json, fingerprint)
                if child.is_file() and child.suffix.lower() == ".json":
                    continue
                sz = _dir_size_bytes(child) if child.is_dir() else child.stat().st_size
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
                reclaimed_bytes += sz
                removed_paths.append(str(child))

    # 5. Emergency Stage 2 pilot optimizer state trim ONLY if disk still low after Stage 2 lock
    cur_free_gb = shutil.disk_usage(root_dir).free / (1024.0 ** 3)
    if s2a_done and s2b_locked and cur_free_gb < aggressive_low_disk_gb:
        lrs_p = root_dir / "results" / "pilots_val94" / "locked_lrs.json"
        pilot_names = {j["run_name"] for j in build_stage2a_jobs(cfg)}
        if lrs_p.is_file():
            from study_registry import build_stage2b_candidates
            lrs = json.loads(lrs_p.read_text(encoding="utf-8")).get("locked_lrs", {})
            if lrs:
                pilot_names.update(c["run_name"] for c in build_stage2b_candidates(cfg, lrs))
        for rname in sorted(pilot_names):
            rdir = ckpt_root / rname
            if (rdir / "DONE.flag").is_file() and (rdir / "epoch_50.pth").is_file():
                lat = rdir / "latest_state.pt"
                if lat.is_file():
                    sz = lat.stat().st_size
                    lat.unlink()
                    reclaimed_bytes += sz
                    removed_paths.append(str(lat))

    reclaimed_gb = round(reclaimed_bytes / (1024.0 ** 3), 2)
    inv = disk_inventory(root_dir)
    if removed_paths:
        print(f"[CLEANUP] Reclaimed {reclaimed_gb:.2f} GB across {len(removed_paths)} unused cache item(s) | Free={inv['free_gb']:.2f} GB")
    return {
        "reclaimed_gb": reclaimed_gb,
        "removed_count": len(removed_paths),
        "removed_paths": removed_paths,
        "inventory": inv,
    }


def preflight_check_cache_784(cfg: dict[str, Any]) -> tuple[int, list[int]]:
    """
    Verify all 784 (690 train + 94 val) volumes are cleanly cached inside cache_fp16/.
    Fails closed with missing case IDs if any case is absent or incomplete.
    """
    splits = load_validated_splits(cfg["study"]["splits_file"])
    expected_ids = sorted(int(c) for c in (splits["train"] + splits["val"]))
    cache_dir = Path(cfg["study"]["root_dir"]) / "cache_fp16"
    missing = [cid for cid in expected_ids if not is_case_cached(cache_dir, cid)]
    if missing:
        raise RuntimeError(
            f"cache_fp16 preflight failed: {len(missing)}/{len(expected_ids)} cases missing "
            f"(first 10: {missing[:10]}). Run `python run_queue.py --stage cache` first."
        )
    val20_file = Path(cfg["study"]["val20_ids_file"])
    if not val20_file.is_file():
        raise FileNotFoundError(f"Missing frozen val20_curves_ids.json: {val20_file}")
    return len(expected_ids), missing


def build_stage2b_jobs(cfg: dict[str, Any], locked_lrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deduplicated Stage 2B training jobs and persist skip/alias ledger."""
    jobs, skipped = _registry_build_stage2b_jobs(cfg, locked_lrs)
    root_dir = Path(cfg["study"]["root_dir"])
    atomic_write_json(
        {
            "jobs_to_run": [j["run_name"] for j in jobs],
            "skipped_or_aliased": skipped,
        },
        root_dir / "logs" / "stage2b_dedup_aliases.json",
    )
    return jobs


def execute_jobs(config_path: str | Path, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    status_file = root_dir / "logs" / "queue_status.json"
    results: list[dict[str, Any]] = []

    for idx, job in enumerate(jobs, 1):
        free_gb = check_disk_headroom(root_dir, min_free_gb=25.0)
        if free_gb < 100.0:
            cleanup_unused_caches(cfg, aggressive_low_disk_gb=60.0)
            free_gb = check_disk_headroom(root_dir, min_free_gb=25.0)
        inv = disk_inventory(root_dir)
        print(
            f"[DISK] Job {idx}/{len(jobs)} ({job['run_name']}) | Free={inv['free_gb']:.2f} GB "
            f"| cache_fp16={inv['cache_fp16_gb']:.2f} GB | nnUNet_prep={inv['nnunet_preprocessed_gb']:.2f} GB "
            f"| ckpts={inv['checkpoints_gb']:.2f} GB"
        )
        atomic_write_json(
            {
                "current_index": idx,
                "total_jobs": len(jobs),
                "current_job": job["run_name"],
                **inv,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            status_file,
        )

        done_flag = root_dir / "checkpoints" / job["run_name"] / "DONE.flag"
        if done_flag.is_file():
            print(f"[QUEUE-SKIP] {job['run_name']} already completed ({done_flag})")
            res = json.loads(done_flag.read_text(encoding="utf-8"))
            results.append(res)
            cleanup_unused_caches(cfg, aggressive_low_disk_gb=60.0)
            continue

        if job["kind"] == "monai":
            res = run_training(
                config_path=config_path,
                run_name=job["run_name"],
                arch=job["arch"],
                init_filters=job["init_filters"],
                lr=job["lr"],
                seed=job["seed"],
                epochs=job.get("epochs"),
                gate_mode=job.get("gate_mode", "conv_bias"),
                enable_ds=job.get("enable_ds", True),
                loss_name=job.get("loss_name", "rasnet_v2"),
                w_cldice_override=job.get("w_cldice"),
            )
        elif job["kind"] == "nnunet":
            res = run_nnunet_v2_seed(config_path=config_path, seed=job["seed"])
        else:
            raise ValueError(f"Unknown job kind: {job['kind']}")

        results.append(res)
        cleanup_unused_caches(cfg, aggressive_low_disk_gb=60.0)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def _default_locked_lrs_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["study"]["root_dir"]) / "results" / "pilots_val94" / "locked_lrs.json"


def _load_locked_lrs(cfg: dict[str, Any], locked_lrs_json: str | Path | None) -> dict[str, float]:
    p = Path(locked_lrs_json) if locked_lrs_json else _default_locked_lrs_path(cfg)
    if not p.is_file():
        raise FileNotFoundError(f"Missing locked LRs JSON: {p}. Run stage2a_eval first.")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if "locked_lrs" in payload and isinstance(payload["locked_lrs"], dict):
        return {k: float(v) for k, v in payload["locked_lrs"].items()}
    return {k: float(v) for k, v in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="RASNet-v2 Resilient Stage Queue Orchestrator (Stages 2-5)")
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "cache",
            "cleanup",
            "stage2a",
            "stage2a_eval",
            "stage2b",
            "stage2b_eval",
            "stage3",
            "stage4a",
            "stage4b",
            "stage4c",
            "stage5",
        ],
    )
    parser.add_argument("--locked-lrs-json", default=None)
    parser.add_argument("--workers", type=int, default=6, help="Cache builder worker count for --stage cache")
    parser.add_argument("--auto-cache", action="store_true", help="Auto-build missing cache_fp16 cases before training")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])

    if args.stage == "cache":
        res = build_cache(args.config, max_workers=args.workers)
        print(json.dumps(res, indent=2))
        return

    if args.stage == "cleanup":
        res = cleanup_unused_caches(cfg, aggressive_low_disk_gb=60.0)
        print(json.dumps(res, indent=2))
        return

    need_cache_check = False
    if args.stage in ("stage2a", "stage2b"):
        need_cache_check = True
    elif args.stage == "stage3":
        s3_monai = [j for j in build_stage3_eval_manifest(cfg) if j["kind"] == "monai"]
        need_cache_check = not _all_jobs_done(root_dir, s3_monai)

    if need_cache_check:
        if args.auto_cache:
            try:
                preflight_check_cache_784(cfg)
            except RuntimeError:
                build_cache(args.config, max_workers=args.workers)
        n_cached, _ = preflight_check_cache_784(cfg)
        print(f"[PREFLIGHT] cache_fp16 verified complete ({n_cached}/784 train+val cases)")

    if args.stage == "stage2a":
        execute_jobs(args.config, build_stage2a_jobs(cfg))
    elif args.stage == "stage2a_eval":
        from stage2_pilot_eval import evaluate_stage2a_pilots
        evaluate_stage2a_pilots(args.config)
    elif args.stage == "stage2b":
        lrs = _load_locked_lrs(cfg, args.locked_lrs_json)
        jobs = build_stage2b_jobs(cfg, lrs)
        execute_jobs(args.config, jobs)
    elif args.stage == "stage2b_eval":
        from stage2_pilot_eval import evaluate_stage2b_pilots
        lrs_path = Path(args.locked_lrs_json) if args.locked_lrs_json else _default_locked_lrs_path(cfg)
        evaluate_stage2b_pilots(args.config, locked_lrs_path=lrs_path)
    elif args.stage == "stage3":
        lock_doc = root_dir / "results" / "pilots_val94" / "pilot_lock_decision.json"
        if not lock_doc.is_file():
            raise FileNotFoundError(f"Missing Stage 2B lock decision: {lock_doc}. Run stage2b_eval first.")
        pilot_lock = json.loads(lock_doc.read_text(encoding="utf-8"))
        lrs = _load_locked_lrs(cfg, args.locked_lrs_json)
        execute_jobs(args.config, build_stage3_jobs(cfg, lrs, pilot_lock=pilot_lock))
    elif args.stage == "stage4a":
        cleanup_unused_caches(cfg, aggressive_low_disk_gb=60.0)
        check_disk_headroom(root_dir, min_free_gb=20.0)
        from stage4a_val20_curves import run_stage4a_val20_curves
        run_stage4a_val20_curves(args.config)
    elif args.stage == "stage4b":
        cleanup_unused_caches(cfg, aggressive_low_disk_gb=60.0)
        check_disk_headroom(root_dir, min_free_gb=30.0)
        from stage4b_val94_lock import run_stage4b_val94_lock
        run_stage4b_val94_lock(args.config)
    elif args.stage == "stage4c":
        cleanup_unused_caches(cfg, aggressive_low_disk_gb=60.0)
        check_disk_headroom(root_dir, min_free_gb=20.0)
        from stage4c_test150_eval import run_stage4c_test150
        run_stage4c_test150(args.config)
    elif args.stage == "stage5":
        from stage5_stats_and_efficiency import run_stage5_analysis
        run_stage5_analysis(args.config)


if __name__ == "__main__":
    main()
