"""
Shared Stage 2–6 Study Registry, Checkpoint Path Resolution, Deduplication & Atomic I/O Helpers.
Single source of truth for all run specs across run_queue.py and Stage 2/4/5/6 evaluators.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_LR_KEYS = (
    "segresnet_f16",
    "segresnet_f32",
    "rasnet_v2_f16",
    "rasnet_v2_f32",
    "vnet",
    "unet3d",
)


def lr_tag(lr: float) -> str:
    return f"{float(lr):.0e}".replace("-0", "-")


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing file for SHA-256: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)
    return p


def atomic_write_json(payload: Any, path: str | Path, sort_keys: bool = False) -> Path:
    return atomic_write_text(json.dumps(payload, indent=2, sort_keys=sort_keys), path)


def atomic_write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)
    return p


def df_to_markdown(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table without requiring tabulate."""
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float,)):
                cells.append("nan" if pd.isna(v) else format(v, float_fmt))
            elif hasattr(v, "dtype") and "float" in str(v.dtype):
                fv = float(v)
                cells.append("nan" if pd.isna(fv) else format(fv, float_fmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_stage2a_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    12 (F=16 across 4 MONAI archs x 3 LRs) + 4 (F=32 mb=1x2 sanity checks on SegResNet-v2 & RASNet-v2)
    = 16 runs locking all 6 entries in locked_lrs.json with zero cross-architecture fallback.
    """
    ep = int(cfg["training_budget"]["pilot_epochs"])
    jobs: list[dict[str, Any]] = []
    for arch in ("segresnet", "rasnet_v2", "vnet", "unet3d"):
        lr_key = f"{arch}_f16" if arch in ("segresnet", "rasnet_v2") else arch
        for lr in (1e-4, 3e-4, 1e-3):
            jobs.append({
                "kind": "monai",
                "stage": "stage2a",
                "lr_key": lr_key,
                "run_name": f"pilot2a_{arch}_f16_lr{lr_tag(lr)}_seed42",
                "arch": arch,
                "init_filters": 32 if arch == "unet3d" else 16,
                "lr": float(lr),
                "seed": 42,
                "epochs": ep,
                "gate_mode": "conv_bias",
                "enable_ds": True,
                "loss_name": "rasnet_v2",
                "w_cldice": float(cfg["loss"]["rasnet_v2"]["w_cldice"]),
            })
    for arch in ("segresnet", "rasnet_v2"):
        lr_key = f"{arch}_f32"
        for lr in (3e-4, 1e-3):
            jobs.append({
                "kind": "monai",
                "stage": "stage2a",
                "lr_key": lr_key,
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
    return jobs


def build_stage2b_candidates(cfg: dict[str, Any], locked_lrs: dict[str, float]) -> list[dict[str, Any]]:
    """
    All 5 Stage 2B RASNet-v2 design candidates over F in {16, 32}, gate_mode, w_cldice,
    annotated with `dedup_source_run` when identical (F, lr, gate_mode, enable_ds=True, w_cldice=0.2)
    was already produced by Stage 2A.
    """
    ep = int(cfg["training_budget"]["pilot_epochs"])
    lr16 = float(locked_lrs["rasnet_v2_f16"])
    lr32 = float(locked_lrs["rasnet_v2_f32"])
    s2a_by_sig = {
        (j["arch"], int(j["init_filters"]), float(j["lr"]), j["gate_mode"], bool(j["enable_ds"]), j["loss_name"], float(j["w_cldice"]), int(j["seed"])): j["run_name"]
        for j in build_stage2a_jobs(cfg)
    }

    specs = [
        {"run_name": "pilot2b_rasnet_v2_f16_convbias_cl020", "init_filters": 16, "lr": lr16, "gate_mode": "conv_bias", "w_cldice": 0.2},
        {"run_name": "pilot2b_rasnet_v2_f16_gnorig_cl020", "init_filters": 16, "lr": lr16, "gate_mode": "groupnorm_orig", "w_cldice": 0.2},
        {"run_name": "pilot2b_rasnet_v2_f16_convbias_cl000", "init_filters": 16, "lr": lr16, "gate_mode": "conv_bias", "w_cldice": 0.0},
        {"run_name": "pilot2b_rasnet_v2_f32_convbias_cl020", "init_filters": 32, "lr": lr32, "gate_mode": "conv_bias", "w_cldice": 0.2},
        {"run_name": "pilot2b_rasnet_v2_f32_gnorig_cl020", "init_filters": 32, "lr": lr32, "gate_mode": "groupnorm_orig", "w_cldice": 0.2},
    ]
    out: list[dict[str, Any]] = []
    for sp in specs:
        sig = ("rasnet_v2", int(sp["init_filters"]), float(sp["lr"]), sp["gate_mode"], True, "rasnet_v2", float(sp["w_cldice"]), 42)
        dedup_src = s2a_by_sig.get(sig)
        out.append({
            "kind": "monai",
            "stage": "stage2b",
            "run_name": sp["run_name"],
            "arch": "rasnet_v2",
            "init_filters": int(sp["init_filters"]),
            "lr": float(sp["lr"]),
            "seed": 42,
            "epochs": ep,
            "gate_mode": sp["gate_mode"],
            "enable_ds": True,
            "loss_name": "rasnet_v2",
            "w_cldice": float(sp["w_cldice"]),
            "dedup_source_run": dedup_src,
        })
    return out


def build_stage2b_jobs(cfg: dict[str, Any], locked_lrs: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return (jobs_to_run, skipped_dedup_records) eliminating duplicate (F, lr, conv_bias, w_cldice=0.2)
    Stage 2B runs already covered by Stage 2A or already carrying DONE.flag.
    """
    root_dir = Path(cfg["study"]["root_dir"])
    candidates = build_stage2b_candidates(cfg, locked_lrs)
    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cand in candidates:
        own_done = (root_dir / "checkpoints" / cand["run_name"] / "DONE.flag").is_file()
        src_run = cand.get("dedup_source_run")
        if own_done:
            skipped.append({"run_name": cand["run_name"], "reason": "own_DONE_flag", "resolved_run": cand["run_name"]})
            continue
        if src_run is not None:
            skipped.append({"run_name": cand["run_name"], "reason": "deduplicated_against_stage2a", "resolved_run": src_run})
            continue
        jobs.append(cand)
    return jobs, skipped


def validate_locked_lrs(locked_lrs: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in REQUIRED_LR_KEYS:
        if k not in locked_lrs:
            raise KeyError(f"locked_lrs.json missing required explicit LR key {k!r} (found {sorted(locked_lrs)})")
        val = float(locked_lrs[k])
        if val <= 0.0:
            raise ValueError(f"Invalid LR for {k}: {val}")
        out[k] = val
    return out


def build_stage3_jobs(
    cfg: dict[str, Any],
    locked_lrs: dict[str, Any],
    pilot_lock: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Exact 20-run Stage 3 matrix: 1+3+3+1+1+3+3+1+1 MONAI runs + 3 nnU-Net V2 seeds."""
    lrs = validate_locked_lrs(locked_lrs)
    seeds = [int(s) for s in cfg["study"]["seeds"]]
    ep = int(cfg["training_budget"]["epochs"])
    default_w_cl = float(cfg["loss"]["rasnet_v2"]["w_cldice"])
    ld = pilot_lock.get("locked_design", {}) if isinstance(pilot_lock, dict) else {}
    lead_gate = str(ld.get("gate_mode", "conv_bias"))
    lead_w_cl = float(ld.get("w_cldice", default_w_cl))

    jobs: list[dict[str, Any]] = [
        {
            "kind": "monai",
            "stage": "stage3",
            "family": "segresnet_f16_baseloss",
            "run_name": "s3_segresnet_f16_baseloss_seed42",
            "arch": "segresnet",
            "init_filters": 16,
            "lr": lrs["segresnet_f16"],
            "seed": 42,
            "epochs": ep,
            "gate_mode": "conv_bias",
            "enable_ds": False,
            "loss_name": "base_dice_focal",
        },
    ]
    for s in seeds:
        jobs.append({
            "kind": "monai",
            "stage": "stage3",
            "family": "segresnet_f16_v2loss",
            "run_name": f"s3_segresnet_f16_v2loss_seed{s}",
            "arch": "segresnet",
            "init_filters": 16,
            "lr": lrs["segresnet_f16"],
            "seed": s,
            "epochs": ep,
            "gate_mode": "conv_bias",
            "enable_ds": False,
            "loss_name": "rasnet_v2",
        })
    for s in seeds:
        jobs.append({
            "kind": "monai",
            "stage": "stage3",
            "family": "segresnet_f32_v2loss",
            "run_name": f"s3_segresnet_f32_v2loss_seed{s}",
            "arch": "segresnet",
            "init_filters": 32,
            "lr": lrs["segresnet_f32"],
            "seed": s,
            "epochs": ep,
            "gate_mode": "conv_bias",
            "enable_ds": False,
            "loss_name": "rasnet_v2",
        })

    jobs.extend([
        {
            "kind": "monai",
            "stage": "stage3",
            "family": "rasnet_orig_ag_f16",
            "run_name": "s3_rasnet_orig_ag_f16_ds_v2loss_seed42",
            "arch": "rasnet_v2",
            "init_filters": 16,
            "lr": lrs["rasnet_v2_f16"],
            "seed": 42,
            "epochs": ep,
            "gate_mode": "groupnorm_orig",
            "enable_ds": True,
            "loss_name": "rasnet_v2",
        },
        {
            "kind": "monai",
            "stage": "stage3",
            "family": "rasnet_fixed_ag_nods_f16",
            "run_name": "s3_rasnet_fixed_ag_f16_nods_v2loss_seed42",
            "arch": "rasnet_v2",
            "init_filters": 16,
            "lr": lrs["rasnet_v2_f16"],
            "seed": 42,
            "epochs": ep,
            "gate_mode": "conv_bias",
            "enable_ds": False,
            "loss_name": "rasnet_v2",
        },
    ])
    for s in seeds:
        jobs.append({
            "kind": "monai",
            "stage": "stage3",
            "family": "rasnet_v2_f16",
            "run_name": f"s3_rasnet_v2_f16_full_seed{s}",
            "arch": "rasnet_v2",
            "init_filters": 16,
            "lr": lrs["rasnet_v2_f16"],
            "seed": s,
            "epochs": ep,
            "gate_mode": lead_gate,
            "enable_ds": True,
            "loss_name": "rasnet_v2",
            "w_cldice": lead_w_cl,
        })
    for s in seeds:
        jobs.append({
            "kind": "monai",
            "stage": "stage3",
            "family": "rasnet_v2_f32",
            "run_name": f"s3_rasnet_v2_f32_full_seed{s}",
            "arch": "rasnet_v2",
            "init_filters": 32,
            "lr": lrs["rasnet_v2_f32"],
            "seed": s,
            "epochs": ep,
            "gate_mode": lead_gate,
            "enable_ds": True,
            "loss_name": "rasnet_v2",
            "w_cldice": lead_w_cl,
        })

    jobs.extend([
        {
            "kind": "monai",
            "stage": "stage3",
            "family": "vnet_v2",
            "run_name": "s3_vnet_v2loss_seed42",
            "arch": "vnet",
            "init_filters": 16,
            "lr": lrs["vnet"],
            "seed": 42,
            "epochs": ep,
            "gate_mode": "conv_bias",
            "enable_ds": False,
            "loss_name": "rasnet_v2",
        },
        {
            "kind": "monai",
            "stage": "stage3",
            "family": "unet3d_v2",
            "run_name": "s3_unet3d_v2loss_seed42",
            "arch": "unet3d",
            "init_filters": 32,
            "lr": lrs["unet3d"],
            "seed": 42,
            "epochs": ep,
            "gate_mode": "conv_bias",
            "enable_ds": False,
            "loss_name": "rasnet_v2",
        },
    ])
    for s in seeds:
        jobs.append({
            "kind": "nnunet",
            "stage": "stage3",
            "family": "nnunet_v2",
            "run_name": f"nnunet_v2_seed{s}",
            "arch": "nnunet_v2",
            "init_filters": 32,
            "seed": s,
            "epochs": ep,
        })
    return jobs


def build_stage3_eval_manifest(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build Stage 3 evaluation job specs without requiring locked_lrs.json
    (uses dummy LRs; Stage 4/5 only read run_name, family, kind, arch, init_filters, gate_mode, enable_ds, seed).
    Honors Stage 2B pilot_lock_decision.json gate_mode/w_cldice if present.
    """
    dummy = {k: 3e-4 for k in REQUIRED_LR_KEYS}
    pilot_lock = None
    lock_p = Path(cfg["study"]["root_dir"]) / "results" / "pilots_val94" / "pilot_lock_decision.json"
    if lock_p.is_file() and lock_p.stat().st_size > 0:
        try:
            pilot_lock = json.loads(lock_p.read_text(encoding="utf-8"))
        except Exception:
            pilot_lock = None
    return build_stage3_jobs(cfg, dummy, pilot_lock=pilot_lock)


def nnunet_model_dir(cfg: dict[str, Any], seed: int) -> Path:
    root_dir = Path(cfg["study"]["root_dir"])
    nn_cfg = cfg["nnunet_v2"]
    return (
        root_dir
        / "nnUNet_work"
        / "nnUNet_results"
        / f"seed_{int(seed)}"
        / str(nn_cfg["dataset_name"])
        / f"{nn_cfg['trainer_base']}__nnUNetPlans__{nn_cfg['configuration']}"
    )


def resolve_run_best_checkpoint(cfg: dict[str, Any], job: dict[str, Any]) -> tuple[Path, str | None]:
    """Return (path_or_model_folder, nnunet_ckpt_name_or_None) for final best checkpoint."""
    root_dir = Path(cfg["study"]["root_dir"])
    if job["kind"] == "monai":
        return root_dir / "checkpoints" / job["run_name"] / "best_model.pth", None
    return nnunet_model_dir(cfg, int(job["seed"])), "checkpoint_best.pth"


def resolve_run_epoch_checkpoint(cfg: dict[str, Any], job: dict[str, Any], epoch: int) -> tuple[Path, str | None]:
    """Return (path_or_model_folder, nnunet_ckpt_name_or_None) for epoch snapshot {50,100,150,200}."""
    root_dir = Path(cfg["study"]["root_dir"])
    if job["kind"] == "monai":
        return root_dir / "checkpoints" / job["run_name"] / f"epoch_{int(epoch)}.pth", None
    return nnunet_model_dir(cfg, int(job["seed"])), f"checkpoint_epoch_{int(epoch)}.pth"
