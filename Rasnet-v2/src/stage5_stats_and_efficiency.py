"""
Stage 5: Multi-Seed Statistical Analysis (Family 1 K=4 & Family 2 K=3 + Non-Inferiority Δ_NI=-0.0050),
Dice-vs-Params Pareto Figure, RTX 4080 SUPER Latency/GFLOPs/VRAM Profiler, and MANIFEST_V2_LOCKED.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_fair_v2
from rasnet_v2_model import build_monai_model, count_parameters
from study_registry import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    build_stage3_eval_manifest,
    df_to_markdown,
    nnunet_model_dir,
    resolve_run_best_checkpoint,
    sha256_file,
)


METRIC_DIRECTIONS = {
    "dice": "higher",
    "iou": "higher",
    "precision": "higher",
    "recall": "higher",
    "hd95": "lower",
    "asd": "lower",
    "cldice": "higher",
    "centerline_recall": "higher",
}


def _load_family_tables(
    test_dir: Path,
    jobs: list[dict[str, Any]],
    arms: tuple[str, ...] = ("A", "B", "C_top2", "C_seeded"),
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Load per-case Test CSVs grouped by (family, arm) -> {
      "per_seed": {seed: df},
      "seed_averaged": df_N150,
      "seed_pooled": df_N450_with_composite_case_id,
    }
    """
    grouped: dict[tuple[str, str], dict[int, pd.DataFrame]] = {}
    for job in jobs:
        rname = job["run_name"]
        fam = job["family"]
        seed = int(job["seed"])
        for arm in arms:
            p = test_dir / f"per_case_{rname}_arm_{arm}.csv"
            if p.is_file() and p.stat().st_size > 0:
                df = pd.read_csv(p)
                if not df.empty:
                    grouped.setdefault((fam, arm), {})[seed] = df

    out: dict[tuple[str, str], dict[str, Any]] = {}
    metric_cols = list(METRIC_DIRECTIONS.keys())
    for (fam, arm), seed_map in grouped.items():
        dfs_pooled = []
        for s, df_s in sorted(seed_map.items()):
            d_tmp = df_s.copy()
            d_tmp["orig_case_id"] = d_tmp["case_id"].astype(int)
            d_tmp["case_id"] = int(s) * 100000 + d_tmp["orig_case_id"]
            dfs_pooled.append(d_tmp)
        df_pooled = pd.concat(dfs_pooled, ignore_index=True)

        df_cat_avg = pd.concat(list(seed_map.values()), ignore_index=True)
        present_metrics = [c for c in metric_cols if c in df_cat_avg.columns]
        df_avg = df_cat_avg.groupby("case_id", as_index=False)[present_metrics].mean()

        out[(fam, arm)] = {
            "per_seed": seed_map,
            "seed_averaged": df_avg,
            "seed_pooled": df_pooled,
        }
    return out


def _run_family1(
    fam_tables: dict[tuple[str, str], dict[str, Any]],
    arms: tuple[str, ...],
) -> pd.DataFrame:
    contrasts = [
        ("rasnet_v2_f16", "segresnet_f16_v2loss", "F16_RASNetV2_vs_SegResNetV2"),
        ("rasnet_v2_f32", "segresnet_f32_v2loss", "F32_RASNetV2_vs_SegResNetV2"),
    ]
    ablation_contrasts_seed42 = [
        ("segresnet_f16_v2loss", "segresnet_f16_baseloss", "Abl_Loss_V2Loss_vs_BaseLoss"),
        ("rasnet_fixed_ag_nods_f16", "segresnet_f16_v2loss", "Abl_AG_FixedAG_vs_SegResNetV2"),
        ("rasnet_v2_f16", "rasnet_orig_ag_f16", "Abl_Gate_ConvBias_vs_OrigGN"),
        ("rasnet_v2_f16", "rasnet_fixed_ag_nods_f16", "Abl_DS_Full_vs_NoDS"),
    ]
    k4_metrics = ("dice", "cldice", "precision", "hd95")
    rows: list[dict[str, Any]] = []

    for arm in arms:
        for fam_a, fam_b, label in contrasts:
            if (fam_a, arm) not in fam_tables or (fam_b, arm) not in fam_tables:
                continue
            for agg in ("seed_averaged", "seed_pooled"):
                df_a = fam_tables[(fam_a, arm)][agg]
                df_b = fam_tables[(fam_b, arm)][agg]
                if set(df_a["case_id"]) != set(df_b["case_id"]):
                    raise RuntimeError(
                        f"Family 1 case_id/seed mismatch for {label} ({arm}, {agg}): "
                        f"len(a)={len(df_a)} vs len(b)={len(df_b)}"
                    )
                fam_res = []
                for m in k4_metrics:
                    st = eval_fair_v2.paired_with_noninferiority(
                        df_a, df_b, metric=m, allow_nan_filtering=True, direction=METRIC_DIRECTIONS[m]
                    )
                    st.update({"family_id": "Family1_K4", "contrast": label, "arm": arm, "aggregation": agg, "metric": m, "direction": METRIC_DIRECTIONS[m]})
                    fam_res.append(st)
                holm_ps = eval_fair_v2.holm_adjust([r["p_wilcoxon"] for r in fam_res])
                for r, hp in zip(fam_res, holm_ps):
                    r["p_holm"] = float(hp)
                    r["significant_holm_005"] = bool(hp < 0.05)
                    rows.append(r)

        # Single-seed (seed=42) causal ablation breakdown
        for fam_a, fam_b, label in ablation_contrasts_seed42:
            if (fam_a, arm) not in fam_tables or (fam_b, arm) not in fam_tables:
                continue
            s_a = fam_tables[(fam_a, arm)]["per_seed"].get(42)
            s_b = fam_tables[(fam_b, arm)]["per_seed"].get(42)
            if s_a is None or s_b is None:
                continue
            fam_res = []
            for m in k4_metrics:
                st = eval_fair_v2.paired_with_noninferiority(
                    s_a, s_b, metric=m, allow_nan_filtering=True, direction=METRIC_DIRECTIONS[m]
                )
                st.update({"family_id": "Ablation_Seed42_K4", "contrast": label, "arm": arm, "aggregation": "seed_42", "metric": m, "direction": METRIC_DIRECTIONS[m]})
                fam_res.append(st)
            holm_ps = eval_fair_v2.holm_adjust([r["p_wilcoxon"] for r in fam_res])
            for r, hp in zip(fam_res, holm_ps):
                r["p_holm"] = float(hp)
                r["significant_holm_005"] = bool(hp < 0.05)
                rows.append(r)

    return pd.DataFrame(rows)


def _run_family2(
    fam_tables: dict[tuple[str, str], dict[str, Any]],
    arms: tuple[str, ...],
    ni_margin: float = -0.0050,
) -> pd.DataFrame:
    k3_metrics = ("precision", "hd95", "cldice")
    rows: list[dict[str, Any]] = []

    for arm in arms:
        if ("nnunet_v2", arm) not in fam_tables:
            continue
        for fam_a in ("rasnet_v2_f16", "rasnet_v2_f32", "segresnet_f16_v2loss", "segresnet_f32_v2loss"):
            if (fam_a, arm) not in fam_tables:
                continue
            for agg in ("seed_averaged", "seed_pooled"):
                df_a = fam_tables[(fam_a, arm)][agg]
                df_b = fam_tables[("nnunet_v2", arm)][agg]
                if set(df_a["case_id"]) != set(df_b["case_id"]):
                    raise RuntimeError(
                        f"Family 2 case_id/seed mismatch for {fam_a}_vs_nnunet_v2 ({arm}, {agg}): "
                        f"len(a)={len(df_a)} vs len(b)={len(df_b)}"
                    )

                sup_res = []
                for m in k3_metrics:
                    st = eval_fair_v2.paired_with_noninferiority(
                        df_a, df_b, metric=m, allow_nan_filtering=True, direction=METRIC_DIRECTIONS[m]
                    )
                    st.update({
                        "family_id": "Family2_Superiority_K3",
                        "contrast": f"{fam_a}_vs_nnunet_v2",
                        "arm": arm,
                        "aggregation": agg,
                        "metric": m,
                        "direction": METRIC_DIRECTIONS[m],
                    })
                    sup_res.append(st)
                holm_ps = eval_fair_v2.holm_adjust([r["p_wilcoxon"] for r in sup_res])
                for r, hp in zip(sup_res, holm_ps):
                    r["p_holm"] = float(hp)
                    r["significant_holm_005"] = bool(hp < 0.05)
                    rows.append(r)

                ni_st = eval_fair_v2.paired_with_noninferiority(
                    df_a, df_b, metric="dice", ni_margin=ni_margin, allow_nan_filtering=True, direction="higher"
                )
                ni_st.update({
                    "family_id": "Family2_NonInferiority_Dice",
                    "contrast": f"{fam_a}_vs_nnunet_v2",
                    "arm": arm,
                    "aggregation": agg,
                    "metric": "dice",
                    "direction": "higher",
                    "p_holm": float(ni_st["p_wilcoxon"]),
                    "significant_holm_005": bool(ni_st["non_inferior"]),
                })
                rows.append(ni_st)

    return pd.DataFrame(rows)


def _profile_model_efficiency(
    cfg: dict[str, Any],
    out_dir: Path,
    skip_gpu_profiler: bool = False,
    n_warmup: int = 10,
    n_iters: int = 30,
) -> pd.DataFrame:
    """
    Measure exact parameter count, GFLOPs per patch, checkpoint MB, FP32 & AMP FP16
    RTX 4080 SUPER patch latency, and Peak Patch Inference VRAM.
    """
    root_dir = Path(cfg["study"]["root_dir"])
    has_cuda = torch.cuda.is_available() and not skip_gpu_profiler
    dev = torch.device("cuda" if has_cuda else "cpu")

    specs = [
        ("SegResNet-v2 (F=16)", "monai", "segresnet", 16, False, "s3_segresnet_f16_v2loss_seed42", (1, 1, 160, 160, 96)),
        ("RASNet-v2 Lite (F=16)", "monai", "rasnet_v2", 16, True, "s3_rasnet_v2_f16_full_seed42", (1, 1, 160, 160, 96)),
        ("Patch 3D U-Net-v2", "monai", "unet3d", 32, False, "s3_unet3d_v2loss_seed42", (1, 1, 160, 160, 96)),
        ("SegResNet-v2 (F=32)", "monai", "segresnet", 32, False, "s3_segresnet_f32_v2loss_seed42", (1, 1, 160, 160, 96)),
        ("RASNet-v2 Scaled (F=32)", "monai", "rasnet_v2", 32, True, "s3_rasnet_v2_f32_full_seed42", (1, 1, 160, 160, 96)),
        ("V-Net-v2", "monai", "vnet", 16, False, "s3_vnet_v2loss_seed42", (1, 1, 160, 160, 96)),
        ("nnU-Net V2 (Ref)", "nnunet", "nnunet_v2", 32, True, "nnunet_v2_seed42", (1, 1, 96, 160, 160)),
    ]

    rows: list[dict[str, Any]] = []
    for label, kind, arch, f_val, enable_ds, sample_run, inp_shape in specs:
        net = None
        ckpt_mb = float("nan")
        try:
            if kind == "monai":
                net = build_monai_model(arch=arch, init_filters=f_val, gate_mode="conv_bias", enable_ds=enable_ds).eval()
                ckpt_p = root_dir / "checkpoints" / sample_run / "best_model.pth"
                if ckpt_p.is_file():
                    ckpt_mb = float(ckpt_p.stat().st_size / (1024.0 ** 2))
            else:
                m_dir = nnunet_model_dir(cfg, 42)
                ckpt_p = m_dir / f"fold_{cfg['nnunet_v2']['fold']}" / "checkpoint_best.pth"
                if ckpt_p.is_file():
                    ckpt_mb = float(ckpt_p.stat().st_size / (1024.0 ** 2))
                    pred = eval_fair_v2.build_nnunet_predictor(m_dir, fold=int(cfg["nnunet_v2"]["fold"]), checkpoint_name="checkpoint_best.pth", device="cpu")
                    net = pred.network.eval()
                    # Disable deep supervision for inference profiling parity
                    if hasattr(net, "decoder") and hasattr(net.decoder, "deep_supervision"):
                        net.decoder.deep_supervision = False
                    del pred
                else:
                    # Fallback: construct PlainConvUNet directly from nnUNetPlans.json if checkpoint absent
                    plans_p = root_dir / "nnUNet_work" / "nnUNet_preprocessed" / cfg["nnunet_v2"]["dataset_name"] / "nnUNetPlans.json"
                    if plans_p.is_file():
                        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
                        plans = json.loads(plans_p.read_text(encoding="utf-8"))
                        c_cfg = plans["configurations"][cfg["nnunet_v2"]["configuration"]]
                        arch_cfg = c_cfg["architecture"]
                        net = get_network_from_plans(
                            arch_cfg["network_class_name"],
                            arch_cfg["arch_kwargs"],
                            arch_cfg["_kw_requires_import"],
                            1,
                            2,
                            deep_supervision=False,
                        ).eval()

            if net is None:
                rows.append({"model": label, "arch": arch, "init_filters": f_val, "params": np.nan, "params_m": np.nan, "weights_mb": np.nan, "ckpt_mb": np.nan, "gflops_per_patch": np.nan, "lat_fp32_ms_mean": np.nan, "lat_fp32_ms_std": np.nan, "lat_amp_ms_mean": np.nan, "lat_amp_ms_std": np.nan, "peak_infer_vram_mb": np.nan})
                continue

            n_params = count_parameters(net)
            weights_mb = float(n_params * 4 / (1024.0 ** 2))
            if np.isnan(ckpt_mb):
                ckpt_mb = weights_mb

            # Measure exact FLOPs on CPU with eval mode (no DS)
            x_cpu = torch.randn(*inp_shape, dtype=torch.float32)
            with torch.no_grad(), FlopCounterMode(display=False) as fc:
                _ = net(x_cpu)
            gflops = float(fc.get_total_flops() / 1e9)
            del x_cpu

            lat_fp32_mean, lat_fp32_std = float("nan"), float("nan")
            lat_amp_mean, lat_amp_std = float("nan"), float("nan")
            peak_vram_mb = float("nan")

            if has_cuda:
                net = net.to(dev)
                x_gpu = torch.randn(*inp_shape, device=dev, dtype=torch.float32)

                with torch.no_grad():
                    for _ in range(n_warmup):
                        _ = net(x_gpu)
                    torch.cuda.synchronize(dev)
                    times_fp32 = []
                    for _ in range(n_iters):
                        t0 = time.perf_counter()
                        _ = net(x_gpu)
                        torch.cuda.synchronize(dev)
                        times_fp32.append((time.perf_counter() - t0) * 1000.0)
                    lat_fp32_mean = float(np.mean(times_fp32))
                    lat_fp32_std = float(np.std(times_fp32, ddof=1))

                    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    torch.cuda.reset_peak_memory_stats(dev)
                    for _ in range(n_warmup):
                        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
                            _ = net(x_gpu)
                    torch.cuda.synchronize(dev)
                    times_amp = []
                    for _ in range(n_iters):
                        t0 = time.perf_counter()
                        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
                            _ = net(x_gpu)
                        torch.cuda.synchronize(dev)
                        times_amp.append((time.perf_counter() - t0) * 1000.0)
                    lat_amp_mean = float(np.mean(times_amp))
                    lat_amp_std = float(np.std(times_amp, ddof=1))
                    peak_vram_mb = float(torch.cuda.max_memory_allocated(dev) / (1024.0 ** 2))
                del x_gpu

            rows.append({
                "model": label,
                "arch": arch,
                "init_filters": int(f_val),
                "patch_shape": "x".join(str(v) for v in inp_shape[2:]),
                "params": int(n_params),
                "params_m": round(n_params / 1e6, 3),
                "weights_mb": round(weights_mb, 2),
                "ckpt_mb": round(ckpt_mb, 2),
                "gflops_per_patch": round(gflops, 2),
                "lat_fp32_ms_mean": round(lat_fp32_mean, 2) if not np.isnan(lat_fp32_mean) else np.nan,
                "lat_fp32_ms_std": round(lat_fp32_std, 2) if not np.isnan(lat_fp32_std) else np.nan,
                "lat_amp_ms_mean": round(lat_amp_mean, 2) if not np.isnan(lat_amp_mean) else np.nan,
                "lat_amp_ms_std": round(lat_amp_std, 2) if not np.isnan(lat_amp_std) else np.nan,
                "peak_infer_vram_mb": round(peak_vram_mb, 1) if not np.isnan(peak_vram_mb) else np.nan,
            })
        finally:
            del net
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_eff = pd.DataFrame(rows)
    atomic_write_csv(df_eff, out_dir / "rtx4080_efficiency_table.csv")
    atomic_write_text(df_to_markdown(df_eff, float_fmt=".2f"), out_dir / "rtx4080_efficiency_table.md")
    return df_eff


def _plot_pareto(
    df_fam: pd.DataFrame,
    df_eff: pd.DataFrame,
    out_dir: Path,
) -> Path | None:
    if df_fam.empty:
        return None
    fam_to_label = {
        "segresnet_f16_v2loss": ("SegResNet-v2 F=16", 4.701),
        "rasnet_v2_f16": ("RASNet-v2 F=16", 4.707),
        "unet3d_v2": ("Patch 3D U-Net-v2", 12.86),
        "segresnet_f32_v2loss": ("SegResNet-v2 F=32", 18.80),
        "rasnet_v2_f32": ("RASNet-v2 F=32", 18.80),
        "nnunet_v2": ("nnU-Net V2", 30.79),
        "vnet_v2": ("V-Net-v2", 45.60),
    }
    eff_lookup = {
        "segresnet_f16_v2loss": "SegResNet-v2 (F=16)",
        "rasnet_v2_f16": "RASNet-v2 Lite (F=16)",
        "unet3d_v2": "Patch 3D U-Net-v2",
        "segresnet_f32_v2loss": "SegResNet-v2 (F=32)",
        "rasnet_v2_f32": "RASNet-v2 Scaled (F=32)",
        "nnunet_v2": "nnU-Net V2 (Ref)",
        "vnet_v2": "V-Net-v2",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, (arm, title) in zip(axes, (("A", "Arm A (Raw 1-Pass, τ=0.50)"), ("C_seeded", "Arm C-Seeded (Val-Locked)"))):
        sub = df_fam[(df_fam["arm"] == arm) & (df_fam["aggregation"] == "seed_averaged")]
        if sub.empty and arm == "C_seeded":
            sub = df_fam[(df_fam["arm"] == "C_top2") & (df_fam["aggregation"] == "seed_averaged")]
            title = "Arm C-Top2 (Val-Locked)"
        for _, row in sub.iterrows():
            fam = str(row["family"])
            if fam not in fam_to_label:
                continue
            disp, default_pm = fam_to_label[fam]
            pm = default_pm
            if not df_eff.empty and eff_lookup.get(fam) in df_eff["model"].values:
                val = df_eff.loc[df_eff["model"] == eff_lookup[fam], "params_m"].iloc[0]
                if pd.notna(val):
                    pm = float(val)
            dice_m = float(row["dice_mean"])
            seed_sd = float(row.get("dice_across_seed_std", 0.0))
            is_ras = "rasnet_v2" in fam
            marker = "*" if is_ras else ("s" if fam == "nnunet_v2" else "o")
            msize = 12 if is_ras else 8
            color = "#d62728" if is_ras else ("#1f77b4" if fam == "nnunet_v2" else "#555555")
            ax.errorbar(pm, dice_m, yerr=seed_sd if seed_sd > 0 else None, fmt=marker, markersize=msize, color=color, capsize=4, label=disp)
            xy_off = (5, 6) if is_ras else ((5, -10) if "segresnet" in fam else (5, 3))
            ax.annotate(disp, (pm, dice_m), textcoords="offset points", xytext=xy_off, fontsize=7.5)
        ax.set_title(title)
        ax.set_xlabel("Parameters (Millions)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test N=150 Mean Dice")
    fig.suptitle("Tier 3 Parameter-Performance Pareto: RASNet-v2 vs Baselines", fontsize=12)
    fig.tight_layout()
    out_png = out_dir / "fig_dice_vs_params_pareto.png"
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_dice_vs_params_pareto.{ext}", dpi=180)
    plt.close(fig)
    return out_png


def run_stage5_analysis(
    config_path: str | Path,
    lead_width: int | None = None,
    skip_gpu_profiler: bool = False,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    test_dir = root_dir / "results" / "test_retrospective_150"
    val94_dir = root_dir / "results" / "val94_final"
    pilot_dir = root_dir / "results" / "pilots_val94"
    out_dir = root_dir / "results" / "stats_and_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_stage3_eval_manifest(cfg)
    fam_tables = _load_family_tables(test_dir, jobs)
    arms_present = tuple(a for a in ("A", "B", "C_top2", "C_seeded") if any(k[1] == a for k in fam_tables))

    df_f1 = _run_family1(fam_tables, arms_present)
    df_f2 = _run_family2(fam_tables, arms_present, ni_margin=-0.0050)
    atomic_write_csv(df_f1, out_dir / "family1_causal_stats.csv")
    atomic_write_csv(df_f2, out_dir / "family2_vs_nnunet_stats.csv")

    df_eff = _profile_model_efficiency(cfg, out_dir, skip_gpu_profiler=skip_gpu_profiler)
    fam_sum_p = test_dir / "summary_by_family_multiseed.csv"
    df_fam_sum = pd.read_csv(fam_sum_p) if (fam_sum_p.is_file() and fam_sum_p.stat().st_size > 0) else pd.DataFrame()
    _plot_pareto(df_fam_sum, df_eff, out_dir)

    # Determine lead width F* from Stage 2B pilot lock decision (fallback 16)
    pilot_lock_p = pilot_dir / "pilot_lock_decision.json"
    if lead_width is None and pilot_lock_p.is_file():
        plock = json.loads(pilot_lock_p.read_text(encoding="utf-8"))
        lead_width = int(plock["locked_design"]["init_filters"])
    lead_f = int(lead_width if lead_width is not None else 16)

    thresh_p = val94_dir / "thresholds_v2.json"
    if not thresh_p.is_file():
        raise FileNotFoundError(f"Missing frozen thresholds_v2.json: {thresh_p}")
    thresh_sha = sha256_file(thresh_p)

    # Freeze canonical Phase I models (RASNet-v2 locked F*, matching SegResNet-v2 F*, nnU-Net V2)
    locked_families = {
        "rasnet_v2": f"rasnet_v2_f{lead_f}",
        "segresnet_v2": f"segresnet_f{lead_f}_v2loss",
        "nnunet_v2": "nnunet_v2",
    }
    locked_ckpts: dict[str, Any] = {}
    for role, fam in locked_families.items():
        fam_jobs = [j for j in jobs if j["family"] == fam]
        seed_entries = {}
        for j in fam_jobs:
            ckpt_or_dir, nn_name = resolve_run_best_checkpoint(cfg, j)
            ckpt_file = ckpt_or_dir if j["kind"] == "monai" else (ckpt_or_dir / f"fold_{cfg['nnunet_v2']['fold']}" / str(nn_name))
            if not ckpt_file.is_file() and not skip_gpu_profiler:
                raise FileNotFoundError(f"Missing locked Phase I candidate checkpoint for {role} ({j['run_name']}): {ckpt_file}")
            seed_entries[str(j["seed"])] = {
                "run_name": j["run_name"],
                "kind": j["kind"],
                "arch": j["arch"],
                "init_filters": int(j["init_filters"]),
                "gate_mode": j.get("gate_mode", "conv_bias"),
                "enable_ds": bool(j.get("enable_ds", True)),
                "checkpoint_or_model_dir": str(ckpt_or_dir),
                "nnunet_checkpoint_name": nn_name,
                "checkpoint_sha256": sha256_file(ckpt_file) if ckpt_file.is_file() else "missing",
            }
        canonical_seed = "42" if "42" in seed_entries else next(iter(seed_entries))
        locked_ckpts[role] = {
            "family": fam,
            "canonical_seed": int(canonical_seed),
            "canonical": seed_entries[canonical_seed],
            "all_seeds": seed_entries,
        }

    prereg_p = root_dir / "PREREGISTRATION_V2.md"
    val20_p = Path(cfg["study"]["val20_ids_file"])
    splits_p = Path(cfg["study"]["splits_file"])

    manifest_locked = {
        "version": "2.0.0",
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lead_init_filters": lead_f,
        "thresholds_v2_path": str(thresh_p),
        "thresholds_v2_sha256": thresh_sha,
        "preregistration_sha256": sha256_file(prereg_p) if prereg_p.is_file() else "missing",
        "val20_ids_sha256": sha256_file(val20_p) if val20_p.is_file() else "missing",
        "splits_sha256": sha256_file(splits_p) if splits_p.is_file() else "missing",
        "locked_models": locked_ckpts,
        "stats_artifacts": {
            "family1_causal_csv": str(out_dir / "family1_causal_stats.csv"),
            "family2_vs_nnunet_csv": str(out_dir / "family2_vs_nnunet_stats.csv"),
            "efficiency_table_csv": str(out_dir / "rtx4080_efficiency_table.csv"),
            "pareto_png": str(out_dir / "fig_dice_vs_params_pareto.png"),
        },
    }
    atomic_write_json(manifest_locked, out_dir / "MANIFEST_V2_LOCKED.json")

    md = [
        "# Stage 5 Statistical Analysis, Efficiency & Manifest Lock",
        "",
        f"- **Locked Lead Width**: `F* = {lead_f}`",
        f"- **Frozen Thresholds (`thresholds_v2.json`) SHA-256**: `{thresh_sha}`",
        "",
        "## Family 1: Tier 1 Causal Architecture Comparison (`K = 4`, Holm-Bonferroni)",
        "",
        df_to_markdown(df_f1) if not df_f1.empty else "_No Family 1 rows._",
        "",
        "## Family 2: Tier 2 Full-System vs `nnU-Net V2` (`K = 3` + Non-Inferiority `Δ_NI = -0.0050`)",
        "",
        df_to_markdown(df_f2) if not df_f2.empty else "_No Family 2 rows._",
        "",
        "## RTX 4080 SUPER Efficiency Table",
        "",
        df_to_markdown(df_eff, float_fmt=".2f"),
        "",
    ]
    atomic_write_text("\n".join(md), out_dir / "STAGE5_SUMMARY_REPORT.md")
    print(f"[Stage 5] Locked MANIFEST_V2_LOCKED.json (lead F*={lead_f})")
    return manifest_locked


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 Statistical Analysis, Pareto & Efficiency Profiler")
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument("--lead-width", type=int, default=None)
    parser.add_argument("--skip-gpu-profiler", action="store_true")
    args = parser.parse_args()
    run_stage5_analysis(args.config, lead_width=args.lead_width, skip_gpu_profiler=args.skip_gpu_profiler)


if __name__ == "__main__":
    main()
