"""
Unified Confounder-Controlled MONAI Trainer (`train_fair_v2.py`).
Enforces identical (0.3496, 0.3496, 0.50) spacing, (160, 160, 96) patch size,
deterministic cyclic 250 steps/epoch x effective batch_size 2 (pos=1:neg=2),
per-step CosineAnnealingLR(T_max=total_steps), val_interval=20 on val20_curves only,
and two-phase atomic checkpointing (latest_state.pt, mid_epoch_state.pt, epoch_*, best_model.pth).
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augmentations import apply_shared_patch_augmentations, sample_pos_neg_patch
from cache_builder import case_cache_paths
from dataset_pairs import load_validated_splits
from rasnet_v2_loss import BaseDiceFocalLoss, RASNetV2Loss
from rasnet_v2_model import build_monai_model, count_parameters


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_csv_write(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class DeterministicCyclicPatchDataset(Dataset):
    """
    Maps global_sample_idx -> deterministic (case_id, SeedSequence([seed, global_sample_idx])).
     guarantees identical patch crops and augmentations across num_workers>=0 and exact resume.
    """

    def __init__(
        self,
        cache_dir: Path,
        train_ids: list[int],
        total_samples: int,
        seed: int,
        patch_size_xyz: tuple[int, int, int] = (160, 160, 96),
        fg_prob: float = 1.0 / 3.0,
        start_sample_idx: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        self.train_ids = sorted(int(c) for c in train_ids)
        self.n_cases = len(self.train_ids)
        self.total_samples = int(total_samples)
        self.seed = int(seed)
        self.patch_size_xyz = tuple(int(v) for v in patch_size_xyz)
        self.fg_prob = float(fg_prob)
        self.start_sample_idx = int(start_sample_idx)

        # Pre-generate deterministic cyclic case permutation stream covering all samples
        n_cycles = (self.total_samples + self.n_cases - 1) // self.n_cases
        stream = []
        ids_arr = np.asarray(self.train_ids, dtype=np.int32)
        for c_idx in range(n_cycles):
            perm_rng = np.random.default_rng(self.seed * 100003 + c_idx)
            stream.append(perm_rng.permutation(ids_arr))
        self.case_stream = np.concatenate(stream)[: self.total_samples]

    def __len__(self) -> int:
        return max(0, self.total_samples - self.start_sample_idx)

    def __getitem__(self, rel_idx: int):
        global_idx = self.start_sample_idx + int(rel_idx)
        cid = int(self.case_stream[global_idx])
        ps = case_cache_paths(self.cache_dir, cid)

        # Memory-mapped headerless/array reads keep OS cache fast and worker RAM tiny
        img = np.load(ps["img"], mmap_mode="r")
        lbl = np.load(ps["lbl"], mmap_mode="r")
        fg = np.load(ps["fg"], mmap_mode="r")
        bg = np.load(ps["bg"], mmap_mode="r")

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, global_idx]))
        p_img, p_lbl = sample_pos_neg_patch(
            img,
            lbl,
            fg,
            bg,
            rng,
            patch_size_xyz=self.patch_size_xyz,
            fg_prob=self.fg_prob,
        )
        p_img, p_lbl = apply_shared_patch_augmentations(p_img, p_lbl, rng)
        return torch.from_numpy(p_img), torch.from_numpy(p_lbl.astype(np.int64, copy=False))


def resolve_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def evaluate_cached_val20(
    model: torch.nn.Module,
    val_ids: list[int],
    cache_dir: Path,
    roi_size_xyz: tuple[int, int, int],
    sw_overlap: float,
    sw_mode: str,
    sw_batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run in-loop 0.50 Gaussian sliding-window validation strictly on frozen val20_curves.
    Keeps full-volume buffers & soft-Dice aggregation on CPU (`sw_device=device, device='cpu'`)
    so validation Peak VRAM never exceeds ~3.5 GB (eliminating Windows WDDM paging).
    """
    model.eval()
    dices = []
    losses = []
    amp_ok = device.type == "cuda"
    amp_dtype = resolve_amp_dtype(device)
    cpu_dev = torch.device("cpu")
    with torch.no_grad():
        for cid in val_ids:
            ps = case_cache_paths(cache_dir, cid)
            img = torch.from_numpy(np.load(ps["img"]).astype(np.float32, copy=False)).unsqueeze(0)
            gt_mask = torch.from_numpy(np.load(ps["lbl"]) > 0).unsqueeze(0)  # (1, 1, X, Y, Z) bool on CPU
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_ok):
                logits = sliding_window_inference(
                    img,
                    roi_size=roi_size_xyz,
                    sw_batch_size=int(sw_batch_size),
                    predictor=model,
                    overlap=float(sw_overlap),
                    mode=str(sw_mode),
                    sw_device=device,
                    device=cpu_dev,
                )
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
            probs_fg = torch.softmax(logits.float(), dim=1)[:, 1:2]
            pred_fg = probs_fg >= 0.5
            inter = int((pred_fg & gt_mask).sum().item())
            denom = int(pred_fg.sum().item() + gt_mask.sum().item())
            case_dice = (2.0 * inter / denom) if denom > 0 else 1.0

            gt_f = gt_mask.float()
            soft_inter = float((probs_fg * gt_f).sum().item())
            soft_denom = float(probs_fg.sum().item() + gt_f.sum().item())
            soft_dice_loss = 1.0 - (2.0 * soft_inter + 1e-5) / (soft_denom + 1e-5)

            dices.append(case_dice)
            losses.append(soft_dice_loss)
    return float(np.mean(dices)), float(np.mean(losses))


def build_criterion(loss_name: str, cfg: dict[str, Any], enable_ds: bool, w_cldice_override: float | None = None):
    if loss_name == "base_dice_focal":
        lc = cfg["loss"]["base_dice_focal"]
        return BaseDiceFocalLoss(gamma=float(lc["focal_gamma"]))
    lc = cfg["loss"]["rasnet_v2"]
    w_cl = float(lc["w_cldice"]) if w_cldice_override is None else float(w_cldice_override)
    w_d = float(lc["w_dice"])
    w_f = float(lc["w_focal"])
    # If w_cldice overridden to 0 during pilot ablation, re-normalize Dice/Focal proportionally
    if w_cldice_override is not None and float(w_cldice_override) == 0.0:
        s = max(w_d + w_f, 1e-8)
        w_d, w_f = w_d / s, w_f / s
    ds_w = tuple(float(w) for w in lc["ds_weights"]) if enable_ds else (0.0, 0.0)
    return RASNetV2Loss(
        w_dice=w_d,
        w_focal=w_f,
        w_cldice=w_cl,
        gamma=float(lc["focal_gamma"]),
        cl_iter=int(lc["cl_iter"]),
        ds_weights=ds_w,
        ds_include_cldice=bool(lc["ds_include_cldice"]),
    )


def run_training(
    config_path: str | Path,
    run_name: str,
    arch: str,
    init_filters: int,
    lr: float,
    seed: int,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    gate_mode: str = "conv_bias",
    enable_ds: bool = True,
    loss_name: str = "rasnet_v2",
    w_cldice_override: float | None = None,
    micro_batch_size: int | None = None,
    grad_accum: int | None = None,
    val_interval: int | None = None,
    num_workers: int = 4,
    mid_epoch_ckpt_every_steps: int | None = None,
    interrupt_after_epoch: int | None = None,
    interrupt_after_step: int | None = None,
    train_ids_override: list[int] | None = None,
    val_ids_override: list[int] | None = None,
    force_device: str | None = None,
    patch_size_override: tuple[int, int, int] | None = None,
    eval_at_end: bool = True,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    cache_dir = root_dir / "cache_fp16"
    ckpt_dir = root_dir / "checkpoints" / run_name
    log_dir = root_dir / "logs" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    done_flag = ckpt_dir / "DONE.flag"
    if done_flag.is_file():
        print(f"[SKIP] {run_name} already finished ({done_flag})")
        return json.loads(done_flag.read_text(encoding="utf-8"))

    if train_ids_override is not None:
        train_ids = list(train_ids_override)
    else:
        splits = load_validated_splits(cfg["study"]["splits_file"])
        train_ids = list(splits["train"])
    if val_ids_override is not None:
        val_ids = list(val_ids_override)
    else:
        val20_payload = json.loads(Path(cfg["study"]["val20_ids_file"]).read_text(encoding="utf-8"))
        val_ids = list(val20_payload["case_ids"])

    tb = cfg["training_budget"]
    n_epochs = int(epochs if epochs is not None else tb["epochs"])
    n_steps_ep = int(steps_per_epoch if steps_per_epoch is not None else tb["steps_per_epoch"])
    eff_bs = int(tb["effective_batch_size"])

    if micro_batch_size is None or grad_accum is None:
        # Automatic policy: F=32 on MONAI uses mb=1 x grad_accum=2 to avoid WDDM paging
        if arch.lower() in ("rasnet_v2", "segresnet") and int(init_filters) >= 32:
            mb, ga = 1, 2
        else:
            mb, ga = 2, 1
    else:
        mb, ga = int(micro_batch_size), int(grad_accum)
    if mb * ga != eff_bs:
        raise ValueError(f"micro_batch_size * grad_accum must equal {eff_bs}, got {mb}*{ga}")

    v_interval = int(val_interval if val_interval is not None else tb["val_interval"])
    mid_every = int(mid_epoch_ckpt_every_steps if mid_epoch_ckpt_every_steps is not None else tb["mid_epoch_ckpt_every_steps"])
    patch_xyz = tuple(int(v) for v in (patch_size_override or cfg["preprocessing"]["patch_size_xyz"]))
    roi_xyz = tuple(int(v) for v in (patch_size_override or cfg["inference_eval"]["roi_size_xyz"]))
    sw_overlap = float(cfg["inference_eval"]["sw_overlap"])
    sw_mode = str(cfg["inference_eval"]["sw_mode"])
    sw_bs = int(cfg["inference_eval"]["sw_batch_size"])

    device = torch.device(force_device) if force_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(device)
    amp_tag = "fp32" if device.type != "cuda" else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")

    cfg_hash = hashlib.sha256(
        json.dumps(
            {
                "run_name": run_name,
                "arch": arch,
                "init_filters": init_filters,
                "lr": lr,
                "seed": seed,
                "epochs": n_epochs,
                "steps_per_epoch": n_steps_ep,
                "gate_mode": gate_mode,
                "enable_ds": enable_ds,
                "loss_name": loss_name,
                "w_cldice_override": w_cldice_override,
                "mb": mb,
                "ga": ga,
                "patch_xyz": list(patch_xyz),
                "amp_dtype": amp_tag,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    set_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    model = build_monai_model(
        arch=arch,
        init_filters=int(init_filters),
        gate_mode=gate_mode,
        enable_ds=enable_ds,
        dropout_prob=0.1,
    ).to(device)
    criterion = build_criterion(loss_name, cfg, enable_ds=(enable_ds and arch.lower() == "rasnet_v2"), w_cldice_override=w_cldice_override).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(tb["weight_decay"]))
    total_opt_steps = n_epochs * n_steps_ep
    scheduler = CosineAnnealingLR(optimizer, T_max=total_opt_steps, eta_min=float(tb["scheduler_eta_min"]))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    latest_path = ckpt_dir / "latest_state.pt"
    mid_path = ckpt_dir / "mid_epoch_state.pt"
    best_path = ckpt_dir / "best_model.pth"
    log_csv = log_dir / "training_log.csv"
    tel_csv = log_dir / "gpu_telemetry.csv"

    start_epoch = 1
    resume_step_in_epoch = 0
    resumed_running_loss = 0.0
    global_opt_step = 0
    best_val_dice = -1.0
    elapsed_gpu_sec = 0.0
    history: list[dict[str, Any]] = []
    tel_rows: list[dict[str, Any]] = []

    # Resume priority: newest valid checkpoint between mid_epoch_state.pt and latest_state.pt
    resume_ckpt: Path | None = None
    if latest_path.is_file() or mid_path.is_file():
        cands = []
        for p in (mid_path, latest_path):
            if p.is_file():
                try:
                    st = torch.load(p, map_location="cpu", weights_only=False)
                    if st.get("config_sha256") != cfg_hash:
                        raise RuntimeError(f"Config mismatch on resume for {p}")
                    cands.append((int(st["global_opt_step"]), p, st))
                except Exception as exc:
                    print(f"[WARN] Ignoring corrupt/incompatible checkpoint {p}: {exc}")
        if cands:
            cands.sort(key=lambda t: t[0], reverse=True)
            _, resume_ckpt, st = cands[0]
            model.load_state_dict(st["model_state_dict"])
            optimizer.load_state_dict(st["optimizer_state_dict"])
            scheduler.load_state_dict(st["scheduler_state_dict"])
            if st.get("scaler_state_dict") is not None:
                scaler.load_state_dict(st["scaler_state_dict"])
            saved_rng_state = st["rng_state"]
            global_opt_step = int(st["global_opt_step"])
            start_epoch = int(st["next_epoch"])
            resume_step_in_epoch = int(st["next_step_in_epoch"])
            resumed_running_loss = float(st.get("running_loss", 0.0)) if resume_step_in_epoch > 0 else 0.0
            best_val_dice = float(st["best_val_dice"])
            elapsed_gpu_sec = float(st.get("elapsed_gpu_sec", 0.0))
            history = list(st.get("history", []))
            tel_rows = list(st.get("tel_rows", []))
            print(
                f"[RESUME] {run_name} from {resume_ckpt.name}: epoch={start_epoch}, "
                f"step_in_ep={resume_step_in_epoch}, global_step={global_opt_step}, best_val_dice={best_val_dice:.4f}"
            )
    else:
        saved_rng_state = None
    if latest_path.is_file() or mid_path.is_file():
        if resume_ckpt is None:
            saved_rng_state = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_samples = total_opt_steps * eff_bs
    snapshots = set(int(e) for e in tb["epoch_snapshots"])
    log_fields = ["epoch", "global_step", "train_loss", "val_loss", "val_dice", "learning_rate", "epoch_time_s", "cumulative_h"]
    tel_fields = ["epoch", "global_step", "peak_vram_mb", "sec_per_step"]

    n_params = count_parameters(model)
    print(
        f"[TRAIN] {run_name} | arch={arch}(F={init_filters}, gate={gate_mode}, ds={enable_ds}) "
        f"| params={n_params:,} | lr={lr:.1e} | mb={mb}x{ga} | steps={n_epochs}x{n_steps_ep}={total_opt_steps} "
        f"| seed={seed} | val20_interval={v_interval} | device={device.type}:{amp_tag}"
    )

    run_start_sample = ((start_epoch - 1) * n_steps_ep + resume_step_in_epoch) * eff_bs
    ds = DeterministicCyclicPatchDataset(
        cache_dir=cache_dir,
        train_ids=train_ids,
        total_samples=total_samples,
        seed=seed,
        patch_size_xyz=patch_xyz,
        fg_prob=float(tb["foreground_ratio"]),
        start_sample_idx=run_start_sample,
    )
    use_workers = int(num_workers) > 0 and start_epoch <= n_epochs and len(ds) > 0
    dl_gen = torch.Generator().manual_seed(int(seed) + 7)
    loader = None
    loader_iter = None
    try:
        loader = DataLoader(
            ds,
            batch_size=mb,
            shuffle=False,
            num_workers=int(num_workers) if use_workers else 0,
            pin_memory=(device.type == "cuda"),
            persistent_workers=use_workers,
            prefetch_factor=2 if use_workers else None,
            generator=dl_gen,
        )
        loader_iter = iter(loader)
        if saved_rng_state is not None:
            _restore_rng_state(saved_rng_state)

        consecutive_nonfinite = 0
        for epoch in range(start_epoch, n_epochs + 1):
            ep_t0 = time.perf_counter()
            model.train()
            running_loss = resumed_running_loss
            resumed_running_loss = 0.0
            steps_done_ep = resume_step_in_epoch
            accum_counter = 0
            optimizer.zero_grad(set_to_none=True)

            micro_batches_needed = (n_steps_ep - resume_step_in_epoch) * ga
            for _ in range(micro_batches_needed):
                micro_img, micro_lbl = next(loader_iter)
                micro_img = micro_img.to(device, non_blocking=True)
                micro_lbl = micro_lbl.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    outputs = model(micro_img)
                loss_unscaled = criterion(outputs, micro_lbl)
                loss_val = float(loss_unscaled.detach().item())
                if not np.isfinite(loss_val):
                    consecutive_nonfinite += 1
                    if consecutive_nonfinite >= 25:
                        raise RuntimeError(
                            f"Non-finite loss detected for {consecutive_nonfinite} consecutive micro-batches "
                            f"in {run_name} (epoch={epoch}, global_step={global_opt_step}, amp={amp_tag})"
                        )
                else:
                    consecutive_nonfinite = 0
                loss_scaled = loss_unscaled / float(ga)
                scaler.scale(loss_scaled).backward()
                running_loss += loss_val / float(ga)
                accum_counter += 1

                if accum_counter < ga:
                    continue

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tb["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

                steps_done_ep += 1
                global_opt_step += 1

                if mid_every > 0 and (steps_done_ep % mid_every == 0) and (steps_done_ep < n_steps_ep):
                    atomic_torch_save(
                        {
                            "config_sha256": cfg_hash,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "rng_state": _capture_rng_state(),
                            "global_opt_step": global_opt_step,
                            "next_epoch": epoch,
                            "next_step_in_epoch": steps_done_ep,
                            "running_loss": float(running_loss),
                            "best_val_dice": best_val_dice,
                            "elapsed_gpu_sec": elapsed_gpu_sec + (time.perf_counter() - ep_t0),
                            "history": history,
                            "tel_rows": tel_rows,
                        },
                        mid_path,
                    )

                if interrupt_after_step is not None and global_opt_step >= interrupt_after_step:
                    print(f"[INTERRUPT-TEST] Intentional exit at global_step={global_opt_step}")
                    return {"status": "interrupted", "global_step": global_opt_step}

            resume_step_in_epoch = 0  # completed full epoch cleanly
            train_loss = running_loss / float(max(steps_done_ep, 1))
            cur_lr = float(optimizer.param_groups[0]["lr"])

            eff_sw_bs = min(sw_bs, 2 if (arch.lower() in ("rasnet_v2", "segresnet") and int(init_filters) >= 32) else 4)
            val_dice = float("nan")
            val_loss = float("nan")
            if (v_interval <= n_epochs and epoch % v_interval == 0) or (eval_at_end and ((epoch == n_epochs) or (epoch in snapshots))):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                val_dice, val_loss = evaluate_cached_val20(
                    model=model,
                    val_ids=val_ids,
                    cache_dir=cache_dir,
                    roi_size_xyz=roi_xyz,
                    sw_overlap=sw_overlap,
                    sw_mode=sw_mode,
                    sw_batch_size=eff_sw_bs,
                    device=device,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    atomic_torch_save(
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": epoch,
                            "global_opt_step": global_opt_step,
                            "val20_dice": best_val_dice,
                            "arch": arch,
                            "init_filters": int(init_filters),
                            "gate_mode": gate_mode,
                            "enable_ds": enable_ds,
                            "seed": int(seed),
                        },
                        best_path,
                    )

            ep_dt = time.perf_counter() - ep_t0
            elapsed_gpu_sec += ep_dt
            peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if device.type == "cuda" else 0.0

            history.append({
                "epoch": epoch,
                "global_step": global_opt_step,
                "train_loss": round(train_loss, 6),
                "val_loss": "" if np.isnan(val_loss) else round(val_loss, 6),
                "val_dice": "" if np.isnan(val_dice) else round(val_dice, 6),
                "learning_rate": f"{cur_lr:.6e}",
                "epoch_time_s": round(ep_dt, 2),
                "cumulative_h": round(elapsed_gpu_sec / 3600.0, 4),
            })
            tel_rows.append({
                "epoch": epoch,
                "global_step": global_opt_step,
                "peak_vram_mb": round(peak_mb, 1),
                "sec_per_step": round(ep_dt / max(n_steps_ep, 1), 4),
            })
            atomic_csv_write(history, log_fields, log_csv)
            atomic_csv_write(tel_rows, tel_fields, tel_csv)

            if epoch in snapshots:
                atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "global_opt_step": global_opt_step,
                        "arch": arch,
                        "init_filters": int(init_filters),
                        "gate_mode": gate_mode,
                        "enable_ds": enable_ds,
                        "seed": int(seed),
                    },
                    ckpt_dir / f"epoch_{epoch}.pth",
                )

            atomic_torch_save(
                {
                    "config_sha256": cfg_hash,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "rng_state": _capture_rng_state(),
                    "global_opt_step": global_opt_step,
                    "next_epoch": epoch + 1,
                    "next_step_in_epoch": 0,
                    "best_val_dice": best_val_dice,
                    "elapsed_gpu_sec": elapsed_gpu_sec,
                    "history": history,
                    "tel_rows": tel_rows,
                },
                latest_path,
            )

            val_msg = f"val20_dice={val_dice:.4f}" if not np.isnan(val_dice) else "val=skip"
            print(
                f"[Ep {epoch:03d}/{n_epochs}] step={global_opt_step}/{total_opt_steps} | loss={train_loss:.4f} | "
                f"{val_msg} | lr={cur_lr:.2e} | vram={peak_mb:.0f}MB | {ep_dt:.1f}s"
            )

            if interrupt_after_epoch is not None and epoch >= interrupt_after_epoch:
                print(f"[INTERRUPT-TEST] Intentional exit after epoch={epoch}")
                return {"status": "interrupted", "epoch": epoch, "global_step": global_opt_step}
    finally:
        if loader_iter is not None and hasattr(loader_iter, "_shutdown_workers"):
            try:
                loader_iter._shutdown_workers()
            except Exception:
                pass
        del loader_iter, loader, model, optimizer, criterion, scheduler, scaler
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    done_payload = {
        "status": "complete",
        "run_name": run_name,
        "arch": arch,
        "init_filters": int(init_filters),
        "gate_mode": gate_mode,
        "enable_ds": enable_ds,
        "loss_name": loss_name,
        "seed": int(seed),
        "epochs": n_epochs,
        "global_opt_step": global_opt_step,
        "best_val20_dice": best_val_dice,
        "elapsed_hours": round(elapsed_gpu_sec / 3600.0, 4),
        "peak_vram_mb": float(max((r["peak_vram_mb"] for r in tel_rows), default=0.0)),
    }
    atomic_torch_save(done_payload, done_flag.with_suffix(".pt"))
    tmp_done = done_flag.with_suffix(".tmp")
    tmp_done.write_text(json.dumps(done_payload, indent=2), encoding="utf-8")
    tmp_done.replace(done_flag)
    if mid_path.is_file():
        mid_path.unlink()
    return done_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arch", required=True, choices=["rasnet_v2", "segresnet", "vnet", "unet3d"])
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--gate-mode", default="conv_bias", choices=["conv_bias", "groupnorm_orig"])
    parser.add_argument("--disable-ds", action="store_true")
    parser.add_argument("--loss-name", default="rasnet_v2", choices=["rasnet_v2", "base_dice_focal"])
    parser.add_argument("--w-cldice", type=float, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mid-epoch-ckpt-every-steps", type=int, default=None)
    parser.add_argument("--interrupt-after-epoch", type=int, default=None)
    parser.add_argument("--interrupt-after-step", type=int, default=None)
    parser.add_argument("--force-device", default=None)
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        run_name=args.run_name,
        arch=args.arch,
        init_filters=args.init_filters,
        lr=args.lr,
        seed=args.seed,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        gate_mode=args.gate_mode,
        enable_ds=not args.disable_ds,
        loss_name=args.loss_name,
        w_cldice_override=args.w_cldice,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        val_interval=args.val_interval,
        num_workers=args.num_workers,
        mid_epoch_ckpt_every_steps=args.mid_epoch_ckpt_every_steps,
        interrupt_after_epoch=args.interrupt_after_epoch,
        interrupt_after_step=args.interrupt_after_step,
        force_device=args.force_device,
    )


if __name__ == "__main__":
    main()

