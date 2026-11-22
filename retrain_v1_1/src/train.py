"""Unified 200-epoch Strategy A MONAI trainer with full atomic resume checkpoints & manifests.

Every trainable model in Strategy A uses the identical:
  - transform-hashed blosc2 cache (`retrain_v1_1/cache/tf_<sha256[:16]>/`)
  - deterministic cyclic 250 steps/ep x batch_size 2 (pos=1:neg=2, 160x160x96)
  - full 3D augmentations (flips, affine rot/scale, noise, blur, intensity scale, contrast)
  - DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, batch=True) + fixed DS
  - AdamW(lr=3e-4, wd=1e-5) + WarmupCosinePerStep + clip_grad_norm_(1.0) + bfloat16 AMP
  - full atomic per-epoch resume checkpoint (`latest_state.pt` with model/optimizer/scaler/scheduler/RNG)
  - standardized full Val-94 validation cadence (every 10 epochs + epoch 200)
  - per-run `train_manifest.json` and atomic `train_log.csv`.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from benchmark.symmetric_eval import env_info, git_info, sha256_file
from retrain_v1_1.src.cache import (
    case_cache_dir,
    check_disk_headroom_gb,
    load_cached_case,
    open_cached_case_handles,
)
from retrain_v1_1.src.losses import StrategyADiceCEDSLoss
from retrain_v1_1.src.models import build_strategy_a_model, count_parameters
from retrain_v1_1.src.transforms import (
    PAD_IMAGE_VAL,
    PAD_LABEL_VAL,
    apply_3d_train_augmentations,
    compute_crop_slices,
    compute_transform_hash,
    pad_patch_if_needed,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_write_csv(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def resolve_amp_dtype(device: torch.device, requested: str = "bfloat16") -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if str(requested).lower() == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    eta_min: float,
) -> LambdaLR:
    min_ratio = float(eta_min) / max(float(base_lr), 1e-12)
    t_steps = max(int(total_steps), 1)
    w_steps = max(0, min(int(warmup_steps), t_steps - 1))

    def lr_lambda(step: int) -> float:
        if w_steps > 0 and step < w_steps:
            return max(min_ratio, float(step + 1) / float(w_steps))
        progress = float(step - w_steps) / float(max(t_steps - w_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _dl_worker_init(worker_id: int) -> None:
    """Prevent thread oversubscription across Windows DataLoader workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import blosc2
        blosc2.set_nthreads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


class DeterministicStrategyADataset(Dataset):
    """Stream-resumable deterministic patch dataset reading directly from chunked blosc2 cache."""

    def __init__(
        self,
        cache_dir: Path,
        train_ids: list[int],
        seed: int,
        start_global_idx: int,
        total_samples: int,
        patch_size_xyz: tuple[int, int, int],
        fg_prob: float,
        aug_cfg: dict[str, Any],
        pad_image_val: float = PAD_IMAGE_VAL,
        pad_label_val: int = PAD_LABEL_VAL,
    ):
        self.cache_dir = Path(cache_dir)
        self.train_ids = sorted(int(c) for c in train_ids)
        self.seed = int(seed)
        self.start_global = int(start_global_idx)
        self.total_samples = int(total_samples)
        self.n_remaining = max(0, self.total_samples - self.start_global)
        self.patch_size_xyz = tuple(int(v) for v in patch_size_xyz)
        self.fg_prob = float(fg_prob)
        self.aug_cfg = dict(aug_cfg)
        self.pad_image_val = float(pad_image_val)
        self.pad_label_val = int(pad_label_val)

        if self.n_remaining > 0:
            n_cases = len(self.train_ids)
            first_cycle = self.start_global // n_cases
            last_cycle = (self.total_samples - 1) // n_cases
            ids_arr = np.asarray(self.train_ids, dtype=np.int32)
            cycles = []
            for c_idx in range(first_cycle, last_cycle + 1):
                perm_rng = np.random.default_rng(self.seed * 100003 + c_idx)
                cycles.append(perm_rng.permutation(ids_arr))
            concat = np.concatenate(cycles)
            offset = self.start_global - first_cycle * n_cases
            self.stream_cases = concat[offset : offset + self.n_remaining]
        else:
            self.stream_cases = np.empty((0,), dtype=np.int32)

    def __len__(self) -> int:
        return self.n_remaining

    def __getitem__(self, idx: int):
        global_idx = self.start_global + int(idx)
        cid = int(self.stream_cases[idx])
        img_h, lbl_h, fg, bg = open_cached_case_handles(self.cache_dir, cid)

        # Sub-stream seeds: crop RNG and augmentation RNG derived strictly from (seed, global_idx)
        ss = np.random.SeedSequence([self.seed, global_idx])
        s_crop, s_aug = ss.spawn(2)
        crop_rng = np.random.default_rng(s_crop)
        aug_seed = int(np.random.default_rng(s_aug).integers(0, 2**31 - 1))

        vol_shape_xyz = (int(img_h.shape[1]), int(img_h.shape[2]), int(img_h.shape[3]))
        sx, sy, sz = compute_crop_slices(
            vol_shape_xyz=vol_shape_xyz,
            fg_coords=fg,
            bg_coords=bg,
            rng=crop_rng,
            patch_size_xyz=self.patch_size_xyz,
            fg_prob=self.fg_prob,
        )
        p_img = np.asarray(img_h[:, sx, sy, sz], dtype=np.float32)
        p_lbl = np.asarray(lbl_h[:, sx, sy, sz], dtype=np.uint8)
        p_img, p_lbl = pad_patch_if_needed(
            p_img, p_lbl, self.patch_size_xyz, self.pad_image_val, self.pad_label_val
        )
        p_img, p_lbl = apply_3d_train_augmentations(p_img, p_lbl, self.aug_cfg, seed=aug_seed)
        return torch.from_numpy(p_img), torch.from_numpy(p_lbl.astype(np.int64, copy=False))


def evaluate_cached_val(
    model: torch.nn.Module,
    val_ids: list[int],
    cache_dir: Path,
    roi_size_xyz: tuple[int, int, int],
    sw_overlap: float,
    sw_mode: str,
    sw_batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """Full-volume sliding-window validation on cached Val-94 (sw_device=GPU, output=CPU)."""
    model.eval()
    dices: list[float] = []
    soft_losses: list[float] = []
    amp_ok = device.type == "cuda"
    cpu_dev = torch.device("cpu")
    with torch.no_grad():
        for cid in val_ids:
            img_np, lbl_np, _, _, _ = load_cached_case(cache_dir, cid)
            img_t = torch.from_numpy(img_np.astype(np.float32, copy=False)).unsqueeze(0)
            gt_mask = torch.from_numpy(lbl_np > 0).unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_ok):
                logits = sliding_window_inference(
                    img_t,
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
            dices.append((2.0 * inter / denom) if denom > 0 else float("nan"))

            gt_f = gt_mask.float()
            soft_inter = float((probs_fg * gt_f).sum().item())
            soft_denom = float(probs_fg.sum().item() + gt_f.sum().item())
            soft_losses.append(1.0 - (2.0 * soft_inter + 1e-5) / (soft_denom + 1e-5))
    return float(np.nanmean(dices)), float(np.nanmean(soft_losses))


def _update_queue_status(
    status_path: Path | None,
    model_name: str,
    epoch: int,
    total_epochs: int,
    best_val_dice: float,
    best_epoch: int,
    last_train_loss: float,
    ckpt_dir: Path,
    state: str = "running",
) -> None:
    if status_path is None:
        return
    free_gb = shutil.disk_usage(str(ckpt_dir)).free / (1024 ** 3)
    atomic_write_json(
        {
            "state": state,
            "current_model": model_name,
            "epoch": int(epoch),
            "total_epochs": int(total_epochs),
            "best_val_dice": float(best_val_dice),
            "best_epoch": int(best_epoch),
            "last_train_loss": float(last_train_loss),
            "free_disk_gb": round(free_gb, 2),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        status_path,
    )


def train_model(
    cfg: dict[str, Any],
    model_spec: dict[str, Any],
    cache_root: Path,
    ckpt_dir: Path,
    log_dir: Path,
    train_ids: list[int],
    val_ids: list[int],
    device_str: str = "cuda",
    stop_after_epoch: int | None = None,
    queue_status_path: Path | None = None,
) -> dict[str, Any]:
    """Train or seamlessly resume a single Strategy A model for 200 epochs."""
    ckpt_dir = Path(ckpt_dir)
    log_dir = Path(log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    done_flag = ckpt_dir / "DONE.flag"
    if done_flag.is_file():
        print(f"[SKIP] {model_spec['name']} already complete ({done_flag})", flush=True)
        return json.loads(done_flag.read_text(encoding="utf-8"))

    tr_cfg = cfg["training"]
    pre_cfg = cfg["preprocessing"]
    loss_cfg = cfg["loss"]
    aug_cfg = cfg["augmentations"]

    tf_hash = compute_transform_hash(pre_cfg)
    cache_dir = case_cache_dir(cache_root, tf_hash)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Transform-hashed cache dir missing: {cache_dir}")

    cfg_sha256 = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    seed = int(cfg["study"]["seed"])
    n_epochs = int(tr_cfg["epochs"])
    steps_per_ep = int(tr_cfg["steps_per_epoch"])
    batch_size = int(tr_cfg["batch_size"])
    samples_per_ep = steps_per_ep * batch_size
    total_steps = n_epochs * steps_per_ep
    warmup_steps = int(tr_cfg.get("warmup_epochs", 5)) * steps_per_ep
    base_lr = float(tr_cfg["lr"])
    eta_min = float(tr_cfg.get("eta_min", 1e-6))
    grad_clip = float(tr_cfg.get("grad_clip_norm", 1.0))
    val_interval = int(tr_cfg.get("val_interval_epochs", 10))
    patch_size_xyz = tuple(int(v) for v in pre_cfg["patch_size_xyz"])
    roi_size_xyz = tuple(int(v) for v in cfg["evaluation"]["roi_size_xyz"])
    fg_prob = float(tr_cfg.get("foreground_ratio", 1.0 / 3.0))
    num_workers = int(tr_cfg.get("num_workers", 4))
    min_free_gb = float(tr_cfg.get("min_free_disk_gb", 25.0))

    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(device_str)
    amp_dtype = resolve_amp_dtype(device, tr_cfg.get("amp_dtype", "bfloat16"))
    amp_enabled = device.type == "cuda"
    use_scaler = amp_enabled and (amp_dtype == torch.float16)

    set_all_seeds(seed)
    model = build_strategy_a_model(
        arch=model_spec["arch"],
        init_filters=int(model_spec.get("init_filters", 16)),
        dropout_prob=float(model_spec.get("dropout_prob", 0.1)),
        enable_attention=bool(model_spec.get("enable_attention", True)),
        enable_ds=bool(model_spec.get("enable_ds", True)),
    ).to(device)
    n_params = count_parameters(model)

    criterion = StrategyADiceCEDSLoss(
        main_weight=float(loss_cfg.get("ds_main_weight", 1.0)),
        ds_weights_quarter_half=tuple(float(v) for v in loss_cfg.get("ds_weights_quarter_half", [0.25, 0.50])),
        normalize_by_sum=bool(loss_cfg.get("ds_normalize_by_sum", True)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(tr_cfg.get("weight_decay", 1e-5)),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        base_lr=base_lr,
        eta_min=eta_min,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch = 1
    global_step = 0
    best_val_dice = -1.0
    best_epoch = 0
    elapsed_prior = 0.0
    log_rows: list[dict[str, Any]] = []

    latest_path = ckpt_dir / "latest_state.pt"
    log_csv = log_dir / "train_log.csv"
    fieldnames = ["epoch", "train_loss", "val_dice", "val_soft_dice_loss", "lr", "epoch_sec", "elapsed_sec"]

    if latest_path.is_file():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        if str(ckpt.get("transform_sha256", "")) != tf_hash:
            raise RuntimeError(
                f"Resume transform_sha256 mismatch for {model_spec['name']}: "
                f"ckpt={ckpt.get('transform_sha256')} != current={tf_hash}"
            )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("global_step", (start_epoch - 1) * steps_per_ep))
        best_val_dice = float(ckpt.get("best_val_dice", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        elapsed_prior = float(ckpt.get("elapsed_sec", 0.0))
        log_rows = list(ckpt.get("log_rows", []))
        _restore_rng_state(ckpt["rng_state"])
        print(
            f"[RESUME] {model_spec['name']} from epoch {start_epoch}/{n_epochs} "
            f"(best_val_dice={best_val_dice:.4f} @ ep {best_epoch})",
            flush=True,
        )
    else:
        print(
            f"[START] {model_spec['name']} ({n_params:,} params, device={device}, "
            f"amp={amp_dtype}, epochs={n_epochs}, steps/ep={steps_per_ep}, lr={base_lr})",
            flush=True,
        )

    if device.type == "cuda":
        check_disk_headroom_gb(ckpt_dir, min_free_gb=min_free_gb)

    t_session = time.time()
    last_epoch_run = start_epoch - 1
    nonfinite_streak = 0

    # Spawn DataLoader workers ONCE across all remaining epochs (avoids 200x Windows spawn overhead)
    remaining_epochs = max(0, n_epochs - start_epoch + 1)
    if remaining_epochs > 0:
        ds = DeterministicStrategyADataset(
            cache_dir=cache_dir,
            train_ids=train_ids,
            seed=seed,
            start_global_idx=(start_epoch - 1) * samples_per_ep,
            total_samples=n_epochs * samples_per_ep,
            patch_size_xyz=patch_size_xyz,
            fg_prob=fg_prob,
            aug_cfg=aug_cfg,
            pad_image_val=float(pre_cfg.get("pad_image_val", PAD_IMAGE_VAL)),
            pad_label_val=int(pre_cfg.get("pad_label_val", PAD_LABEL_VAL)),
        )
        dl_gen = torch.Generator()
        dl_gen.manual_seed(seed)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=_dl_worker_init if num_workers > 0 else None,
            generator=dl_gen,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
            persistent_workers=False,
        )
        loader_iter = iter(loader)
    else:
        loader_iter = iter(())

    for epoch in range(start_epoch, n_epochs + 1):
        ep_t0 = time.time()
        model.train()

        loss_sum = 0.0
        n_batches = 0
        for _ in range(steps_per_ep):
            imgs, lbls = next(loader_iter)
            imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
            lbls = lbls.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            if not torch.isfinite(loss):
                nonfinite_streak += 1
                if nonfinite_streak >= 5:
                    raise RuntimeError(
                        f"Non-finite loss for 5 consecutive steps in {model_spec['name']} at epoch {epoch}"
                    )
                continue
            nonfinite_streak = 0

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            scheduler.step()

            loss_sum += float(loss.item())
            n_batches += 1
            global_step += 1

        mean_loss = loss_sum / max(n_batches, 1)
        cur_lr = float(optimizer.param_groups[0]["lr"])

        val_dice_str = ""
        val_loss_str = ""
        is_val_epoch = (epoch % val_interval == 0) or (epoch == n_epochs)
        if is_val_epoch:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                check_disk_headroom_gb(ckpt_dir, min_free_gb=min_free_gb)
            val_dice, val_soft_loss = evaluate_cached_val(
                model=model,
                val_ids=val_ids,
                cache_dir=cache_dir,
                roi_size_xyz=roi_size_xyz,
                sw_overlap=float(tr_cfg.get("val_sw_overlap", 0.50)),
                sw_mode=str(tr_cfg.get("val_sw_mode", "gaussian")),
                sw_batch_size=int(tr_cfg.get("val_sw_batch_size", 4)),
                device=device,
                amp_dtype=amp_dtype,
            )
            val_dice_str = f"{val_dice:.6f}"
            val_loss_str = f"{val_soft_loss:.6f}"
            if val_dice > best_val_dice:
                best_val_dice = float(val_dice)
                best_epoch = int(epoch)
                atomic_torch_save(model.state_dict(), ckpt_dir / "best_model.pth")
            if device.type == "cuda":
                torch.cuda.empty_cache()

        ep_sec = time.time() - ep_t0
        total_elapsed = elapsed_prior + (time.time() - t_session)
        log_rows = [r for r in log_rows if int(r["epoch"]) < epoch]
        log_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": f"{mean_loss:.6f}",
                "val_dice": val_dice_str,
                "val_soft_dice_loss": val_loss_str,
                "lr": f"{cur_lr:.8f}",
                "epoch_sec": f"{ep_sec:.2f}",
                "elapsed_sec": f"{total_elapsed:.2f}",
            }
        )
        atomic_write_csv(log_rows, fieldnames, log_csv)

        # Full atomic resume checkpoint persisted after EVERY epoch so user can stop/resume cleanly
        atomic_torch_save(
            {
                "model_name": model_spec["name"],
                "epoch": int(epoch),
                "global_step": int(global_step),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_scaler else None,
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_dice": float(best_val_dice),
                "best_epoch": int(best_epoch),
                "elapsed_sec": float(total_elapsed),
                "rng_state": _capture_rng_state(),
                "transform_sha256": tf_hash,
                "config_sha256": cfg_sha256,
                "log_rows": log_rows,
            },
            latest_path,
        )
        _update_queue_status(
            queue_status_path,
            model_name=model_spec["name"],
            epoch=epoch,
            total_epochs=n_epochs,
            best_val_dice=best_val_dice,
            best_epoch=best_epoch,
            last_train_loss=mean_loss,
            ckpt_dir=ckpt_dir,
            state="running",
        )
        last_epoch_run = epoch

        if is_val_epoch or epoch == 1:
            msg = (
                f"[{model_spec['name']}] ep {epoch:3d}/{n_epochs} | loss={mean_loss:.4f} | "
                f"lr={cur_lr:.2e} | ep_s={ep_sec:.1f}s"
            )
            if val_dice_str:
                msg += f" | val94_dice={float(val_dice_str):.4f} (best={best_val_dice:.4f} @ ep {best_epoch})"
            print(msg, flush=True)

        if stop_after_epoch is not None and epoch >= int(stop_after_epoch):
            print(f"[STOPPED EARLY] {model_spec['name']} paused cleanly at epoch {epoch}", flush=True)
            return {
                "model_name": model_spec["name"],
                "status": "paused",
                "epoch": int(epoch),
                "best_val_dice": float(best_val_dice),
                "best_epoch": int(best_epoch),
            }

    # Final epoch 200 completion artifacts
    final_ckpt_path = ckpt_dir / "final_epoch200.pth"
    atomic_torch_save(model.state_dict(), final_ckpt_path)
    best_ckpt_path = ckpt_dir / "best_model.pth"
    if not best_ckpt_path.is_file():
        atomic_torch_save(model.state_dict(), best_ckpt_path)

    total_elapsed = elapsed_prior + (time.time() - t_session)
    manifest = {
        "model_name": model_spec["name"],
        "model_spec": model_spec,
        "n_params": int(n_params),
        "seed": int(seed),
        "epochs_completed": int(last_epoch_run),
        "best_epoch": int(best_epoch),
        "best_val_dice": float(best_val_dice),
        "best_model_path": str(best_ckpt_path),
        "best_model_sha256": sha256_file(best_ckpt_path),
        "final_model_path": str(final_ckpt_path),
        "final_model_sha256": sha256_file(final_ckpt_path),
        "transform_sha256": tf_hash,
        "config_sha256": cfg_sha256,
        "amp_dtype": str(amp_dtype),
        "elapsed_sec": round(total_elapsed, 2),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_info(REPO_ROOT, code_paths=("retrain_v1_1/src", "retrain_v1_1/configs", "benchmark")),
        "env": env_info(),
    }
    atomic_write_json(manifest, log_dir / "train_manifest.json")
    atomic_write_json(manifest, done_flag)
    _update_queue_status(
        queue_status_path,
        model_name=model_spec["name"],
        epoch=n_epochs,
        total_epochs=n_epochs,
        best_val_dice=best_val_dice,
        best_epoch=best_epoch,
        last_train_loss=float(log_rows[-1]["train_loss"]) if log_rows else 0.0,
        ckpt_dir=ckpt_dir,
        state="model_done",
    )
    print(
        f"[DONE] {model_spec['name']} | best_val94_dice={best_val_dice:.4f} @ ep {best_epoch} | "
        f"best_sha256={manifest['best_model_sha256'][:12]} | total={total_elapsed / 3600:.2f}h",
        flush=True,
    )
    del loader_iter, model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest
