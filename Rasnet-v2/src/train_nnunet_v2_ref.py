"""
Isolated Multi-Seed Native nnU-Net V2 Trainer Wrapper (`train_nnunet_v2_ref.py`).
Fixes two verified nnunetv2 limitations without editing site-packages:
  1. Applies explicit per-seed RNG control (seeds 42, 123, 456) in isolated per-seed
     nnUNet_results directories (`Rasnet-v2/nnUNet_work/results/seed_{seed}/`).
  2. Saves immutable checkpoint_epoch_{50,100,150,200}.pth snapshots atomically via
     a FairV2nnUNetTrainer(nnUNetTrainer_200epochs) subclass while tagging
     'trainer_name': 'nnUNetTrainer_200epochs' so unmodified nnUNetPredictor loads cleanly.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Match nnunetv2/run/run_training.py BLAS/OpenMP single-thread policy before numpy/torch init
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_nnunet_env(work_dir: Path, seed: int) -> tuple[Path, Path, Path]:
    nn_raw = work_dir / "nnUNet_raw"
    nn_prep = work_dir / "nnUNet_preprocessed"
    nn_res = work_dir / "nnUNet_results" / f"seed_{int(seed)}"
    nn_raw.mkdir(parents=True, exist_ok=True)
    nn_prep.mkdir(parents=True, exist_ok=True)
    nn_res.mkdir(parents=True, exist_ok=True)

    # MUST be set before importing nnunetv2
    os.environ["nnUNet_raw"] = str(nn_raw)
    os.environ["nnUNet_preprocessed"] = str(nn_prep)
    os.environ["nnUNet_results"] = str(nn_res)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    return nn_raw, nn_prep, nn_res


def link_and_inject_dataset(
    cfg: dict[str, Any],
    nn_raw: Path,
    nn_prep: Path,
    dataset_name: str,
) -> tuple[list[int], list[int]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_pairs import load_validated_splits, resolve_case_pairs

    splits = load_validated_splits(cfg["study"]["splits_file"])
    train_ids = [int(c) for c in splits["train"]]
    val_ids = [int(c) for c in splits["val"]]
    all_ids = sorted(train_ids + val_ids)
    pairs = resolve_case_pairs(cfg["study"]["dataset_root"], all_ids, check_headers=False)

    dset_dir = nn_raw / dataset_name
    img_dir = dset_dir / "imagesTr"
    lbl_dir = dset_dir / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for cid in all_ids:
        src_img, src_lbl = pairs[cid]
        dst_img = img_dir / f"case_{cid:04d}_0000.nii.gz"
        dst_lbl = lbl_dir / f"case_{cid:04d}.nii.gz"
        if not dst_img.is_file():
            try:
                os.link(src_img, dst_img)
            except Exception:
                shutil.copy2(src_img, dst_img)
        if not dst_lbl.is_file():
            try:
                os.link(src_lbl, dst_lbl)
            except Exception:
                shutil.copy2(src_lbl, dst_lbl)

    dset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": len(all_ids),
        "file_ending": ".nii.gz",
    }
    (dset_dir / "dataset.json").write_text(json.dumps(dset_json, indent=4), encoding="utf-8")

    prep_dset_dir = nn_prep / dataset_name
    prep_dset_dir.mkdir(parents=True, exist_ok=True)
    (prep_dset_dir / "dataset.json").write_text(json.dumps(dset_json, indent=4), encoding="utf-8")
    nn_splits = [{"train": [f"case_{c:04d}" for c in train_ids], "val": [f"case_{c:04d}" for c in val_ids]}]
    (prep_dset_dir / "splits_final.json").write_text(json.dumps(nn_splits, indent=4), encoding="utf-8")
    return train_ids, val_ids


def check_preprocessed_complete(
    dataset_name: str,
    nn_prep: Path,
    train_ids: list[int],
    val_ids: list[int],
) -> tuple[bool, list[str]]:
    """Verify top-level JSONs plus all 784 train+val (.b2nd, _seg.b2nd, .pkl) files exist and are non-empty."""
    prep_dset = nn_prep / dataset_name
    prep_cfg_dir = prep_dset / "nnUNetPlans_3d_fullres"
    missing: list[str] = []

    for req_name in ("nnUNetPlans.json", "dataset_fingerprint.json", "splits_final.json", "dataset.json"):
        p = prep_dset / req_name
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(str(p))

    if not prep_cfg_dir.is_dir():
        missing.append(str(prep_cfg_dir))
        return False, missing

    all_ids = sorted(set(int(c) for c in (train_ids + val_ids)))
    for cid in all_ids:
        stem = f"case_{cid:04d}"
        for ext in (".b2nd", "_seg.b2nd", ".pkl"):
            fp = prep_cfg_dir / f"{stem}{ext}"
            if not fp.is_file() or fp.stat().st_size == 0:
                missing.append(fp.name)
                if len(missing) >= 20:
                    return False, missing
    return len(missing) == 0, missing


def ensure_preprocessed(
    dataset_id: int,
    dataset_name: str,
    nn_prep: Path,
    nn_raw: Path,
    train_ids: list[int],
    val_ids: list[int],
) -> None:
    prep_dset = nn_prep / dataset_name
    is_ok, missing = check_preprocessed_complete(dataset_name, nn_prep, train_ids, val_ids)
    if not is_ok:
        print(f"[nnU-Net Prep] Missing/incomplete preprocessed artifacts ({len(missing)} shown: {missing[:5]})")
        cmd = [
            sys.executable,
            "-m",
            "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
            "-d",
            str(dataset_id),
            "-c",
            "3d_fullres",
            "-np",
            "8",
            "--clean",
        ]
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        print(f"[nnU-Net Prep] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
        raw_dset_json = nn_raw / dataset_name / "dataset.json"
        if raw_dset_json.is_file():
            shutil.copy2(raw_dset_json, prep_dset / "dataset.json")
        nn_splits = [{"train": [f"case_{c:04d}" for c in train_ids], "val": [f"case_{c:04d}" for c in val_ids]}]
        (prep_dset / "splits_final.json").write_text(json.dumps(nn_splits, indent=4), encoding="utf-8")
        ok_after, missing_after = check_preprocessed_complete(dataset_name, nn_prep, train_ids, val_ids)
        if not ok_after:
            raise RuntimeError(f"nnU-Net preprocessing incomplete after run; missing: {missing_after[:10]}")


def parse_nnunet_log(log_path: Path, output_csv: Path) -> None:
    if not log_path.is_file():
        return
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    history = []
    cur_ep = None
    ep_loss = np.nan
    val_loss = np.nan
    val_dice = np.nan
    lr = np.nan
    ep_time = np.nan
    cum_time = 0.0

    for line in lines:
        m_ep = re.search(r"\bEpoch\s+(\d+)\s*$", line)
        if m_ep:
            if cur_ep is not None:
                history.append({
                    "epoch": cur_ep + 1,
                    "train_loss": ep_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "learning_rate": lr,
                    "epoch_time_seconds": ep_time,
                    "cumulative_time_hours": cum_time / 3600.0,
                })
            cur_ep = int(m_ep.group(1))
            ep_loss = np.nan
            val_loss = np.nan
            val_dice = np.nan
            lr = np.nan
            ep_time = np.nan
            continue
        if m := re.search(r"Current learning rate:\s*([0-9eE\.\-]+)", line):
            lr = float(m.group(1))
        if m := re.search(r"train_loss\s*([0-9eE\.\-]+)", line):
            ep_loss = float(m.group(1))
        if m := re.search(r"val_loss\s*([0-9eE\.\-]+)", line):
            val_loss = float(m.group(1))
        if m := re.search(r"Pseudo dice\s*\[(?:np\.float32\()?([0-9\.]+)", line):
            val_dice = float(m.group(1))
        if m := re.search(r"Epoch time:\s*([0-9\.]+)\s*s", line):
            ep_time = float(m.group(1))
            cum_time += ep_time

    if cur_ep is not None:
        history.append({
            "epoch": cur_ep + 1,
            "train_loss": ep_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "learning_rate": lr,
            "epoch_time_seconds": ep_time,
            "cumulative_time_hours": cum_time / 3600.0,
        })
    if history:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_csv.with_suffix(".tmp")
        pd.DataFrame(history).to_csv(tmp, index=False)
        tmp.replace(output_csv)


def run_nnunet_v2_seed(
    config_path: str | Path,
    seed: int = 42,
    num_epochs: int | None = None,
    epoch_snapshots: tuple[int, ...] | None = None,
    interrupt_after_epoch: int | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root_dir = Path(cfg["study"]["root_dir"])
    work_dir = root_dir / "nnUNet_work"
    nn_cfg = cfg["nnunet_v2"]
    dataset_id = int(nn_cfg["dataset_id"])
    dataset_name = str(nn_cfg["dataset_name"])
    configuration = str(nn_cfg["configuration"])
    fold = int(nn_cfg["fold"])
    target_epochs = int(num_epochs if num_epochs is not None else cfg["training_budget"]["epochs"])
    snaps = set(int(e) for e in (epoch_snapshots or nn_cfg["epoch_snapshots"]))

    run_name = f"nnunet_v2_seed{int(seed)}"
    ckpt_mirror_dir = root_dir / "checkpoints" / run_name
    log_mirror_dir = root_dir / "logs" / run_name
    ckpt_mirror_dir.mkdir(parents=True, exist_ok=True)
    log_mirror_dir.mkdir(parents=True, exist_ok=True)
    done_flag = ckpt_mirror_dir / "DONE.flag"
    if done_flag.is_file():
        print(f"[SKIP] {run_name} already completed ({done_flag})")
        return json.loads(done_flag.read_text(encoding="utf-8"))

    nn_raw, nn_prep, nn_res = setup_nnunet_env(work_dir, seed)
    train_ids, val_ids = link_and_inject_dataset(cfg, nn_raw, nn_prep, dataset_name)
    ensure_preprocessed(dataset_id, dataset_name, nn_prep, nn_raw, train_ids, val_ids)

    # Import nnunetv2 strictly AFTER env vars are set
    import torch.backends.cudnn as cudnn
    from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
    from nnunetv2.run.run_training import maybe_load_checkpoint
    from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
        nnUNetTrainer_200epochs as _Base200Epochs,
    )
    from nnunetv2.utilities.file_path_utilities import get_output_folder

    class nnUNetTrainer_200epochs(_Base200Epochs):
        def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device("cuda"),
        ):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.num_epochs = target_epochs
            self._fair_seed = int(seed)
            self._fair_snapshots = snaps
            self._interrupt_after_epoch = interrupt_after_epoch
            self._interrupted = False

        def initialize(self):
            seed_everything(self._fair_seed)
            super().initialize()

        def get_dataloaders(self):
            import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as tr_mod
            orig_nondet = tr_mod.NonDetMultiThreadedAugmenter
            base_seed = int(self._fair_seed)
            call_counter = [0]

            class _SeededNonDetAugmenter(orig_nondet):
                def __init__(self, *args, **kwargs):
                    if kwargs.get("seeds") is None:
                        n_proc = kwargs.get("num_processes", args[2] if len(args) > 2 else 1)
                        idx = call_counter[0]
                        call_counter[0] += 1
                        kwargs["seeds"] = [base_seed * 10000 + idx * 100 + w for w in range(int(n_proc))]
                    super().__init__(*args, **kwargs)

            seed_everything(base_seed)
            tr_mod.NonDetMultiThreadedAugmenter = _SeededNonDetAugmenter
            try:
                return super().get_dataloaders()
            finally:
                tr_mod.NonDetMultiThreadedAugmenter = orig_nondet

        def save_checkpoint(self, filename: str) -> None:
            if self.local_rank != 0 or self.disable_checkpointing:
                return
            mod = self.network.module if self.is_ddp else self.network
            if hasattr(mod, "_orig_mod"):
                mod = mod._orig_mod
            checkpoint = {
                "network_weights": mod.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                "logging": self.logger.get_checkpoint(),
                "_best_ema": self._best_ema,
                "current_epoch": self.current_epoch + 1,
                "init_args": self.my_init_kwargs,
                "trainer_name": "nnUNetTrainer_200epochs",
                "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
                "fair_v2_seed": self._fair_seed,
            }
            out_p = Path(filename)
            tmp_p = out_p.with_suffix(out_p.suffix + ".tmp")
            with open(tmp_p, "wb") as f:
                torch.save(checkpoint, f)
                f.flush()
                os.fsync(f.fileno())
            tmp_p.replace(out_p)

        def on_epoch_end(self):
            ep_one_based = int(self.current_epoch) + 1
            super().on_epoch_end()
            # super().on_epoch_end() increments current_epoch; temporarily restore it so saved
            # checkpoints store the exact completed epoch index matching nnUNetTrainer.on_train_end.
            self.current_epoch -= 1
            try:
                already_saved_latest = (ep_one_based % self.save_every == 0) and (ep_one_based != self.num_epochs)
                if not already_saved_latest:
                    self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_latest.pth"))
                if ep_one_based in self._fair_snapshots:
                    snap_path = os.path.join(self.output_folder, f"checkpoint_epoch_{ep_one_based}.pth")
                    self.save_checkpoint(snap_path)
            finally:
                self.current_epoch += 1
            if self._interrupt_after_epoch is not None and ep_one_based >= self._interrupt_after_epoch:
                self._interrupted = True
                for dl in (self.dataloader_train, self.dataloader_val):
                    if dl is not None and isinstance(dl, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                        dl._finish()
                raise KeyboardInterrupt(f"Intentional Gate 3 stop at epoch {ep_one_based}")

        def on_train_end(self):
            # Call parent cleanly so dataloaders finish without worker leaks, then preserve latest snapshot
            super().on_train_end()
            final_ckpt = os.path.join(self.output_folder, "checkpoint_final.pth")
            latest_ckpt = os.path.join(self.output_folder, "checkpoint_latest.pth")
            if os.path.isfile(final_ckpt) and not os.path.isfile(latest_ckpt):
                shutil.copy2(final_ckpt, latest_ckpt)

    prep_dset_dir = nn_prep / dataset_name
    plans = json.loads((prep_dset_dir / "nnUNetPlans.json").read_text(encoding="utf-8"))
    dataset_json = json.loads((prep_dset_dir / "dataset.json").read_text(encoding="utf-8"))

    model_dir = get_output_folder(dataset_id, "nnUNetTrainer_200epochs", "nnUNetPlans", configuration)
    fold_dir = Path(model_dir) / f"fold_{fold}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # Persist clean unmodified plans.json & dataset.json before injecting continue_training
    shutil.copy2(prep_dset_dir / "nnUNetPlans.json", Path(model_dir) / "plans.json")
    (Path(model_dir) / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    latest_ckpt = fold_dir / "checkpoint_latest.pth"
    continue_training = latest_ckpt.is_file()
    plans["continue_training"] = bool(continue_training)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prev_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if device.type == "cuda":
        cudnn.deterministic = False
        cudnn.benchmark = True

    trainer = nnUNetTrainer_200epochs(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device,
    )
    fold_dir.mkdir(parents=True, exist_ok=True)

    maybe_load_checkpoint(trainer, continue_training=continue_training, validation_only=False)

    t0 = time.perf_counter()
    try:
        trainer.run_training()
    except KeyboardInterrupt as exc:
        if trainer._interrupted:
            print(f"[INTERRUPT-TEST] {exc}")
            return {"status": "interrupted", "fold_dir": str(fold_dir), "model_dir": str(model_dir)}
        raise
    finally:
        for dl in (getattr(trainer, "dataloader_train", None), getattr(trainer, "dataloader_val", None)):
            if dl is not None and isinstance(dl, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                try:
                    dl._finish()
                except Exception:
                    pass
        del trainer
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.set_num_threads(prev_num_threads)

    dt = time.perf_counter() - t0
    ckpt_index = {
        f.name: str(f)
        for f in sorted(fold_dir.glob("checkpoint_*.pth"))
    }
    (ckpt_mirror_dir / "checkpoints_index.json").write_text(
        json.dumps({"seed_results_dir": str(fold_dir), "checkpoints": ckpt_index}, indent=2),
        encoding="utf-8",
    )
    log_files = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if log_files:
        parse_nnunet_log(log_files[0], log_mirror_dir / "training_log.csv")

    done_payload = {
        "status": "complete",
        "run_name": run_name,
        "seed": int(seed),
        "epochs": target_epochs,
        "model_dir": str(model_dir),
        "fold_dir": str(fold_dir),
        "elapsed_hours": round(dt / 3600.0, 4),
    }
    tmp_done = done_flag.with_suffix(".tmp")
    tmp_done.write_text(json.dumps(done_payload, indent=2), encoding="utf-8")
    tmp_done.replace(done_flag)
    return done_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"C:\Thesis_RASNET\Rasnet-v2\configs\fair_v2.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--snapshots", type=int, nargs="*", default=None)
    parser.add_argument("--interrupt-after-epoch", type=int, default=None)
    args = parser.parse_args()
    run_nnunet_v2_seed(
        config_path=args.config,
        seed=args.seed,
        num_epochs=args.epochs,
        epoch_snapshots=tuple(args.snapshots) if args.snapshots else None,
        interrupt_after_epoch=args.interrupt_after_epoch,
    )


if __name__ == "__main__":
    main()
