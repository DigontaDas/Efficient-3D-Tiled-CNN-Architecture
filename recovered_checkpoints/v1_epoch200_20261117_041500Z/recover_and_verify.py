#!/usr/bin/env python3
"""Recover and verify archived V1 epoch-200 final checkpoints (RASNet & nnU-Net V2).

Self-contained, idempotent, non-destructive recovery & provenance verifier:
- Streams ONLY `rasnet_last.pth` and `nnunet_last.pt` via a seekable SplitZipReader
  over `checkpoints_200ep.zip.001`..`.010` (no 900 MB combined ZIP, no `extractall()`).
- Validates local file headers (`PK\x03\x04`, flag_bits == 0, filename, CRC-32, size),
  central directory metadata, streamed CRC-32, byte sizes, and SHA-256 digests before
  atomic `os.replace`.
- Constructs an isolated native nnU-Net V2 model folder with verified `plans.json`,
  `dataset.json`, and `fold_0/checkpoint_final.pth`.
- Verifies nnU-Net V2 epoch-200 embedded metadata, trainer convention, log match,
  and CPU `nnUNetPredictor` initialization.
- Verifies RASNet V1 strict loading, script/log/archive epoch-200 provenance (explicitly
  noting absence of embedded checkpoint metadata), and weight differences vs best.
- Checks pre- and post-run non-interference on all protected paths and atomically writes
  `provenance_verification.json` and `PROVENANCE_REPORT.md`.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import datetime as dt
import hashlib
import inspect
import io
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any

# Expected archive & artifact constants (R1)
EXPECTED_PART_COUNT = 10
EXPECTED_COMBINED_ARCHIVE_BYTES = 901_178_285
EXPECTED_ARCHIVE_MEMBERS = {
    "segresnet_best.pt",
    "segresnet_last.pt",
    "3dunet_best.pth",
    "3dunet_last.pth",
    "vnet_best.pt",
    "vnet_last.pt",
    "nnunet_best.pt",
    "nnunet_last.pt",
    "rasnet_best.pth",
    "rasnet_last.pth",
}

TARGET_SPECS: dict[str, dict[str, Any]] = {
    "rasnet_last.pth": {
        "size": 18_871_718,
        "crc32": 0x382CC853,
        "sha256": "71c9e5ef4a02849686c99ccb2d1544ff78c78187e1c0d53e7b22884c708c1b95",
    },
    "nnunet_last.pt": {
        "size": 246_472_688,
        "crc32": 0xAE7F79E9,
        "sha256": "2105bd7238fff7485f1a8d72622f977f635f441e6d44863920c85709b02ffec2",
    },
}

EXPECTED_PLANS_SHA256 = "c67af4aaeb24f5a72b58c83e351a9c62cf026f8baf1bbb727e3022491d5733d8"
EXPECTED_DATASET_SHA256 = "d81f40253d7c2d88b9d1cdbd4bd7319a7a0206fce5b2350287317bd34f9c7389"
EXPECTED_RASNET_BEST_SHA256 = "8d4c25cd12d07c40423062b27b1d36836e5fd6cecb30c20c34055eeac976d1b8"
EXPECTED_NNUNET_BEST_SHA256 = "9accddf283b1c480e386f30f426b9200ca8d4aa1ecd70d5b5af0b8aad6058711"

FORBIDDEN_OUTPUT_PARENTS = (
    "Q1_Publication_Package",
    "results",
    "Rasnet-v2",
    "benchmark",
    "scripts",
    "tests",
    "Phase3_Local_Integration",
    "secondary-imagcas-dataset",
)

PROTECTED_ANCHOR_FILES = (
    "LAB_RUN_PLAN.md",
    "CHECKPOINTS.md",
    "Phase3_Local_Integration/splits_final.json",
    "Phase3_Local_Integration/rasnet_model.py",
    "results/thresholds.json",
    "Rasnet-v2/logs/queue_status.json",
    "Rasnet-v2/results/pilots_val94/pilot_lock_decision.md",
    "benchmark/adapters/nnunet_adapter.py",
    "benchmark/run_benchmark.py",
    "benchmark/symmetric_eval.py",
    "Q1_Publication_Package/matched_200ep_benchmark/train_rasnet_200ep.py",
    "Q1_Publication_Package/matched_200ep_benchmark/prepare_and_run_nnunet_200ep.py",
    "Q1_Publication_Package/matched_200ep_benchmark/logs/rasnet_200ep_log.csv",
    "Q1_Publication_Package/matched_200ep_benchmark/logs/nnunet_200ep_log.csv",
    "Q1_Publication_Package/matched_200ep_benchmark/checkpoints/rasnet_best.pth",
    "Q1_Publication_Package/matched_200ep_benchmark/checkpoints/nnunet_best.pt",
    "Q1_Publication_Package/matched_200ep_benchmark/nnUNet_results/Dataset501_CoronarySeg/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres/plans.json",
    "Q1_Publication_Package/matched_200ep_benchmark/nnUNet_results/Dataset501_CoronarySeg/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres/dataset.json",
    "Q1_Publication_Package/matched_200ep_benchmark/nnUNet_results/Dataset501_CoronarySeg/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
)

PROTECTED_DIRS = (
    "Q1_Publication_Package/matched_200ep_benchmark/checkpoints",
    "Q1_Publication_Package/matched_200ep_benchmark/checkpoints_best_and_last",
    "Q1_Publication_Package/matched_200ep_benchmark/nnUNet_results",
    "Q1_Publication_Package/matched_200ep_benchmark/logs",
    "results",
    "benchmark",
    "scripts",
    "tests",
)

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB


class SplitZipReader(io.RawIOBase):
    """Seekable read-only RawIOBase stream over contiguous multi-part archive files."""

    def __init__(self, part_paths: list[Path]) -> None:
        super().__init__()
        if not part_paths:
            raise ValueError("part_paths must not be empty")
        self._paths = [Path(p) for p in part_paths]
        self._handles: list[io.BufferedReader | Any] = []
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._pos = 0
        offset = 0
        try:
            for p in self._paths:
                size = p.stat().st_size
                if size <= 0:
                    raise ValueError(f"Archive part is empty: {p}")
                fh = p.open("rb")
                self._handles.append(fh)
                self._starts.append(offset)
                offset += size
                self._ends.append(offset)
        except Exception:
            for fh in self._handles:
                try:
                    fh.close()
                except Exception:
                    pass
            raise
        self._total_size = offset

    @property
    def total_size(self) -> int:
        return self._total_size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._total_size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if new_pos < 0:
            raise ValueError(f"Negative seek position: {new_pos}")
        self._pos = new_pos
        return self._pos

    def readinto(self, b) -> int:  # type: ignore[override]
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if self._pos >= self._total_size:
            return 0
        mv = memoryview(b).cast("B")
        total_req = len(mv)
        if total_req == 0:
            return 0
        total_read = 0
        while total_read < total_req and self._pos < self._total_size:
            idx = bisect.bisect_right(self._ends, self._pos)
            if idx >= len(self._handles):
                break
            part_start = self._starts[idx]
            part_end = self._ends[idx]
            local_offset = self._pos - part_start
            avail = part_end - self._pos
            to_read = min(avail, total_req - total_read)
            fh = self._handles[idx]
            fh.seek(local_offset, io.SEEK_SET)
            chunk = fh.read(to_read)
            if not chunk:
                break
            n = len(chunk)
            mv[total_read : total_read + n] = chunk
            total_read += n
            self._pos += n
        return total_read

    def close(self) -> None:
        if not self.closed:
            for fh in self._handles:
                try:
                    fh.close()
                except Exception:
                    pass
            self._handles.clear()
            super().close()


def sha256_and_crc32_file(path: Path) -> tuple[int, int, str]:
    h = hashlib.sha256()
    crc = 0
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            crc = zlib.crc32(chunk, crc)
            h.update(chunk)
    return total, (crc & 0xFFFFFFFF), h.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def atomic_copy_verified(src: Path, dst: Path, expected_sha256: str) -> str:
    src_size, _, src_sha = sha256_and_crc32_file(src)
    if src_sha != expected_sha256:
        raise RuntimeError(f"Source SHA-256 mismatch for {src}: got {src_sha}, expected {expected_sha256}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst_size, _, dst_sha = sha256_and_crc32_file(dst)
        if dst_sha == expected_sha256 and dst_size == src_size:
            return "verified_existing"
        raise RuntimeError(
            f"Existing destination file {dst} has mismatched digest/size: "
            f"got sha256={dst_sha} size={dst_size}, expected sha256={expected_sha256} size={src_size}"
        )
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        shutil.copyfile(src, tmp)
        tmp_size, _, tmp_sha = sha256_and_crc32_file(tmp)
        if tmp_sha != expected_sha256 or tmp_size != src_size:
            raise RuntimeError(f"Copied temp SHA-256 mismatch for {dst}: {tmp_sha}")
        os.replace(tmp, dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    # Independent on-disk re-read verification
    final_size, _, final_sha = sha256_and_crc32_file(dst)
    if final_sha != expected_sha256 or final_size != src_size:
        raise RuntimeError(f"Post-replace verification failed for {dst}: {final_sha}")
    return "copied_and_verified"


def validate_output_dir(repo_root: Path, output_dir: Path) -> None:
    out_resolved = output_dir.resolve()
    for forbidden in FORBIDDEN_OUTPUT_PARENTS:
        forb_resolved = (repo_root / forbidden).resolve()
        if out_resolved == forb_resolved or forb_resolved in out_resolved.parents:
            raise ValueError(
                f"Refusing to write inside protected directory '{forbidden}': {out_resolved}"
            )


def snapshot_protected_state(repo_root: Path) -> dict[str, Any]:
    file_digests: dict[str, dict[str, Any]] = {}
    for rel in PROTECTED_ANCHOR_FILES:
        p = repo_root / rel
        if not p.is_file():
            raise FileNotFoundError(f"Protected anchor file missing: {p}")
        size, _, digest = sha256_and_crc32_file(p)
        file_digests[rel] = {"size_bytes": size, "sha256": digest}

    dir_stats: dict[str, dict[str, int]] = {}
    for rel in PROTECTED_DIRS:
        d = repo_root / rel
        if not d.is_dir():
            raise FileNotFoundError(f"Protected directory missing: {d}")
        files = [x for x in d.rglob("*") if x.is_file()]
        dir_stats[rel] = {
            "file_count": len(files),
            "total_bytes": sum(x.stat().st_size for x in files),
        }
    return {"files": file_digests, "dirs": dir_stats}


def discover_split_parts(archive_dir: Path) -> list[Path]:
    parts = [archive_dir / f"checkpoints_200ep.zip.{i:03d}" for i in range(1, EXPECTED_PART_COUNT + 1)]
    for p in parts:
        if not p.is_file():
            raise FileNotFoundError(f"Missing split archive part: {p}")
        if p.stat().st_size <= 0:
            raise RuntimeError(f"Empty split archive part: {p}")
    extra = sorted(
        p.name for p in archive_dir.glob("checkpoints_200ep.zip.*") if p not in set(parts)
    )
    if extra:
        raise RuntimeError(f"Unexpected extra split archive parts found: {extra}")
    total_bytes = sum(p.stat().st_size for p in parts)
    if total_bytes != EXPECTED_COMBINED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Split archive total size {total_bytes} != expected {EXPECTED_COMBINED_ARCHIVE_BYTES}"
        )
    return parts


def verify_local_file_header(part_paths: list[Path], info: zipfile.ZipInfo) -> dict[str, Any]:
    with SplitZipReader(part_paths) as raw:
        raw.seek(info.header_offset, io.SEEK_SET)
        hdr = raw.read(30)
        if len(hdr) != 30:
            raise RuntimeError(f"Truncated local file header for {info.filename}")
        (
            sig,
            _ver_needed,
            flag_bits,
            comp_method,
            _mod_time,
            _mod_date,
            lh_crc,
            lh_comp_sz,
            lh_uncomp_sz,
            fname_len,
            extra_len,
        ) = struct.unpack("<4sHHHHHIIIHH", hdr)
        lh_fname = raw.read(fname_len).decode("utf-8", errors="strict")
    if sig != b"PK\x03\x04":
        raise RuntimeError(f"Bad local header signature for {info.filename}: {sig!r}")
    if lh_fname != info.filename:
        raise RuntimeError(f"Local header filename {lh_fname!r} != CD filename {info.filename!r}")
    if flag_bits != 0:
        raise RuntimeError(f"Unexpected flag_bits={flag_bits:#x} for {info.filename}; expected 0")
    if lh_crc != info.CRC:
        raise RuntimeError(f"Local CRC {lh_crc:#010x} != CD CRC {info.CRC:#010x} for {info.filename}")
    if lh_uncomp_sz != info.file_size:
        raise RuntimeError(
            f"Local uncomp size {lh_uncomp_sz} != CD file_size {info.file_size} for {info.filename}"
        )
    return {
        "header_offset": info.header_offset,
        "flag_bits": flag_bits,
        "compress_type": comp_method,
        "local_crc32_hex": f"0x{lh_crc:08x}",
        "local_compress_size": lh_comp_sz,
        "local_file_size": lh_uncomp_sz,
        "extra_len": extra_len,
    }


def extract_and_verify_targets(
    archive_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parts = discover_split_parts(archive_dir)
    inventory: list[dict[str, Any]] = []
    zip_best_metadata: dict[str, Any] = {}
    recovered: dict[str, Any] = {}

    with SplitZipReader(parts) as raw:
        with io.BufferedReader(raw, buffer_size=CHUNK_SIZE) as buf:
            with zipfile.ZipFile(buf, "r") as zf:
                infos = zf.infolist()
                names = [info.filename for info in infos]
                for name in names:
                    if "/" in name or "\\" in name or ".." in name or os.path.isabs(name):
                        raise RuntimeError(f"Unsafe/non-flat archive member name: {name!r}")
                if len(names) != len(EXPECTED_ARCHIVE_MEMBERS) or set(names) != EXPECTED_ARCHIVE_MEMBERS:
                    raise RuntimeError(
                        f"Archive member inventory mismatch: got {sorted(names)}, "
                        f"expected {sorted(EXPECTED_ARCHIVE_MEMBERS)}"
                    )

                for info in infos:
                    inventory.append(
                        {
                            "filename": info.filename,
                            "file_size": info.file_size,
                            "compress_size": info.compress_size,
                            "compress_type": info.compress_type,
                            "flag_bits": info.flag_bits,
                            "crc32_hex": f"0x{info.CRC:08x}",
                            "zip_date_time": f"{info.date_time[0]:04d}-{info.date_time[1]:02d}-{info.date_time[2]:02d} {info.date_time[3]:02d}:{info.date_time[4]:02d}:{info.date_time[5]:02d}",
                        }
                    )

                # Cross-check rasnet_best.pth inside ZIP against disk (purely in-memory)
                for best_name, expected_best_sha in (
                    ("rasnet_best.pth", EXPECTED_RASNET_BEST_SHA256),
                    ("nnunet_best.pt", EXPECTED_NNUNET_BEST_SHA256),
                ):
                    b_info = zf.getinfo(best_name)
                    h = hashlib.sha256()
                    crc = 0
                    b_sz = 0
                    with zf.open(b_info, "r") as src:
                        while True:
                            chunk = src.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            b_sz += len(chunk)
                            crc = zlib.crc32(chunk, crc)
                            h.update(chunk)
                    crc &= 0xFFFFFFFF
                    sha = h.hexdigest()
                    if b_sz != b_info.file_size or crc != b_info.CRC or sha != expected_best_sha:
                        raise RuntimeError(
                            f"In-archive best checkpoint mismatch for {best_name}: "
                            f"size={b_sz}/{b_info.file_size}, crc={crc:#x}/{b_info.CRC:#x}, sha={sha}"
                        )
                    zip_best_metadata[best_name] = {
                        "file_size": b_sz,
                        "crc32_hex": f"0x{crc:08x}",
                        "sha256": sha,
                        "matches_disk_best": True,
                        "zip_date_time": f"{b_info.date_time[0]:04d}-{b_info.date_time[1]:02d}-{b_info.date_time[2]:02d} {b_info.date_time[3]:02d}:{b_info.date_time[4]:02d}:{b_info.date_time[5]:02d}",
                    }

                # Stream-extract ONLY rasnet_last.pth and nnunet_last.pt
                for target_name, spec in TARGET_SPECS.items():
                    info = zf.getinfo(target_name)
                    lh_info = verify_local_file_header(parts, info)
                    if info.file_size != spec["size"]:
                        raise RuntimeError(
                            f"CD size mismatch for {target_name}: {info.file_size} != {spec['size']}"
                        )
                    if info.CRC != spec["crc32"]:
                        raise RuntimeError(
                            f"CD CRC mismatch for {target_name}: {info.CRC:#010x} != {spec['crc32']:#010x}"
                        )

                    dest = output_dir / target_name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        disk_size, disk_crc, disk_sha = sha256_and_crc32_file(dest)
                        if (
                            disk_size != spec["size"]
                            or disk_crc != spec["crc32"]
                            or disk_sha != spec["sha256"]
                        ):
                            raise RuntimeError(
                                f"Existing {dest} failed integrity verification: "
                                f"size={disk_size}, crc={disk_crc:#010x}, sha256={disk_sha}"
                            )
                        action = "verified_existing"
                        computed_crc = disk_crc
                        computed_sha = disk_sha
                        bytes_written = disk_size
                    else:
                        tmp = dest.with_name(dest.name + ".tmp")
                        h = hashlib.sha256()
                        crc = 0
                        bytes_written = 0
                        try:
                            with zf.open(info, "r") as zsrc, tmp.open("wb") as dst_f:
                                while True:
                                    chunk = zsrc.read(CHUNK_SIZE)
                                    if not chunk:
                                        break
                                    dst_f.write(chunk)
                                    bytes_written += len(chunk)
                                    crc = zlib.crc32(chunk, crc)
                                    h.update(chunk)
                                dst_f.flush()
                                os.fsync(dst_f.fileno())
                            computed_crc = crc & 0xFFFFFFFF
                            computed_sha = h.hexdigest()
                            if bytes_written != spec["size"]:
                                raise RuntimeError(
                                    f"Extracted size mismatch for {target_name}: {bytes_written} != {spec['size']}"
                                )
                            if computed_crc != spec["crc32"]:
                                raise RuntimeError(
                                    f"Extracted CRC mismatch for {target_name}: {computed_crc:#010x} != {spec['crc32']:#010x}"
                                )
                            if computed_sha != spec["sha256"]:
                                raise RuntimeError(
                                    f"Extracted SHA-256 mismatch for {target_name}: {computed_sha} != {spec['sha256']}"
                                )
                            os.replace(tmp, dest)
                        except Exception:
                            tmp.unlink(missing_ok=True)
                            raise
                        # Independent on-disk re-read check
                        r_size, r_crc, r_sha = sha256_and_crc32_file(dest)
                        if r_size != spec["size"] or r_crc != spec["crc32"] or r_sha != spec["sha256"]:
                            raise RuntimeError(f"Post-replace re-read failed for {dest}")
                        action = "extracted_from_split_archive"

                    key = target_name.split(".")[0]
                    recovered[key] = {
                        "filename": target_name,
                        "path": str(dest),
                        "action": action,
                        "size_bytes": bytes_written,
                        "expected_crc32_hex": f"0x{spec['crc32']:08x}",
                        "computed_crc32_hex": f"0x{computed_crc:08x}",
                        "crc_match": True,
                        "expected_sha256": spec["sha256"],
                        "actual_sha256": computed_sha,
                        "sha256_match": True,
                        "zip_timestamp": f"{info.date_time[0]:04d}-{info.date_time[1]:02d}-{info.date_time[2]:02d} {info.date_time[3]:02d}:{info.date_time[4]:02d}:{info.date_time[5]:02d}",
                        "local_header": lh_info,
                    }

    split_meta = {
        "directory": str(archive_dir),
        "part_count": len(parts),
        "part_files": [p.name for p in parts],
        "combined_bytes": EXPECTED_COMBINED_ARCHIVE_BYTES,
        "recovery_method": "seekable_split_zip_stream_no_combined_zip_no_extractall",
        "members_inventory": inventory,
        "archive_best_cross_check": zip_best_metadata,
    }
    return split_meta, recovered, zip_best_metadata


def setup_isolated_nnunet_folder(repo_root: Path, output_dir: Path, nnunet_last_path: Path) -> dict[str, Any]:
    src_model_dir = (
        repo_root
        / "Q1_Publication_Package"
        / "matched_200ep_benchmark"
        / "nnUNet_results"
        / "Dataset501_CoronarySeg"
        / "nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres"
    )
    native_dir = (
        output_dir
        / "nnunet_native_model"
        / "Dataset501_CoronarySeg"
        / "nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres"
    )
    plans_action = atomic_copy_verified(
        src_model_dir / "plans.json",
        native_dir / "plans.json",
        EXPECTED_PLANS_SHA256,
    )
    dataset_action = atomic_copy_verified(
        src_model_dir / "dataset.json",
        native_dir / "dataset.json",
        EXPECTED_DATASET_SHA256,
    )
    ckpt_action = atomic_copy_verified(
        nnunet_last_path,
        native_dir / "fold_0" / "checkpoint_final.pth",
        TARGET_SPECS["nnunet_last.pt"]["sha256"],
    )
    return {
        "path": str(native_dir),
        "plans_path": str(native_dir / "plans.json"),
        "plans_action": plans_action,
        "plans_sha256": EXPECTED_PLANS_SHA256,
        "dataset_path": str(native_dir / "dataset.json"),
        "dataset_action": dataset_action,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "checkpoint_final_path": str(native_dir / "fold_0" / "checkpoint_final.pth"),
        "checkpoint_final_action": ckpt_action,
        "checkpoint_final_sha256": TARGET_SPECS["nnunet_last.pt"]["sha256"],
    }


def verify_nnunet_provenance(
    repo_root: Path,
    nnunet_last_path: Path,
    native_info: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from benchmark.adapters.nnunet_adapter import ensure_nnunet_trainer_registered

    ckpt = torch.load(str(nnunet_last_path), map_location="cpu", weights_only=False)
    current_epoch = int(ckpt["current_epoch"])
    if current_epoch != 200:
        raise RuntimeError(f"Expected nnunet_last.pt current_epoch == 200, got {current_epoch}")

    trainer_name = ckpt["trainer_name"]
    if trainer_name != "nnUNetTrainer_200epochs":
        raise RuntimeError(f"Unexpected trainer_name: {trainer_name}")

    init_args = ckpt["init_args"]
    configuration = init_args["configuration"]
    fold = int(init_args["fold"])
    mirroring_axes = tuple(ckpt["inference_allowed_mirroring_axes"])
    if configuration != "3d_fullres":
        raise RuntimeError(f"Unexpected configuration: {configuration}")
    if fold != 0:
        raise RuntimeError(f"Unexpected fold: {fold}")
    if mirroring_axes != (0, 1, 2):
        raise RuntimeError(f"Unexpected mirroring axes: {mirroring_axes}")

    disk_plans = json.loads(Path(native_info["plans_path"]).read_text(encoding="utf-8"))
    disk_dataset = json.loads(Path(native_info["dataset_path"]).read_text(encoding="utf-8"))
    if init_args["plans"] != disk_plans:
        raise RuntimeError("Checkpoint init_args['plans'] does not match on-disk plans.json")
    if init_args["dataset_json"] != disk_dataset:
        raise RuntimeError("Checkpoint init_args['dataset_json'] does not match on-disk dataset.json")

    network_weights = ckpt["network_weights"]
    if len(network_weights) != 292:
        raise RuntimeError(f"Expected 292 network_weights keys, got {len(network_weights)}")

    # Verify trainer 0-based+1 convention directly in installed nnunetv2
    on_train_end_src = inspect.getsource(nnUNetTrainer.on_train_end)
    save_ckpt_src = inspect.getsource(nnUNetTrainer.save_checkpoint)
    if "self.current_epoch -= 1" not in on_train_end_src or "'current_epoch': self.current_epoch + 1" not in save_ckpt_src:
        raise RuntimeError("Could not confirm nnUNetTrainer 0-based+1 current_epoch saving convention")

    # Verify logging arrays & zero timestamp gaps > 1s
    logging_dict = ckpt["logging"]
    expected_log_keys = (
        "mean_fg_dice",
        "ema_fg_dice",
        "dice_per_class_or_region",
        "train_losses",
        "val_losses",
        "lrs",
        "epoch_start_timestamps",
        "epoch_end_timestamps",
    )
    key_lengths = {}
    for k in expected_log_keys:
        if k not in logging_dict:
            raise RuntimeError(f"Missing logging key: {k}")
        key_lengths[k] = len(logging_dict[k])
        if key_lengths[k] != 200:
            raise RuntimeError(f"Logging key {k} length {key_lengths[k]} != 200")

    starts = logging_dict["epoch_start_timestamps"]
    ends = logging_dict["epoch_end_timestamps"]
    gaps = [float(starts[i] - ends[i - 1]) for i in range(1, len(starts))]
    gaps_gt_1s = sum(1 for g in gaps if g > 1.0)
    if gaps_gt_1s != 0:
        raise RuntimeError(f"Found {gaps_gt_1s} inter-epoch timestamp gaps > 1s in nnunet_last.pt")

    # Compare against logs/nnunet_200ep_log.csv
    csv_path = repo_root / "Q1_Publication_Package" / "matched_200ep_benchmark" / "logs" / "nnunet_200ep_log.csv"
    valid_rows: list[dict[str, float]] = []
    empty_marker_epochs: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["train_loss"] == "" or row["val_dice"] == "":
                empty_marker_epochs.append(int(row["epoch"]))
            else:
                valid_rows.append(
                    {
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"]),
                        "val_dice": float(row["val_dice"]),
                        "learning_rate": float(row["learning_rate"]),
                    }
                )
    if len(valid_rows) != 200:
        raise RuntimeError(f"Expected 200 valid rows in nnunet_200ep_log.csv, got {len(valid_rows)}")
    if empty_marker_epochs != [51, 101, 151, 201]:
        raise RuntimeError(f"Unexpected empty resume-marker epochs in nnunet_200ep_log.csv: {empty_marker_epochs}")

    max_diff_train = 0.0
    max_diff_val = 0.0
    max_diff_dice = 0.0
    max_diff_lr = 0.0
    max_rnd_diff_train = 0.0
    max_rnd_diff_val = 0.0
    max_rnd_diff_dice = 0.0
    max_rnd_diff_lr = 0.0
    for idx, row in enumerate(valid_rows):
        if row["epoch"] != idx + 1:
            raise RuntimeError(f"Non-contiguous nnU-Net log epoch at idx {idx}: {row['epoch']}")
        c_tr = float(logging_dict["train_losses"][idx])
        c_va = float(logging_dict["val_losses"][idx])
        c_di = float(logging_dict["mean_fg_dice"][idx])
        c_lr = float(logging_dict["lrs"][idx])
        max_diff_train = max(max_diff_train, abs(row["train_loss"] - c_tr))
        max_diff_val = max(max_diff_val, abs(row["val_loss"] - c_va))
        max_diff_dice = max(max_diff_dice, abs(row["val_dice"] - c_di))
        max_diff_lr = max(max_diff_lr, abs(row["learning_rate"] - c_lr))
        max_rnd_diff_train = max(max_rnd_diff_train, abs(row["train_loss"] - round(c_tr, 4)))
        max_rnd_diff_val = max(max_rnd_diff_val, abs(row["val_loss"] - round(c_va, 4)))
        max_rnd_diff_dice = max(max_rnd_diff_dice, abs(row["val_dice"] - round(c_di, 4)))
        max_rnd_diff_lr = max(max_rnd_diff_lr, abs(row["learning_rate"] - round(c_lr, 5)))

    if (
        max(max_diff_train, max_diff_val, max_diff_dice) >= 5.001e-5
        or max_diff_lr >= 5.001e-6
        or max(max_rnd_diff_train, max_rnd_diff_val, max_rnd_diff_dice, max_rnd_diff_lr) >= 1e-6
    ):
        raise RuntimeError(
            f"nnunet_200ep_log.csv discrepancy beyond 4/5-decimal rounding: "
            f"raw=(train={max_diff_train}, val={max_diff_val}, dice={max_diff_dice}, lr={max_diff_lr}), "
            f"rounded=(train={max_rnd_diff_train}, val={max_rnd_diff_val}, dice={max_rnd_diff_dice}, lr={max_rnd_diff_lr})"
        )

    # Compare against nnunet_best.pt
    best_path = repo_root / "Q1_Publication_Package" / "matched_200ep_benchmark" / "checkpoints" / "nnunet_best.pt"
    best_ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
    best_epoch = int(best_ckpt["current_epoch"])
    if best_epoch != 199:
        raise RuntimeError(f"Expected nnunet_best.pt current_epoch == 199, got {best_epoch}")
    best_weights = best_ckpt["network_weights"]
    changed_keys = 0
    max_abs_diff = 0.0
    total_l2_sq = 0.0
    for k, t_last in network_weights.items():
        t_best = best_weights[k]
        if not torch.equal(t_last, t_best):
            changed_keys += 1
            diff = (t_last.float() - t_best.float()).abs()
            max_abs_diff = max(max_abs_diff, float(diff.max().item()))
            total_l2_sq += float((diff * diff).sum().item())

    # Initialize nnUNetPredictor on CPU
    ensure_nnunet_trainer_registered()
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        native_info["path"],
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )

    net_class_name = type(predictor.network).__name__
    unique_params = sum(p.numel() for p in predictor.network.parameters())
    patch_size = [int(x) for x in predictor.configuration_manager.patch_size]
    spacing = [float(x) for x in predictor.configuration_manager.spacing]
    norm_schemes = list(predictor.configuration_manager.normalization_schemes)
    pred_mirror = tuple(predictor.allowed_mirroring_axes)
    num_heads = int(predictor.label_manager.num_segmentation_heads)

    if net_class_name != "PlainConvUNet":
        raise RuntimeError(f"Expected PlainConvUNet, got {net_class_name}")
    if unique_params != 30_785_994:
        raise RuntimeError(f"Expected 30,785,994 unique params, got {unique_params}")
    if len(predictor.list_of_parameters) != 1 or len(predictor.list_of_parameters[0]) != 292:
        raise RuntimeError("Unexpected predictor.list_of_parameters structure")
    if patch_size != [96, 160, 160]:
        raise RuntimeError(f"Unexpected patch_size: {patch_size}")
    if spacing != [0.5, 0.349609375, 0.349609375]:
        raise RuntimeError(f"Unexpected spacing: {spacing}")
    if norm_schemes != ["CTNormalization"]:
        raise RuntimeError(f"Unexpected normalization_schemes: {norm_schemes}")
    if pred_mirror != (0, 1, 2):
        raise RuntimeError(f"Unexpected predictor.allowed_mirroring_axes: {pred_mirror}")
    if num_heads != 2:
        raise RuntimeError(f"Expected num_segmentation_heads == 2, got {num_heads}")

    return {
        "checkpoint_role": "final_epoch200",
        "epoch_identity_supported_by": "embedded_checkpoint_metadata_and_trainer_convention",
        "embedded_current_epoch": current_epoch,
        "trainer_epoch_convention": "0-based internal current_epoch (0..199); on_train_end() decrements by 1 and save_checkpoint() writes current_epoch + 1 == 200",
        "trainer_name": trainer_name,
        "configuration": configuration,
        "fold": fold,
        "inference_allowed_mirroring_axes": list(mirroring_axes),
        "architecture_class": net_class_name,
        "parameter_count": unique_params,
        "network_weights_key_count": len(network_weights),
        "logging_key_lengths": key_lengths,
        "inter_epoch_gaps_gt_1s": gaps_gt_1s,
        "total_training_hours_from_timestamps": float((ends[-1] - starts[0]) / 3600.0),
        "csv_log_verification": {
            "path": str(csv_path),
            "valid_rows": len(valid_rows),
            "empty_resume_marker_rows": empty_marker_epochs,
            "max_abs_diff_train_loss_raw": max_diff_train,
            "max_abs_diff_val_loss_raw": max_diff_val,
            "max_abs_diff_val_dice_raw": max_diff_dice,
            "max_abs_diff_lr_raw": max_diff_lr,
            "max_abs_diff_train_loss_rounded4": max_rnd_diff_train,
            "max_abs_diff_val_loss_rounded4": max_rnd_diff_val,
            "max_abs_diff_val_dice_rounded4": max_rnd_diff_dice,
            "max_abs_diff_lr_rounded5": max_rnd_diff_lr,
        },
        "final_epoch_metrics_from_checkpoint": {
            "mean_fg_dice": float(logging_dict["mean_fg_dice"][-1]),
            "ema_fg_dice": float(logging_dict["ema_fg_dice"][-1]),
            "train_loss": float(logging_dict["train_losses"][-1]),
            "val_loss": float(logging_dict["val_losses"][-1]),
            "lr": float(logging_dict["lrs"][-1]),
        },
        "best_ema_fg_dice": float(ckpt.get("_best_ema")),
        "diff_vs_nnunet_best": {
            "best_checkpoint_path": str(best_path),
            "best_checkpoint_sha256": EXPECTED_NNUNET_BEST_SHA256,
            "best_current_epoch": best_epoch,
            "total_keys": len(network_weights),
            "changed_keys": changed_keys,
            "max_abs_diff": max_abs_diff,
            "total_l2_diff": math.sqrt(total_l2_sq),
        },
        "native_model_folder": {
            **native_info,
            "plans_matches_checkpoint_init_args": True,
            "dataset_matches_checkpoint_init_args": True,
            "predictor_initialized": True,
            "predictor_trainer_name": predictor.trainer_name,
            "predictor_patch_size": patch_size,
            "predictor_spacing": spacing,
            "predictor_normalization_schemes": norm_schemes,
            "predictor_num_segmentation_heads": num_heads,
        },
    }


def verify_rasnet_provenance(
    repo_root: Path,
    rasnet_last_path: Path,
    recovered_rasnet_meta: dict[str, Any],
    zip_best_metadata: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from Phase3_Local_Integration.rasnet_model import RASNet

    train_script_path = repo_root / "Q1_Publication_Package" / "matched_200ep_benchmark" / "train_rasnet_200ep.py"
    train_script_text = train_script_path.read_text(encoding="utf-8")
    train_script_lines = train_script_text.splitlines()
    best_save_lines = [i + 1 for i, ln in enumerate(train_script_lines) if "torch.save(model.state_dict(), best_ckpt_path)" in ln]
    last_save_lines = [i + 1 for i, ln in enumerate(train_script_lines) if "torch.save(model.state_dict(), last_ckpt_path)" in ln]
    if not best_save_lines or not last_save_lines:
        raise RuntimeError("Could not locate RASNet best/last torch.save statements in train_rasnet_200ep.py")
    _, _, train_script_sha = sha256_and_crc32_file(train_script_path)

    model_file_path = repo_root / "Phase3_Local_Integration" / "rasnet_model.py"
    _, _, model_file_sha = sha256_and_crc32_file(model_file_path)

    # Parse logs/rasnet_200ep_log.csv
    log_path = repo_root / "Q1_Publication_Package" / "matched_200ep_benchmark" / "logs" / "rasnet_200ep_log.csv"
    _, _, log_sha = sha256_and_crc32_file(log_path)
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 rows in rasnet_200ep_log.csv, got {len(rows)}")

    epochs = [int(r["epoch"]) for r in rows]
    if epochs != list(range(1, 201)):
        raise RuntimeError("rasnet_200ep_log.csv epochs are not contiguous 1..200")

    cum_hours = [float(r["cumulative_time_hours"]) for r in rows]
    if any(math.isnan(x) for x in cum_hours) or any(cum_hours[i] <= cum_hours[i - 1] for i in range(1, 200)):
        raise RuntimeError("rasnet_200ep_log.csv cumulative_time_hours is not strictly monotonic")

    lr_max_err = 0.0
    val_epochs: list[int] = []
    best_ep = -1
    best_val_dice = -1.0
    for idx, r in enumerate(rows):
        ep = int(r["epoch"])
        tr_loss = float(r["train_loss"])
        if math.isnan(tr_loss):
            raise RuntimeError(f"NaN train_loss at epoch {ep}")
        lr = float(r["learning_rate"])
        expected_lr = 1e-6 + 0.5 * (2e-4 - 1e-6) * (1.0 + math.cos(math.pi * (ep - 1) / 200.0))
        lr_max_err = max(lr_max_err, abs(lr - expected_lr))
        if r["val_dice"] != "":
            vd = float(r["val_dice"])
            if not math.isnan(vd):
                val_epochs.append(ep)
                if vd > best_val_dice:
                    best_val_dice = vd
                    best_ep = ep

    if lr_max_err >= 1e-12:
        raise RuntimeError(f"CosineAnnealingLR max error {lr_max_err} >= 1e-12")
    if len(val_epochs) != 12 or val_epochs != [20, 40, 50, 60, 80, 100, 120, 140, 150, 160, 180, 200]:
        raise RuntimeError(f"Unexpected RASNet validation epochs: {val_epochs}")
    if best_ep != 160 or abs(best_val_dice - 0.7769595980644226) >= 1e-12:
        raise RuntimeError(f"Unexpected RASNet best epoch/dice: ep={best_ep}, dice={best_val_dice}")

    final_val_dice = float(rows[-1]["val_dice"])
    if abs(final_val_dice - 0.7761295437812805) >= 1e-12:
        raise RuntimeError(f"Unexpected RASNet final epoch-200 val_dice: {final_val_dice}")

    # ZIP timestamp delta vs log elapsed time (ep 160 -> ep 200)
    t_last = dt.datetime.strptime(recovered_rasnet_meta["zip_timestamp"], "%Y-%m-%d %H:%M:%S")
    t_best = dt.datetime.strptime(zip_best_metadata["rasnet_best.pth"]["zip_date_time"], "%Y-%m-%d %H:%M:%S")
    zip_delta_s = (t_last - t_best).total_seconds()
    log_delta_s = (cum_hours[199] - cum_hours[159]) * 3600.0
    ts_discrepancy_s = abs(zip_delta_s - log_delta_s)
    if ts_discrepancy_s > 2.0:
        raise RuntimeError(
            f"ZIP timestamp delta {zip_delta_s}s disagrees with log elapsed {log_delta_s}s by {ts_discrepancy_s}s"
        )

    # Strict checkpoint loading & forward signature check
    sd_last = torch.load(str(rasnet_last_path), map_location="cpu", weights_only=True)
    if not isinstance(sd_last, collections.OrderedDict):
        raise RuntimeError(f"Expected OrderedDict in rasnet_last.pth, got {type(sd_last)}")
    if len(sd_last) != 114:
        raise RuntimeError(f"Expected 114 state_dict keys in rasnet_last.pth, got {len(sd_last)}")
    dtypes = sorted({str(v.dtype) for v in sd_last.values()})
    if dtypes != ["torch.float32"]:
        raise RuntimeError(f"Unexpected tensor dtypes in rasnet_last.pth: {dtypes}")
    all_finite = all(bool(torch.isfinite(v).all().item()) for v in sd_last.values())
    if not all_finite:
        raise RuntimeError("Non-finite tensor values detected in rasnet_last.pth")

    ctor_args = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
        "init_filters": 16,
        "dropout_prob": 0.1,
    }
    model = RASNet(**ctor_args)
    param_count = sum(p.numel() for p in model.parameters())
    buffer_count = sum(1 for _ in model.buffers())
    if param_count != 4_706_772:
        raise RuntimeError(f"Expected 4,706,772 parameters in RASNet, got {param_count}")
    if len(model.state_dict()) != 114:
        raise RuntimeError(f"Expected 114 keys in RASNet.state_dict(), got {len(model.state_dict())}")

    model.load_state_dict(sd_last, strict=True)

    dummy = torch.zeros((1, 1, 32, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        eval_out = model(dummy)
        if not isinstance(eval_out, torch.Tensor) or tuple(eval_out.shape) != (1, 2, 32, 32, 32):
            raise RuntimeError(f"Unexpected RASNet.eval() output: {type(eval_out)}")
        model.train()
        train_out = model(dummy)
        if not (isinstance(train_out, tuple) and len(train_out) == 2 and isinstance(train_out[0], torch.Tensor) and isinstance(train_out[1], list)):
            raise RuntimeError(f"Unexpected RASNet.train() output: {type(train_out)}")
        model.eval()

    # Diff vs rasnet_best.pth
    best_path = repo_root / "Q1_Publication_Package" / "matched_200ep_benchmark" / "checkpoints" / "rasnet_best.pth"
    sd_best = torch.load(str(best_path), map_location="cpu", weights_only=True)
    changed_keys = 0
    max_abs_diff = 0.0
    total_l2_sq = 0.0
    for k, t_last in sd_last.items():
        t_best = sd_best[k]
        if not torch.equal(t_last, t_best):
            changed_keys += 1
            diff = (t_last - t_best).abs()
            max_abs_diff = max(max_abs_diff, float(diff.max().item()))
            total_l2_sq += float((diff * diff).sum().item())

    if changed_keys != 114 or max_abs_diff <= 0.0:
        raise RuntimeError(f"Unexpected diff vs rasnet_best.pth: changed_keys={changed_keys}, max_abs_diff={max_abs_diff}")

    return {
        "checkpoint_role": "final_epoch200",
        "epoch_identity_supported_by": "script_log_archive_provenance",
        "embedded_epoch_metadata_present": False,
        "embedded_metadata_note": (
            "Raw OrderedDict state_dict saved via torch.save(model.state_dict(), last_ckpt_path) "
            f"in train_rasnet_200ep.py:{last_save_lines[0]}; epoch-200 identity is established by "
            "script saving logic, uninterrupted 200-epoch rasnet_200ep_log.csv, and ZIP archive "
            "timestamp alignment rather than embedded checkpoint metadata."
        ),
        "state_dict_format": "OrderedDict",
        "tensor_dtypes": dtypes,
        "state_dict_key_count": len(sd_last),
        "all_tensors_finite": all_finite,
        "architecture": {
            "module": "Phase3_Local_Integration.rasnet_model.RASNet",
            "source_sha256": model_file_sha,
            "constructor_args": ctor_args,
            "parameter_count": param_count,
            "buffer_count": buffer_count,
            "strict_load_passed": True,
            "eval_return_type": "torch.Tensor",
            "train_return_type": "tuple[torch.Tensor, list[torch.Tensor]]",
        },
        "diff_vs_rasnet_best": {
            "best_checkpoint_path": str(best_path),
            "best_checkpoint_sha256": EXPECTED_RASNET_BEST_SHA256,
            "best_zip_matches_disk": zip_best_metadata["rasnet_best.pth"]["matches_disk_best"],
            "total_keys": len(sd_last),
            "changed_keys": changed_keys,
            "max_abs_diff": max_abs_diff,
            "total_l2_diff": math.sqrt(total_l2_sq),
        },
        "training_script": {
            "path": str(train_script_path),
            "sha256": train_script_sha,
            "target_epochs": 200,
            "val_interval": 20,
            "best_save_lines": best_save_lines,
            "last_save_lines": last_save_lines,
        },
        "training_log": {
            "path": str(log_path),
            "sha256": log_sha,
            "row_count": len(rows),
            "epoch_min": epochs[0],
            "epoch_max": epochs[-1],
            "contiguous_epochs": True,
            "validation_epochs": val_epochs,
            "best_epoch": best_ep,
            "best_val_dice": best_val_dice,
            "final_epoch": 200,
            "final_val_dice": final_val_dice,
            "final_val_loss": float(rows[-1]["val_loss"]),
            "final_train_loss": float(rows[-1]["train_loss"]),
            "cosine_lr_max_abs_error": lr_max_err,
            "zip_timestamp_delta_seconds": zip_delta_s,
            "log_elapsed_ep160_to_ep200_seconds": log_delta_s,
            "timestamp_delta_discrepancy_seconds": ts_discrepancy_s,
        },
    }


def collect_environment(repo_root: Path) -> dict[str, Any]:
    import importlib.metadata
    import monai
    import nnunetv2
    import torch

    try:
        nnunetv2_ver = importlib.metadata.version("nnunetv2")
    except Exception:
        nnunetv2_ver = getattr(nnunetv2, "__version__", "unknown")

    git_head = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    git_dirty_tracked = bool(
        subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short", "--untracked-files=no"], text=True
        ).strip()
    )
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "nnunetv2_version": nnunetv2_ver,
        "git_head": git_head,
        "git_dirty_tracked": git_dirty_tracked,
    }


def render_provenance_report(payload: dict[str, Any]) -> str:
    ras_rec = payload["recovered_checkpoints"]["rasnet_last"]
    nnu_rec = payload["recovered_checkpoints"]["nnunet_last"]
    ras_prov = payload["rasnet_provenance"]
    nnu_prov = payload["nnunet_provenance"]
    native = nnu_prov["native_model_folder"]
    env = payload["environment"]

    lines = [
        "# V1 Epoch-200 Checkpoint Recovery & Provenance Report",
        "",
        f"- **Run ID:** `{payload['run_id']}`",
        f"- **Generated (UTC):** `{payload['generated_at_utc']}`",
        f"- **Overall Status:** `{payload['overall_status']}`",
        f"- **Git HEAD:** `{env['git_head']}` (tracked dirty: `{env['git_dirty_tracked']}`)",
        f"- **Environment:** Python `{env['python_version']}`, torch `{env['torch_version']}`, MONAI `{env['monai_version']}`, nnunetv2 `{env['nnunetv2_version']}`",
        "",
        "## 1. Seekable Split-Archive Recovery Integrity",
        "",
        f"Recovered **only** `rasnet_last.pth` and `nnunet_last.pt` from `{payload['split_archive']['directory']}` "
        f"(`10` parts, `{payload['split_archive']['combined_bytes']:,}` bytes) using `SplitZipReader(io.RawIOBase)` + "
        "`io.BufferedReader` + `zipfile.ZipFile` (no 900 MB combined ZIP created, no `extractall()` into historical folders).",
        "",
        "| Member | Action | Size (B) | CRC-32 (CD/LH/Computed) | SHA-256 | ZIP Timestamp |",
        "|---|---|---:|---|---|---|",
        f"| `rasnet_last.pth` | `{ras_rec['action']}` | {ras_rec['size_bytes']:,} | `{ras_rec['computed_crc32_hex']}` | `{ras_rec['actual_sha256']}` | `{ras_rec['zip_timestamp']}` |",
        f"| `nnunet_last.pt` | `{nnu_rec['action']}` | {nnu_rec['size_bytes']:,} | `{nnu_rec['computed_crc32_hex']}` | `{nnu_rec['actual_sha256']}` | `{nnu_rec['zip_timestamp']}` |",
        "",
        "## 2. nnU-Net V2 Epoch-200 Provenance & Isolated Native Folder",
        "",
        f"- **Checkpoint Role:** `{nnu_prov['checkpoint_role']}` (`{nnu_prov['epoch_identity_supported_by']}`)",
        f"- **Embedded `current_epoch`:** `{nnu_prov['embedded_current_epoch']}` ({nnu_prov['trainer_epoch_convention']})",
        f"- **Trainer / Config / Fold:** `{nnu_prov['trainer_name']}` / `{nnu_prov['configuration']}` / fold `{nnu_prov['fold']}`",
        f"- **Mirroring Axes:** `{tuple(nnu_prov['inference_allowed_mirroring_axes'])}`",
        f"- **Architecture:** `{nnu_prov['architecture_class']}` (`{nnu_prov['parameter_count']:,}` unique params, `{nnu_prov['network_weights_key_count']}` keys)",
        f"- **Logging Verification:** all 8 logging keys have length `{nnu_prov['logging_key_lengths']['mean_fg_dice']}`, `{nnu_prov['inter_epoch_gaps_gt_1s']}` inter-epoch gaps > 1s; "
        f"matches `nnunet_200ep_log.csv` (`{nnu_prov['csv_log_verification']['valid_rows']}` valid rows + `4` empty resume-marker rows `{nnu_prov['csv_log_verification']['empty_resume_marker_rows']}`) with "
        f"`max_raw_diff={max(nnu_prov['csv_log_verification']['max_abs_diff_train_loss_raw'], nnu_prov['csv_log_verification']['max_abs_diff_val_loss_raw'], nnu_prov['csv_log_verification']['max_abs_diff_val_dice_raw']):.2e}` "
        f"(4/5-decimal log rounding; `max_rounded_diff={max(nnu_prov['csv_log_verification']['max_abs_diff_train_loss_rounded4'], nnu_prov['csv_log_verification']['max_abs_diff_val_loss_rounded4'], nnu_prov['csv_log_verification']['max_abs_diff_val_dice_rounded4'], nnu_prov['csv_log_verification']['max_abs_diff_lr_rounded5']):.2e}`)",
        f"- **Diff vs `nnunet_best.pt` (`current_epoch={nnu_prov['diff_vs_nnunet_best']['best_current_epoch']}`):** "
        f"`{nnu_prov['diff_vs_nnunet_best']['changed_keys']}/{nnu_prov['diff_vs_nnunet_best']['total_keys']}` tensors differ, "
        f"`max_abs_diff={nnu_prov['diff_vs_nnunet_best']['max_abs_diff']:.6f}`, `total_l2_diff={nnu_prov['diff_vs_nnunet_best']['total_l2_diff']:.6f}`",
        f"- **Isolated Native Folder:** `{native['path']}`",
        f"  - `plans.json` (`{native['plans_action']}`): `{native['plans_sha256']}` (equals checkpoint `init_args['plans']`: `{native['plans_matches_checkpoint_init_args']}`)",
        f"  - `dataset.json` (`{native['dataset_action']}`): `{native['dataset_sha256']}` (equals checkpoint `init_args['dataset_json']`: `{native['dataset_matches_checkpoint_init_args']}`)",
        f"  - `fold_0/checkpoint_final.pth` (`{native['checkpoint_final_action']}`): `{native['checkpoint_final_sha256']}`",
        f"  - **`nnUNetPredictor` CPU Initialization:** `{native['predictor_initialized']}` (`patch_size={native['predictor_patch_size']}`, `spacing={native['predictor_spacing']}`, `normalization={native['predictor_normalization_schemes']}`)",
        "",
        "## 3. RASNet V1 Epoch-200 Provenance & Strict Loading",
        "",
        f"- **Checkpoint Role:** `{ras_prov['checkpoint_role']}` (`{ras_prov['epoch_identity_supported_by']}`)",
        f"- **Embedded Metadata Note:** **No embedded epoch/optimizer metadata is present in `rasnet_last.pth`** (`embedded_epoch_metadata_present={ras_prov['embedded_epoch_metadata_present']}`). "
        f"{ras_prov['embedded_metadata_note']}",
        f"- **Training Script & Log Provenance:** `train_rasnet_200ep.py` (SHA-256 `{ras_prov['training_script']['sha256']}`) saves `rasnet_best.pth` on validation improvement (line `{ras_prov['training_script']['best_save_lines'][0]}`) and unconditionally saves `rasnet_last.pth` after every epoch (line `{ras_prov['training_script']['last_save_lines'][0]}`). "
        f"`rasnet_200ep_log.csv` records `200` contiguous epochs (`1..200`), `12` validation evaluations, best epoch `160` (`val_dice={ras_prov['training_log']['best_val_dice']:.16f}`), final epoch `200` (`val_dice={ras_prov['training_log']['final_val_dice']:.16f}`), and exact `CosineAnnealingLR` alignment (`max_abs_err={ras_prov['training_log']['cosine_lr_max_abs_error']:.2e}`).",
        f"- **Archive Timestamp Alignment:** ZIP timestamp delta (`rasnet_last.pth` minus `rasnet_best.pth`) = `{ras_prov['training_log']['zip_timestamp_delta_seconds']:.2f} s` vs log elapsed time ep `160->200` = `{ras_prov['training_log']['log_elapsed_ep160_to_ep200_seconds']:.2f} s` (discrepancy `{ras_prov['training_log']['timestamp_delta_discrepancy_seconds']:.4f} s`, within 2s DOS ZIP timestamp granularity).",
        f"- **Strict Model Loading:** `OrderedDict` with `{ras_prov['state_dict_key_count']}` `float32` tensors (all finite), loaded with `strict=True` into `RASNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16, dropout_prob=0.1)` (`{ras_prov['architecture']['parameter_count']:,}` params). "
        f"`eval()` returns `{ras_prov['architecture']['eval_return_type']}`; `train()` returns `{ras_prov['architecture']['train_return_type']}`.",
        f"- **Diff vs `rasnet_best.pth`:** `{ras_prov['diff_vs_rasnet_best']['changed_keys']}/{ras_prov['diff_vs_rasnet_best']['total_keys']}` tensors differ, "
        f"`max_abs_diff={ras_prov['diff_vs_rasnet_best']['max_abs_diff']:.6f}`, `total_l2_diff={ras_prov['diff_vs_rasnet_best']['total_l2_diff']:.6f}`.",
        "",
        "## 4. Non-Interference & Historical Doc Discrepancies",
        "",
        f"- **Protected Paths Unmodified (Before == After):** `{payload['non_interference_checks']['protected_paths_verified_unmodified']}` (`{len(payload['non_interference_checks']['protected_hashes_before']['files'])}` anchor files and `{len(payload['non_interference_checks']['protected_hashes_before']['dirs'])}` protected directories verified).",
        "- **Known Historical Doc Discrepancies (read-only observation; `CHECKPOINTS.md` left untouched):**",
    ]
    for item in payload["documentation_discrepancies_noted"]:
        lines.append(f"  - {item}")
    lines.extend(
        [
            "",
            "## 5. Reproducibility Command",
            "",
            "```powershell",
            f"C:\\Thesis_RASNET_env\\bench\\Scripts\\python.exe {Path(__file__).resolve()}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover and verify V1 epoch-200 final checkpoints.")
    parser.add_argument("--repo-root", type=Path, default=Path(r"C:\Thesis_RASNET"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    validate_output_dir(repo_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_snapshot = snapshot_protected_state(repo_root)
    archive_dir = repo_root / "Q1_Publication_Package" / "matched_200ep_benchmark" / "checkpoints_best_and_last"

    split_meta, recovered, zip_best_metadata = extract_and_verify_targets(archive_dir, output_dir)
    native_info = setup_isolated_nnunet_folder(
        repo_root, output_dir, Path(recovered["nnunet_last"]["path"])
    )
    nnu_prov = verify_nnunet_provenance(repo_root, Path(recovered["nnunet_last"]["path"]), native_info)
    ras_prov = verify_rasnet_provenance(
        repo_root,
        Path(recovered["rasnet_last"]["path"]),
        recovered["rasnet_last"],
        zip_best_metadata,
    )
    post_snapshot = snapshot_protected_state(repo_root)

    if pre_snapshot != post_snapshot:
        raise RuntimeError("Non-interference check failed: protected files or directories changed during execution")

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": output_dir.name.replace("v1_epoch200_", ""),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_status": "PASS",
        "environment": collect_environment(repo_root),
        "split_archive": split_meta,
        "recovered_checkpoints": recovered,
        "rasnet_provenance": ras_prov,
        "nnunet_provenance": nnu_prov,
        "non_interference_checks": {
            "protected_paths_verified_unmodified": True,
            "protected_hashes_before": pre_snapshot,
            "protected_hashes_after": post_snapshot,
        },
        "documentation_discrepancies_noted": [
            "CHECKPOINTS.md:9 lists 400 keys for rasnet_best.pth; verified key count in both rasnet_best.pth and rasnet_last.pth is 114.",
            "CHECKPOINTS.md:13,31 lists 402 keys, patch_size=[128,192,192], spacing=[0.5,0.359375,0.359375] for nnU-Net V2; verified plans.json and checkpoints have 292 keys, patch_size=[96,160,160], spacing=[0.5,0.349609375,0.349609375], and nnunet_best.pt has current_epoch=199.",
        ],
    }

    json_path = output_dir / "provenance_verification.json"
    md_path = output_dir / "PROVENANCE_REPORT.md"
    atomic_write_text(json_path, json.dumps(payload, indent=2) + "\n")
    atomic_write_text(md_path, render_provenance_report(payload))

    summary = {
        "overall_status": payload["overall_status"],
        "rasnet_last_action": recovered["rasnet_last"]["action"],
        "rasnet_last_sha256": recovered["rasnet_last"]["actual_sha256"],
        "nnunet_last_action": recovered["nnunet_last"]["action"],
        "nnunet_last_sha256": recovered["nnunet_last"]["actual_sha256"],
        "nnunet_current_epoch": nnu_prov["embedded_current_epoch"],
        "nnunet_predictor_initialized": nnu_prov["native_model_folder"]["predictor_initialized"],
        "rasnet_strict_load": ras_prov["architecture"]["strict_load_passed"],
        "non_interference_ok": True,
        "provenance_json": str(json_path),
        "provenance_report": str(md_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
