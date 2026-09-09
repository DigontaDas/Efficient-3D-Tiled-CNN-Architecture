#!/usr/bin/env python3
"""
Recombination and extraction utility for the 200-Epoch Matched Benchmark checkpoints.

Because GitHub enforces a 100 MB single-file upload limit, the authoritative 10 checkpoints
(best + last for SegResNet, 3D U-Net, V-Net, nnU-Net V2, and RASNet) are archived into a
multi-part split zip with parts under 95 MB (checkpoints_200ep.zip.001 ... .010).

Usage:
    python recombine_and_unzip.py
"""

import glob
import os
import sys
import zipfile

def recombine_and_extract(output_dir=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = current_dir

    parts = sorted(glob.glob(os.path.join(current_dir, "checkpoints_200ep.zip.[0-9][0-9][0-9]")))
    if not parts:
        print("[ERROR] No split parts found matching checkpoints_200ep.zip.XXX")
        sys.exit(1)

    recombined_zip = os.path.join(output_dir, "checkpoints_matched_200ep.zip")
    print(f"[*] Found {len(parts)} split parts. Recombining into {recombined_zip}...")

    total_bytes = 0
    with open(recombined_zip, "wb") as f_out:
        for part in parts:
            part_name = os.path.basename(part)
            size = os.path.getsize(part)
            total_bytes += size
            print(f"  -> Reading {part_name} ({size / (1024*1024):.2f} MB)...")
            with open(part, "rb") as f_in:
                f_out.write(f_in.read())

    print(f"[OK] Recombined {recombined_zip} ({total_bytes / (1024*1024):.2f} MB).")
    print("[*] Verifying zip integrity and extracting...")

    with zipfile.ZipFile(recombined_zip, "r") as zf:
        members = zf.namelist()
        print(f"  Archive contains {len(members)} models:")
        for m in members:
            info = zf.getinfo(m)
            print(f"    - {m:25s} ({info.file_size / (1024*1024):.2f} MB)")
        zf.extractall(output_dir)

    print("[OK] Extraction successful!")
    print(f"All 10 checkpoints extracted to: {output_dir}")

    # Remove temporary combined zip
    if os.path.exists(recombined_zip):
        os.remove(recombined_zip)
        print("[OK] Removed intermediate combined zip file.")

if __name__ == "__main__":
    recombine_and_extract()
