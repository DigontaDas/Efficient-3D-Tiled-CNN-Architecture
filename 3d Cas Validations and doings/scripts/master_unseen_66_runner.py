r"""
master_unseen_66_runner.py — Master Pipeline Orchestrator for Pure Unseen External Cohort (N=66)
Sequentially runs:
  1. RASNet evaluation across 66 unseen cases
  2. SegResNet evaluation across 66 unseen cases
  3. V-Net evaluation across 66 unseen cases
  4. nnU-Net evaluation across 66 unseen cases
  5. 3D U-Net evaluation across 66 unseen cases
  6. Stenosis & Block Detection Gallery (20 visual panels)
  7. Downstream Statistical Synthesis & Master Tables (Wilcoxon, 95% CIs)
"""

import os
import sys
import time
import subprocess
import datetime

PYTHON_EXE = sys.executable
WORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(WORK_DIR, "scripts")


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def run_command(cmd_list, desc: str):
    log(f"\n{'='*90}\n[*] STARTING STEP: {desc}\n[*] Command: {' '.join(cmd_list)}\n{'='*90}")
    t0 = time.time()
    res = subprocess.run(cmd_list, cwd=WORK_DIR)
    elapsed = round((time.time() - t0) / 60.0, 2)
    if res.returncode != 0:
        log(f"[!] FAILED: {desc} exited with code {res.returncode} after {elapsed} mins.")
        sys.exit(res.returncode)
    log(f"[+] COMPLETED: {desc} in {elapsed} mins.\n")


def main():
    log("=" * 90)
    log("[*] LAUNCHING MASTER PIPELINE: Pure Unseen External Cohort (N=66)")
    log("=" * 90)

    total_start = time.time()

    # Step 1: RASNet
    run_command([PYTHON_EXE, "-u", os.path.join(SCRIPTS_DIR, "evaluate_unseen_66_models.py"), "--model", "rasnet"], "RASNet Unseen 66 Evaluation")

    # Step 2: SegResNet
    run_command([PYTHON_EXE, "-u", os.path.join(SCRIPTS_DIR, "evaluate_unseen_66_models.py"), "--model", "segresnet"], "SegResNet Unseen 66 Evaluation")

    # Step 3: V-Net
    run_command([PYTHON_EXE, "-u", os.path.join(SCRIPTS_DIR, "evaluate_unseen_66_models.py"), "--model", "vnet"], "V-Net Unseen 66 Evaluation")

    # Step 4: nnU-Net
    run_command([PYTHON_EXE, "-u", os.path.join(SCRIPTS_DIR, "evaluate_unseen_66_models.py"), "--model", "nnunet"], "nnU-Net Unseen 66 Evaluation")

    # Step 4b: RASNet (Threshold 0.5 - Pure Comparison)
    run_command([PYTHON_EXE, "-u", os.path.join(SCRIPTS_DIR, "evaluate_unseen_66_models.py"), "--model", "rasnet", "--threshold", "0.5"], "RASNet Threshold 0.5 Unseen 66 Evaluation")

    # Step 5: 3D U-Net
    run_command([PYTHON_EXE, "-u", os.path.join(SCRIPTS_DIR, "evaluate_unseen_66_models.py"), "--model", "3dunet"], "3D U-Net Unseen 66 Evaluation")

    # Step 6: Stenosis & Block Detection Gallery (20 visual panels)
    run_command([PYTHON_EXE, "-u", os.path.join(SCRIPTS_DIR, "unseen_66_stenosis_detection.py")], "20-Case Visual Stenosis Block Detection Gallery")

    # Step 7: Downstream Statistical Synthesis & Master Tables
    run_command([PYTHON_EXE, "-u", os.path.join(SCRIPTS_DIR, "unseen_66_statistical_synthesis.py")], "Statistical Synthesis & Master Tables")

    total_elapsed = round((time.time() - total_start) / 60.0, 2)
    log("=" * 90)
    log(f"[+] MASTER UNSEEN COHORT PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed} MINS!")
    log("=" * 90)


if __name__ == "__main__":
    main()
