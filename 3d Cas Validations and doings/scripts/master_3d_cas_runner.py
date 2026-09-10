r"""
master_3d_cas_runner.py — Master Unattended Orchestrator for 3D CAS Validation
Runs all 5 models (RASNet, SegResNet, V-Net, 3D U-Net, nnU-Net) sequentially to completion,
followed by automated statistical significance testing, failure analysis, stenosis sanity checks,
and master table / figure compilation.
"""

import os
import sys
import time
import subprocess
import pandas as pd

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(WORK_DIR, "results")
LOG_FILE = os.path.join(RESULTS_DIR, "master_pipeline.log")

MODELS = ["rasnet", "segresnet", "vnet", "3dunet", "nnunet"]
TOTAL_EVALUABLE_CASES = 134


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(formatted + "\n")


def get_completed_cases(model_name: str) -> int:
    csv_p = os.path.join(RESULTS_DIR, f"3d_cas_{model_name}_case_metrics.csv")
    if not os.path.exists(csv_p):
        return 0
    try:
        df = pd.read_csv(csv_p)
        return len(df)
    except Exception:
        return 0


def wait_for_rasnet():
    """Waits for currently running RASNet process to complete all 134 evaluable cases."""
    log("[*] Checking status of active RASNet evaluation...")
    while True:
        completed = get_completed_cases("rasnet")
        log(f"  [RASNet Progress] {completed}/{TOTAL_EVALUABLE_CASES} cases recorded in CSV.")
        if completed >= TOTAL_EVALUABLE_CASES:
            log("[OK] RASNet has finished all 134 cases!")
            break
        
        # Check if python process 04_evaluate_models_3d_cas.py is running
        try:
            out = subprocess.check_output(
                'powershell -Command "Get-CimInstance Win32_Process -Filter \\"Name=\'python.exe\'\\" | Select-Object -ExpandProperty CommandLine"',
                shell=True, text=True
            )
            if "04_evaluate_models_3d_cas.py" not in out:
                log("[!] RASNet process exited or not running. Resuming...")
                break
        except Exception:
            break

        time.sleep(15)


def run_command_with_logging(cmd: list, desc: str):
    log(f"[*] Starting: {desc}")
    log(f"    Command: {' '.join(cmd)}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        cmd,
        cwd=WORK_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        encoding="utf-8",
        errors="replace"
    )

    for line in proc.stdout:
        line_clean = line.strip()
        if line_clean:
            print(f"    {line_clean}", flush=True)
            with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"    {line_clean}\n")

    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log(f"[ERROR] {desc} failed with exit code {proc.returncode} after {elapsed:.1f}s")
        raise RuntimeError(f"Command failed: {desc}")
    log(f"[OK] Completed: {desc} in {elapsed:.1f}s ({elapsed/60.0:.2f} mins)")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log("=" * 80)
    log("[START] MASTER 3D CAS VALIDATION & EVALUATION PIPELINE")
    log("=" * 80)

    # Step 1: Wait for or complete RASNet
    wait_for_rasnet()
    completed_rasnet = get_completed_cases("rasnet")
    if completed_rasnet < TOTAL_EVALUABLE_CASES:
        log(f"[*] Resuming RASNet from case {completed_rasnet} to complete all {TOTAL_EVALUABLE_CASES}...")
        run_command_with_logging(
            [sys.executable, "-u", os.path.join(SCRIPT_DIR, "04_evaluate_models_3d_cas.py"), "--model", "rasnet", "--sw-batch-size", "4"],
            "RASNet Full 3D CAS Evaluation"
        )

    # Step 2: Run baseline models sequentially
    remaining_models = ["segresnet", "vnet", "3dunet", "nnunet"]
    for m in remaining_models:
        done = get_completed_cases(m)
        if done >= TOTAL_EVALUABLE_CASES:
            log(f"[OK] Model {m.upper()} is already completely evaluated ({done}/{TOTAL_EVALUABLE_CASES} cases). Skipping.")
            continue
        
        log(f"\n{'='*80}\n[*] Launching {m.upper()} Evaluation (Progress: {done}/{TOTAL_EVALUABLE_CASES})\n{'='*80}")
        run_command_with_logging(
            [sys.executable, "-u", os.path.join(SCRIPT_DIR, "04_evaluate_models_3d_cas.py"), "--model", m, "--sw-batch-size", "4"],
            f"{m.upper()} Full 3D CAS Evaluation"
        )

    # Step 3: Run downstream synthesis and analysis scripts
    log("\n" + "=" * 80)
    log("[*] ALL 5 MODELS EVALUATED! Running Downstream Statistical & Publication Synthesis...")
    log("=" * 80)

    # 3a. Statistical Significance
    run_command_with_logging(
        [sys.executable, "-u", os.path.join(SCRIPT_DIR, "05_statistical_analysis.py")],
        "Wilcoxon Significance Testing & Bootstrap 95% CIs"
    )

    # 3b. Failure Case Analysis
    run_command_with_logging(
        [sys.executable, "-u", os.path.join(SCRIPT_DIR, "08_failure_case_analysis.py")],
        "3D CAS Failure Case Analysis & Rendering"
    )

    # 3c. Stenosis Sanity Check
    run_command_with_logging(
        [sys.executable, "-u", os.path.join(SCRIPT_DIR, "09_stenosis_sanity_check.py")],
        "3D CAS Stenosis Luminal Narrowing Sanity Check"
    )

    # 3d. Master Tables and Figures
    run_command_with_logging(
        [sys.executable, "-u", os.path.join(SCRIPT_DIR, "12_master_tables_and_figures.py")],
        "Master Tables, Figures & FINAL_RESULTS_SUMMARY Compilation"
    )

    log("=" * 80)
    log("[COMPLETE] MASTER PIPELINE FINISHED SUCCESSFULLY! ALL 5 MODELS & DOWNSTREAM ARTIFACTS COMPLETE.")
    log("=" * 80)


if __name__ == "__main__":
    main()
