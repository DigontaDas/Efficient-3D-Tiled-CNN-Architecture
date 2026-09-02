"""
08b_export_training_metrics.py
==============================
Parses the verified champion RASNet 70-epoch training log and exports
both CSV and Markdown tables of per-epoch metrics.
"""

import os
import re
import pandas as pd

LOG_PATH = r"H:\Thesis_Trainings\_Archive_Historical_Experiments_and_Logs\Antigravity-brain-from-lab-pc\Antigravity-brain-from-lab-pc\701162ec-7b45-4eab-8e23-cc134a37a12c\.system_generated\tasks\task-170.log"
OUT_CSV_1 = r"H:\Thesis_Trainings\results-after-hallucin-fix\training_metrics_per_epoch.csv"
OUT_MD_1  = r"H:\Thesis_Trainings\results-after-hallucin-fix\training_metrics_per_epoch.md"
OUT_CSV_2 = r"H:\Thesis_Trainings\Q1_Publication_Package\stats\training_metrics_per_epoch.csv"
OUT_MD_2  = r"H:\Thesis_Trainings\Q1_Publication_Package\stats\training_metrics_per_epoch.md"

def parse_and_export():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

    pattern = re.compile(r'^\s*(\d+)\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*MB')
    
    rows = []
    with open(LOG_PATH, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                epoch = int(match.group(1))
                time_sec = float(match.group(2))
                loss = float(match.group(3))
                lr = float(match.group(4))
                vram_mb = float(match.group(5))
                rows.append({
                    "Epoch": epoch,
                    "Time_Seconds": round(time_sec, 3),
                    "Training_Loss": round(loss, 4),
                    "Learning_Rate": lr,
                    "Max_VRAM_MB": round(vram_mb, 1)
                })

    df = pd.DataFrame(rows)
    print(f"[OK] Parsed {len(df)} epochs from training log.")
    assert len(df) == 70, f"Expected 70 epochs, found {len(df)}"

    best_row = df.loc[df['Training_Loss'].idxmin()]
    print(f"Best Loss: {best_row['Training_Loss']} at Epoch {int(best_row['Epoch'])}")
    assert best_row['Training_Loss'] == 0.1099, f"Expected best loss 0.1099, got {best_row['Training_Loss']}"
    assert int(best_row['Epoch']) == 41, f"Expected best epoch 41, got {int(best_row['Epoch'])}"

    # Add cumulative training time (hours)
    df["Cumulative_Time_Hours"] = (df["Time_Seconds"].cumsum() / 3600.0).round(3)

    # Save CSVs
    for p in [OUT_CSV_1, OUT_CSV_2]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df.to_csv(p, index=False)
        print(f"[SAVED] CSV -> {p}")

    # Build formatted Markdown Table
    total_time_h = df["Time_Seconds"].sum() / 3600.0
    avg_epoch_s = df["Time_Seconds"].iloc[1:].mean() # exclude epoch 1 cache warm-up

    md = f"""# RASNet 70-Epoch Training Metrics Table

**Model**: RASNet (Residual Attention Segmentation Network with Deep Supervision & StenosisAwareLoss)  
**Dataset**: ImageCAS (690 Training Scans, 3D Patches $96\\times 96\\times 96$, Foreground Ratio 2:1)  
**Hardware Profile**: NVIDIA GeForce RTX 4080 SUPER (16 GB VRAM), AMP FP16 + TF32  
**Total Training Time**: {total_time_h:.2f} hours (Average epoch time after initial caching: {avg_epoch_s:.1f}s)  
**Best Model Loss**: **0.1099** achieved at **Epoch 41** (Checkpoint saved as `rasnet_best.pth`)

---

## Epoch-by-Epoch Progress

| Epoch | Time (s) | Training Loss | Learning Rate | Max VRAM (MB) | Cumulative Time (h) | Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
"""
    for _, r in df.iterrows():
        ep = int(r["Epoch"])
        t = r["Time_Seconds"]
        l = r["Training_Loss"]
        lr = r["Learning_Rate"]
        mem = r["Max_VRAM_MB"]
        cum_t = r["Cumulative_Time_Hours"]
        note = ""
        if ep == 41:
            note = "**🏆 Champion Checkpoint (Min Loss: 0.1099)**"
        elif ep == 1:
            note = "Data Caching & Warm-up"
        elif ep % 10 == 0:
            note = f"Checkpoint saved (`rasnet_epoch_{ep}.pth`)"

        md += f"| {ep:02d} | {t:.1f} | {l:.4f} | {lr:.6f} | {mem:.1f} | {cum_t:.3f} | {note} |\n"

    for p in [OUT_MD_1, OUT_MD_2]:
        with open(p, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"[SAVED] Markdown -> {p}")

    print("Task 1 complete successfully.")

if __name__ == "__main__":
    parse_and_export()
