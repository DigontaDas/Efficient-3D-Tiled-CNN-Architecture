# 🧩 Table 8: Component-Wise Ablation Study (Primary Benchmark N=150)
**Updated with Authentic 200-Epoch Lab PC Retraining Results**  
**Benchmark Target**: Primary ImageCAS Test Set ($N=150$) evaluated at 0.5 mm isotropic resolution.  

---

| Step | Configuration | Attention Gates | Deep Supervision | Loss Function | TTA (4-Pass) | cc3d Pruning | Dice | HD95 (mm) | IoU | Precision | Recall | Δ Dice | Δ HD95 | Notes |
|:---:|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | SegResNet Baseline | No | No | Standard Dice+CE | No | No | 0.7469 | 31.48 | 0.6001 | 0.7313 | 0.7713 | Ref (0.00%) | Ref | Baseline encoder-decoder with residual units |
| **2** | + AttentionGate3D Only | Yes (3-Level) | No | Standard Dice+CE | No | No | 0.7440 | 23.44 | 0.5964 | 0.7450 | 0.7512 | -0.0029 (-0.39%) | -8.04 mm | Suppresses irrelevant non-vascular structures; boundary error drops by 8.04 mm |
| **3** | + Deep Supervision (aux2, aux3) | Yes (3-Level) | Yes (aux2, aux3) | Standard Dice+CE | No | No | 0.7535 | 21.61 | 0.6083 | 0.7700 | 0.7448 | +0.0066 (+0.88%) | -9.87 mm | Accelerates gradient propagation through multi-scale auxiliary heads |
| **4** | **+ StenosisAwareLoss (Raw Single Pass)** | **Yes (3-Level)** | **Yes (aux2, aux3)** | **StenosisAware (α=0.4, γ=2.5)** | **No** | **No** | **0.7681** | **13.45** | **0.6271** | **0.8127** | **0.7340** | **+0.0212 (+2.84%)** | **-18.03 mm** | **Focal penalization heavily suppresses background noise; HD95 drops 8.16 mm** |
| **5** | + 4-Pass TTA (Inference) | Yes (3-Level) | Yes (aux2, aux3) | StenosisAware (α=0.4, γ=2.5) | Yes (3 Flips + Orig) | No | 0.7745 | 12.20 | 0.6355 | 0.8260 | 0.7347 | +0.0276 (+3.70%) | -19.28 mm | Orthogonal test-time flip averaging stabilizes boundary contours |
| **6** | **+ cc3d Top-2 Pruning (= Champion RASNet)** | **Yes (3-Level)** | **Yes (aux2, aux3)** | **StenosisAware (α=0.4, γ=2.5)** | **Yes (3 Flips + Orig)** | **Yes (Top-2 Trees)** | **0.7765** | **10.29** | **0.6396** | **0.8801** | **0.7016** | **+0.0296 (+3.96%)** | **-21.19 mm** | **Connected component filtering removes isolated cardiac false positives; Precision reaches 88.01%** |

---

### Scientific Synthesis:
1. **Architectural vs Post-Processing Gain**:
   - The combination of **Attention Gates + Deep Supervision + StenosisAwareLoss (Step 4, Raw Model)** increases Precision from **0.7313 to 0.8127** (+8.14%) and drops HD95 by more than half (from **31.48 mm down to 13.45 mm**) *without any TTA or connected-component post-processing*.
   - Even without TTA or `cc3d`, Raw RASNet (Step 4) achieves Dice = **0.7681**, matching nnU-Net V2 (**0.7687**) while delivering significantly superior boundary accuracy (HD95 **13.45 mm** vs **21.10 mm**) and Precision (**0.8127** vs **0.7391**).
2. **The Role of Inference Refinement (TTA & cc3d)**:
   - 4-Pass TTA (Step 5) adds **+0.64% Dice** and further stabilizes boundaries (HD95: **12.20 mm**).
   - cc3d Top-2 Pruning (Step 6) filters disconnected extracardiac clutter, driving Precision to **88.01%** and achieving champion HD95 (**10.29 mm**).
