# GPU Optimization Benchmark Report

- **Device**: NVIDIA GeForce RTX 3060 Ti
- **Mode**: Real ImageCAS NIfTI Data
- **Patch Size**: (96, 96, 96)
- **Batch Size**: 2
- **TF32 Matmul**: True
- **TF32 CuDNN**: True

## Epoch Metrics

| Epoch | Time (s) | Mean Loss | Learning Rate | Max VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- |
| 01 | 5.653s | 1.5327 | 0.000095 | 9597.7 MB |
| 02 | 4.605s | 1.1980 | 0.000087 | 9599.7 MB |
| 03 | 4.630s | 1.1145 | 0.000046 | 9599.7 MB |
| 04 | 4.581s | 1.0940 | 0.000009 | 9599.7 MB |
| 05 | 4.324s | 1.0933 | 0.000002 | 9599.7 MB |

---
*Benchmark completed successfully with all 5 active GPU optimizations (AMP, TF32, Pinned Memory, Channels-Last 3D, and OneCycleLR).* 
