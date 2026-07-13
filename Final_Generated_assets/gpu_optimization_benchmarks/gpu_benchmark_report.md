# GPU Optimization Benchmark Report

- **Device**: NVIDIA GeForce RTX 3060 Ti
- **Patch Size**: (96, 96, 96)
- **Batch Size**: 2
- **TF32 Matmul**: True
- **TF32 CuDNN**: True

## Epoch Metrics

| Epoch | Time (s) | Mean Loss | Learning Rate | Max VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- |
| 01 | 1.101s | 1.2204 | 0.000091 | 2238.5 MB |
| 02 | 0.749s | 1.1976 | 0.000089 | 2238.5 MB |
| 03 | 0.760s | 1.1949 | 0.000050 | 2238.5 MB |
| 04 | 0.762s | 1.1943 | 0.000011 | 2238.5 MB |
| 05 | 0.769s | 1.1941 | 0.000001 | 2238.5 MB |

---
*Benchmark completed successfully with all 5 active GPU optimizations (AMP, TF32, Pinned Memory, Channels-Last 3D, and OneCycleLR).* 
