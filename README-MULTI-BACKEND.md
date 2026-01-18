# AutoGluon Multi-Backend Patch for Time Series Models

This repository contains a patch for AutoGluon to enable multi-backend support specifically for time series models. The primary focus is on extending hardware acceleration beyond CUDA to include MPS (Metal Performance Shaders for Apple Silicon) and XPU (Intel GPU via oneAPI).

## Key Features
- **Multi-Backend Support**: Instead of relying solely on CUDA, this patched version attempts to support MPS and XPU backends for improved compatibility across different hardware platforms.
- **Tested Models**: The patch has been tested only with Temporal Fusion Transformer (TFT) and Chronos 2 models. Other time series models in AutoGluon may not work as expected and have not been verified.

## Limitations and Disclaimer
- **Testing Scope**: Due to limited resources for testing and expertise, this patch is provided "as is" without guarantees of stability, performance, or compatibility with all AutoGluon features.
- **Backend-Specific Notes**:
    - **MPS**: TFT and Chronos 2 work fine on MPS-enabled devices (e.g., Apple M-series chips).
    - **XPU**: The official PyTorch Lightning package does not support XPU accelerators, so the patch will fallback to CPU for these operations. For Chronos 2, which is a large model, fine-tuning on Intel GPUs (XPU) can still take a significant amount of time and may encounter hiccups. For more details on XPU-specific issues and workarounds, refer to [README-XPU.md](README-XPU.md).
- This patch is experimental and intended for users who need multi-backend flexibility. Extensive testing on production workloads is recommended before deployment.