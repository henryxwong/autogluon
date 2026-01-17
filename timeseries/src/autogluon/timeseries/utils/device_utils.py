"""
Device detection utilities for multi-backend GPU support.

This module provides functions to automatically detect and select the best available
device for PyTorch operations, supporting NVIDIA CUDA, AMD ROCm (treated as CUDA),
Intel XPU, Apple MPS, and CPU backends.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)


def get_device(preferred_backend: Literal["cuda", "mps", "xpu", "cpu"] | None = None) -> str:
    """
    Detects the best available device for PyTorch operations.
    
    Priority: CUDA/ROCm > XPU > MPS > CPU
    
    Parameters
    ----------
    preferred_backend : str or None, default = None
        Optional string to force selection of a specific backend.
        Valid values: 'cuda', 'mps', 'xpu', 'cpu'.
        If None, automatically detects the best available device.
    
    Returns
    -------
    str
        The detected or selected device type ('cuda', 'mps', 'xpu', or 'cpu').
    
    Raises
    ------
    ValueError
        If preferred_backend is specified but not available.
    
    Examples
    --------
    >>> get_device()
    'cuda'  # if CUDA/ROCm is available
    
    >>> get_device(preferred_backend='mps')
    'mps'  # if MPS is available, otherwise raises ValueError
    """
    import torch
    
    # Check for preferred backend if specified
    if preferred_backend is not None:
        if preferred_backend == "cuda":
            if torch.cuda.is_available():
                logger.info("Using CUDA device (preferred backend)")
                return "cuda"
            else:
                raise ValueError(
                    f"Preferred backend 'cuda' is not available. "
                    "Please ensure PyTorch is installed with CUDA support."
                )
        elif preferred_backend == "mps":
            if torch.backends.mps.is_available():
                logger.info("Using MPS device (preferred backend)")
                return "mps"
            else:
                raise ValueError(
                    f"Preferred backend 'mps' is not available. "
                    "MPS is only supported on Apple Silicon (M1/M2/M3) Macs."
                )
        elif preferred_backend == "xpu":
            if torch.xpu.is_available():
                logger.info("Using XPU device (preferred backend)")
                return "xpu"
            else:
                raise ValueError(
                    f"Preferred backend 'xpu' is not available. "
                    "Please ensure PyTorch with Intel XPU support is installed."
                )
        elif preferred_backend == "cpu":
            logger.info("Using CPU device (preferred backend)")
            return "cpu"
        else:
            raise ValueError(
                f"Invalid preferred_backend '{preferred_backend}'. "
                "Valid values are: 'cuda', 'mps', 'xpu', 'cpu'"
            )
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        # Check for ROCm (AMD GPU)
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if is_rocm:
            logger.info("Using ROCm device (AMD GPU)")
        else:
            logger.info("Using CUDA device (NVIDIA GPU)")
        return "cuda"
    elif torch.xpu.is_available():
        logger.info("Using XPU device (Intel GPU)")
        return "xpu"
    elif torch.backends.mps.is_available():
        logger.info("Using MPS device (Apple Silicon)")
        return "mps"
    else:
        logger.info("No GPU detected, using CPU device")
        return "cpu"


def get_torch_device(preferred_backend: Literal["cuda", "mps", "xpu", "cpu"] | None = None) -> "torch.device":
    """
    Returns a torch.device object for the best available device.
    
    This is a convenience wrapper around get_device() that returns an actual
    torch.device object instead of a string.
    
    Parameters
    ----------
    preferred_backend : str or None, default = None
        Optional string to force selection of a specific backend.
    
    Returns
    -------
    torch.device
        A torch.device object for the detected or selected backend.
    
    Examples
    --------
    >>> device = get_torch_device()
    >>> model.to(device)
    """
    import torch
    return torch.device(get_device(preferred_backend=preferred_backend))


def get_lightning_accelerator(device_type: str) -> str:
    """
    Maps a device type to the appropriate PyTorch Lightning accelerator string.
    
    Parameters
    ----------
    device_type : str
        The device type ('cuda', 'mps', 'xpu', or 'cpu').
    
    Returns
    -------
    str
        The PyTorch Lightning accelerator string.
    
    Examples
    --------
    >>> get_lightning_accelerator('cuda')
    'gpu'
    
    >>> get_lightning_accelerator('mps')
    'mps'
    """
    if device_type == "cuda":
        return "gpu"
    elif device_type == "mps":
        return "mps"
    elif device_type == "xpu":
        # PyTorch Lightning may not support 'xpu' accelerator directly
        # Fall back to CPU with a warning, or use 'gpu' if XPU is treated as GPU
        logger.warning(
            "XPU accelerator may not be fully supported by PyTorch Lightning. "
            "Falling back to CPU for training."
        )
        return "cpu"
    elif device_type == "cpu":
        return "cpu"
    else:
        logger.warning(
            f"Unknown device type '{device_type}', defaulting to CPU accelerator"
        )
        return "cpu"


def is_gpu_available() -> bool:
    """
    Checks if any GPU backend (CUDA/ROCm, XPU, or MPS) is available.
    
    Returns
    -------
    bool
        True if any GPU backend is available, False otherwise.
    
    Examples
    --------
    >>> if is_gpu_available():
    ...     device = get_torch_device()
    """
    import torch
    return (
        torch.cuda.is_available() 
        or torch.xpu.is_available() 
        or torch.backends.mps.is_available()
    )


def get_device_info() -> dict:
    """
    Returns detailed information about available devices.
    
    This is useful for debugging and logging device availability.
    
    Returns
    -------
    dict
        A dictionary containing information about available devices:
        - 'cuda': bool indicating CUDA/ROCm availability
        - 'xpu': bool indicating XPU availability
        - 'mps': bool indicating MPS availability
        - 'selected_device': the selected device type
        - 'cuda_device_count': number of CUDA devices (if available)
    
    Examples
    --------
    >>> info = get_device_info()
    >>> print(f"Using device: {info['selected_device']}")
    """
    import torch
    
    info = {
        'cuda': torch.cuda.is_available(),
        'xpu': torch.xpu.is_available(),
        'mps': torch.backends.mps.is_available(),
    }
    
    if info['cuda']:
        info['cuda_device_count'] = torch.cuda.device_count()
        # Check for ROCm
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        info['is_rocm'] = is_rocm
    
    info['selected_device'] = get_device()
    
    return info


def supports_mem_get_info(device: str) -> bool:
    if device == "cpu":
        return False
    elif device in ["cuda", "mps"]:
        return True
    elif device == "xpu":
        return False
    else:
        return False  # Fallback for unknown backends
