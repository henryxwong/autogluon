"""
Unit tests for device detection utilities.
"""

import pytest
import torch

from autogluon.timeseries.utils.device_utils import (
    get_device,
    get_torch_device,
    get_lightning_accelerator,
    is_gpu_available,
    get_device_info,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_returns_string(self):
        """Test that get_device returns a string."""
        device = get_device()
        assert isinstance(device, str)
        assert device in ["cuda", "mps", "xpu", "cpu"]

    def test_get_device_with_preferred_backend_cpu(self):
        """Test get_device with preferred backend 'cpu'."""
        device = get_device(preferred_backend="cpu")
        assert device == "cpu"

    def test_get_device_with_invalid_preferred_backend(self):
        """Test get_device with invalid preferred backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid preferred_backend"):
            get_device(preferred_backend="invalid")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_with_preferred_backend_cuda(self):
        """Test get_device with preferred backend 'cuda' when available."""
        device = get_device(preferred_backend="cuda")
        assert device == "cuda"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_get_device_with_preferred_backend_mps(self):
        """Test get_device with preferred backend 'mps' when available."""
        device = get_device(preferred_backend="mps")
        assert device == "mps"

    @pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
    def test_get_device_with_preferred_backend_xpu(self):
        """Test get_device with preferred backend 'xpu' when available."""
        device = get_device(preferred_backend="xpu")
        assert device == "xpu"

    def test_get_device_priority(self):
        """Test that get_device respects priority: CUDA > XPU > MPS > CPU."""
        device = get_device()
        
        # If multiple backends are available, verify priority
        if torch.cuda.is_available():
            assert device == "cuda"
        elif torch.xpu.is_available():
            assert device == "xpu"
        elif torch.backends.mps.is_available():
            assert device == "mps"
        else:
            assert device == "cpu"


class TestGetTorchDevice:
    """Tests for get_torch_device function."""

    def test_get_torch_device_returns_torch_device(self):
        """Test that get_torch_device returns a torch.device object."""
        device = get_torch_device()
        assert isinstance(device, torch.device)

    def test_get_torch_device_with_preferred_backend(self):
        """Test get_torch_device with preferred backend."""
        device = get_torch_device(preferred_backend="cpu")
        assert isinstance(device, torch.device)
        assert device.type == "cpu"


class TestGetLightningAccelerator:
    """Tests for get_lightning_accelerator function."""

    def test_get_lightning_accelerator_cuda(self):
        """Test get_lightning_accelerator for CUDA."""
        accelerator = get_lightning_accelerator("cuda")
        assert accelerator == "gpu"

    def test_get_lightning_accelerator_mps(self):
        """Test get_lightning_accelerator for MPS."""
        accelerator = get_lightning_accelerator("mps")
        assert accelerator == "mps"

    def test_get_lightning_accelerator_xpu(self):
        """Test get_lightning_accelerator for XPU."""
        accelerator = get_lightning_accelerator("xpu")
        # XPU may fall back to CPU
        assert accelerator in ["cpu", "gpu"]

    def test_get_lightning_accelerator_cpu(self):
        """Test get_lightning_accelerator for CPU."""
        accelerator = get_lightning_accelerator("cpu")
        assert accelerator == "cpu"

    def test_get_lightning_accelerator_unknown(self):
        """Test get_lightning_accelerator for unknown device type."""
        accelerator = get_lightning_accelerator("unknown")
        assert accelerator == "cpu"


class TestIsGpuAvailable:
    """Tests for is_gpu_available function."""

    def test_is_gpu_available_returns_bool(self):
        """Test that is_gpu_available returns a boolean."""
        available = is_gpu_available()
        assert isinstance(available, bool)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_is_gpu_available_with_cuda(self):
        """Test that is_gpu_available returns True when CUDA is available."""
        assert is_gpu_available() is True

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_is_gpu_available_with_mps(self):
        """Test that is_gpu_available returns True when MPS is available."""
        assert is_gpu_available() is True

    @pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
    def test_is_gpu_available_with_xpu(self):
        """Test that is_gpu_available returns True when XPU is available."""
        assert is_gpu_available() is True

    @pytest.mark.skipif(
        torch.cuda.is_available() or torch.backends.mps.is_available() or torch.xpu.is_available(),
        reason="GPU backend available"
    )
    def test_is_gpu_available_no_gpu(self):
        """Test that is_gpu_available returns False when no GPU backend is available."""
        assert is_gpu_available() is False


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_get_device_info_returns_dict(self):
        """Test that get_device_info returns a dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)

    def test_get_device_info_has_required_keys(self):
        """Test that get_device_info has all required keys."""
        info = get_device_info()
        assert "cuda" in info
        assert "xpu" in info
        assert "mps" in info
        assert "selected_device" in info

    def test_get_device_info_cuda_details(self):
        """Test that get_device_info includes CUDA details when available."""
        info = get_device_info()
        
        if torch.cuda.is_available():
            assert info["cuda"] is True
            assert "cuda_device_count" in info
            assert isinstance(info["cuda_device_count"], int)
        else:
            assert info["cuda"] is False

    def test_get_device_info_rocm_detection(self):
        """Test that get_device_info detects ROCm when present."""
        info = get_device_info()
        
        if torch.cuda.is_available():
            # Check for ROCm flag
            assert "is_rocm" in info
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                assert info["is_rocm"] is True
            else:
                assert info["is_rocm"] is False

    def test_get_device_info_selected_device_matches_get_device(self):
        """Test that selected_device in info matches get_device()."""
        info = get_device_info()
        device = get_device()
        assert info["selected_device"] == device


class TestDeviceIntegration:
    """Integration tests for device utilities."""

    def test_device_consistency(self):
        """Test that all device functions return consistent results."""
        device_str = get_device()
        device_obj = get_torch_device()
        info = get_device_info()
        
        assert device_str == str(device_obj)
        assert device_str == info["selected_device"]

    def test_lightning_accelerator_matches_device(self):
        """Test that Lightning accelerator is appropriate for the device."""
        device = get_device()
        accelerator = get_lightning_accelerator(device)
        
        # Verify accelerator is valid
        assert accelerator in ["gpu", "mps", "cpu"]
        
        # CUDA should map to gpu
        if device == "cuda":
            assert accelerator == "gpu"
        
        # MPS should map to mps
        if device == "mps":
            assert accelerator == "mps"
        
        # CPU should map to cpu
        if device == "cpu":
            assert accelerator == "cpu"