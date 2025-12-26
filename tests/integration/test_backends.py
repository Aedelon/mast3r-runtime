"""Integration tests for mast3r_runtime.backends.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import platform
import sys
from unittest.mock import MagicMock, patch

import pytest

from mast3r_runtime.core.config import MASt3RRuntimeConfig

from ..conftest import (
    has_cpu_backend,
    has_metal_backend,
    skip_no_cpu,
    skip_no_metal,
    skip_not_macos,
)


# ==============================================================================
# Mock helper - prevent native module import issues in editable mode
# ==============================================================================


def mock_native_imports():
    """Create mocks for native modules that may not be compiled."""
    mock_cpu = MagicMock()
    mock_cpu.is_available.return_value = False

    mock_metal = MagicMock()
    mock_metal.is_available.return_value = False

    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = False

    mock_jetson = MagicMock()
    mock_jetson.is_available.return_value = False

    return {
        "mast3r_runtime.backends._cpu": mock_cpu,
        "mast3r_runtime.backends._metal": mock_metal,
        "mast3r_runtime.backends._cuda": mock_cuda,
        "mast3r_runtime.backends._jetson": mock_jetson,
    }


# ==============================================================================
# get_available_backends Tests
# ==============================================================================


class TestGetAvailableBackends:
    """Tests for get_available_backends function."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        with patch.dict(sys.modules, mock_native_imports()):
            # Clear cached imports
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            backends = backends_module.get_available_backends()
            assert isinstance(backends, dict)

    def test_python_always_available(self):
        """Python backend is always available."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            backends = backends_module.get_available_backends()
            assert backends.get("python") is True

    def test_contains_expected_keys(self):
        """Contains expected backend keys."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            backends = backends_module.get_available_backends()
            expected_keys = {"python", "cpu", "metal", "cuda", "jetson"}
            assert set(backends.keys()) == expected_keys

    def test_values_are_bool(self):
        """All values are boolean."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            backends = backends_module.get_available_backends()
            for name, available in backends.items():
                assert isinstance(available, bool), f"{name} should be bool"

    @skip_not_macos
    def test_metal_detection_on_macos(self):
        """Metal detection works on macOS."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            backends = backends_module.get_available_backends()
            # Just verify no crash during detection
            assert "metal" in backends


# ==============================================================================
# get_best_backend Tests
# ==============================================================================


class TestGetBestBackend:
    """Tests for get_best_backend function."""

    def test_returns_string(self):
        """Returns a string backend name."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            backend = backends_module.get_best_backend()
            assert isinstance(backend, str)

    def test_returns_valid_backend(self):
        """Returns a valid backend name."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            backend = backends_module.get_best_backend()
            valid_backends = {"python", "cpu", "metal", "cuda", "jetson"}
            assert backend in valid_backends

    @skip_not_macos
    def test_macos_arm64_prefers_metal(self):
        """macOS arm64 prefers Metal if available."""
        if platform.machine() != "arm64":
            pytest.skip("Not arm64")

        # Mock Metal as available
        mocks = mock_native_imports()
        mocks["mast3r_runtime.backends._metal"].is_available.return_value = True

        with patch.dict(sys.modules, mocks):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            backend = backends_module.get_best_backend()
            assert backend == "metal"

    def test_fallback_to_python(self):
        """Falls back to Python if nothing else available."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            backend = backends_module.get_best_backend()
            # With all native backends mocked as unavailable, should be python
            assert backend == "python"


# ==============================================================================
# get_backend_info Tests
# ==============================================================================


class TestGetBackendInfo:
    """Tests for get_backend_info function."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            info = backends_module.get_backend_info()
            assert isinstance(info, dict)

    def test_contains_platform_info(self):
        """Contains platform information."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            info = backends_module.get_backend_info()
            assert "platform" in info
            assert "system" in info["platform"]
            assert "machine" in info["platform"]
            assert "python" in info["platform"]

    def test_contains_backends_dict(self):
        """Contains backends availability dict."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            info = backends_module.get_backend_info()
            assert "backends" in info
            assert isinstance(info["backends"], dict)

    def test_contains_best_backend(self):
        """Contains best_backend field."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            info = backends_module.get_backend_info()
            assert "best_backend" in info
            assert isinstance(info["best_backend"], str)

    def test_contains_details(self):
        """Contains details dict."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            info = backends_module.get_backend_info()
            assert "details" in info
            assert isinstance(info["details"], dict)

    @skip_no_metal
    def test_metal_details(self):
        """Metal details are populated when available."""
        # This test only runs if Metal is truly available
        import importlib

        import mast3r_runtime.backends as backends_module

        importlib.reload(backends_module)
        info = backends_module.get_backend_info()
        if info["backends"].get("metal"):
            assert "metal" in info["details"]
            assert "device" in info["details"]["metal"]


# ==============================================================================
# get_runtime Tests
# ==============================================================================


class TestGetRuntime:
    """Tests for get_runtime function."""

    def test_auto_backend(self):
        """AUTO backend selects best available."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            config = MASt3RRuntimeConfig(runtime={"backend": "auto"})
            try:
                engine = backends_module.get_runtime(config)
                assert engine is not None
            except RuntimeError as e:
                # Expected if backend not available
                assert "not available" in str(e).lower()

    def test_python_backend(self):
        """Python backend is always available."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            config = MASt3RRuntimeConfig(runtime={"backend": "python"})
            engine = backends_module.get_runtime(config)
            assert engine is not None
            assert "Python" in engine.name or "python" in engine.name.lower()

    @skip_no_metal
    def test_metal_backend(self):
        """Metal backend can be requested."""
        from mast3r_runtime.backends import get_runtime

        config = MASt3RRuntimeConfig(runtime={"backend": "metal"})
        engine = get_runtime(config)
        assert engine is not None

    @skip_no_cpu
    def test_cpu_backend(self):
        """CPU backend can be requested."""
        from mast3r_runtime.backends import get_runtime

        config = MASt3RRuntimeConfig(runtime={"backend": "cpu"})
        engine = get_runtime(config)
        assert engine is not None

    def test_invalid_backend_raises(self):
        """Invalid backend raises RuntimeError."""
        # For CUDA backend on non-CUDA system, the import will fail
        # We mock all modules but set _cuda to cause ImportError
        mocks = mock_native_imports()
        # Set CUDA to None to cause import failure
        mocks["mast3r_runtime.backends._cuda"] = None

        with patch.dict(sys.modules, mocks):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            config = MASt3RRuntimeConfig(runtime={"backend": "cuda"})
            with pytest.raises(RuntimeError) as exc_info:
                backends_module.get_runtime(config)
            assert "not available" in str(exc_info.value).lower()


# ==============================================================================
# Native Backend Import Tests (only run if backends actually available)
# ==============================================================================


class TestNativeBackendImports:
    """Tests for native backend module imports."""

    @skip_no_cpu
    def test_cpu_module_import(self):
        """_cpu module can be imported and has expected attributes."""
        from mast3r_runtime import _cpu

        assert hasattr(_cpu, "is_available")
        assert hasattr(_cpu, "CpuEngine")

    @skip_no_metal
    def test_metal_module_import(self):
        """_metal module can be imported and has expected attributes."""
        from mast3r_runtime import _metal

        assert hasattr(_metal, "is_available")
        assert hasattr(_metal, "MetalEngine")
        assert hasattr(_metal, "get_device_name")

    @skip_no_cpu
    def test_cpu_is_available(self):
        """_cpu.is_available() works."""
        from mast3r_runtime import _cpu

        result = _cpu.is_available()
        assert isinstance(result, bool)

    @skip_no_metal
    def test_metal_is_available(self):
        """_metal.is_available() works."""
        from mast3r_runtime import _metal

        result = _metal.is_available()
        assert isinstance(result, bool)

    @skip_no_metal
    def test_metal_device_name(self):
        """_metal.get_device_name() returns a string."""
        from mast3r_runtime import _metal

        name = _metal.get_device_name()
        assert isinstance(name, str)
        assert len(name) > 0


# ==============================================================================
# Native Engine Instantiation Tests
# ==============================================================================


class TestNativeEngineInstantiation:
    """Tests for native engine instantiation."""

    @skip_no_cpu
    def test_cpu_engine_creation(self):
        """CpuEngine can be instantiated."""
        from mast3r_runtime import _cpu

        engine = _cpu.CpuEngine(
            variant="dune_vit_small_336",
            resolution=336,
            precision="fp16",
            num_threads=4,
        )
        assert engine is not None

    @skip_no_cpu
    def test_cpu_engine_name(self):
        """CpuEngine has a name."""
        from mast3r_runtime import _cpu

        engine = _cpu.CpuEngine(
            variant="dune_vit_small_336",
            resolution=336,
            precision="fp16",
            num_threads=4,
        )
        name = engine.name()
        assert isinstance(name, str)
        assert len(name) > 0

    @skip_no_metal
    def test_metal_engine_creation(self):
        """MetalEngine can be instantiated."""
        from mast3r_runtime import _metal

        engine = _metal.MetalEngine(
            variant="dune_vit_small_336",
            resolution=336,
            precision="fp16",
            num_threads=4,
        )
        assert engine is not None

    @skip_no_metal
    def test_metal_engine_name(self):
        """MetalEngine has a name with device info."""
        from mast3r_runtime import _metal

        engine = _metal.MetalEngine(
            variant="dune_vit_small_336",
            resolution=336,
            precision="fp16",
            num_threads=4,
        )
        name = engine.name()
        assert isinstance(name, str)
        # Should include "Metal" and device name
        assert "Metal" in name
