"""Unit tests for mast3r_runtime package exports.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


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
# Package Import Tests
# ==============================================================================


class TestPackageImports:
    """Tests for package-level imports."""

    def test_import_package(self):
        """Package can be imported."""
        import mast3r_runtime

        assert mast3r_runtime is not None

    def test_version(self):
        """Package has version."""
        from mast3r_runtime import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_author(self):
        """Package has author."""
        from mast3r_runtime import __author__

        assert "Delanoe" in __author__ or "Pirard" in __author__

    def test_license(self):
        """Package has license."""
        from mast3r_runtime import __license__

        assert "Apache" in __license__


# ==============================================================================
# Configuration Exports
# ==============================================================================


class TestConfigExports:
    """Tests for configuration exports."""

    def test_model_variant_export(self):
        """ModelVariant is exported."""
        from mast3r_runtime import ModelVariant

        assert hasattr(ModelVariant, "DUNE_VIT_SMALL_336")

    def test_backend_type_export(self):
        """BackendType is exported."""
        from mast3r_runtime import BackendType

        assert hasattr(BackendType, "AUTO")
        assert hasattr(BackendType, "METAL")

    def test_precision_export(self):
        """Precision is exported."""
        from mast3r_runtime import Precision

        assert hasattr(Precision, "FP16")
        assert hasattr(Precision, "FP32")

    def test_model_config_export(self):
        """ModelConfig is exported."""
        from mast3r_runtime import ModelConfig

        config = ModelConfig()
        assert config.variant is not None

    def test_runtime_config_export(self):
        """RuntimeConfig is exported."""
        from mast3r_runtime import RuntimeConfig

        config = RuntimeConfig()
        assert config.backend is not None

    def test_matching_config_export(self):
        """MatchingConfig is exported."""
        from mast3r_runtime import MatchingConfig

        config = MatchingConfig()
        assert config.top_k > 0

    def test_mast3r_runtime_config_export(self):
        """MASt3RRuntimeConfig is exported."""
        from mast3r_runtime import MASt3RRuntimeConfig

        config = MASt3RRuntimeConfig()
        assert config.model is not None
        assert config.runtime is not None
        assert config.matching is not None


# ==============================================================================
# Preset Exports
# ==============================================================================


class TestPresetExports:
    """Tests for preset configuration exports."""

    def test_preset_drone_fast(self):
        """PRESET_DRONE_FAST is exported."""
        from mast3r_runtime import PRESET_DRONE_FAST

        assert PRESET_DRONE_FAST.model.variant.value == "dune_vit_small_336"

    def test_preset_drone_quality(self):
        """PRESET_DRONE_QUALITY is exported."""
        from mast3r_runtime import PRESET_DRONE_QUALITY

        assert PRESET_DRONE_QUALITY.model.variant.value == "dune_vit_base_448"

    def test_preset_desktop_precision(self):
        """PRESET_DESKTOP_PRECISION is exported."""
        from mast3r_runtime import PRESET_DESKTOP_PRECISION

        assert PRESET_DESKTOP_PRECISION.model.variant.value == "mast3r_vit_large"


# ==============================================================================
# Model Specs Export
# ==============================================================================


class TestModelSpecsExport:
    """Tests for MODEL_SPECS export."""

    def test_model_specs_export(self):
        """MODEL_SPECS is exported."""
        from mast3r_runtime import MODEL_SPECS

        assert isinstance(MODEL_SPECS, dict)
        assert len(MODEL_SPECS) > 0


# ==============================================================================
# Checkpoint Helper Exports
# ==============================================================================


class TestCheckpointHelperExports:
    """Tests for checkpoint helper function exports."""

    def test_get_checkpoint_paths_export(self):
        """get_checkpoint_paths is exported."""
        from mast3r_runtime import ModelVariant, get_checkpoint_paths

        paths = get_checkpoint_paths(ModelVariant.DUNE_VIT_SMALL_336)
        assert isinstance(paths, dict)

    def test_get_checkpoint_urls_export(self):
        """get_checkpoint_urls is exported."""
        from mast3r_runtime import ModelVariant, get_checkpoint_urls

        urls = get_checkpoint_urls(ModelVariant.DUNE_VIT_SMALL_336)
        assert isinstance(urls, dict)

    def test_get_total_checkpoint_size_mb_export(self):
        """get_total_checkpoint_size_mb is exported."""
        from mast3r_runtime import ModelVariant, get_total_checkpoint_size_mb

        size = get_total_checkpoint_size_mb(ModelVariant.DUNE_VIT_SMALL_336)
        assert isinstance(size, int)
        assert size > 0


# ==============================================================================
# Engine Interface Exports
# ==============================================================================


class TestEngineExports:
    """Tests for engine interface exports."""

    def test_engine_interface_export(self):
        """EngineInterface is exported."""
        from mast3r_runtime import EngineInterface

        assert EngineInterface is not None

    def test_inference_result_export(self):
        """InferenceResult is exported."""
        from mast3r_runtime import InferenceResult

        assert InferenceResult is not None

    def test_match_result_export(self):
        """MatchResult is exported."""
        from mast3r_runtime import MatchResult

        assert MatchResult is not None


# ==============================================================================
# Backend Exports
# ==============================================================================


class TestBackendExports:
    """Tests for backend function exports."""

    def test_get_available_backends_export(self):
        """get_available_backends is exported."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            from mast3r_runtime import get_available_backends

            backends = get_available_backends()
            assert isinstance(backends, dict)

    def test_get_backend_info_export(self):
        """get_backend_info is exported."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            from mast3r_runtime import get_backend_info

            info = get_backend_info()
            assert isinstance(info, dict)

    def test_get_best_backend_export(self):
        """get_best_backend is exported."""
        with patch.dict(sys.modules, mock_native_imports()):
            import importlib

            import mast3r_runtime.backends as backends_module

            importlib.reload(backends_module)
            from mast3r_runtime import get_best_backend

            backend = get_best_backend()
            assert isinstance(backend, str)

    def test_get_runtime_export(self):
        """get_runtime is exported."""
        from mast3r_runtime import get_runtime

        assert callable(get_runtime)


# ==============================================================================
# __all__ Tests
# ==============================================================================


class TestAllExports:
    """Tests for __all__ list."""

    def test_all_defined(self):
        """__all__ is defined."""
        import mast3r_runtime

        assert hasattr(mast3r_runtime, "__all__")
        assert isinstance(mast3r_runtime.__all__, list)

    def test_all_exports_are_accessible(self):
        """All items in __all__ are accessible."""
        import mast3r_runtime

        for name in mast3r_runtime.__all__:
            assert hasattr(mast3r_runtime, name), f"Missing export: {name}"

    def test_expected_exports_in_all(self):
        """Expected exports are in __all__."""
        import mast3r_runtime

        expected = [
            "__version__",
            "ModelVariant",
            "BackendType",
            "MASt3RRuntimeConfig",
            "get_runtime",
            "InferenceResult",
            "MatchResult",
        ]

        for name in expected:
            assert name in mast3r_runtime.__all__, f"Missing from __all__: {name}"
