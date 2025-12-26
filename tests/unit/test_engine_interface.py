"""Unit tests for mast3r_runtime.core.engine_interface.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from mast3r_runtime.core.engine_interface import (
    EngineInterface,
    InferenceResult,
    MatchResult,
)

from ..conftest import skip_no_torch


# ==============================================================================
# InferenceResult Tests
# ==============================================================================


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    @pytest.fixture
    def inference_result(self):
        """Create a sample InferenceResult."""
        H, W, D = 24, 24, 256
        return InferenceResult(
            pts3d_1=np.random.rand(H, W, 3).astype(np.float32),
            pts3d_2=np.random.rand(H, W, 3).astype(np.float32),
            desc_1=np.random.rand(H, W, D).astype(np.float32),
            desc_2=np.random.rand(H, W, D).astype(np.float32),
            conf_1=np.random.rand(H, W).astype(np.float32),
            conf_2=np.random.rand(H, W).astype(np.float32),
            timing_ms={"preprocess": 1.0, "inference": 10.0},
        )

    def test_creation(self, inference_result):
        """InferenceResult can be created."""
        assert inference_result is not None
        assert inference_result.pts3d_1 is not None

    def test_height_property(self, inference_result):
        """height property returns correct value."""
        assert inference_result.height == 24

    def test_width_property(self, inference_result):
        """width property returns correct value."""
        assert inference_result.width == 24

    def test_desc_dim_property(self, inference_result):
        """desc_dim property returns correct value."""
        assert inference_result.desc_dim == 256

    def test_optional_desc_conf(self):
        """desc_conf fields are optional."""
        result = InferenceResult(
            pts3d_1=np.zeros((10, 10, 3), dtype=np.float32),
            pts3d_2=np.zeros((10, 10, 3), dtype=np.float32),
            desc_1=np.zeros((10, 10, 64), dtype=np.float32),
            desc_2=np.zeros((10, 10, 64), dtype=np.float32),
            conf_1=np.zeros((10, 10), dtype=np.float32),
            conf_2=np.zeros((10, 10), dtype=np.float32),
        )
        assert result.desc_conf_1 is None
        assert result.desc_conf_2 is None

    def test_with_desc_conf(self):
        """desc_conf fields can be set."""
        result = InferenceResult(
            pts3d_1=np.zeros((10, 10, 3), dtype=np.float32),
            pts3d_2=np.zeros((10, 10, 3), dtype=np.float32),
            desc_1=np.zeros((10, 10, 64), dtype=np.float32),
            desc_2=np.zeros((10, 10, 64), dtype=np.float32),
            conf_1=np.zeros((10, 10), dtype=np.float32),
            conf_2=np.zeros((10, 10), dtype=np.float32),
            desc_conf_1=np.ones((10, 10), dtype=np.float32),
            desc_conf_2=np.ones((10, 10), dtype=np.float32),
        )
        assert result.desc_conf_1 is not None
        assert result.desc_conf_2 is not None

    def test_timing_default(self):
        """timing_ms defaults to empty dict."""
        result = InferenceResult(
            pts3d_1=np.zeros((10, 10, 3), dtype=np.float32),
            pts3d_2=np.zeros((10, 10, 3), dtype=np.float32),
            desc_1=np.zeros((10, 10, 64), dtype=np.float32),
            desc_2=np.zeros((10, 10, 64), dtype=np.float32),
            conf_1=np.zeros((10, 10), dtype=np.float32),
            conf_2=np.zeros((10, 10), dtype=np.float32),
        )
        assert result.timing_ms == {}

    @skip_no_torch
    def test_to_torch(self, inference_result):
        """to_torch converts arrays to tensors."""
        import torch

        torch_result = inference_result.to_torch()

        assert isinstance(torch_result.pts3d_1, torch.Tensor)
        assert isinstance(torch_result.desc_1, torch.Tensor)
        assert isinstance(torch_result.conf_1, torch.Tensor)

    @skip_no_torch
    def test_to_torch_device(self, inference_result):
        """to_torch accepts device parameter."""
        import torch

        torch_result = inference_result.to_torch(device="cpu")

        assert torch_result.pts3d_1.device.type == "cpu"

    @skip_no_torch
    def test_to_torch_preserves_shapes(self, inference_result):
        """to_torch preserves array shapes."""
        torch_result = inference_result.to_torch()

        assert tuple(torch_result.pts3d_1.shape) == (24, 24, 3)
        assert tuple(torch_result.desc_1.shape) == (24, 24, 256)
        assert tuple(torch_result.conf_1.shape) == (24, 24)

    @skip_no_torch
    def test_to_torch_with_desc_conf(self):
        """to_torch converts desc_conf if present."""
        result = InferenceResult(
            pts3d_1=np.zeros((10, 10, 3), dtype=np.float32),
            pts3d_2=np.zeros((10, 10, 3), dtype=np.float32),
            desc_1=np.zeros((10, 10, 64), dtype=np.float32),
            desc_2=np.zeros((10, 10, 64), dtype=np.float32),
            conf_1=np.zeros((10, 10), dtype=np.float32),
            conf_2=np.zeros((10, 10), dtype=np.float32),
            desc_conf_1=np.ones((10, 10), dtype=np.float32),
            desc_conf_2=np.ones((10, 10), dtype=np.float32),
        )
        import torch

        torch_result = result.to_torch()
        assert isinstance(torch_result.desc_conf_1, torch.Tensor)

    def test_to_torch_without_pytorch(self, inference_result, monkeypatch):
        """to_torch raises ImportError when PyTorch not available."""
        # Mock import to fail
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError) as exc_info:
            inference_result.to_torch()
        assert "PyTorch is required" in str(exc_info.value)


# ==============================================================================
# MatchResult Tests
# ==============================================================================


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    @pytest.fixture
    def match_result(self):
        """Create a sample MatchResult."""
        N = 100
        return MatchResult(
            idx_1=np.arange(N, dtype=np.int64),
            idx_2=np.arange(N, dtype=np.int64),
            pts2d_1=np.random.rand(N, 2).astype(np.float32),
            pts2d_2=np.random.rand(N, 2).astype(np.float32),
            pts3d_1=np.random.rand(N, 3).astype(np.float32),
            pts3d_2=np.random.rand(N, 3).astype(np.float32),
            confidence=np.random.rand(N).astype(np.float32),
            timing_ms={"match": 5.0},
        )

    def test_creation(self, match_result):
        """MatchResult can be created."""
        assert match_result is not None

    def test_num_matches_property(self, match_result):
        """num_matches property returns correct count."""
        assert match_result.num_matches == 100

    def test_empty_result(self):
        """Empty MatchResult has num_matches=0."""
        result = MatchResult(
            idx_1=np.array([], dtype=np.int64),
            idx_2=np.array([], dtype=np.int64),
            pts2d_1=np.zeros((0, 2), dtype=np.float32),
            pts2d_2=np.zeros((0, 2), dtype=np.float32),
            pts3d_1=np.zeros((0, 3), dtype=np.float32),
            pts3d_2=np.zeros((0, 3), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
        )
        assert result.num_matches == 0

    def test_timing_default(self):
        """timing_ms defaults to empty dict."""
        result = MatchResult(
            idx_1=np.array([0], dtype=np.int64),
            idx_2=np.array([0], dtype=np.int64),
            pts2d_1=np.zeros((1, 2), dtype=np.float32),
            pts2d_2=np.zeros((1, 2), dtype=np.float32),
            pts3d_1=np.zeros((1, 3), dtype=np.float32),
            pts3d_2=np.zeros((1, 3), dtype=np.float32),
            confidence=np.array([1.0], dtype=np.float32),
        )
        assert result.timing_ms == {}

    @skip_no_torch
    def test_to_torch(self, match_result):
        """to_torch converts arrays to tensors."""
        import torch

        torch_result = match_result.to_torch()

        assert isinstance(torch_result.idx_1, torch.Tensor)
        assert isinstance(torch_result.pts2d_1, torch.Tensor)
        assert isinstance(torch_result.confidence, torch.Tensor)

    @skip_no_torch
    def test_to_torch_preserves_shapes(self, match_result):
        """to_torch preserves array shapes."""
        torch_result = match_result.to_torch()

        assert tuple(torch_result.idx_1.shape) == (100,)
        assert tuple(torch_result.pts2d_1.shape) == (100, 2)
        assert tuple(torch_result.pts3d_1.shape) == (100, 3)


# ==============================================================================
# EngineInterface Tests
# ==============================================================================


class TestEngineInterface:
    """Tests for EngineInterface abstract class."""

    def test_is_abstract(self):
        """EngineInterface cannot be instantiated directly."""
        from mast3r_runtime import MASt3RRuntimeConfig

        config = MASt3RRuntimeConfig()

        with pytest.raises(TypeError):
            EngineInterface(config)  # type: ignore

    def test_subclass_must_implement_abstract_methods(self):
        """Subclasses must implement abstract methods."""

        class IncompleteEngine(EngineInterface):
            pass

        from mast3r_runtime import MASt3RRuntimeConfig

        config = MASt3RRuntimeConfig()

        with pytest.raises(TypeError):
            IncompleteEngine(config)

    def test_complete_subclass(self):
        """Complete subclass can be instantiated."""

        class CompleteEngine(EngineInterface):
            @property
            def name(self) -> str:
                return "Test Engine"

            @property
            def is_ready(self) -> bool:
                return True

            def load(self) -> None:
                pass

            def warmup(self, num_iterations: int = 3) -> None:
                pass

            def infer(self, img1, img2) -> InferenceResult:
                H, W = 24, 24
                return InferenceResult(
                    pts3d_1=np.zeros((H, W, 3), dtype=np.float32),
                    pts3d_2=np.zeros((H, W, 3), dtype=np.float32),
                    desc_1=np.zeros((H, W, 64), dtype=np.float32),
                    desc_2=np.zeros((H, W, 64), dtype=np.float32),
                    conf_1=np.ones((H, W), dtype=np.float32),
                    conf_2=np.ones((H, W), dtype=np.float32),
                )

            def match(
                self, desc_1, desc_2, conf_1=None, conf_2=None, pts3d_1=None, pts3d_2=None
            ) -> MatchResult:
                return MatchResult(
                    idx_1=np.array([0], dtype=np.int64),
                    idx_2=np.array([0], dtype=np.int64),
                    pts2d_1=np.zeros((1, 2), dtype=np.float32),
                    pts2d_2=np.zeros((1, 2), dtype=np.float32),
                    pts3d_1=np.zeros((1, 3), dtype=np.float32),
                    pts3d_2=np.zeros((1, 3), dtype=np.float32),
                    confidence=np.array([1.0], dtype=np.float32),
                )

            def release(self) -> None:
                pass

        from mast3r_runtime import MASt3RRuntimeConfig

        config = MASt3RRuntimeConfig()
        engine = CompleteEngine(config)

        assert engine.name == "Test Engine"
        assert engine.is_ready is True
        assert engine.config == config

    def test_context_manager(self):
        """Engine can be used as context manager."""

        class MockEngine(EngineInterface):
            def __init__(self, config):
                super().__init__(config)
                self._loaded = False
                self._released = False

            @property
            def name(self) -> str:
                return "Mock"

            @property
            def is_ready(self) -> bool:
                return self._loaded

            def load(self) -> None:
                self._loaded = True

            def warmup(self, num_iterations: int = 3) -> None:
                pass

            def infer(self, img1, img2) -> InferenceResult:
                return InferenceResult(
                    pts3d_1=np.zeros((1, 1, 3), dtype=np.float32),
                    pts3d_2=np.zeros((1, 1, 3), dtype=np.float32),
                    desc_1=np.zeros((1, 1, 64), dtype=np.float32),
                    desc_2=np.zeros((1, 1, 64), dtype=np.float32),
                    conf_1=np.ones((1, 1), dtype=np.float32),
                    conf_2=np.ones((1, 1), dtype=np.float32),
                )

            def match(self, desc_1, desc_2, **kwargs) -> MatchResult:
                return MatchResult(
                    idx_1=np.array([], dtype=np.int64),
                    idx_2=np.array([], dtype=np.int64),
                    pts2d_1=np.zeros((0, 2), dtype=np.float32),
                    pts2d_2=np.zeros((0, 2), dtype=np.float32),
                    pts3d_1=np.zeros((0, 3), dtype=np.float32),
                    pts3d_2=np.zeros((0, 3), dtype=np.float32),
                    confidence=np.array([], dtype=np.float32),
                )

            def release(self) -> None:
                self._released = True

        from mast3r_runtime import MASt3RRuntimeConfig

        config = MASt3RRuntimeConfig()

        with MockEngine(config) as engine:
            assert engine._loaded is True
            assert engine._released is False

        assert engine._released is True

    def test_infer_and_match(self):
        """infer_and_match convenience method works."""

        class MockEngine(EngineInterface):
            @property
            def name(self) -> str:
                return "Mock"

            @property
            def is_ready(self) -> bool:
                return True

            def load(self) -> None:
                pass

            def warmup(self, num_iterations: int = 3) -> None:
                pass

            def infer(self, img1, img2) -> InferenceResult:
                return InferenceResult(
                    pts3d_1=np.zeros((24, 24, 3), dtype=np.float32),
                    pts3d_2=np.zeros((24, 24, 3), dtype=np.float32),
                    desc_1=np.random.rand(24, 24, 64).astype(np.float32),
                    desc_2=np.random.rand(24, 24, 64).astype(np.float32),
                    conf_1=np.ones((24, 24), dtype=np.float32),
                    conf_2=np.ones((24, 24), dtype=np.float32),
                )

            def match(
                self, desc_1, desc_2, conf_1=None, conf_2=None, pts3d_1=None, pts3d_2=None
            ) -> MatchResult:
                return MatchResult(
                    idx_1=np.array([0, 1], dtype=np.int64),
                    idx_2=np.array([0, 1], dtype=np.int64),
                    pts2d_1=np.zeros((2, 2), dtype=np.float32),
                    pts2d_2=np.zeros((2, 2), dtype=np.float32),
                    pts3d_1=np.zeros((2, 3), dtype=np.float32),
                    pts3d_2=np.zeros((2, 3), dtype=np.float32),
                    confidence=np.array([0.9, 0.8], dtype=np.float32),
                )

            def release(self) -> None:
                pass

        from mast3r_runtime import MASt3RRuntimeConfig

        config = MASt3RRuntimeConfig()
        engine = MockEngine(config)

        img = np.zeros((336, 336, 3), dtype=np.uint8)
        result, matches = engine.infer_and_match(img, img)

        assert isinstance(result, InferenceResult)
        assert isinstance(matches, MatchResult)
        assert matches.num_matches == 2
