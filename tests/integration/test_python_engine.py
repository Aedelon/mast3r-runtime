"""Integration tests for mast3r_runtime.backends.python.python_engine.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from mast3r_runtime import MASt3RRuntimeConfig
from mast3r_runtime.backends.python.python_engine import PythonEngine
from mast3r_runtime.core.engine_interface import InferenceResult, MatchResult


# ==============================================================================
# PythonEngine Tests
# ==============================================================================


class TestPythonEngine:
    """Tests for PythonEngine."""

    @pytest.fixture
    def config(self):
        """Create a test config with small resolution for fast tests."""
        # Use 224 (minimum valid) to make tests faster
        # Default 448x448 creates 200K descriptors which is too slow for matching
        return MASt3RRuntimeConfig(model={"resolution": 224})

    @pytest.fixture
    def engine(self, config):
        """Create a PythonEngine instance in placeholder mode for fast tests."""
        return PythonEngine(config, placeholder=True)

    def test_creation(self, engine):
        """Engine can be created."""
        assert engine is not None
        assert isinstance(engine, PythonEngine)

    def test_name(self, engine):
        """Engine has correct name."""
        assert engine.name == "Python (numpy)"

    def test_not_ready_initially(self, engine):
        """Engine is not ready before load."""
        assert engine.is_ready is False

    def test_load(self, engine):
        """Engine can be loaded."""
        engine.load()
        assert engine.is_ready is True

    def test_warmup(self, engine):
        """Warmup runs without error."""
        engine.warmup(num_iterations=1)
        # Warmup should also load the engine
        assert engine.is_ready is True

    def test_infer(self, engine, config, random_rgb_image):
        """Inference produces correct output shapes (placeholder mode)."""
        engine.load()
        result = engine.infer(random_rgb_image, random_rgb_image)

        # Placeholder mode outputs at feature map size (resolution / patch_size)
        feat_size = config.model.resolution // 14
        assert isinstance(result, InferenceResult)
        assert result.pts3d_1.shape == (feat_size, feat_size, 3)
        assert result.pts3d_2.shape == (feat_size, feat_size, 3)
        assert result.desc_1.shape == (feat_size, feat_size, 256)
        assert result.conf_1.shape == (feat_size, feat_size)

    def test_infer_auto_loads(self, engine, random_rgb_image):
        """Inference auto-loads if not ready."""
        assert engine.is_ready is False
        result = engine.infer(random_rgb_image, random_rgb_image)
        assert engine.is_ready is True
        assert isinstance(result, InferenceResult)

    def test_infer_timing(self, engine, random_rgb_image):
        """Inference records timing."""
        engine.load()
        result = engine.infer(random_rgb_image, random_rgb_image)

        assert "preprocess_ms" in result.timing_ms
        assert "inference_ms" in result.timing_ms
        assert "total_ms" in result.timing_ms
        assert result.timing_ms["total_ms"] >= 0

    def test_infer_with_resize(self, engine, config):
        """Inference handles images needing resize (placeholder mode)."""
        engine.load()
        # Non-square image that needs resize
        img = np.zeros((240, 640, 3), dtype=np.uint8)
        result = engine.infer(img, img)

        # Placeholder mode outputs at feature map size
        feat_size = config.model.resolution // 14
        assert result.pts3d_1.shape == (feat_size, feat_size, 3)

    def test_match(self, engine, random_descriptors):
        """Matching produces correct output."""
        engine.load()
        desc_1, desc_2 = random_descriptors
        result = engine.match(desc_1, desc_2)

        assert isinstance(result, MatchResult)
        assert result.num_matches >= 0

    def test_match_with_confidence(self, engine, random_descriptors, confidence_maps):
        """Matching works with confidence maps."""
        engine.load()
        desc_1, desc_2 = random_descriptors
        conf_1, conf_2 = confidence_maps
        result = engine.match(desc_1, desc_2, conf_1, conf_2)

        assert isinstance(result, MatchResult)

    def test_match_with_pts3d(self, engine, random_descriptors, pts3d_maps):
        """Matching works with 3D points."""
        engine.load()
        desc_1, desc_2 = random_descriptors
        pts3d_1, pts3d_2 = pts3d_maps
        result = engine.match(desc_1, desc_2, pts3d_1=pts3d_1, pts3d_2=pts3d_2)

        assert isinstance(result, MatchResult)
        if result.num_matches > 0:
            assert result.pts3d_1.shape[1] == 3

    def test_release(self, engine):
        """Release frees resources."""
        engine.load()
        assert engine.is_ready is True

        engine.release()
        assert engine.is_ready is False

    def test_context_manager(self, config, random_rgb_image):
        """Engine works as context manager (placeholder mode)."""
        with PythonEngine(config, placeholder=True) as engine:
            assert engine.is_ready is True
            result = engine.infer(random_rgb_image, random_rgb_image)
            assert isinstance(result, InferenceResult)

        # After context, engine should be released
        assert engine.is_ready is False

    def test_config_access(self, engine, config):
        """Config is accessible."""
        assert engine.config == config

    def test_infer_and_match_convenience(self, engine, random_rgb_image):
        """infer_and_match convenience method works."""
        engine.load()
        result, matches = engine.infer_and_match(random_rgb_image, random_rgb_image)

        assert isinstance(result, InferenceResult)
        assert isinstance(matches, MatchResult)


# ==============================================================================
# Preprocessing Tests
# ==============================================================================


class TestPythonEnginePreprocessing:
    """Tests for PythonEngine preprocessing."""

    @pytest.fixture
    def config(self):
        """Create config with small resolution for fast tests."""
        return MASt3RRuntimeConfig(model={"resolution": 224})

    @pytest.fixture
    def engine(self, config):
        """Create engine in placeholder mode for fast tests."""
        return PythonEngine(config, placeholder=True)

    def test_preprocess_normalizes(self, engine, config):
        """Preprocessing normalizes to expected range."""
        res = config.model.resolution
        img = np.ones((res, res, 3), dtype=np.uint8) * 128
        preprocessed = engine._preprocess(img)

        # Should be in roughly [-2, 2] range after normalization
        assert preprocessed.min() > -3
        assert preprocessed.max() < 3

    def test_preprocess_shape(self, engine, config):
        """Preprocessing produces correct shape."""
        res = config.model.resolution
        img = np.zeros((res, res, 3), dtype=np.uint8)
        preprocessed = engine._preprocess(img)

        assert preprocessed.shape == (1, 3, res, res)
        assert preprocessed.dtype == np.float32

    def test_preprocess_resizes_small(self, engine, config):
        """Preprocessing resizes small images."""
        res = config.model.resolution
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        preprocessed = engine._preprocess(img)

        assert preprocessed.shape == (1, 3, res, res)

    def test_preprocess_resizes_large(self, engine, config):
        """Preprocessing resizes large images."""
        res = config.model.resolution
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        preprocessed = engine._preprocess(img)

        assert preprocessed.shape == (1, 3, res, res)

    def test_preprocess_handles_wide(self, engine, config):
        """Preprocessing handles wide images."""
        res = config.model.resolution
        img = np.zeros((200, 800, 3), dtype=np.uint8)
        preprocessed = engine._preprocess(img)

        assert preprocessed.shape == (1, 3, res, res)

    def test_preprocess_handles_tall(self, engine, config):
        """Preprocessing handles tall images."""
        res = config.model.resolution
        img = np.zeros((800, 200, 3), dtype=np.uint8)
        preprocessed = engine._preprocess(img)

        assert preprocessed.shape == (1, 3, res, res)
