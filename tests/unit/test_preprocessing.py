"""Unit tests for mast3r_runtime.core.preprocessing.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from mast3r_runtime.core.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NumpyPreprocessor,
    PreprocessorBase,
    prepare_image_numpy,
)


# ==============================================================================
# Constants Tests
# ==============================================================================


class TestConstants:
    """Tests for preprocessing constants."""

    def test_imagenet_mean(self):
        """ImageNet mean values are correct."""
        assert len(IMAGENET_MEAN) == 3
        assert IMAGENET_MEAN == (0.485, 0.456, 0.406)

    def test_imagenet_std(self):
        """ImageNet std values are correct."""
        assert len(IMAGENET_STD) == 3
        assert IMAGENET_STD == (0.229, 0.224, 0.225)


# ==============================================================================
# PreprocessorBase Tests
# ==============================================================================


class TestPreprocessorBase:
    """Tests for PreprocessorBase abstract class."""

    def test_is_abstract(self):
        """PreprocessorBase is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            PreprocessorBase()  # type: ignore

    def test_resolution_validation(self):
        """Resolution must be divisible by patch_size."""

        class ConcretePreprocessor(PreprocessorBase):
            def __call__(self, image):
                return image

        # Valid: 336 % 14 == 0
        prep = ConcretePreprocessor(resolution=336, patch_size=14)
        assert prep.resolution == 336

        # Invalid: 337 % 14 != 0
        with pytest.raises(ValueError) as exc_info:
            ConcretePreprocessor(resolution=337, patch_size=14)
        assert "divisible by patch_size" in str(exc_info.value)


# ==============================================================================
# NumpyPreprocessor Tests
# ==============================================================================


class TestNumpyPreprocessor:
    """Tests for NumpyPreprocessor."""

    def test_init_default(self):
        """Default initialization."""
        prep = NumpyPreprocessor()
        assert prep.resolution == 336
        assert prep.patch_size == 14
        assert prep.mean == IMAGENET_MEAN
        assert prep.std == IMAGENET_STD

    def test_init_custom(self):
        """Custom initialization."""
        prep = NumpyPreprocessor(
            resolution=448,
            patch_size=14,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        assert prep.resolution == 448
        assert prep.mean == (0.5, 0.5, 0.5)

    def test_output_shape(self, random_rgb_image):
        """Output has correct shape."""
        prep = NumpyPreprocessor(resolution=336)
        output = prep(random_rgb_image)

        assert output.shape == (1, 3, 336, 336)
        assert output.dtype == np.float32

    def test_output_shape_different_resolution(self, random_rgb_image):
        """Output shape matches resolution."""
        prep = NumpyPreprocessor(resolution=224)
        output = prep(random_rgb_image)

        assert output.shape == (1, 3, 224, 224)

    def test_normalization_applied(self, gradient_image):
        """ImageNet normalization is applied."""
        prep = NumpyPreprocessor(resolution=336)
        output = prep(gradient_image)

        # After normalization, values should be roughly in [-3, 3] range
        assert output.min() < 0
        assert output.max() > 0
        assert output.min() > -10
        assert output.max() < 10

    def test_invalid_input_shape(self):
        """Invalid input shape raises error."""
        prep = NumpyPreprocessor()

        # 2D array
        with pytest.raises(ValueError) as exc_info:
            prep(np.zeros((336, 336), dtype=np.uint8))
        assert "Expected [H, W, 3]" in str(exc_info.value)

        # 4 channels
        with pytest.raises(ValueError) as exc_info:
            prep(np.zeros((336, 336, 4), dtype=np.uint8))
        assert "Expected [H, W, 3]" in str(exc_info.value)

    def test_handles_non_square_input(self):
        """Handles non-square input images."""
        prep = NumpyPreprocessor(resolution=336)
        # Wide image
        wide = np.zeros((240, 640, 3), dtype=np.uint8)
        output = prep(wide)
        assert output.shape == (1, 3, 336, 336)

        # Tall image
        tall = np.zeros((640, 240, 3), dtype=np.uint8)
        output = prep(tall)
        assert output.shape == (1, 3, 336, 336)

    def test_handles_small_input(self):
        """Handles input smaller than target resolution."""
        prep = NumpyPreprocessor(resolution=336)
        small = np.zeros((100, 100, 3), dtype=np.uint8)
        output = prep(small)
        assert output.shape == (1, 3, 336, 336)

    def test_handles_large_input(self):
        """Handles input larger than target resolution."""
        prep = NumpyPreprocessor(resolution=336)
        large = np.zeros((1000, 1000, 3), dtype=np.uint8)
        output = prep(large)
        assert output.shape == (1, 3, 336, 336)

    def test_deterministic(self, random_rgb_image):
        """Output is deterministic for same input."""
        prep = NumpyPreprocessor(resolution=336)
        output1 = prep(random_rgb_image)
        output2 = prep(random_rgb_image)
        np.testing.assert_array_equal(output1, output2)

    def test_preserves_batch_content(self):
        """Different images produce different outputs."""
        prep = NumpyPreprocessor(resolution=336)

        img1 = np.zeros((336, 336, 3), dtype=np.uint8)
        img2 = np.ones((336, 336, 3), dtype=np.uint8) * 255

        output1 = prep(img1)
        output2 = prep(img2)

        assert not np.allclose(output1, output2)


# ==============================================================================
# prepare_image_numpy Tests
# ==============================================================================


class TestPrepareImageNumpy:
    """Tests for prepare_image_numpy convenience function."""

    def test_default_resolution(self, random_rgb_image):
        """Default resolution is 336."""
        output = prepare_image_numpy(random_rgb_image)
        assert output.shape == (1, 3, 336, 336)

    def test_custom_resolution(self, random_rgb_image):
        """Custom resolution is respected."""
        output = prepare_image_numpy(random_rgb_image, resolution=224)
        assert output.shape == (1, 3, 224, 224)

    def test_custom_patch_size(self, random_rgb_image):
        """Custom patch_size is validated."""
        # 336 is divisible by 16
        output = prepare_image_numpy(random_rgb_image, resolution=336, patch_size=16)
        assert output.shape == (1, 3, 336, 336)

    def test_invalid_resolution_patch_size(self, random_rgb_image):
        """Invalid resolution/patch_size combination raises error."""
        with pytest.raises(ValueError):
            prepare_image_numpy(random_rgb_image, resolution=337, patch_size=14)

    def test_output_dtype(self, random_rgb_image):
        """Output is float32."""
        output = prepare_image_numpy(random_rgb_image)
        assert output.dtype == np.float32


# ==============================================================================
# Edge Cases and Performance Tests
# ==============================================================================


class TestPreprocessingEdgeCases:
    """Edge case tests for preprocessing."""

    def test_all_zeros_image(self):
        """Black image (all zeros) is processed correctly."""
        prep = NumpyPreprocessor(resolution=336)
        black = np.zeros((336, 336, 3), dtype=np.uint8)
        output = prep(black)

        # After normalization: (0.0 - mean) / std
        expected_r = (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        expected_g = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        expected_b = (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

        np.testing.assert_allclose(output[0, 0].mean(), expected_r, rtol=0.01)
        np.testing.assert_allclose(output[0, 1].mean(), expected_g, rtol=0.01)
        np.testing.assert_allclose(output[0, 2].mean(), expected_b, rtol=0.01)

    def test_all_ones_image(self):
        """White image (all 255) is processed correctly."""
        prep = NumpyPreprocessor(resolution=336)
        white = np.ones((336, 336, 3), dtype=np.uint8) * 255
        output = prep(white)

        # After normalization: (1.0 - mean) / std
        expected_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        expected_g = (1.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        expected_b = (1.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

        np.testing.assert_allclose(output[0, 0].mean(), expected_r, rtol=0.01)
        np.testing.assert_allclose(output[0, 1].mean(), expected_g, rtol=0.01)
        np.testing.assert_allclose(output[0, 2].mean(), expected_b, rtol=0.01)

    def test_single_color_channels(self):
        """Each color channel is processed independently."""
        prep = NumpyPreprocessor(resolution=336)

        # Red image
        red = np.zeros((336, 336, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        output = prep(red)
        assert output[0, 0].mean() > output[0, 1].mean()  # Red > Green
        assert output[0, 0].mean() > output[0, 2].mean()  # Red > Blue

        # Green image
        green = np.zeros((336, 336, 3), dtype=np.uint8)
        green[:, :, 1] = 255
        output = prep(green)
        assert output[0, 1].mean() > output[0, 0].mean()  # Green > Red
        assert output[0, 1].mean() > output[0, 2].mean()  # Green > Blue

    def test_1x1_image(self):
        """1x1 image is processed (upscaled)."""
        prep = NumpyPreprocessor(resolution=336)
        tiny = np.array([[[128, 64, 32]]], dtype=np.uint8)  # 1x1x3
        output = prep(tiny)
        assert output.shape == (1, 3, 336, 336)

    def test_chw_format(self, random_rgb_image):
        """Output is in CHW format (channels first)."""
        prep = NumpyPreprocessor(resolution=336)
        output = prep(random_rgb_image)

        # Shape should be (1, 3, H, W)
        assert output.shape[1] == 3  # Channels
        assert output.shape[2] == 336  # Height
        assert output.shape[3] == 336  # Width
