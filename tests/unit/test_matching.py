"""Unit tests for mast3r_runtime.backends.matching.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from mast3r_runtime.backends.matching import (
    compute_similarity_matrix,
    normalize_descriptors,
    reciprocal_match,
    spatial_filter_matches,
)


# ==============================================================================
# normalize_descriptors Tests
# ==============================================================================


class TestNormalizeDescriptors:
    """Tests for normalize_descriptors function."""

    def test_2d_input(self):
        """Normalizes 2D input [N, D]."""
        desc = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        normalized = normalize_descriptors(desc)

        # First row: [3, 4] / 5 = [0.6, 0.8]
        np.testing.assert_allclose(normalized[0], [0.6, 0.8])
        # Second row: [1, 0] / 1 = [1, 0]
        np.testing.assert_allclose(normalized[1], [1.0, 0.0])

    def test_3d_input(self):
        """Normalizes 3D input [H, W, D]."""
        desc = np.zeros((2, 2, 3), dtype=np.float32)
        desc[0, 0] = [3.0, 4.0, 0.0]  # norm = 5
        desc[0, 1] = [1.0, 0.0, 0.0]  # norm = 1

        normalized = normalize_descriptors(desc)

        np.testing.assert_allclose(normalized[0, 0], [0.6, 0.8, 0.0])
        np.testing.assert_allclose(normalized[0, 1], [1.0, 0.0, 0.0])

    def test_output_has_unit_norm(self, random_descriptors):
        """Output vectors have unit norm."""
        desc_1, _ = random_descriptors
        normalized = normalize_descriptors(desc_1)

        # Flatten to [N, D] for norm calculation
        flat = normalized.reshape(-1, normalized.shape[-1])
        norms = np.linalg.norm(flat, axis=-1)

        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_handles_zero_vectors(self):
        """Handles zero vectors without NaN."""
        desc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        normalized = normalize_descriptors(desc)

        # Should not contain NaN (epsilon prevents division by zero)
        assert not np.any(np.isnan(normalized))

    def test_preserves_dtype(self):
        """Preserves float32 dtype."""
        desc = np.random.rand(10, 256).astype(np.float32)
        normalized = normalize_descriptors(desc)
        assert normalized.dtype == np.float32

    def test_preserves_shape(self, random_descriptors):
        """Preserves input shape."""
        desc_1, _ = random_descriptors
        normalized = normalize_descriptors(desc_1)
        assert normalized.shape == desc_1.shape


# ==============================================================================
# compute_similarity_matrix Tests
# ==============================================================================


class TestComputeSimilarityMatrix:
    """Tests for compute_similarity_matrix function."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1."""
        desc = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sim = compute_similarity_matrix(desc, desc)

        # Diagonal should be 1.0 (identical vectors)
        np.testing.assert_allclose(sim.diagonal(), 1.0, rtol=1e-5)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0."""
        desc_1 = np.array([[1.0, 0.0]], dtype=np.float32)
        desc_2 = np.array([[0.0, 1.0]], dtype=np.float32)
        sim = compute_similarity_matrix(desc_1, desc_2)

        np.testing.assert_allclose(sim[0, 0], 0.0, atol=1e-5)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1."""
        desc_1 = np.array([[1.0, 0.0]], dtype=np.float32)
        desc_2 = np.array([[-1.0, 0.0]], dtype=np.float32)
        sim = compute_similarity_matrix(desc_1, desc_2)

        np.testing.assert_allclose(sim[0, 0], -1.0, rtol=1e-5)

    def test_output_shape(self, random_descriptors):
        """Output has shape [N1, N2]."""
        desc_1, desc_2 = random_descriptors
        N1 = desc_1.shape[0] * desc_1.shape[1]
        N2 = desc_2.shape[0] * desc_2.shape[1]

        flat_1 = desc_1.reshape(-1, desc_1.shape[-1])
        flat_2 = desc_2.reshape(-1, desc_2.shape[-1])

        sim = compute_similarity_matrix(flat_1, flat_2)
        assert sim.shape == (N1, N2)

    def test_symmetry(self):
        """Similarity matrix is symmetric for same input."""
        desc = np.random.rand(10, 64).astype(np.float32)
        sim = compute_similarity_matrix(desc, desc)

        np.testing.assert_allclose(sim, sim.T, rtol=1e-5)

    def test_values_in_range(self, random_descriptors):
        """Similarity values are in [-1, 1]."""
        desc_1, desc_2 = random_descriptors
        flat_1 = desc_1.reshape(-1, desc_1.shape[-1])
        flat_2 = desc_2.reshape(-1, desc_2.shape[-1])

        sim = compute_similarity_matrix(flat_1, flat_2)

        assert sim.min() >= -1.0 - 1e-5
        assert sim.max() <= 1.0 + 1e-5


# ==============================================================================
# reciprocal_match Tests
# ==============================================================================


class TestReciprocalMatch:
    """Tests for reciprocal_match function."""

    def test_basic_match(self, random_descriptors):
        """Basic matching returns MatchResult."""
        desc_1, desc_2 = random_descriptors
        result = reciprocal_match(desc_1, desc_2)

        assert hasattr(result, "idx_1")
        assert hasattr(result, "idx_2")
        assert hasattr(result, "pts2d_1")
        assert hasattr(result, "pts2d_2")
        assert hasattr(result, "confidence")

    def test_result_shapes(self, random_descriptors):
        """Result arrays have consistent shapes."""
        desc_1, desc_2 = random_descriptors
        result = reciprocal_match(desc_1, desc_2)

        N = len(result.idx_1)
        assert len(result.idx_2) == N
        assert result.pts2d_1.shape == (N, 2)
        assert result.pts2d_2.shape == (N, 2)
        assert result.confidence.shape == (N,)

    def test_with_confidence(self, random_descriptors, confidence_maps):
        """Matching with confidence maps."""
        desc_1, desc_2 = random_descriptors
        conf_1, conf_2 = confidence_maps
        result = reciprocal_match(desc_1, desc_2, conf_1, conf_2)

        assert result.num_matches >= 0

    def test_with_pts3d(self, random_descriptors, pts3d_maps):
        """Matching with 3D points."""
        desc_1, desc_2 = random_descriptors
        pts3d_1, pts3d_2 = pts3d_maps
        result = reciprocal_match(desc_1, desc_2, pts3d_1=pts3d_1, pts3d_2=pts3d_2)

        assert result.pts3d_1.shape[1] == 3
        assert result.pts3d_2.shape[1] == 3

    def test_top_k_limit(self, random_descriptors):
        """top_k limits number of matches."""
        desc_1, desc_2 = random_descriptors
        result = reciprocal_match(desc_1, desc_2, top_k=10)

        assert result.num_matches <= 10

    def test_reciprocal_true(self, random_descriptors):
        """reciprocal=True filters non-mutual matches."""
        desc_1, desc_2 = random_descriptors
        result_recip = reciprocal_match(desc_1, desc_2, reciprocal=True)
        result_no_recip = reciprocal_match(desc_1, desc_2, reciprocal=False)

        # Reciprocal matching typically returns fewer matches
        assert result_recip.num_matches <= result_no_recip.num_matches

    def test_confidence_threshold(self, random_descriptors):
        """confidence_threshold filters low-confidence matches."""
        desc_1, desc_2 = random_descriptors
        result_low = reciprocal_match(desc_1, desc_2, confidence_threshold=0.0)
        result_high = reciprocal_match(desc_1, desc_2, confidence_threshold=0.9)

        assert result_high.num_matches <= result_low.num_matches

    def test_timing_info(self, random_descriptors):
        """Timing information is recorded."""
        desc_1, desc_2 = random_descriptors
        result = reciprocal_match(desc_1, desc_2)

        assert "match_ms" in result.timing_ms
        assert result.timing_ms["match_ms"] >= 0

    def test_pts2d_coordinates_valid(self, random_descriptors):
        """2D coordinates are within image bounds."""
        desc_1, desc_2 = random_descriptors
        H, W, _ = desc_1.shape
        result = reciprocal_match(desc_1, desc_2)

        if result.num_matches > 0:
            assert result.pts2d_1[:, 0].min() >= 0  # x >= 0
            assert result.pts2d_1[:, 0].max() < W  # x < W
            assert result.pts2d_1[:, 1].min() >= 0  # y >= 0
            assert result.pts2d_1[:, 1].max() < H  # y < H

    def test_identical_descriptors(self):
        """Identical descriptors produce diagonal matches."""
        desc = np.random.rand(8, 8, 64).astype(np.float32)
        result = reciprocal_match(desc, desc, reciprocal=True, confidence_threshold=0.0)

        # Should match each point to itself
        np.testing.assert_array_equal(result.idx_1, result.idx_2)


# ==============================================================================
# spatial_filter_matches Tests
# ==============================================================================


class TestSpatialFilterMatches:
    """Tests for spatial_filter_matches function."""

    def test_filters_large_displacements(self, random_descriptors):
        """Filters matches with large 2D displacements."""
        desc_1, desc_2 = random_descriptors
        result = reciprocal_match(desc_1, desc_2, confidence_threshold=0.0)

        filtered = spatial_filter_matches(result, max_displacement=5.0)

        assert filtered.num_matches <= result.num_matches

    def test_preserves_close_matches(self):
        """Preserves matches with small displacements."""
        # Create a match result with known displacements
        from mast3r_runtime.core.engine_interface import MatchResult

        result = MatchResult(
            idx_1=np.array([0, 1, 2]),
            idx_2=np.array([0, 1, 2]),
            pts2d_1=np.array([[0.0, 0.0], [10.0, 0.0], [50.0, 50.0]], dtype=np.float32),
            pts2d_2=np.array([[1.0, 1.0], [15.0, 0.0], [100.0, 100.0]], dtype=np.float32),
            pts3d_1=np.zeros((3, 3), dtype=np.float32),
            pts3d_2=np.zeros((3, 3), dtype=np.float32),
            confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
            timing_ms={},
        )

        filtered = spatial_filter_matches(result, max_displacement=10.0)

        # First two matches should pass (displacement < 10)
        # Third match should be filtered (displacement ~70.7)
        assert filtered.num_matches == 2

    def test_no_filtering_with_large_threshold(self, random_descriptors):
        """No filtering with very large threshold."""
        desc_1, desc_2 = random_descriptors
        result = reciprocal_match(desc_1, desc_2, confidence_threshold=0.0)

        filtered = spatial_filter_matches(result, max_displacement=10000.0)

        assert filtered.num_matches == result.num_matches

    def test_preserves_timing(self, random_descriptors):
        """Preserves timing information."""
        desc_1, desc_2 = random_descriptors
        result = reciprocal_match(desc_1, desc_2)

        filtered = spatial_filter_matches(result, max_displacement=50.0)

        assert filtered.timing_ms == result.timing_ms


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestMatchingEdgeCases:
    """Edge case tests for matching functions."""

    def test_empty_descriptors(self):
        """Handles empty descriptor arrays."""
        desc = np.zeros((0, 0, 64), dtype=np.float32)

        # This might raise or return empty - depends on implementation
        try:
            result = reciprocal_match(desc, desc)
            assert result.num_matches == 0
        except (ValueError, IndexError):
            pass  # Expected for empty input

    def test_single_pixel(self):
        """Handles single-pixel descriptor maps."""
        desc = np.random.rand(1, 1, 64).astype(np.float32)
        result = reciprocal_match(desc, desc)

        # Single pixel matches itself
        assert result.num_matches <= 1

    def test_very_high_dimension(self):
        """Handles high-dimensional descriptors."""
        desc = np.random.rand(4, 4, 1024).astype(np.float32)
        result = reciprocal_match(desc, desc)

        assert result.num_matches >= 0

    def test_all_same_descriptor(self):
        """Handles all identical descriptors."""
        desc = np.ones((8, 8, 64), dtype=np.float32)
        result = reciprocal_match(desc, desc)

        # All descriptors are identical, so any match is valid
        assert result.num_matches >= 0

    def test_orthogonal_descriptors(self):
        """Handles orthogonal descriptor sets."""
        # Create two orthogonal sets
        desc_1 = np.zeros((4, 4, 64), dtype=np.float32)
        desc_1[:, :, :32] = 1.0  # First half
        desc_2 = np.zeros((4, 4, 64), dtype=np.float32)
        desc_2[:, :, 32:] = 1.0  # Second half

        result = reciprocal_match(desc_1, desc_2, confidence_threshold=0.5)

        # Orthogonal descriptors should have low similarity
        # Most matches should be filtered by confidence
        assert result.num_matches < 16  # Less than total pixels
