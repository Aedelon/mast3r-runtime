"""Descriptor matching utilities for MASt3R runtime.

Pure numpy implementation for lightweight deployment.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from ..core.engine_interface import MatchResult


def normalize_descriptors(desc: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2 normalize descriptors.

    Args:
        desc: Descriptors [H, W, D] or [N, D]

    Returns:
        Normalized descriptors with same shape
    """
    norm = np.linalg.norm(desc, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)  # Avoid division by zero
    return desc / norm


def compute_similarity_matrix(
    desc_1: NDArray[np.float32],
    desc_2: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute cosine similarity between all descriptor pairs.

    Args:
        desc_1: Descriptors [N1, D]
        desc_2: Descriptors [N2, D]

    Returns:
        Similarity matrix [N1, N2]
    """
    # Normalize
    desc_1_norm = normalize_descriptors(desc_1)
    desc_2_norm = normalize_descriptors(desc_2)

    # Cosine similarity via dot product
    return desc_1_norm @ desc_2_norm.T


def reciprocal_match(
    desc_1: NDArray[np.float32],
    desc_2: NDArray[np.float32],
    conf_1: NDArray[np.float32] | None = None,
    conf_2: NDArray[np.float32] | None = None,
    pts3d_1: NDArray[np.float32] | None = None,
    pts3d_2: NDArray[np.float32] | None = None,
    top_k: int = 512,
    reciprocal: bool = True,
    confidence_threshold: float = 0.5,
) -> MatchResult:
    """Reciprocal top-K matching between descriptors.

    Finds mutual nearest neighbors between two sets of descriptors.

    Args:
        desc_1: Descriptors from view 1 [H, W, D]
        desc_2: Descriptors from view 2 [H, W, D]
        conf_1: Optional confidence from view 1 [H, W]
        conf_2: Optional confidence from view 2 [H, W]
        pts3d_1: Optional 3D points from view 1 [H, W, 3]
        pts3d_2: Optional 3D points from view 2 [H, W, 3]
        top_k: Maximum number of matches to return
        reciprocal: Whether to use reciprocal matching
        confidence_threshold: Minimum match confidence

    Returns:
        MatchResult with correspondences
    """
    timing = {}
    t0 = time.perf_counter()

    _H, W, D = desc_1.shape

    # Flatten spatial dimensions
    desc_1_flat = desc_1.reshape(-1, D)  # [N, D] where N = H * W
    desc_2_flat = desc_2.reshape(-1, D)

    # Compute similarity matrix
    sim = compute_similarity_matrix(desc_1_flat, desc_2_flat)  # [N, N]

    # Apply confidence weighting if provided
    if conf_1 is not None and conf_2 is not None:
        conf_1_flat = conf_1.reshape(-1)  # [N]
        conf_2_flat = conf_2.reshape(-1)  # [N]
        conf_weight = np.outer(conf_1_flat, conf_2_flat)  # [N, N]
        sim = sim * conf_weight

    if reciprocal:
        # Find mutual nearest neighbors
        # Forward: for each in desc_1, find best match in desc_2
        nn_12 = np.argmax(sim, axis=1)  # [N]
        # Backward: for each in desc_2, find best match in desc_1
        nn_21 = np.argmax(sim, axis=0)  # [N]

        # Reciprocal check: nn_21[nn_12[i]] == i
        idx_1_all = np.arange(len(nn_12))
        reciprocal_mask = nn_21[nn_12] == idx_1_all

        # Get reciprocal matches
        idx_1 = idx_1_all[reciprocal_mask]
        idx_2 = nn_12[reciprocal_mask]

        # Get match confidences
        confidence = sim[idx_1, idx_2]

    else:
        # Simple top-K matching without reciprocity
        # For each point in desc_1, find best match in desc_2
        nn_12 = np.argmax(sim, axis=1)  # [N]
        idx_1 = np.arange(len(nn_12))
        idx_2 = nn_12
        confidence = sim[idx_1, idx_2]

    # Filter by confidence threshold
    valid_mask = confidence >= confidence_threshold
    idx_1 = idx_1[valid_mask]
    idx_2 = idx_2[valid_mask]
    confidence = confidence[valid_mask]

    # Keep top-K by confidence
    if len(idx_1) > top_k:
        top_indices = np.argsort(confidence)[-top_k:]
        idx_1 = idx_1[top_indices]
        idx_2 = idx_2[top_indices]
        confidence = confidence[top_indices]

    # Convert flat indices to 2D coordinates
    pts2d_1 = np.stack([idx_1 % W, idx_1 // W], axis=1).astype(np.float32)  # [N, 2] (x, y)
    pts2d_2 = np.stack([idx_2 % W, idx_2 // W], axis=1).astype(np.float32)

    # Get 3D points at match locations
    if pts3d_1 is not None and pts3d_2 is not None:
        pts3d_1_flat = pts3d_1.reshape(-1, 3)
        pts3d_2_flat = pts3d_2.reshape(-1, 3)
        matched_pts3d_1 = pts3d_1_flat[idx_1]
        matched_pts3d_2 = pts3d_2_flat[idx_2]
    else:
        matched_pts3d_1 = np.zeros((len(idx_1), 3), dtype=np.float32)
        matched_pts3d_2 = np.zeros((len(idx_2), 3), dtype=np.float32)

    timing["match_ms"] = (time.perf_counter() - t0) * 1000

    return MatchResult(
        idx_1=idx_1.astype(np.int64),
        idx_2=idx_2.astype(np.int64),
        pts2d_1=pts2d_1,
        pts2d_2=pts2d_2,
        pts3d_1=matched_pts3d_1,
        pts3d_2=matched_pts3d_2,
        confidence=confidence.astype(np.float32),
        timing_ms=timing,
    )


def spatial_filter_matches(
    match_result: MatchResult,
    max_displacement: float = 100.0,
) -> MatchResult:
    """Filter matches by spatial consistency.

    Removes matches where the 2D displacement is too large.

    Args:
        match_result: Input matches
        max_displacement: Maximum allowed pixel displacement

    Returns:
        Filtered MatchResult
    """
    # Compute 2D displacement
    displacement = np.linalg.norm(match_result.pts2d_1 - match_result.pts2d_2, axis=1)
    valid_mask = displacement <= max_displacement

    return MatchResult(
        idx_1=match_result.idx_1[valid_mask],
        idx_2=match_result.idx_2[valid_mask],
        pts2d_1=match_result.pts2d_1[valid_mask],
        pts2d_2=match_result.pts2d_2[valid_mask],
        pts3d_1=match_result.pts3d_1[valid_mask],
        pts3d_2=match_result.pts3d_2[valid_mask],
        confidence=match_result.confidence[valid_mask],
        timing_ms=match_result.timing_ms.copy(),
    )
