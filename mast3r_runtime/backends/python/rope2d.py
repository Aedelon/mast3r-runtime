"""2D Rotary Position Embedding (RoPE) implementation.

Pure numpy implementation of RoPE for CroCoNet/MASt3R models.
Based on the RoPE paper and NAVER's CroCo implementation.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PositionGetter:
    """Generate 2D position coordinates for patches.

    Creates (y, x) coordinates for each patch in the grid.
    Caches results for efficiency.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int], NDArray[np.int64]] = {}

    def __call__(self, batch_size: int, height: int, width: int) -> NDArray[np.int64]:
        """Get position coordinates for a grid of patches.

        Args:
            batch_size: Batch size.
            height: Grid height in patches.
            width: Grid width in patches.

        Returns:
            Position coordinates [B, H*W, 2] with (y, x) for each patch.
        """
        key = (height, width)
        if key not in self._cache:
            # Create coordinate grids
            y = np.arange(height, dtype=np.int64)
            x = np.arange(width, dtype=np.int64)
            # Cartesian product: all (y, x) pairs
            yy, xx = np.meshgrid(y, x, indexing="ij")
            positions = np.stack([yy.ravel(), xx.ravel()], axis=-1)  # [H*W, 2]
            self._cache[key] = positions

        # Expand for batch
        positions = self._cache[key]  # [H*W, 2]
        positions = np.broadcast_to(
            positions[np.newaxis, :, :], (batch_size, height * width, 2)
        ).copy()
        return positions


class RoPE2D:
    """2D Rotary Position Embedding.

    Applies rotary embeddings to query/key tensors based on 2D positions.
    Uses separate frequency bases for y and x dimensions.
    """

    def __init__(self, freq: float = 100.0, f0: float = 1.0) -> None:
        """Initialize RoPE 2D.

        Args:
            freq: Base frequency for position encoding.
            f0: Frequency scaling factor.
        """
        self.freq = freq
        self.f0 = f0
        # Cache for frequencies
        self._freq_cache: dict[int, NDArray[np.float32]] = {}

    def _get_freqs(self, dim: int) -> NDArray[np.float32]:
        """Get frequency bases for given dimension.

        Args:
            dim: Dimension of the embeddings (head_dim).

        Returns:
            Frequencies [dim//4] for each pair of dimensions.
        """
        if dim not in self._freq_cache:
            # Each position (y, x) uses dim/4 frequencies each
            # Total dim = 4 * num_freqs (2 pos * 2 for sin/cos rotation)
            num_freqs = dim // 4
            freqs = 1.0 / (self.freq ** (np.arange(num_freqs, dtype=np.float32) / num_freqs))
            self._freq_cache[dim] = freqs
        return self._freq_cache[dim]

    def __call__(
        self, tokens: NDArray[np.float32], positions: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        """Apply RoPE to tokens based on positions.

        Args:
            tokens: Query or key tokens [B, num_heads, N, head_dim].
            positions: 2D positions [B, N, 2] with (y, x) coordinates.

        Returns:
            Rotated tokens [B, num_heads, N, head_dim].
        """
        B, num_heads, N, head_dim = tokens.shape

        freqs = self._get_freqs(head_dim)  # [dim//4]
        num_freqs = len(freqs)

        # Get y and x positions [B, N]
        pos_y = positions[:, :, 0].astype(np.float32)  # [B, N]
        pos_x = positions[:, :, 1].astype(np.float32)  # [B, N]

        # Compute angles: pos * freq [B, N, num_freqs]
        angles_y = pos_y[:, :, np.newaxis] * freqs * self.f0  # [B, N, num_freqs]
        angles_x = pos_x[:, :, np.newaxis] * freqs * self.f0  # [B, N, num_freqs]

        # Stack angles for y and x: [B, N, dim//2]
        # First half for y, second half for x
        angles = np.concatenate([angles_y, angles_x], axis=-1)  # [B, N, dim//2]

        # Compute sin and cos
        cos = np.cos(angles)  # [B, N, dim//2]
        sin = np.sin(angles)  # [B, N, dim//2]

        # Expand for num_heads
        cos = cos[:, np.newaxis, :, :]  # [B, 1, N, dim//2]
        sin = sin[:, np.newaxis, :, :]  # [B, 1, N, dim//2]

        # Split tokens into two halves
        # First half uses dim//2, second half uses dim//2
        d2 = head_dim // 2
        tokens_1 = tokens[:, :, :, :d2]  # [B, num_heads, N, dim//2]
        tokens_2 = tokens[:, :, :, d2:]  # [B, num_heads, N, dim//2]

        # Apply rotation:
        # rotated_1 = tokens_1 * cos - tokens_2 * sin
        # rotated_2 = tokens_1 * sin + tokens_2 * cos
        rotated_1 = tokens_1 * cos - tokens_2 * sin
        rotated_2 = tokens_1 * sin + tokens_2 * cos

        # Concatenate back
        result = np.concatenate([rotated_1, rotated_2], axis=-1)

        return result.astype(np.float32)
