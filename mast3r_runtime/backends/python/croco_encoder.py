"""CroCoNet Encoder implementation for MASt3R models.

Pure numpy/Python implementation of CroCo Vision Transformer encoder.
Uses RoPE 2D positional embeddings instead of learnable position embeddings.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .rope2d import PositionGetter, RoPE2D
from .vit_encoder import conv2d, gelu, layer_norm, linear, softmax

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CroCoAttention:
    """Multi-head self-attention with RoPE 2D.

    Unlike standard ViT attention, this uses rotary position embeddings
    applied to queries and keys for position-aware attention.
    """

    def __init__(
        self,
        qkv_weight: NDArray[np.float32],
        qkv_bias: NDArray[np.float32],
        proj_weight: NDArray[np.float32],
        proj_bias: NDArray[np.float32],
        num_heads: int,
        rope: RoPE2D,
    ):
        self.qkv_weight = qkv_weight
        self.qkv_bias = qkv_bias
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias
        self.num_heads = num_heads
        self.rope = rope

        # Infer dimensions
        self.embed_dim = proj_weight.shape[0]
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: NDArray[np.float32],
        positions: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        """Forward pass with RoPE.

        Args:
            x: Input [B, N, D].
            positions: 2D positions [B, N, 2].

        Returns:
            Output [B, N, D].
        """
        B, N, D = x.shape

        # QKV projection [B, N, 3*D]
        qkv = linear(x, self.qkv_weight, self.qkv_bias)

        # Reshape to [B, N, 3, num_heads, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        # Transpose to [3, B, num_heads, N, head_dim]
        qkv = np.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q and K
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        # Attention scores [B, num_heads, N, N]
        attn = (q @ np.swapaxes(k, -2, -1)) * self.scale
        attn = softmax(attn, axis=-1)

        # Apply attention to values [B, num_heads, N, head_dim]
        out = attn @ v

        # Transpose and reshape [B, N, D]
        out = np.transpose(out, (0, 2, 1, 3)).reshape(B, N, D)

        # Output projection
        out = linear(out, self.proj_weight, self.proj_bias)

        return out


class CroCoMLP:
    """MLP block with GELU activation."""

    def __init__(
        self,
        fc1_weight: NDArray[np.float32],
        fc1_bias: NDArray[np.float32],
        fc2_weight: NDArray[np.float32],
        fc2_bias: NDArray[np.float32],
    ):
        self.fc1_weight = fc1_weight
        self.fc1_bias = fc1_bias
        self.fc2_weight = fc2_weight
        self.fc2_bias = fc2_bias

    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass."""
        x = linear(x, self.fc1_weight, self.fc1_bias)
        x = gelu(x)
        x = linear(x, self.fc2_weight, self.fc2_bias)
        return x


class CroCoBlock:
    """CroCo Transformer encoder block.

    Pre-norm architecture with RoPE attention.
    No LayerScale (unlike DINOv2).
    """

    def __init__(
        self,
        norm1_weight: NDArray[np.float32],
        norm1_bias: NDArray[np.float32],
        attn: CroCoAttention,
        norm2_weight: NDArray[np.float32],
        norm2_bias: NDArray[np.float32],
        mlp: CroCoMLP,
    ):
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.attn = attn
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.mlp = mlp

    def __call__(
        self,
        x: NDArray[np.float32],
        positions: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        """Forward pass with pre-norm.

        Args:
            x: Input [B, N, D].
            positions: 2D positions [B, N, 2].

        Returns:
            Output [B, N, D].
        """
        # Self-attention with residual
        residual = x
        x = layer_norm(x, self.norm1_weight, self.norm1_bias)
        x = self.attn(x, positions)
        x = residual + x

        # MLP with residual
        residual = x
        x = layer_norm(x, self.norm2_weight, self.norm2_bias)
        x = self.mlp(x)
        x = residual + x

        return x


class CroCoEncoder:
    """CroCo Vision Transformer encoder for MASt3R models.

    Key differences from DINOv2 ViT:
    - Uses RoPE 2D instead of learnable position embeddings
    - No CLS token or register tokens
    - patch_size = 16 (vs 14 for DINOv2)
    """

    def __init__(
        self,
        weights: dict[str, NDArray[np.float32]],
        num_heads: int = 16,
        rope_freq: float = 100.0,
    ):
        """Initialize encoder from weight dictionary.

        Args:
            weights: Dictionary of numpy arrays from safetensors.
            num_heads: Number of attention heads.
            rope_freq: RoPE frequency base.
        """
        self.num_heads = num_heads

        # Patch embedding
        self.patch_embed_weight = weights["patch_embed.proj.weight"]
        self.patch_embed_bias = weights["patch_embed.proj.bias"]

        # Final norm
        self.norm_weight = weights["enc_norm.weight"]
        self.norm_bias = weights["enc_norm.bias"]

        # RoPE and position getter
        self.rope = RoPE2D(freq=rope_freq)
        self.position_getter = PositionGetter()

        # Build transformer blocks
        self.blocks: list[CroCoBlock] = []
        block_idx = 0
        while f"enc_blocks.{block_idx}.norm1.weight" in weights:
            prefix = f"enc_blocks.{block_idx}"

            attn = CroCoAttention(
                qkv_weight=weights[f"{prefix}.attn.qkv.weight"],
                qkv_bias=weights[f"{prefix}.attn.qkv.bias"],
                proj_weight=weights[f"{prefix}.attn.proj.weight"],
                proj_bias=weights[f"{prefix}.attn.proj.bias"],
                num_heads=num_heads,
                rope=self.rope,
            )

            mlp = CroCoMLP(
                fc1_weight=weights[f"{prefix}.mlp.fc1.weight"],
                fc1_bias=weights[f"{prefix}.mlp.fc1.bias"],
                fc2_weight=weights[f"{prefix}.mlp.fc2.weight"],
                fc2_bias=weights[f"{prefix}.mlp.fc2.bias"],
            )

            block = CroCoBlock(
                norm1_weight=weights[f"{prefix}.norm1.weight"],
                norm1_bias=weights[f"{prefix}.norm1.bias"],
                attn=attn,
                norm2_weight=weights[f"{prefix}.norm2.weight"],
                norm2_bias=weights[f"{prefix}.norm2.bias"],
                mlp=mlp,
            )

            self.blocks.append(block)
            block_idx += 1

        # Infer dimensions
        self.embed_dim = self.patch_embed_weight.shape[0]
        self.patch_size = self.patch_embed_weight.shape[2]

    def forward(
        self,
        x: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Encode image to features.

        Args:
            x: Input image [B, 3, H, W] normalized float32.

        Returns:
            Tuple of:
                - Features [B, N, D] where N = (H/patch_size)*(W/patch_size)
                - Positions [B, N, 2] with (y, x) coordinates
        """
        B, C, H, W = x.shape

        # Patch embedding [B, D, H/p, W/p]
        x = conv2d(x, self.patch_embed_weight, self.patch_embed_bias, stride=self.patch_size)

        # Get grid dimensions
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches = h_patches * w_patches

        # Reshape to sequence [B, N, D]
        x = x.reshape(B, self.embed_dim, -1)  # [B, D, N]
        x = np.transpose(x, (0, 2, 1))  # [B, N, D]

        # Generate positions
        positions = self.position_getter(B, h_patches, w_patches)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, positions)

        # Final layer norm
        x = layer_norm(x, self.norm_weight, self.norm_bias)

        return x, positions

    def __call__(
        self,
        x: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Alias for forward."""
        return self.forward(x)
