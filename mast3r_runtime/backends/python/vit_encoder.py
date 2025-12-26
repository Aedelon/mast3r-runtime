"""ViT Encoder implementation for DUNE/MASt3R models.

Pure numpy/Python implementation of Vision Transformer encoder.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def gelu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """GELU activation function."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def interpolate_pos_embed(
    pos_embed: NDArray[np.float32],
    target_num_patches: int,
) -> NDArray[np.float32]:
    """Interpolate position embeddings to new resolution.

    Used when running inference at a different resolution than training.
    This is standard practice in ViT/DINOv2 for resolution flexibility.

    Args:
        pos_embed: Original position embeddings [1, 1+N, D] with CLS token.
        target_num_patches: Target number of patches (H/patch_size * W/patch_size).

    Returns:
        Interpolated position embeddings [1, 1+target_num_patches, D].
    """
    # Separate CLS token from patch embeddings
    cls_embed = pos_embed[:, :1]  # [1, 1, D]
    patch_embed = pos_embed[:, 1:]  # [1, N, D]

    src_num_patches = patch_embed.shape[1]

    # If sizes match, no interpolation needed
    if src_num_patches == target_num_patches:
        return pos_embed

    # Infer source grid size (assume square)
    src_size = int(src_num_patches**0.5)
    tgt_size = int(target_num_patches**0.5)
    embed_dim = patch_embed.shape[2]

    # Reshape to spatial [1, H, W, D]
    patch_embed = patch_embed.reshape(1, src_size, src_size, embed_dim)

    # Transpose to [1, D, H, W] for interpolation
    patch_embed = np.transpose(patch_embed, (0, 3, 1, 2))

    # Bilinear interpolation
    B, D, H, W = patch_embed.shape
    H_out, W_out = tgt_size, tgt_size

    output = np.zeros((B, D, H_out, W_out), dtype=patch_embed.dtype)
    scale_h = H / H_out
    scale_w = W / W_out

    for i in range(H_out):
        for j in range(W_out):
            src_h = i * scale_h
            src_w = j * scale_w
            h0, w0 = int(src_h), int(src_w)
            h1, w1 = min(h0 + 1, H - 1), min(w0 + 1, W - 1)
            fh, fw = src_h - h0, src_w - w0

            output[:, :, i, j] = (
                (1 - fh) * (1 - fw) * patch_embed[:, :, h0, w0]
                + (1 - fh) * fw * patch_embed[:, :, h0, w1]
                + fh * (1 - fw) * patch_embed[:, :, h1, w0]
                + fh * fw * patch_embed[:, :, h1, w1]
            )

    # Reshape back to [1, N, D]
    output = np.transpose(output, (0, 2, 3, 1))  # [1, H, W, D]
    output = output.reshape(1, target_num_patches, embed_dim)

    # Concatenate CLS token back
    return np.concatenate([cls_embed, output], axis=1)


def softmax(x: NDArray[np.float32], axis: int = -1) -> NDArray[np.float32]:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(
    x: NDArray[np.float32],
    weight: NDArray[np.float32],
    bias: NDArray[np.float32],
    eps: float = 1e-6,
) -> NDArray[np.float32]:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return weight * x_norm + bias


def linear(
    x: NDArray[np.float32],
    weight: NDArray[np.float32],
    bias: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Linear layer: y = xW^T + b."""
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def conv2d(
    x: NDArray[np.float32],
    weight: NDArray[np.float32],
    bias: NDArray[np.float32] | None = None,
    stride: int = 1,
) -> NDArray[np.float32]:
    """2D convolution for patch embedding.

    Args:
        x: Input [B, C, H, W]
        weight: Kernel [out_channels, in_channels, kH, kW]
        bias: Optional bias [out_channels]
        stride: Stride (same for H and W)

    Returns:
        Output [B, out_channels, H', W']
    """
    B, C, H, W = x.shape
    out_channels, in_channels, kH, kW = weight.shape

    assert C == in_channels
    assert kH == kW  # Square kernel
    assert stride == kH  # Non-overlapping patches

    H_out = H // stride
    W_out = W // stride

    # Extract patches and apply conv as matrix multiply
    output = np.zeros((B, out_channels, H_out, W_out), dtype=x.dtype)

    for i in range(H_out):
        for j in range(W_out):
            # Extract patch [B, C, kH, kW]
            patch = x[:, :, i * stride : i * stride + kH, j * stride : j * stride + kW]
            # Reshape to [B, C * kH * kW]
            patch_flat = patch.reshape(B, -1)
            # Weight is [out_channels, C * kH * kW]
            weight_flat = weight.reshape(out_channels, -1)
            # Output [B, out_channels]
            output[:, :, i, j] = patch_flat @ weight_flat.T

    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)

    return output


class MultiHeadAttention:
    """Multi-head self-attention."""

    def __init__(
        self,
        qkv_weight: NDArray[np.float32],
        qkv_bias: NDArray[np.float32],
        proj_weight: NDArray[np.float32],
        proj_bias: NDArray[np.float32],
        num_heads: int,
    ):
        self.qkv_weight = qkv_weight
        self.qkv_bias = qkv_bias
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias
        self.num_heads = num_heads

        # Infer dimensions
        self.embed_dim = proj_weight.shape[0]
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim**-0.5

    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass.

        Args:
            x: Input [B, N, D]

        Returns:
            Output [B, N, D]
        """
        B, N, D = x.shape

        # QKV projection [B, N, 3*D]
        qkv = linear(x, self.qkv_weight, self.qkv_bias)

        # Reshape to [B, N, 3, num_heads, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        # Transpose to [3, B, num_heads, N, head_dim]
        qkv = np.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

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


class MLP:
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


class TransformerBlock:
    """Transformer encoder block with LayerScale."""

    def __init__(
        self,
        norm1_weight: NDArray[np.float32],
        norm1_bias: NDArray[np.float32],
        attn: MultiHeadAttention,
        ls1_gamma: NDArray[np.float32],
        norm2_weight: NDArray[np.float32],
        norm2_bias: NDArray[np.float32],
        mlp: MLP,
        ls2_gamma: NDArray[np.float32],
    ):
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.attn = attn
        self.ls1_gamma = ls1_gamma
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.mlp = mlp
        self.ls2_gamma = ls2_gamma

    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass with pre-norm and LayerScale."""
        # Self-attention with residual
        residual = x
        x = layer_norm(x, self.norm1_weight, self.norm1_bias)
        x = self.attn(x)
        x = x * self.ls1_gamma  # LayerScale
        x = residual + x

        # MLP with residual
        residual = x
        x = layer_norm(x, self.norm2_weight, self.norm2_bias)
        x = self.mlp(x)
        x = x * self.ls2_gamma  # LayerScale
        x = residual + x

        return x


class ViTEncoder:
    """Vision Transformer encoder for DUNE models."""

    def __init__(self, weights: dict[str, NDArray], num_heads: int = 6):
        """Initialize encoder from weight dictionary.

        Args:
            weights: Dictionary of numpy arrays from safetensors
            num_heads: Number of attention heads
        """
        self.num_heads = num_heads

        # Patch embedding
        self.patch_embed_weight = weights["encoder.patch_embed.proj.weight"]
        self.patch_embed_bias = weights["encoder.patch_embed.proj.bias"]

        # Position and special tokens
        self.pos_embed = weights["encoder.pos_embed"]
        self.cls_token = weights["encoder.cls_token"]

        # Register tokens (optional, for DINOv2)
        self.register_tokens = weights.get("encoder.register_tokens")

        # Final norm
        self.norm_weight = weights["encoder.norm.weight"]
        self.norm_bias = weights["encoder.norm.bias"]

        # Build transformer blocks
        self.blocks = []
        block_idx = 0
        while f"encoder.blocks.0.{block_idx}.norm1.weight" in weights:
            prefix = f"encoder.blocks.0.{block_idx}"

            attn = MultiHeadAttention(
                qkv_weight=weights[f"{prefix}.attn.qkv.weight"],
                qkv_bias=weights[f"{prefix}.attn.qkv.bias"],
                proj_weight=weights[f"{prefix}.attn.proj.weight"],
                proj_bias=weights[f"{prefix}.attn.proj.bias"],
                num_heads=num_heads,
            )

            mlp = MLP(
                fc1_weight=weights[f"{prefix}.mlp.fc1.weight"],
                fc1_bias=weights[f"{prefix}.mlp.fc1.bias"],
                fc2_weight=weights[f"{prefix}.mlp.fc2.weight"],
                fc2_bias=weights[f"{prefix}.mlp.fc2.bias"],
            )

            block = TransformerBlock(
                norm1_weight=weights[f"{prefix}.norm1.weight"],
                norm1_bias=weights[f"{prefix}.norm1.bias"],
                attn=attn,
                ls1_gamma=weights[f"{prefix}.ls1.gamma"],
                norm2_weight=weights[f"{prefix}.norm2.weight"],
                norm2_bias=weights[f"{prefix}.norm2.bias"],
                mlp=mlp,
                ls2_gamma=weights[f"{prefix}.ls2.gamma"],
            )

            self.blocks.append(block)
            block_idx += 1

        # Infer dimensions
        self.embed_dim = self.patch_embed_weight.shape[0]
        self.patch_size = self.patch_embed_weight.shape[2]

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Encode image to features.

        Args:
            x: Input image [B, 3, H, W] normalized float32

        Returns:
            Features [B, N, D] where N = (H/patch_size)*(W/patch_size) + 1 (cls) + registers
        """
        B, C, H, W = x.shape

        # Patch embedding [B, D, H/p, W/p]
        x = conv2d(x, self.patch_embed_weight, self.patch_embed_bias, stride=self.patch_size)

        # Reshape to sequence [B, N, D]
        num_patches = x.shape[2] * x.shape[3]  # (H/p) * (W/p)
        x = x.reshape(B, self.embed_dim, -1)  # [B, D, N]
        x = np.transpose(x, (0, 2, 1))  # [B, N, D]

        # Prepend CLS token
        cls_tokens = np.broadcast_to(self.cls_token, (B, 1, self.embed_dim)).copy()
        x = np.concatenate([cls_tokens, x], axis=1)

        # Get position embeddings (interpolate if resolution differs from training)
        # pos_embed has shape [1, 1 + train_patches, D]
        train_patches = self.pos_embed.shape[1] - 1  # minus CLS token
        if num_patches != train_patches:
            pos_embed = interpolate_pos_embed(self.pos_embed, num_patches)
        else:
            pos_embed = self.pos_embed

        # Add position embedding BEFORE register tokens
        x = x + pos_embed

        # Add register tokens AFTER position embedding (they don't have positional info)
        if self.register_tokens is not None:
            num_registers = self.register_tokens.shape[1]
            reg_tokens = np.broadcast_to(
                self.register_tokens, (B, num_registers, self.embed_dim)
            ).copy()
            # Insert after CLS token
            x = np.concatenate([x[:, :1], reg_tokens, x[:, 1:]], axis=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = layer_norm(x, self.norm_weight, self.norm_bias)

        return x

    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Alias for forward."""
        return self.forward(x)
