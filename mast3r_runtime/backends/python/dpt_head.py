"""DPT Head implementation for DUNE/MASt3R models.

Dense Prediction Transformer heads for pts3d, descriptors, and confidence.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .vit_encoder import gelu, linear

if TYPE_CHECKING:
    from numpy.typing import NDArray


def conv2d_simple(
    x: NDArray[np.float32],
    weight: NDArray[np.float32],
    bias: NDArray[np.float32] | None = None,
    stride: int = 1,
    padding: int = 0,
) -> NDArray[np.float32]:
    """Simple 2D convolution with padding.

    Args:
        x: Input [B, C, H, W]
        weight: Kernel [out_channels, in_channels, kH, kW]
        bias: Optional bias [out_channels]
        stride: Stride
        padding: Zero padding

    Returns:
        Output [B, out_channels, H', W']
    """
    B, C, H, W = x.shape
    out_channels, in_channels, kH, kW = weight.shape

    # Apply padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        H += 2 * padding
        W += 2 * padding

    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    output = np.zeros((B, out_channels, H_out, W_out), dtype=x.dtype)

    for i in range(H_out):
        for j in range(W_out):
            # Extract patch
            h_start = i * stride
            w_start = j * stride
            patch = x[:, :, h_start : h_start + kH, w_start : w_start + kW]
            # [B, in_channels, kH, kW] -> [B, in_channels * kH * kW]
            patch_flat = patch.reshape(B, -1)
            # weight: [out_channels, in_channels * kH * kW]
            weight_flat = weight.reshape(out_channels, -1)
            # [B, out_channels]
            output[:, :, i, j] = patch_flat @ weight_flat.T

    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)

    return output


def interpolate_bilinear(x: NDArray[np.float32], size: tuple[int, int]) -> NDArray[np.float32]:
    """Bilinear interpolation (simplified).

    Args:
        x: Input [B, C, H, W]
        size: Target (H_out, W_out)

    Returns:
        Interpolated [B, C, H_out, W_out]
    """
    B, C, H, W = x.shape
    H_out, W_out = size

    # Create output grid
    output = np.zeros((B, C, H_out, W_out), dtype=x.dtype)

    # Scale factors
    scale_h = H / H_out
    scale_w = W / W_out

    for i in range(H_out):
        for j in range(W_out):
            # Source coordinates
            src_h = i * scale_h
            src_w = j * scale_w

            # Integer and fractional parts
            h0 = int(src_h)
            w0 = int(src_w)
            h1 = min(h0 + 1, H - 1)
            w1 = min(w0 + 1, W - 1)

            # Weights
            fh = src_h - h0
            fw = src_w - w0

            # Bilinear interpolation
            output[:, :, i, j] = (
                (1 - fh) * (1 - fw) * x[:, :, h0, w0]
                + (1 - fh) * fw * x[:, :, h0, w1]
                + fh * (1 - fw) * x[:, :, h1, w0]
                + fh * fw * x[:, :, h1, w1]
            )

    return output


class LocalFeaturesHead:
    """Head for local feature descriptors."""

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

        # Output dimension (typically 4900 = 49 * 100 or similar)
        self.output_dim = fc2_weight.shape[0]

    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass.

        Args:
            x: Input features [B, N, D]

        Returns:
            Descriptors [B, N, output_dim]
        """
        x = linear(x, self.fc1_weight, self.fc1_bias)
        x = gelu(x)
        x = linear(x, self.fc2_weight, self.fc2_bias)
        return x


class SimpleDPTHead:
    """Simplified DPT head for pts3d and confidence.

    This is a simplified version that skips the full RefineNet.
    For production, implement the full DPT architecture.
    """

    def __init__(self, weights: dict[str, NDArray], prefix: str):
        """Initialize from weights.

        Args:
            weights: Weight dictionary
            prefix: Weight prefix (e.g., 'mast3r.downstream_head1')
        """
        # Final prediction head: Conv -> ReLU -> Conv -> ReLU -> Conv
        self.head_0_weight = weights[f"{prefix}.dpt.head.0.weight"]
        self.head_0_bias = weights[f"{prefix}.dpt.head.0.bias"]
        self.head_2_weight = weights[f"{prefix}.dpt.head.2.weight"]
        self.head_2_bias = weights[f"{prefix}.dpt.head.2.bias"]
        self.head_4_weight = weights[f"{prefix}.dpt.head.4.weight"]
        self.head_4_bias = weights[f"{prefix}.dpt.head.4.bias"]

        # Local features head
        self.local_head = LocalFeaturesHead(
            fc1_weight=weights[f"{prefix}.head_local_features.fc1.weight"],
            fc1_bias=weights[f"{prefix}.head_local_features.fc1.bias"],
            fc2_weight=weights[f"{prefix}.head_local_features.fc2.weight"],
            fc2_bias=weights[f"{prefix}.head_local_features.fc2.bias"],
        )

    def forward(
        self,
        dec_feat: NDArray[np.float32],
        enc_feat: NDArray[np.float32],
        output_size: tuple[int, int],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Forward pass.

        Args:
            dec_feat: Decoder features [B, N, D_dec] where N includes cls+reg+patches
            enc_feat: Encoder features [B, N, D_enc]
            output_size: Target output (H, W) typically (H_input, W_input)

        Returns:
            pts3d: 3D points [B, H, W, 3]
            descriptors: Descriptors [B, H, W, desc_dim]
            confidence: Confidence [B, H, W]
        """
        B = dec_feat.shape[0]

        # Remove CLS and register tokens (keep only patch tokens)
        # Assuming 1 cls + 4 registers + patches
        num_prefix_tokens = 5
        dec_patches = dec_feat[:, num_prefix_tokens:]  # [B, num_patches, D_dec]
        enc_patches = enc_feat[:, num_prefix_tokens:]  # [B, num_patches, D_enc]

        num_patches = dec_patches.shape[1]
        patch_h = patch_w = int(np.sqrt(num_patches))

        # Concatenate encoder and decoder features for local features head
        # [B, num_patches, D_dec + D_enc]
        combined = np.concatenate([dec_patches, enc_patches], axis=-1)

        # Compute local descriptors
        raw_desc = self.local_head(combined)  # [B, num_patches, output_dim]

        # Reshape descriptors to spatial format
        # Assuming output_dim = patch_h * patch_w * desc_dim_per_patch
        # or we need to interpolate
        desc_dim = 256  # Target descriptor dimension
        H_out, W_out = output_size

        # Reshape patches to spatial [B, D, patch_h, patch_w]
        dec_spatial = dec_patches.reshape(B, patch_h, patch_w, -1)
        dec_spatial = np.transpose(dec_spatial, (0, 3, 1, 2))  # [B, D, H, W]

        # For pts3d: need to go through DPT refinenet (simplified here)
        # Just use a simple projection + upsample for now
        feat_channels = dec_spatial.shape[1]

        # Create a simple 256-channel feature map
        feat_256 = np.zeros((B, 256, patch_h, patch_w), dtype=dec_spatial.dtype)
        feat_256[:, :feat_channels] = (
            dec_spatial[:, :256]
            if feat_channels >= 256
            else np.tile(dec_spatial, (1, 256 // feat_channels + 1, 1, 1))[:, :256]
        )

        # Apply head convolutions
        x = conv2d_simple(feat_256, self.head_0_weight, self.head_0_bias, padding=1)
        x = np.maximum(x, 0)  # ReLU
        x = conv2d_simple(x, self.head_2_weight, self.head_2_bias, padding=1)
        x = np.maximum(x, 0)  # ReLU
        x = conv2d_simple(x, self.head_4_weight, self.head_4_bias, padding=0)
        # x: [B, 4, patch_h, patch_w] -> pts3d (3) + conf (1)

        # Upsample to output size
        x = interpolate_bilinear(x, (H_out, W_out))

        # Split into pts3d and confidence
        pts3d = np.transpose(x[:, :3], (0, 2, 3, 1))  # [B, H, W, 3]
        confidence = x[:, 3]  # [B, H, W]

        # Reshape descriptors to [B, H, W, desc_dim]
        # For now, just create placeholder descriptors from raw_desc
        descriptors = np.zeros((B, H_out, W_out, desc_dim), dtype=np.float32)

        # Simple reshape if dimensions work out
        if raw_desc.shape[-1] >= desc_dim:
            desc_reshaped = raw_desc[:, :, :desc_dim]  # [B, num_patches, desc_dim]
            desc_spatial = desc_reshaped.reshape(B, patch_h, patch_w, desc_dim)
            # Upsample descriptors
            desc_transposed = np.transpose(desc_spatial, (0, 3, 1, 2))  # [B, D, H, W]
            desc_upsampled = interpolate_bilinear(desc_transposed, (H_out, W_out))
            descriptors = np.transpose(desc_upsampled, (0, 2, 3, 1))  # [B, H, W, D]

        return pts3d, descriptors, confidence

    def __call__(
        self,
        dec_feat: NDArray[np.float32],
        enc_feat: NDArray[np.float32],
        output_size: tuple[int, int],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Alias for forward."""
        return self.forward(dec_feat, enc_feat, output_size)
