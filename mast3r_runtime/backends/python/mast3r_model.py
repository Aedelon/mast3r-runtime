"""MASt3R Model implementation.

Complete MASt3R model combining CroCo encoder, decoder, and DPT heads.
Uses RoPE 2D positional embeddings.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from safetensors import safe_open

from .croco_decoder import CroCoDecoder
from .croco_encoder import CroCoEncoder
from .dpt_head import SimpleDPTHead, conv2d_simple, interpolate_bilinear
from .vit_encoder import gelu, linear

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MASt3RLocalFeaturesHead:
    """Head for local feature descriptors (MASt3R version)."""

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


class MASt3RDPTHead:
    """Simplified DPT head for MASt3R pts3d and confidence.

    Uses decoder features to predict 3D points and confidence.
    """

    def __init__(self, weights: dict[str, NDArray[np.float32]], prefix: str):
        """Initialize from weights.

        Args:
            weights: Weight dictionary.
            prefix: Weight prefix (e.g., 'downstream_head1').
        """
        # Final prediction head: Conv -> ReLU -> Conv -> ReLU -> Conv
        self.head_0_weight = weights[f"{prefix}.dpt.head.0.weight"]
        self.head_0_bias = weights[f"{prefix}.dpt.head.0.bias"]
        self.head_2_weight = weights[f"{prefix}.dpt.head.2.weight"]
        self.head_2_bias = weights[f"{prefix}.dpt.head.2.bias"]
        self.head_4_weight = weights[f"{prefix}.dpt.head.4.weight"]
        self.head_4_bias = weights[f"{prefix}.dpt.head.4.bias"]

        # Local features head for descriptors
        self.local_head = MASt3RLocalFeaturesHead(
            fc1_weight=weights[f"{prefix}.head_local_features.fc1.weight"],
            fc1_bias=weights[f"{prefix}.head_local_features.fc1.bias"],
            fc2_weight=weights[f"{prefix}.head_local_features.fc2.weight"],
            fc2_bias=weights[f"{prefix}.head_local_features.fc2.bias"],
        )

    def forward(
        self,
        dec_feat: NDArray[np.float32],
        enc_feat: NDArray[np.float32],
        patch_h: int,
        patch_w: int,
        output_size: tuple[int, int],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Forward pass.

        Args:
            dec_feat: Decoder features [B, N, D_dec].
            enc_feat: Encoder features [B, N, D_enc].
            patch_h: Patch grid height.
            patch_w: Patch grid width.
            output_size: Target output (H, W).

        Returns:
            pts3d: 3D points [B, H, W, 3].
            descriptors: Descriptors [B, H, W, desc_dim].
            confidence: Confidence [B, H, W].
        """
        B = dec_feat.shape[0]
        H_out, W_out = output_size

        # Concatenate encoder and decoder features for local features
        combined = np.concatenate([dec_feat, enc_feat], axis=-1)

        # Compute local descriptors
        raw_desc = self.local_head(combined)  # [B, N, output_dim]

        # Reshape decoder features to spatial [B, D, patch_h, patch_w]
        dec_spatial = dec_feat.reshape(B, patch_h, patch_w, -1)
        dec_spatial = np.transpose(dec_spatial, (0, 3, 1, 2))

        # Create 256-channel feature map for DPT head
        feat_channels = dec_spatial.shape[1]
        if feat_channels >= 256:
            feat_256 = dec_spatial[:, :256]
        else:
            # Tile to get 256 channels
            repeats = 256 // feat_channels + 1
            feat_256 = np.tile(dec_spatial, (1, repeats, 1, 1))[:, :256]

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

        # Process descriptors
        desc_dim = 256
        descriptors = np.zeros((B, H_out, W_out, desc_dim), dtype=np.float32)

        if raw_desc.shape[-1] >= desc_dim:
            desc_reshaped = raw_desc[:, :, :desc_dim]  # [B, N, desc_dim]
            desc_spatial = desc_reshaped.reshape(B, patch_h, patch_w, desc_dim)
            desc_transposed = np.transpose(desc_spatial, (0, 3, 1, 2))
            desc_upsampled = interpolate_bilinear(desc_transposed, (H_out, W_out))
            descriptors = np.transpose(desc_upsampled, (0, 2, 3, 1))

        return pts3d, descriptors, confidence

    def __call__(
        self,
        dec_feat: NDArray[np.float32],
        enc_feat: NDArray[np.float32],
        patch_h: int,
        patch_w: int,
        output_size: tuple[int, int],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Alias for forward."""
        return self.forward(dec_feat, enc_feat, patch_h, patch_w, output_size)


class MASt3RModel:
    """Complete MASt3R model for inference.

    Combines:
    - CroCo encoder with RoPE 2D
    - CroCo decoder with cross-attention
    - DPT heads for pts3d and descriptors
    """

    def __init__(
        self,
        weights: dict[str, NDArray[np.float32]],
        encoder_num_heads: int = 16,
        decoder_num_heads: int = 12,
        rope_freq: float = 100.0,
    ):
        """Initialize MASt3R model from weights.

        Args:
            weights: Weight dictionary from safetensors.
            encoder_num_heads: Number of attention heads in encoder.
            decoder_num_heads: Number of attention heads in decoder.
            rope_freq: RoPE frequency base.
        """
        # Encoder
        self.encoder = CroCoEncoder(
            weights=weights,
            num_heads=encoder_num_heads,
            rope_freq=rope_freq,
        )

        # Decoder
        self.decoder = CroCoDecoder(
            weights=weights,
            num_heads=decoder_num_heads,
            rope_freq=rope_freq,
        )

        # DPT heads for each view
        self.head_1 = MASt3RDPTHead(weights, "downstream_head1")
        self.head_2 = MASt3RDPTHead(weights, "downstream_head2")

        # Infer dimensions
        self.patch_size = self.encoder.patch_size  # 16 for MASt3R

    @classmethod
    def from_safetensors(
        cls,
        unified_path: Path | str,
        encoder_num_heads: int = 16,
        decoder_num_heads: int = 12,
    ) -> "MASt3RModel":
        """Load model from safetensors file.

        Args:
            unified_path: Path to unified.safetensors.
            encoder_num_heads: Number of encoder attention heads.
            decoder_num_heads: Number of decoder attention heads.

        Returns:
            Loaded MASt3R model.
        """
        weights = {}
        with safe_open(str(unified_path), framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

        return cls(
            weights=weights,
            encoder_num_heads=encoder_num_heads,
            decoder_num_heads=decoder_num_heads,
        )

    def forward(
        self,
        img1: NDArray[np.float32],
        img2: NDArray[np.float32],
    ) -> dict[str, NDArray[np.float32]]:
        """Run inference on a stereo pair.

        Args:
            img1: First image [B, 3, H, W] normalized float32.
            img2: Second image [B, 3, H, W] normalized float32.

        Returns:
            Dictionary with:
                - pts3d_1: 3D points for view 1 [B, H, W, 3]
                - pts3d_2: 3D points for view 2 [B, H, W, 3]
                - desc_1: Descriptors for view 1 [B, H, W, D]
                - desc_2: Descriptors for view 2 [B, H, W, D]
                - conf_1: Confidence for view 1 [B, H, W]
                - conf_2: Confidence for view 2 [B, H, W]
        """
        B, C, H, W = img1.shape

        # Encode both images
        enc_1, pos_1 = self.encoder(img1)
        enc_2, pos_2 = self.encoder(img2)

        # Decode with cross-attention
        dec_1, dec_2 = self.decoder(enc_1, enc_2, pos_1, pos_2)

        # Compute patch grid size
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # Apply DPT heads
        pts3d_1, desc_1, conf_1 = self.head_1(dec_1, enc_1, patch_h, patch_w, (H, W))
        pts3d_2, desc_2, conf_2 = self.head_2(dec_2, enc_2, patch_h, patch_w, (H, W))

        return {
            "pts3d_1": pts3d_1,
            "pts3d_2": pts3d_2,
            "desc_1": desc_1,
            "desc_2": desc_2,
            "conf_1": conf_1,
            "conf_2": conf_2,
        }

    def __call__(
        self,
        img1: NDArray[np.float32],
        img2: NDArray[np.float32],
    ) -> dict[str, NDArray[np.float32]]:
        """Alias for forward."""
        return self.forward(img1, img2)
