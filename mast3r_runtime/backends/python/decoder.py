"""DUNE/MASt3R Decoder implementation.

Decoder blocks with self-attention and cross-attention between views.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .vit_encoder import MLP, MultiHeadAttention, gelu, layer_norm, linear, softmax

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CrossAttention:
    """Cross-attention module (Q from decoder, K/V from encoder)."""

    def __init__(
        self,
        projq_weight: NDArray[np.float32],
        projq_bias: NDArray[np.float32],
        projk_weight: NDArray[np.float32],
        projk_bias: NDArray[np.float32],
        projv_weight: NDArray[np.float32],
        projv_bias: NDArray[np.float32],
        proj_weight: NDArray[np.float32],
        proj_bias: NDArray[np.float32],
        num_heads: int,
    ):
        self.projq_weight = projq_weight
        self.projq_bias = projq_bias
        self.projk_weight = projk_weight
        self.projk_bias = projk_bias
        self.projv_weight = projv_weight
        self.projv_bias = projv_bias
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias
        self.num_heads = num_heads

        self.embed_dim = proj_weight.shape[0]
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim**-0.5

    def __call__(self, x: NDArray[np.float32], context: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass.

        Args:
            x: Query input [B, N, D]
            context: Key/Value context [B, M, D]

        Returns:
            Output [B, N, D]
        """
        B, N, D = x.shape
        M = context.shape[1]

        # Separate projections for Q, K, V
        q = linear(x, self.projq_weight, self.projq_bias)
        k = linear(context, self.projk_weight, self.projk_bias)
        v = linear(context, self.projv_weight, self.projv_bias)

        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, M, self.num_heads, self.head_dim)
        v = v.reshape(B, M, self.num_heads, self.head_dim)

        # Transpose to [B, num_heads, seq_len, head_dim]
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))

        # Attention scores [B, num_heads, N, M]
        attn = (q @ np.swapaxes(k, -2, -1)) * self.scale
        attn = softmax(attn, axis=-1)

        # Apply attention to values [B, num_heads, N, head_dim]
        out = attn @ v

        # Transpose and reshape [B, N, D]
        out = np.transpose(out, (0, 2, 1, 3)).reshape(B, N, D)

        # Output projection
        out = linear(out, self.proj_weight, self.proj_bias)

        return out


class DecoderBlock:
    """Decoder block with self-attention, cross-attention, and MLP."""

    def __init__(
        self,
        # Self-attention
        norm1_weight: NDArray[np.float32],
        norm1_bias: NDArray[np.float32],
        attn: MultiHeadAttention,
        # Cross-attention
        norm_y_weight: NDArray[np.float32],
        norm_y_bias: NDArray[np.float32],
        cross_attn: CrossAttention,
        # MLP
        norm3_weight: NDArray[np.float32],
        norm3_bias: NDArray[np.float32],
        mlp: MLP,
        # Optional norm2 (between self-attn and cross-attn)
        norm2_weight: NDArray[np.float32] | None = None,
        norm2_bias: NDArray[np.float32] | None = None,
    ):
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.attn = attn
        self.norm_y_weight = norm_y_weight
        self.norm_y_bias = norm_y_bias
        self.cross_attn = cross_attn
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.norm3_weight = norm3_weight
        self.norm3_bias = norm3_bias
        self.mlp = mlp

    def __call__(self, x: NDArray[np.float32], context: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass.

        Args:
            x: Decoder input [B, N, D]
            context: Encoder features from other view [B, M, D]

        Returns:
            Output [B, N, D]
        """
        # Self-attention
        residual = x
        x = layer_norm(x, self.norm1_weight, self.norm1_bias)
        x = self.attn(x)
        x = residual + x

        # Cross-attention
        residual = x
        if self.norm2_weight is not None:
            x = layer_norm(x, self.norm2_weight, self.norm2_bias)
        context_norm = layer_norm(context, self.norm_y_weight, self.norm_y_bias)
        x = self.cross_attn(x, context_norm)
        x = residual + x

        # MLP
        residual = x
        x = layer_norm(x, self.norm3_weight, self.norm3_bias)
        x = self.mlp(x)
        x = residual + x

        return x


class DUNEDecoder:
    """DUNE decoder with symmetric cross-attention between two views."""

    def __init__(self, weights: dict[str, NDArray], num_heads: int = 12):
        """Initialize decoder from weight dictionary.

        Args:
            weights: Dictionary of numpy arrays from safetensors
            num_heads: Number of attention heads (768 / 64 = 12)
        """
        self.num_heads = num_heads

        # Decoder embedding projection (encoder dim -> decoder dim)
        self.decoder_embed_weight = weights["mast3r.decoder_embed.weight"]
        self.decoder_embed_bias = weights["mast3r.decoder_embed.bias"]

        # Encoder normalization
        self.enc_norm_weight = weights.get("mast3r.enc_norm.weight")
        self.enc_norm_bias = weights.get("mast3r.enc_norm.bias")

        # Decoder normalization
        self.dec_norm_weight = weights.get("mast3r.dec_norm.weight")
        self.dec_norm_bias = weights.get("mast3r.dec_norm.bias")

        # Build decoder blocks (first set)
        self.dec_blocks = self._build_decoder_blocks(weights, "mast3r.dec_blocks")

        # Build decoder blocks (second set)
        self.dec_blocks2 = self._build_decoder_blocks(weights, "mast3r.dec_blocks2")

        # Infer dimensions
        self.encoder_dim = self.decoder_embed_weight.shape[1]
        self.decoder_dim = self.decoder_embed_weight.shape[0]

    def _build_decoder_blocks(self, weights: dict[str, NDArray], prefix: str) -> list[DecoderBlock]:
        """Build decoder blocks from weights."""
        blocks = []
        block_idx = 0

        while f"{prefix}.{block_idx}.norm1.weight" in weights:
            p = f"{prefix}.{block_idx}"

            # Self-attention
            attn = MultiHeadAttention(
                qkv_weight=weights[f"{p}.attn.qkv.weight"],
                qkv_bias=weights[f"{p}.attn.qkv.bias"],
                proj_weight=weights[f"{p}.attn.proj.weight"],
                proj_bias=weights[f"{p}.attn.proj.bias"],
                num_heads=self.num_heads,
            )

            # Cross-attention
            cross_attn = CrossAttention(
                projq_weight=weights[f"{p}.cross_attn.projq.weight"],
                projq_bias=weights[f"{p}.cross_attn.projq.bias"],
                projk_weight=weights[f"{p}.cross_attn.projk.weight"],
                projk_bias=weights[f"{p}.cross_attn.projk.bias"],
                projv_weight=weights[f"{p}.cross_attn.projv.weight"],
                projv_bias=weights[f"{p}.cross_attn.projv.bias"],
                proj_weight=weights[f"{p}.cross_attn.proj.weight"],
                proj_bias=weights[f"{p}.cross_attn.proj.bias"],
                num_heads=self.num_heads,
            )

            # MLP
            mlp = MLP(
                fc1_weight=weights[f"{p}.mlp.fc1.weight"],
                fc1_bias=weights[f"{p}.mlp.fc1.bias"],
                fc2_weight=weights[f"{p}.mlp.fc2.weight"],
                fc2_bias=weights[f"{p}.mlp.fc2.bias"],
            )

            block = DecoderBlock(
                norm1_weight=weights[f"{p}.norm1.weight"],
                norm1_bias=weights[f"{p}.norm1.bias"],
                attn=attn,
                norm_y_weight=weights[f"{p}.norm_y.weight"],
                norm_y_bias=weights[f"{p}.norm_y.bias"],
                cross_attn=cross_attn,
                norm2_weight=weights.get(f"{p}.norm2.weight"),
                norm2_bias=weights.get(f"{p}.norm2.bias"),
                norm3_weight=weights[f"{p}.norm3.weight"],
                norm3_bias=weights[f"{p}.norm3.bias"],
                mlp=mlp,
            )

            blocks.append(block)
            block_idx += 1

        return blocks

    def forward(
        self,
        enc_feat1: NDArray[np.float32],
        enc_feat2: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Decode features from two views with cross-attention.

        Args:
            enc_feat1: Encoder features from view 1 [B, N, D_enc]
            enc_feat2: Encoder features from view 2 [B, N, D_enc]

        Returns:
            Tuple of decoder outputs for both views [B, N, D_dec]
        """
        # Normalize encoder features
        if self.enc_norm_weight is not None:
            enc_feat1 = layer_norm(enc_feat1, self.enc_norm_weight, self.enc_norm_bias)
            enc_feat2 = layer_norm(enc_feat2, self.enc_norm_weight, self.enc_norm_bias)

        # Project to decoder dimension
        dec1 = linear(enc_feat1, self.decoder_embed_weight, self.decoder_embed_bias)
        dec2 = linear(enc_feat2, self.decoder_embed_weight, self.decoder_embed_bias)

        # First set of decoder blocks with symmetric cross-attention
        for block in self.dec_blocks:
            # View 1 attends to view 2
            dec1_new = block(dec1, dec2)
            # View 2 attends to view 1
            dec2_new = block(dec2, dec1)
            dec1, dec2 = dec1_new, dec2_new

        # Second set of decoder blocks
        for block in self.dec_blocks2:
            dec1_new = block(dec1, dec2)
            dec2_new = block(dec2, dec1)
            dec1, dec2 = dec1_new, dec2_new

        # Final normalization
        if self.dec_norm_weight is not None:
            dec1 = layer_norm(dec1, self.dec_norm_weight, self.dec_norm_bias)
            dec2 = layer_norm(dec2, self.dec_norm_weight, self.dec_norm_bias)

        return dec1, dec2

    def __call__(
        self,
        enc_feat1: NDArray[np.float32],
        enc_feat2: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Alias for forward."""
        return self.forward(enc_feat1, enc_feat2)
