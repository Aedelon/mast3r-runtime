"""CroCoNet Decoder implementation for MASt3R models.

Pure numpy/Python implementation of CroCo decoder with cross-attention.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .rope2d import RoPE2D
from .vit_encoder import gelu, layer_norm, linear, softmax

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CroCoSelfAttention:
    """Self-attention for decoder (no RoPE, uses standard attention)."""

    def __init__(
        self,
        qkv_weight: NDArray[np.float32],
        qkv_bias: NDArray[np.float32],
        proj_weight: NDArray[np.float32],
        proj_bias: NDArray[np.float32],
        num_heads: int,
        rope: RoPE2D | None = None,
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
        positions: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float32]:
        """Forward pass.

        Args:
            x: Input [B, N, D].
            positions: Optional 2D positions [B, N, 2] for RoPE.

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

        # Apply RoPE if available
        if self.rope is not None and positions is not None:
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


class CrocoCrossAttention:
    """Cross-attention for decoder.

    Query from decoder, Key/Value from encoder.
    Uses separate projections for Q, K, V.
    """

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
        rope: RoPE2D | None = None,
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
        self.rope = rope

        # Infer dimensions
        self.embed_dim = proj_weight.shape[0]
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        query: NDArray[np.float32],
        key_value: NDArray[np.float32],
        query_pos: NDArray[np.int64] | None = None,
        key_pos: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float32]:
        """Forward pass.

        Args:
            query: Query input [B, N_q, D] from decoder.
            key_value: Key/Value input [B, N_kv, D] from encoder.
            query_pos: Optional positions for query [B, N_q, 2].
            key_pos: Optional positions for key [B, N_kv, 2].

        Returns:
            Output [B, N_q, D].
        """
        B, N_q, D = query.shape
        _, N_kv, _ = key_value.shape

        # Project Q, K, V separately
        q = linear(query, self.projq_weight, self.projq_bias)  # [B, N_q, D]
        k = linear(key_value, self.projk_weight, self.projk_bias)  # [B, N_kv, D]
        v = linear(key_value, self.projv_weight, self.projv_bias)  # [B, N_kv, D]

        # Reshape for multi-head attention
        q = q.reshape(B, N_q, self.num_heads, self.head_dim)
        k = k.reshape(B, N_kv, self.num_heads, self.head_dim)
        v = v.reshape(B, N_kv, self.num_heads, self.head_dim)

        # Transpose to [B, num_heads, N, head_dim]
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))

        # Apply RoPE if available
        if self.rope is not None:
            if query_pos is not None:
                q = self.rope(q, query_pos)
            if key_pos is not None:
                k = self.rope(k, key_pos)

        # Attention scores [B, num_heads, N_q, N_kv]
        attn = (q @ np.swapaxes(k, -2, -1)) * self.scale
        attn = softmax(attn, axis=-1)

        # Apply attention to values [B, num_heads, N_q, head_dim]
        out = attn @ v

        # Transpose and reshape [B, N_q, D]
        out = np.transpose(out, (0, 2, 1, 3)).reshape(B, N_q, D)

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


class CroCoDecoderBlock:
    """CroCo Decoder block with self-attention, cross-attention, and MLP.

    Structure:
    1. x = x + self_attn(norm1(x))
    2. x = x + cross_attn(query=norm2(x), kv=norm_y(encoder_out))
    3. x = x + mlp(norm3(x))
    """

    def __init__(
        self,
        norm1_weight: NDArray[np.float32],
        norm1_bias: NDArray[np.float32],
        self_attn: CroCoSelfAttention,
        norm2_weight: NDArray[np.float32],
        norm2_bias: NDArray[np.float32],
        norm_y_weight: NDArray[np.float32],
        norm_y_bias: NDArray[np.float32],
        cross_attn: CrocoCrossAttention,
        norm3_weight: NDArray[np.float32],
        norm3_bias: NDArray[np.float32],
        mlp: CroCoMLP,
    ):
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.self_attn = self_attn
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.norm_y_weight = norm_y_weight
        self.norm_y_bias = norm_y_bias
        self.cross_attn = cross_attn
        self.norm3_weight = norm3_weight
        self.norm3_bias = norm3_bias
        self.mlp = mlp

    def __call__(
        self,
        x: NDArray[np.float32],
        encoder_out: NDArray[np.float32],
        x_pos: NDArray[np.int64] | None = None,
        enc_pos: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float32]:
        """Forward pass.

        Args:
            x: Decoder input [B, N, D].
            encoder_out: Encoder output [B, N_enc, D_enc].
            x_pos: Optional positions for decoder [B, N, 2].
            enc_pos: Optional positions for encoder [B, N_enc, 2].

        Returns:
            Output [B, N, D].
        """
        # Self-attention
        residual = x
        x_norm = layer_norm(x, self.norm1_weight, self.norm1_bias)
        x = residual + self.self_attn(x_norm, x_pos)

        # Cross-attention
        residual = x
        x_norm = layer_norm(x, self.norm2_weight, self.norm2_bias)
        y_norm = layer_norm(encoder_out, self.norm_y_weight, self.norm_y_bias)
        x = residual + self.cross_attn(x_norm, y_norm, x_pos, enc_pos)

        # MLP
        residual = x
        x_norm = layer_norm(x, self.norm3_weight, self.norm3_bias)
        x = residual + self.mlp(x_norm)

        return x


class CroCoDecoder:
    """CroCo Decoder for MASt3R models.

    Contains two parallel decoder branches (for view 1 and view 2).
    Each branch has the same architecture but different weights.
    """

    def __init__(
        self,
        weights: dict[str, NDArray[np.float32]],
        num_heads: int = 12,
        rope_freq: float = 100.0,
    ):
        """Initialize decoder from weight dictionary.

        Args:
            weights: Dictionary of numpy arrays from safetensors.
            num_heads: Number of attention heads.
            rope_freq: RoPE frequency base.
        """
        self.num_heads = num_heads

        # RoPE for decoder
        self.rope = RoPE2D(freq=rope_freq)

        # Decoder embedding (projects encoder dim to decoder dim)
        self.decoder_embed_weight = weights["decoder_embed.weight"]
        self.decoder_embed_bias = weights["decoder_embed.bias"]

        # Mask token for reconstruction (not used in inference)
        self.mask_token = weights["mask_token"]

        # Final norm
        self.dec_norm_weight = weights["dec_norm.weight"]
        self.dec_norm_bias = weights["dec_norm.bias"]

        # Build decoder blocks for view 1
        self.blocks_1 = self._build_blocks(weights, "dec_blocks")

        # Build decoder blocks for view 2
        self.blocks_2 = self._build_blocks(weights, "dec_blocks2")

        # Infer dimensions
        self.decoder_dim = self.decoder_embed_weight.shape[0]
        self.encoder_dim = self.decoder_embed_weight.shape[1]

    def _build_blocks(
        self,
        weights: dict[str, NDArray[np.float32]],
        prefix: str,
    ) -> list[CroCoDecoderBlock]:
        """Build decoder blocks from weights.

        Args:
            weights: Weight dictionary.
            prefix: Block prefix (e.g., "dec_blocks" or "dec_blocks2").

        Returns:
            List of decoder blocks.
        """
        blocks: list[CroCoDecoderBlock] = []
        block_idx = 0
        while f"{prefix}.{block_idx}.norm1.weight" in weights:
            block_prefix = f"{prefix}.{block_idx}"

            self_attn = CroCoSelfAttention(
                qkv_weight=weights[f"{block_prefix}.attn.qkv.weight"],
                qkv_bias=weights[f"{block_prefix}.attn.qkv.bias"],
                proj_weight=weights[f"{block_prefix}.attn.proj.weight"],
                proj_bias=weights[f"{block_prefix}.attn.proj.bias"],
                num_heads=self.num_heads,
                rope=self.rope,
            )

            cross_attn = CrocoCrossAttention(
                projq_weight=weights[f"{block_prefix}.cross_attn.projq.weight"],
                projq_bias=weights[f"{block_prefix}.cross_attn.projq.bias"],
                projk_weight=weights[f"{block_prefix}.cross_attn.projk.weight"],
                projk_bias=weights[f"{block_prefix}.cross_attn.projk.bias"],
                projv_weight=weights[f"{block_prefix}.cross_attn.projv.weight"],
                projv_bias=weights[f"{block_prefix}.cross_attn.projv.bias"],
                proj_weight=weights[f"{block_prefix}.cross_attn.proj.weight"],
                proj_bias=weights[f"{block_prefix}.cross_attn.proj.bias"],
                num_heads=self.num_heads,
                rope=self.rope,
            )

            mlp = CroCoMLP(
                fc1_weight=weights[f"{block_prefix}.mlp.fc1.weight"],
                fc1_bias=weights[f"{block_prefix}.mlp.fc1.bias"],
                fc2_weight=weights[f"{block_prefix}.mlp.fc2.weight"],
                fc2_bias=weights[f"{block_prefix}.mlp.fc2.bias"],
            )

            block = CroCoDecoderBlock(
                norm1_weight=weights[f"{block_prefix}.norm1.weight"],
                norm1_bias=weights[f"{block_prefix}.norm1.bias"],
                self_attn=self_attn,
                norm2_weight=weights[f"{block_prefix}.norm2.weight"],
                norm2_bias=weights[f"{block_prefix}.norm2.bias"],
                norm_y_weight=weights[f"{block_prefix}.norm_y.weight"],
                norm_y_bias=weights[f"{block_prefix}.norm_y.bias"],
                cross_attn=cross_attn,
                norm3_weight=weights[f"{block_prefix}.norm3.weight"],
                norm3_bias=weights[f"{block_prefix}.norm3.bias"],
                mlp=mlp,
            )

            blocks.append(block)
            block_idx += 1

        return blocks

    def forward(
        self,
        encoder_out_1: NDArray[np.float32],
        encoder_out_2: NDArray[np.float32],
        positions_1: NDArray[np.int64],
        positions_2: NDArray[np.int64],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Decode encoder outputs for both views.

        Args:
            encoder_out_1: Encoder output for view 1 [B, N, D_enc].
            encoder_out_2: Encoder output for view 2 [B, N, D_enc].
            positions_1: Positions for view 1 [B, N, 2].
            positions_2: Positions for view 2 [B, N, 2].

        Returns:
            Tuple of decoded features for view 1 and view 2, each [B, N, D_dec].
        """
        # Project ALL encoder outputs to decoder dimension (1024 -> 768)
        # This is needed for cross-attention which expects decoder dim
        enc_proj_1 = linear(encoder_out_1, self.decoder_embed_weight, self.decoder_embed_bias)
        enc_proj_2 = linear(encoder_out_2, self.decoder_embed_weight, self.decoder_embed_bias)

        # Initialize decoder states from projected encoder outputs
        dec_1 = enc_proj_1.copy()
        dec_2 = enc_proj_2.copy()

        # Decode view 1: cross-attend to view 2 (projected) encoder output
        for block in self.blocks_1:
            dec_1 = block(dec_1, enc_proj_2, positions_1, positions_2)

        # Decode view 2: cross-attend to view 1 (projected) encoder output
        for block in self.blocks_2:
            dec_2 = block(dec_2, enc_proj_1, positions_2, positions_1)

        # Final norm
        dec_1 = layer_norm(dec_1, self.dec_norm_weight, self.dec_norm_bias)
        dec_2 = layer_norm(dec_2, self.dec_norm_weight, self.dec_norm_bias)

        return dec_1, dec_2

    def __call__(
        self,
        encoder_out_1: NDArray[np.float32],
        encoder_out_2: NDArray[np.float32],
        positions_1: NDArray[np.int64],
        positions_2: NDArray[np.int64],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Alias for forward."""
        return self.forward(encoder_out_1, encoder_out_2, positions_1, positions_2)
