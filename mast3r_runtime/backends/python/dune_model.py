"""Complete DUNE model implementation.

Combines ViT encoder, cross-attention decoder, and DPT heads.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .decoder import DUNEDecoder
from .dpt_head import SimpleDPTHead
from .vit_encoder import ViTEncoder


class DUNEModel:
    """Complete DUNE model for stereo 3D reconstruction."""

    def __init__(
        self,
        encoder_weights: dict[str, NDArray],
        decoder_weights: dict[str, NDArray],
        encoder_num_heads: int = 6,
        decoder_num_heads: int = 12,
    ):
        """Initialize model from weight dictionaries.

        Args:
            encoder_weights: Encoder weight dictionary
            decoder_weights: Decoder weight dictionary
            encoder_num_heads: Number of encoder attention heads
            decoder_num_heads: Number of decoder attention heads
        """
        self.encoder = ViTEncoder(encoder_weights, num_heads=encoder_num_heads)
        self.decoder = DUNEDecoder(decoder_weights, num_heads=decoder_num_heads)

        # Prediction heads for each view
        self.head1 = SimpleDPTHead(decoder_weights, "mast3r.downstream_head1")
        self.head2 = SimpleDPTHead(decoder_weights, "mast3r.downstream_head2")

        # Store dimensions
        self.patch_size = self.encoder.patch_size
        self.encoder_dim = self.encoder.embed_dim
        self.decoder_dim = self.decoder.decoder_dim

    @classmethod
    def from_safetensors(
        cls,
        encoder_path: str | Path,
        decoder_path: str | Path,
        encoder_num_heads: int = 6,
        decoder_num_heads: int = 12,
    ) -> "DUNEModel":
        """Load model from safetensors files.

        Args:
            encoder_path: Path to encoder.safetensors
            decoder_path: Path to decoder.safetensors
            encoder_num_heads: Number of encoder attention heads
            decoder_num_heads: Number of decoder attention heads

        Returns:
            Initialized model
        """
        from safetensors import safe_open

        # Load encoder weights
        encoder_weights = {}
        with safe_open(str(encoder_path), framework="numpy") as f:
            for name in f.keys():
                encoder_weights[name] = f.get_tensor(name).astype(np.float32)

        # Load decoder weights
        decoder_weights = {}
        with safe_open(str(decoder_path), framework="numpy") as f:
            for name in f.keys():
                decoder_weights[name] = f.get_tensor(name).astype(np.float32)

        return cls(
            encoder_weights,
            decoder_weights,
            encoder_num_heads,
            decoder_num_heads,
        )

    def forward(
        self,
        img1: NDArray[np.float32],
        img2: NDArray[np.float32],
    ) -> dict[str, NDArray[np.float32]]:
        """Run full inference on stereo pair.

        Args:
            img1: First image [B, 3, H, W] normalized float32
            img2: Second image [B, 3, H, W] normalized float32

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

        # Encode both views
        enc_feat1 = self.encoder(img1)  # [B, N, D_enc]
        enc_feat2 = self.encoder(img2)  # [B, N, D_enc]

        # Decode with cross-attention
        dec_feat1, dec_feat2 = self.decoder(enc_feat1, enc_feat2)

        # Apply prediction heads
        output_size = (H, W)
        pts3d_1, desc_1, conf_1 = self.head1(dec_feat1, enc_feat1, output_size)
        pts3d_2, desc_2, conf_2 = self.head2(dec_feat2, enc_feat2, output_size)

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


def load_dune_model(variant: str = "dune_vit_small_336") -> DUNEModel:
    """Load DUNE model from cache.

    Args:
        variant: Model variant name

    Returns:
        Loaded model
    """
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "mast3r_runtime" / "safetensors" / variant
    encoder_path = cache_dir / "encoder.safetensors"
    decoder_path = cache_dir / "decoder.safetensors"

    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder not found: {decoder_path}")

    # Determine num_heads based on variant
    if "small" in variant:
        encoder_num_heads = 6  # 384 / 64
        decoder_num_heads = 12  # 768 / 64
    elif "base" in variant:
        encoder_num_heads = 12  # 768 / 64
        decoder_num_heads = 12  # 768 / 64
    else:
        encoder_num_heads = 16  # 1024 / 64
        decoder_num_heads = 16  # 1024 / 64

    return DUNEModel.from_safetensors(
        encoder_path,
        decoder_path,
        encoder_num_heads,
        decoder_num_heads,
    )
