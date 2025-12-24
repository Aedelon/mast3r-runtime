"""PyTorch backend for MASt3R runtime.

Reference implementation and fallback backend.
Slower than CoreML/TensorRT but provides maximum compatibility.

Loads models via:
- torch.hub for DUNE encoders
- huggingface_hub for MASt3R/DUSt3R models

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as functional

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Relative imports for standalone package
from ...core.config import (
    MODEL_SPECS,
    MASt3RRuntimeConfig,
    ModelVariant,
    Precision,
)
from ...core.engine_interface import (
    EngineInterface,
    InferenceResult,
    MatchResult,
)
from ...core.preprocessing import Preprocessor


def _load_mast3r_from_hub(checkpoint_path: str, device: str = "cpu"):
    """Load MASt3R model from checkpoint using torch.hub.

    This uses naver/mast3r repo directly via torch.hub.
    """
    # Try loading via torch.hub (clones repo if needed)
    try:
        model = torch.hub.load(
            "naver/mast3r",
            "AsymmetricMASt3R",
            pretrained=True,
            weights=checkpoint_path,
            trust_repo=True,
        )
        return model
    except Exception:
        pass

    # Fallback: try direct import if mast3r is installed
    try:
        from mast3r.model import AsymmetricMASt3R

        return AsymmetricMASt3R.from_pretrained(checkpoint_path)
    except ImportError as e:
        msg = (
            "MASt3R not available. Install via:\n"
            "  pip install git+https://github.com/naver/mast3r.git\n"
            "Or use --encoder-only mode for ONNX export."
        )
        raise ImportError(msg) from e


def _load_dust3r_from_hub(checkpoint_path: str, device: str = "cpu"):
    """Load DUSt3R model from checkpoint."""
    try:
        from dust3r.model import AsymmetricCroCo3DStereo

        return AsymmetricCroCo3DStereo.from_pretrained(checkpoint_path)
    except ImportError as e:
        msg = (
            "DUSt3R not available. Install via:\n"
            "  pip install git+https://github.com/naver/dust3r.git"
        )
        raise ImportError(msg) from e


class PyTorchEngine(EngineInterface):
    """PyTorch backend for MASt3R inference.

    Uses MASt3R/DUNE PyTorch implementation via torch.hub.
    Serves as reference and fallback when optimized backends are unavailable.

    Example:
        >>> from mast3r_runtime import MASt3RRuntimeConfig, ModelConfig, ModelVariant
        >>> config = MASt3RRuntimeConfig(model=ModelConfig(variant=ModelVariant.DUNE_VIT_SMALL_14))
        >>> engine = PyTorchEngine(config)
        >>> result = engine.infer(img1, img2)
    """

    def __init__(
        self,
        config: MASt3RRuntimeConfig,
        device: str | torch.device = "auto",
    ) -> None:
        """Initialize PyTorch engine.

        Args:
            config: Runtime configuration
            device: Device to use ("auto", "cuda", "mps", "cpu")
        """
        super().__init__(config)

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        # Determine dtype
        if config.model.precision == Precision.FP16:
            self._dtype = torch.float16
        elif config.model.precision == Precision.FP32:
            self._dtype = torch.float32
        else:
            # INT8/INT4 not supported in pure PyTorch, use FP16
            self._dtype = torch.float16

        # Initialize model and preprocessor
        self._model = None
        self._preprocessor = None
        self._is_ready = False

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load the MASt3R/DUNE model."""
        variant = self._config.model.variant
        spec = MODEL_SPECS[variant]

        print(f"Loading {variant.value} model...")

        # Determine checkpoint path
        checkpoint_dir = self._config.checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = self._config.cache_dir / "checkpoints"

        if variant in (ModelVariant.DUNE_VIT_SMALL_14, ModelVariant.DUNE_VIT_BASE_14):
            # Load DUNE-MASt3R model
            decoder_name = (
                "dunemast3r_cvpr25_vitsmall.pth"
                if variant == ModelVariant.DUNE_VIT_SMALL_14
                else "dunemast3r_cvpr25_vitbase.pth"
            )
            checkpoint_path = checkpoint_dir / "decoders" / decoder_name

            if not checkpoint_path.exists():
                msg = (
                    f"Checkpoint not found: {checkpoint_path}\n"
                    "Run: mast3r-download --variant {variant.value}"
                )
                raise FileNotFoundError(msg)

            self._model = _load_mast3r_from_hub(str(checkpoint_path), str(self._device))

        elif variant == ModelVariant.MAST3R_VIT_LARGE:
            checkpoint_path = (
                checkpoint_dir / "original" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
            )
            if not checkpoint_path.exists():
                msg = f"Checkpoint not found: {checkpoint_path}"
                raise FileNotFoundError(msg)

            self._model = _load_mast3r_from_hub(str(checkpoint_path), str(self._device))

        elif variant == ModelVariant.DUST3R_224_LINEAR:
            checkpoint_path = (
                checkpoint_dir / "original" / "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
            )
            if not checkpoint_path.exists():
                msg = f"Checkpoint not found: {checkpoint_path}"
                raise FileNotFoundError(msg)

            self._model = _load_dust3r_from_hub(str(checkpoint_path), str(self._device))

        else:
            msg = f"Unsupported variant: {variant}"
            raise ValueError(msg)

        # Move to device and set eval mode
        self._model = self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()

        # Initialize preprocessor
        self._preprocessor = Preprocessor(
            resolution=self._config.model.resolution,
            patch_size=spec["patch_size"],
            device=self._device,
            dtype=self._dtype,
        )

        self._is_ready = True
        print(f"Model loaded on {self._device} ({self._dtype})")

    @property
    def name(self) -> str:
        """Human-readable engine name."""
        return f"PyTorch ({self._device})"

    @property
    def is_ready(self) -> bool:
        """Whether the engine is ready for inference."""
        return self._is_ready

    def warmup(self, num_iterations: int = 3) -> None:
        """Warm up the engine with dummy inputs."""
        if not self._is_ready:
            msg = "Engine not ready"
            raise RuntimeError(msg)

        resolution = self._config.model.resolution
        dummy = torch.randn(1, 3, resolution, resolution, device=self._device, dtype=self._dtype)

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.infer(dummy, dummy)

        # Sync
        if self._device.type == "cuda":
            torch.cuda.synchronize()

    def infer(
        self,
        img1: NDArray[np.uint8] | torch.Tensor,
        img2: NDArray[np.uint8] | torch.Tensor,
    ) -> InferenceResult:
        """Run inference on a stereo pair."""
        if not self._is_ready:
            msg = "Engine not ready"
            raise RuntimeError(msg)

        timing = {}

        # Preprocessing
        t0 = time.perf_counter()
        tensor1 = self._preprocessor(img1)
        tensor2 = self._preprocessor(img2)
        timing["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        # Prepare inputs in MASt3R format
        B, _, H, W = tensor1.shape
        true_shape = torch.tensor([[H, W]], device=self._device).expand(B, 2)

        view1 = {"img": tensor1, "true_shape": true_shape}
        view2 = {"img": tensor2, "true_shape": true_shape}

        # Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            result = self._model(view1, view2)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        timing["inference_ms"] = (time.perf_counter() - t0) * 1000

        # Extract outputs
        # MASt3R output format: dict with 'pred1' and 'pred2'
        pred1 = result["pred1"]
        pred2 = result["pred2"]

        return InferenceResult(
            pts3d_1=pred1["pts3d"],
            pts3d_2=pred2["pts3d_in_other_view"],
            desc_1=pred1["desc"],
            desc_2=pred2["desc"],
            conf_1=pred1["conf"],
            conf_2=pred2["conf"],
            desc_conf_1=pred1.get("desc_conf"),
            desc_conf_2=pred2.get("desc_conf"),
            timing_ms=timing,
        )

    def match(
        self,
        desc_1: torch.Tensor,
        desc_2: torch.Tensor,
        conf_1: torch.Tensor | None = None,
        conf_2: torch.Tensor | None = None,
    ) -> MatchResult:
        """Match descriptors between two views using top-K reciprocal matching."""
        timing = {}
        t0 = time.perf_counter()

        B, H, W, D = desc_1.shape
        top_k = self._config.matching.top_k
        conf_threshold = self._config.matching.confidence_threshold

        # Flatten spatial dimensions
        desc_1_flat = desc_1.view(B, H * W, D)  # [B, N, D]
        desc_2_flat = desc_2.view(B, H * W, D)  # [B, N, D]

        # Normalize descriptors
        desc_1_norm = functional.normalize(desc_1_flat, dim=-1)
        desc_2_norm = functional.normalize(desc_2_flat, dim=-1)

        # Compute similarity matrix
        sim = torch.bmm(desc_1_norm, desc_2_norm.transpose(1, 2))  # [B, N, N]

        # Apply confidence weighting if available
        if conf_1 is not None and conf_2 is not None:
            conf_1_flat = conf_1.view(B, H * W, 1)
            conf_2_flat = conf_2.view(B, 1, H * W)
            conf_weight = conf_1_flat * conf_2_flat
            sim = sim * conf_weight

        # Top-K matches from view1 to view2
        topk_sim_12, topk_idx_12 = sim.topk(min(top_k, H * W), dim=2)  # [B, N, K]

        # Top-K matches from view2 to view1
        _topk_sim_21, topk_idx_21 = sim.transpose(1, 2).topk(min(top_k, H * W), dim=2)

        # Reciprocal matching (only keep mutual nearest neighbors if enabled)
        if self._config.matching.reciprocal:
            matches_1 = []
            matches_2 = []
            confidences = []

            for b in range(B):
                for i in range(H * W):
                    j = topk_idx_12[b, i, 0].item()
                    if topk_idx_21[b, j, 0].item() == i:
                        match_conf = topk_sim_12[b, i, 0].item()
                        if match_conf >= conf_threshold:
                            matches_1.append(i)
                            matches_2.append(j)
                            confidences.append(match_conf)

            idx_1 = torch.tensor(matches_1, device=self._device)
            idx_2 = torch.tensor(matches_2, device=self._device)
            confidence = torch.tensor(confidences, device=self._device)
        else:
            idx_1 = torch.arange(H * W, device=self._device).repeat(top_k)
            idx_2 = topk_idx_12[0].flatten()
            confidence = topk_sim_12[0].flatten()

            mask = confidence >= conf_threshold
            idx_1 = idx_1[mask]
            idx_2 = idx_2[mask]
            confidence = confidence[mask]

        # Convert flat indices to 2D coordinates
        pts2d_1 = torch.stack([idx_1 % W, idx_1 // W], dim=1).float()
        pts2d_2 = torch.stack([idx_2 % W, idx_2 // W], dim=1).float()

        timing["match_ms"] = (time.perf_counter() - t0) * 1000

        pts3d_1 = torch.zeros(len(idx_1), 3, device=self._device)
        pts3d_2 = torch.zeros(len(idx_2), 3, device=self._device)

        return MatchResult(
            idx_1=idx_1,
            idx_2=idx_2,
            pts2d_1=pts2d_1,
            pts2d_2=pts2d_2,
            pts3d_1=pts3d_1,
            pts3d_2=pts3d_2,
            confidence=confidence,
            timing_ms=timing,
        )

    def release(self) -> None:
        """Release engine resources."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        self._is_ready = False
