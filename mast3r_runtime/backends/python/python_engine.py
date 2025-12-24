"""Pure Python inference engine (fallback).

Uses numpy for all operations. Intended for development and debugging.
Not recommended for production due to performance limitations.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ...core.engine_interface import EngineInterface, InferenceResult, MatchResult
from ...core.preprocessing import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.config import MASt3RRuntimeConfig


class PythonEngine(EngineInterface):
    """Pure Python/numpy inference engine.

    This is a fallback implementation that works on any platform.
    It uses numpy for all operations and is significantly slower
    than the native backends.

    Use this for:
    - Development and debugging
    - Platforms without native backend support
    - Testing and validation

    Do NOT use this for:
    - Production inference
    - Real-time applications
    """

    def __init__(self, config: MASt3RRuntimeConfig) -> None:
        """Initialize Python engine.

        Args:
            config: Runtime configuration.
        """
        super().__init__(config)
        self._weights = None
        self._resolution = config.model.resolution

    @property
    def name(self) -> str:
        """Human-readable engine name."""
        return "Python (numpy)"

    @property
    def is_ready(self) -> bool:
        """Whether the engine is loaded and ready."""
        return self._is_ready

    def load(self) -> None:
        """Load model weights.

        Note: For the Python fallback, we don't actually load weights
        since we can't run the full model. This is a stub that marks
        the engine as ready for testing purposes.
        """
        # TODO: Load safetensors weights and implement forward pass
        # For now, just mark as ready (returns placeholder outputs)
        self._is_ready = True

    def warmup(self, num_iterations: int = 3) -> None:
        """Warm up the engine."""
        if not self.is_ready:
            self.load()

        rng = np.random.default_rng(42)
        dummy = rng.integers(0, 255, (self._resolution, self._resolution, 3), dtype=np.uint8)

        for _ in range(num_iterations):
            _ = self.infer(dummy, dummy)

    def _preprocess(self, img: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Preprocess image for inference.

        Args:
            img: Input image [H, W, 3] RGB uint8.

        Returns:
            Preprocessed array [1, 3, H, W] float32.
        """
        # Resize if needed
        if img.shape[0] != self._resolution or img.shape[1] != self._resolution:
            pil_img = Image.fromarray(img)

            # Aspect-ratio preserving resize + center crop
            h, w = img.shape[:2]
            scale = max(self._resolution / h, self._resolution / w)
            new_h = int(h * scale)
            new_w = int(w * scale)

            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            # Center crop
            left = (new_w - self._resolution) // 2
            top = (new_h - self._resolution) // 2
            pil_img = pil_img.crop((left, top, left + self._resolution, top + self._resolution))

            img = np.array(pil_img)

        # Convert to float [0, 1]
        img_float = img.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        img_norm = (img_float - mean) / std

        # HWC -> CHW
        img_chw = np.transpose(img_norm, (2, 0, 1))

        # Add batch dimension
        return np.expand_dims(img_chw, axis=0)

    def infer(
        self,
        img1: NDArray[np.uint8],
        img2: NDArray[np.uint8],
    ) -> InferenceResult:
        """Run inference on a stereo pair.

        Note: This is a placeholder implementation that returns
        zeros for all outputs. A real implementation would need
        to load and run the actual model weights.

        Args:
            img1: First image [H, W, 3] RGB uint8.
            img2: Second image [H, W, 3] RGB uint8.

        Returns:
            Inference result (placeholder zeros).
        """
        if not self.is_ready:
            self.load()

        timing = {}

        # Preprocess
        t0 = time.perf_counter()
        _ = self._preprocess(img1)
        _ = self._preprocess(img2)
        timing["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        # Placeholder outputs
        t0 = time.perf_counter()

        res = self._resolution
        desc_dim = 256  # DUNE descriptor dimension

        pts3d_1 = np.zeros((res, res, 3), dtype=np.float32)
        pts3d_2 = np.zeros((res, res, 3), dtype=np.float32)
        desc_1 = np.random.randn(res, res, desc_dim).astype(np.float32)
        desc_2 = np.random.randn(res, res, desc_dim).astype(np.float32)
        conf_1 = np.ones((res, res), dtype=np.float32)
        conf_2 = np.ones((res, res), dtype=np.float32)

        timing["inference_ms"] = (time.perf_counter() - t0) * 1000
        timing["total_ms"] = timing["preprocess_ms"] + timing["inference_ms"]

        return InferenceResult(
            pts3d_1=pts3d_1,
            pts3d_2=pts3d_2,
            desc_1=desc_1,
            desc_2=desc_2,
            conf_1=conf_1,
            conf_2=conf_2,
            timing_ms=timing,
        )

    def match(
        self,
        desc_1: NDArray[np.float32],
        desc_2: NDArray[np.float32],
        conf_1: NDArray[np.float32] | None = None,
        conf_2: NDArray[np.float32] | None = None,
        pts3d_1: NDArray[np.float32] | None = None,
        pts3d_2: NDArray[np.float32] | None = None,
    ) -> MatchResult:
        """Match descriptors between two views.

        Uses the numpy matching implementation.

        Args:
            desc_1: Descriptors from view 1 [H, W, D].
            desc_2: Descriptors from view 2 [H, W, D].
            conf_1: Optional confidence from view 1 [H, W].
            conf_2: Optional confidence from view 2 [H, W].
            pts3d_1: Optional 3D points from view 1 [H, W, 3].
            pts3d_2: Optional 3D points from view 2 [H, W, 3].

        Returns:
            Match result with correspondences.
        """
        from ..matching import reciprocal_match

        return reciprocal_match(
            desc_1=desc_1,
            desc_2=desc_2,
            conf_1=conf_1,
            conf_2=conf_2,
            pts3d_1=pts3d_1,
            pts3d_2=pts3d_2,
            top_k=self._config.matching.top_k,
            reciprocal=self._config.matching.reciprocal,
            confidence_threshold=self._config.matching.confidence_threshold,
        )

    def release(self) -> None:
        """Release resources."""
        self._weights = None
        self._is_ready = False
