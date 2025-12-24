"""Wrapper to adapt native C++ engines to Python EngineInterface.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.engine_interface import EngineInterface, InferenceResult, MatchResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.config import MASt3RRuntimeConfig


class NativeEngineWrapper(EngineInterface):
    """Wraps a native C++ engine (from pybind11) as EngineInterface.

    Handles conversion between Python types and C++ types.
    """

    def __init__(self, native_engine, config: MASt3RRuntimeConfig) -> None:
        """Initialize wrapper.

        Args:
            native_engine: Native engine instance from _cpu, _metal, etc.
            config: Runtime configuration.
        """
        super().__init__(config)
        self._native = native_engine
        self._is_ready = False

    @property
    def name(self) -> str:
        """Human-readable engine name."""
        return self._native.name()

    @property
    def is_ready(self) -> bool:
        """Whether the engine is loaded and ready."""
        return self._is_ready and self._native.is_ready()

    def load(self) -> None:
        """Load the model."""
        from ..core.config import get_default_model_path

        model_path = get_default_model_path(
            self._config.model.variant,
            self._config.model.precision,
            self._config.cache_dir,
        )

        self._native.load(str(model_path))
        self._is_ready = True

    def warmup(self, num_iterations: int = 3) -> None:
        """Warm up the engine."""
        if not self.is_ready:
            self.load()
        self._native.warmup(num_iterations)

    def infer(
        self,
        img1: NDArray[np.uint8],
        img2: NDArray[np.uint8],
    ) -> InferenceResult:
        """Run inference on a stereo pair.

        Args:
            img1: First image [H, W, 3] RGB uint8.
            img2: Second image [H, W, 3] RGB uint8.

        Returns:
            Inference result with 3D points, descriptors, and confidence.
        """
        if not self.is_ready:
            self.load()

        # Call native inference
        result = self._native.infer(img1, img2)

        # Convert native result to Python InferenceResult
        return InferenceResult(
            pts3d_1=np.asarray(result.pts3d_1),
            pts3d_2=np.asarray(result.pts3d_2),
            desc_1=np.asarray(result.desc_1),
            desc_2=np.asarray(result.desc_2),
            conf_1=np.asarray(result.conf_1),
            conf_2=np.asarray(result.conf_2),
            desc_conf_1=None,
            desc_conf_2=None,
            timing_ms=dict(result.timing),
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
        # Call native matching
        result = self._native.match(
            desc_1,
            desc_2,
            top_k=self._config.matching.top_k,
            reciprocal=self._config.matching.reciprocal,
            confidence_threshold=self._config.matching.confidence_threshold,
        )

        # Convert native result to Python MatchResult
        return MatchResult(
            idx_1=np.asarray(result.idx_1),
            idx_2=np.asarray(result.idx_2),
            pts2d_1=np.asarray(result.pts2d_1),
            pts2d_2=np.asarray(result.pts2d_2),
            pts3d_1=np.asarray(result.pts3d_1),
            pts3d_2=np.asarray(result.pts3d_2),
            confidence=np.asarray(result.confidence),
            timing_ms=dict(result.timing),
        )

    def release(self) -> None:
        """Release engine resources."""
        self._is_ready = False
        # Native engine cleanup happens in destructor
