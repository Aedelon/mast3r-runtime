"""MASt3R Runtime - Unified Engine API.

High-level Python API that wraps native C++ backends (Metal, CUDA, CPU).

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core.config import ModelVariant


class Backend(str, Enum):
    """Available inference backends."""

    AUTO = "auto"
    MPSGRAPH = "mpsgraph"  # Fastest on macOS 15+ (~21x speedup)
    METAL = "metal"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass
class InferenceResult:
    """Result from model inference."""

    pts3d_1: NDArray[np.float32]  # [H, W, 3]
    pts3d_2: NDArray[np.float32]  # [H, W, 3]
    desc_1: NDArray[np.float32]  # [H, W, D]
    desc_2: NDArray[np.float32]  # [H, W, D]
    conf_1: NDArray[np.float32]  # [H, W]
    conf_2: NDArray[np.float32]  # [H, W]

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class MatchResult:
    """Result from feature matching."""

    pts2d_1: NDArray[np.float32]  # [N, 2]
    pts2d_2: NDArray[np.float32]  # [N, 2]
    pts3d_1: NDArray[np.float32]  # [N, 3]
    pts3d_2: NDArray[np.float32]  # [N, 3]
    confidence: NDArray[np.float32]  # [N]

    match_ms: float = 0.0

    @property
    def num_matches(self) -> int:
        """Number of matches found."""
        return len(self.confidence)


def _detect_backend() -> Backend:
    """Auto-detect the best available backend."""
    system = platform.system()

    if system == "Darwin":
        # macOS - prefer MPSGraph (21x faster with native SDPA on macOS 15+)
        try:
            from . import _mpsgraph

            if _mpsgraph.is_available():
                return Backend.MPSGRAPH
        except ImportError:
            pass

        # Fallback to Metal kernels
        try:
            from . import _metal

            if _metal.is_available():
                return Backend.METAL
        except ImportError:
            pass

    # Try CUDA
    try:
        from . import _cuda

        if _cuda.is_available():
            return Backend.CUDA
    except ImportError:
        pass

    # Fallback to CPU
    return Backend.CPU


def _get_backend_module(backend: Backend):
    """Get the backend module."""
    if backend == Backend.MPSGRAPH:
        from . import _mpsgraph

        return _mpsgraph
    elif backend == Backend.METAL:
        from . import _metal

        return _metal
    elif backend == Backend.CUDA:
        from . import _cuda

        return _cuda
    elif backend == Backend.CPU:
        from . import _cpu

        return _cpu
    else:
        raise ValueError(f"Unknown backend: {backend}")


class Engine:
    """MASt3R inference engine.

    Provides a unified interface to native C++ backends.

    Example:
        >>> engine = Engine("dune_vit_small_336", backend="metal")
        >>> result = engine.infer(img1, img2)
        >>> matches = engine.match(result, top_k=512)
    """

    def __init__(
        self,
        variant: str = "dune_vit_small_336",
        *,
        backend: str | Backend = Backend.AUTO,
        resolution: int | None = None,
        precision: str = "fp16",
        num_threads: int = 4,
        cache_dir: Path | str | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            variant: Model variant to use.
            backend: Backend to use (auto, metal, cuda, cpu).
            resolution: Input resolution (default: native for variant).
            precision: Weight precision (fp32, fp16).
            num_threads: Number of CPU threads.
            cache_dir: Model cache directory.
        """
        self.variant = variant
        self.precision = precision
        self.num_threads = num_threads

        # Resolve backend
        if isinstance(backend, str):
            backend = Backend(backend)
        if backend == Backend.AUTO:
            backend = _detect_backend()
        self.backend = backend

        # Resolve resolution
        if resolution is None:
            resolution = self._get_native_resolution(variant)
        self.resolution = resolution

        # Resolve cache dir
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "mast3r_runtime"
        self.cache_dir = Path(cache_dir)

        # Create native engine
        self._engine = self._create_engine()

        # Load model
        model_path = self._get_model_path()
        self._engine.load(str(model_path))

    def _get_native_resolution(self, variant: str) -> int:
        """Get native resolution for variant."""
        if "336" in variant:
            return 336
        elif "448" in variant:
            return 448
        elif "512" in variant or "large" in variant:
            return 512
        return 336

    def _create_engine(self):
        """Create the native engine."""
        module = _get_backend_module(self.backend)

        if self.backend == Backend.MPSGRAPH:
            return module.MPSGraphEngine(
                variant=self.variant,
                resolution=self.resolution,
                precision=self.precision,
                num_threads=self.num_threads,
            )
        elif self.backend == Backend.METAL:
            return module.MetalEngine(
                variant=self.variant,
                resolution=self.resolution,
                precision=self.precision,
                num_threads=self.num_threads,
            )
        elif self.backend == Backend.CUDA:
            return module.CudaEngine(
                variant=self.variant,
                resolution=self.resolution,
                precision=self.precision,
            )
        elif self.backend == Backend.CPU:
            return module.CpuEngine(
                variant=self.variant,
                resolution=self.resolution,
                precision=self.precision,
                num_threads=self.num_threads,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _get_model_path(self) -> Path:
        """Get path to model weights."""
        return self.cache_dir / "safetensors" / self.variant

    @property
    def name(self) -> str:
        """Engine name (includes device info)."""
        return self._engine.name()

    @property
    def is_ready(self) -> bool:
        """Check if engine is ready for inference."""
        return self._engine.is_ready()

    def warmup(self, num_iterations: int = 3) -> None:
        """Warmup the engine with dummy data."""
        self._engine.warmup(num_iterations)

    def infer(
        self,
        img1: NDArray[np.uint8],
        img2: NDArray[np.uint8],
    ) -> InferenceResult:
        """Run inference on an image pair.

        Args:
            img1: First image [H, W, 3] uint8.
            img2: Second image [H, W, 3] uint8.

        Returns:
            InferenceResult with 3D points, descriptors, and confidence.
        """
        # Validate inputs
        if img1.dtype != np.uint8 or img2.dtype != np.uint8:
            raise ValueError("Images must be uint8")
        if img1.ndim != 3 or img2.ndim != 3:
            raise ValueError("Images must be [H, W, 3]")
        if img1.shape[2] != 3 or img2.shape[2] != 3:
            raise ValueError("Images must have 3 channels")

        # Ensure contiguous
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)

        # Run inference
        result = self._engine.infer(img1, img2)

        return InferenceResult(
            pts3d_1=np.asarray(result.pts3d_1),
            pts3d_2=np.asarray(result.pts3d_2),
            desc_1=np.asarray(result.desc_1),
            desc_2=np.asarray(result.desc_2),
            conf_1=np.asarray(result.conf_1),
            conf_2=np.asarray(result.conf_2),
            preprocess_ms=result.timing.get("preprocess_ms", 0.0),
            inference_ms=result.timing.get("inference_ms", 0.0),
            total_ms=result.timing.get("total_ms", 0.0),
        )

    def match(
        self,
        result: InferenceResult,
        *,
        top_k: int = 512,
        reciprocal: bool = True,
        confidence_threshold: float = 0.5,
    ) -> MatchResult:
        """Find correspondences from inference result.

        Args:
            result: InferenceResult from infer().
            top_k: Maximum matches per pixel.
            reciprocal: Require mutual nearest neighbors.
            confidence_threshold: Minimum confidence for matches.

        Returns:
            MatchResult with 2D/3D correspondences.
        """
        match_result = self._engine.match(
            result.desc_1,
            result.desc_2,
            top_k=top_k,
            reciprocal=reciprocal,
            confidence_threshold=confidence_threshold,
        )

        return MatchResult(
            pts2d_1=np.asarray(match_result.pts2d_1),
            pts2d_2=np.asarray(match_result.pts2d_2),
            pts3d_1=np.asarray(match_result.pts3d_1),
            pts3d_2=np.asarray(match_result.pts3d_2),
            confidence=np.asarray(match_result.confidence),
            match_ms=match_result.timing.get("match_ms", 0.0),
        )

    def __repr__(self) -> str:
        return (
            f"Engine(variant={self.variant!r}, backend={self.backend.value!r}, "
            f"resolution={self.resolution}, precision={self.precision!r})"
        )


def get_available_backends() -> list[Backend]:
    """Get list of available backends on this system."""
    available = []

    # Check MPSGraph (fastest on macOS 15+)
    try:
        from . import _mpsgraph

        if _mpsgraph.is_available():
            available.append(Backend.MPSGRAPH)
    except ImportError:
        pass

    # Check Metal
    try:
        from . import _metal

        if _metal.is_available():
            available.append(Backend.METAL)
    except ImportError:
        pass

    # Check CUDA
    try:
        from . import _cuda

        if _cuda.is_available():
            available.append(Backend.CUDA)
    except ImportError:
        pass

    # CPU is always available
    available.append(Backend.CPU)

    return available
