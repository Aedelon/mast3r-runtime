"""Abstract base class for MASt3R inference engines.

Defines the interface that all backends (ONNX, CoreML, TensorRT) must implement.
Uses numpy arrays by default for lightweight inference.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .config import MASt3RRuntimeConfig


@dataclass
class InferenceResult:
    """Result of MASt3R inference on a stereo pair.

    Contains 3D points, descriptors, and confidence maps for both views.
    All arrays are numpy arrays by default.
    """

    # 3D points in view1 coordinate frame
    pts3d_1: NDArray[np.float32]  # [H, W, 3]
    pts3d_2: NDArray[np.float32]  # [H, W, 3]

    # Dense descriptors for matching
    desc_1: NDArray[np.float32]  # [H, W, D]
    desc_2: NDArray[np.float32]  # [H, W, D]

    # Confidence maps
    conf_1: NDArray[np.float32]  # [H, W]
    conf_2: NDArray[np.float32]  # [H, W]

    # Optional: descriptor confidence (MASt3R two_confs mode)
    desc_conf_1: NDArray[np.float32] | None = None  # [H, W]
    desc_conf_2: NDArray[np.float32] | None = None  # [H, W]

    # Timing info
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def height(self) -> int:
        """Height of the output maps."""
        return self.pts3d_1.shape[0]

    @property
    def width(self) -> int:
        """Width of the output maps."""
        return self.pts3d_1.shape[1]

    @property
    def desc_dim(self) -> int:
        """Descriptor dimension."""
        return self.desc_1.shape[-1]

    def to_torch(self, device: str = "cpu") -> InferenceResult:
        """Convert all arrays to PyTorch tensors.

        Requires torch to be installed.

        Args:
            device: Target device ("cpu", "cuda", "mps")

        Returns:
            New InferenceResult with torch tensors
        """
        try:
            import torch
        except ImportError as e:
            msg = "PyTorch is required for to_torch(). Install with: pip install torch"
            raise ImportError(msg) from e

        return InferenceResult(
            pts3d_1=torch.from_numpy(self.pts3d_1).to(device),
            pts3d_2=torch.from_numpy(self.pts3d_2).to(device),
            desc_1=torch.from_numpy(self.desc_1).to(device),
            desc_2=torch.from_numpy(self.desc_2).to(device),
            conf_1=torch.from_numpy(self.conf_1).to(device),
            conf_2=torch.from_numpy(self.conf_2).to(device),
            desc_conf_1=(
                torch.from_numpy(self.desc_conf_1).to(device)
                if self.desc_conf_1 is not None
                else None
            ),
            desc_conf_2=(
                torch.from_numpy(self.desc_conf_2).to(device)
                if self.desc_conf_2 is not None
                else None
            ),
            timing_ms=self.timing_ms.copy(),
        )


@dataclass
class MatchResult:
    """Result of descriptor matching between two views.

    Contains correspondences and their confidence scores.
    """

    # Matching indices (flat indices into H*W)
    idx_1: NDArray[np.int64]  # [N]
    idx_2: NDArray[np.int64]  # [N]

    # 2D coordinates of matches
    pts2d_1: NDArray[np.float32]  # [N, 2] (x, y)
    pts2d_2: NDArray[np.float32]  # [N, 2] (x, y)

    # 3D points at match locations
    pts3d_1: NDArray[np.float32]  # [N, 3]
    pts3d_2: NDArray[np.float32]  # [N, 3]

    # Match confidence
    confidence: NDArray[np.float32]  # [N]

    # Timing info
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def num_matches(self) -> int:
        """Number of matches."""
        return len(self.idx_1)

    def to_torch(self, device: str = "cpu") -> MatchResult:
        """Convert all arrays to PyTorch tensors.

        Args:
            device: Target device

        Returns:
            New MatchResult with torch tensors
        """
        try:
            import torch
        except ImportError as e:
            msg = "PyTorch is required for to_torch(). Install with: pip install torch"
            raise ImportError(msg) from e

        return MatchResult(
            idx_1=torch.from_numpy(self.idx_1).to(device),
            idx_2=torch.from_numpy(self.idx_2).to(device),
            pts2d_1=torch.from_numpy(self.pts2d_1).to(device),
            pts2d_2=torch.from_numpy(self.pts2d_2).to(device),
            pts3d_1=torch.from_numpy(self.pts3d_1).to(device),
            pts3d_2=torch.from_numpy(self.pts3d_2).to(device),
            confidence=torch.from_numpy(self.confidence).to(device),
            timing_ms=self.timing_ms.copy(),
        )


class EngineInterface(ABC):
    """Abstract base class for MASt3R inference engines.

    All backends (ONNX, CoreML, TensorRT) must implement this interface.
    This ensures consistent behavior across platforms.

    Example:
        >>> from mast3r_runtime import get_runtime, MASt3RRuntimeConfig
        >>> config = MASt3RRuntimeConfig()
        >>> with get_runtime(config) as engine:
        ...     result = engine.infer(img1, img2)
        ...     matches = engine.match(result.desc_1, result.desc_2)
    """

    def __init__(self, config: MASt3RRuntimeConfig) -> None:
        """Initialize engine with configuration.

        Args:
            config: Runtime configuration
        """
        self._config = config
        self._is_ready = False

    @property
    def config(self) -> MASt3RRuntimeConfig:
        """Runtime configuration."""
        return self._config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name (e.g., 'ONNX CPU', 'CoreML ANE')."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the engine is loaded and ready for inference."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the model and prepare for inference.

        Should be called before infer() if not using context manager.
        """
        ...

    @abstractmethod
    def warmup(self, num_iterations: int = 3) -> None:
        """Warm up the engine with dummy inputs.

        Args:
            num_iterations: Number of warmup iterations
        """
        ...

    @abstractmethod
    def infer(
        self,
        img1: NDArray[np.uint8],
        img2: NDArray[np.uint8],
    ) -> InferenceResult:
        """Run inference on a stereo pair.

        Args:
            img1: First image [H, W, 3] RGB uint8
            img2: Second image [H, W, 3] RGB uint8

        Returns:
            Inference result with 3D points, descriptors, and confidence
        """
        ...

    @abstractmethod
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

        Uses reciprocal top-K matching with optional confidence weighting.

        Args:
            desc_1: Descriptors from view 1 [H, W, D]
            desc_2: Descriptors from view 2 [H, W, D]
            conf_1: Optional confidence from view 1 [H, W]
            conf_2: Optional confidence from view 2 [H, W]
            pts3d_1: Optional 3D points from view 1 [H, W, 3]
            pts3d_2: Optional 3D points from view 2 [H, W, 3]

        Returns:
            Match result with correspondences
        """
        ...

    def infer_and_match(
        self,
        img1: NDArray[np.uint8],
        img2: NDArray[np.uint8],
    ) -> tuple[InferenceResult, MatchResult]:
        """Convenience method: run inference and matching in one call.

        Args:
            img1: First image [H, W, 3] RGB uint8
            img2: Second image [H, W, 3] RGB uint8

        Returns:
            Tuple of (inference_result, match_result)
        """
        result = self.infer(img1, img2)
        matches = self.match(
            result.desc_1,
            result.desc_2,
            result.conf_1,
            result.conf_2,
            result.pts3d_1,
            result.pts3d_2,
        )
        return result, matches

    @abstractmethod
    def release(self) -> None:
        """Release engine resources.

        Should be called when the engine is no longer needed.
        """
        ...

    def __enter__(self) -> EngineInterface:
        """Context manager entry - load model."""
        if not self.is_ready:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - release resources."""
        self.release()

    def benchmark(
        self,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        resolution: int | None = None,
    ) -> dict[str, float]:
        """Benchmark engine performance.

        Args:
            num_iterations: Number of timed iterations
            warmup_iterations: Number of warmup iterations
            resolution: Image resolution (uses config default if None)

        Returns:
            Dict with timing statistics (mean, std, min, max in ms)
        """
        import time

        if resolution is None:
            resolution = self._config.model.resolution

        # Create dummy inputs (random RGB images)
        rng = np.random.default_rng(42)
        dummy1 = rng.integers(0, 255, (resolution, resolution, 3), dtype=np.uint8)
        dummy2 = rng.integers(0, 255, (resolution, resolution, 3), dtype=np.uint8)

        # Ensure model is loaded
        if not self.is_ready:
            self.load()

        # Warmup
        for _ in range(warmup_iterations):
            _ = self.infer(dummy1, dummy2)

        # Timed iterations
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.infer(dummy1, dummy2)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        times_arr = np.array(times)
        return {
            "mean_ms": float(times_arr.mean()),
            "std_ms": float(times_arr.std()),
            "min_ms": float(times_arr.min()),
            "max_ms": float(times_arr.max()),
            "p50_ms": float(np.percentile(times_arr, 50)),
            "p95_ms": float(np.percentile(times_arr, 95)),
            "p99_ms": float(np.percentile(times_arr, 99)),
            "fps": float(1000.0 / times_arr.mean()),
        }
