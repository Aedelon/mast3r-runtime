"""ONNX Runtime backend for MASt3R inference.

Lightweight inference using ONNX Runtime with multiple execution providers:
- CPU (default)
- CUDA (onnxruntime-gpu)
- CoreML (macOS)
- TensorRT (Linux/Jetson)

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ...core.config import MODEL_SPECS, BackendType, MASt3RRuntimeConfig, Precision
from ...core.engine_interface import EngineInterface, InferenceResult, MatchResult
from ..matching import reciprocal_match

if TYPE_CHECKING:
    import onnxruntime as ort


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_execution_providers(backend: BackendType) -> list[str]:
    """Get ONNX Runtime execution providers based on backend type.

    Args:
        backend: Requested backend type

    Returns:
        List of execution provider names in priority order
    """
    if backend == BackendType.TENSORRT:
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    elif backend == BackendType.COREML:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    elif backend == BackendType.ONNX:
        # Try GPU providers first, fall back to CPU
        return ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    elif backend == BackendType.AUTO:
        # Try all available providers
        return [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        return ["CPUExecutionProvider"]


class ONNXEngine(EngineInterface):
    """ONNX Runtime backend for MASt3R inference.

    Supports multiple execution providers for cross-platform deployment.

    Example:
        >>> from mast3r_runtime import MASt3RRuntimeConfig
        >>> from mast3r_runtime.backends.onnx import ONNXEngine
        >>>
        >>> config = MASt3RRuntimeConfig()
        >>> with ONNXEngine(config) as engine:
        ...     result = engine.infer(img1, img2)
        ...     print(f"Inference took {result.timing_ms['total_ms']:.1f}ms")
    """

    def __init__(
        self,
        config: MASt3RRuntimeConfig,
        model_path: str | Path | None = None,
    ) -> None:
        """Initialize ONNX engine.

        Args:
            config: Runtime configuration
            model_path: Path to ONNX model (uses default cache location if None)
        """
        super().__init__(config)

        self._model_path = model_path
        self._session: ort.InferenceSession | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._provider_name: str = "Unknown"

        # Preprocessing parameters
        self._resolution = config.model.resolution
        spec = MODEL_SPECS[config.model.variant]
        self._patch_size = spec["patch_size"]

    def _get_model_path(self) -> Path:
        """Get path to ONNX model file."""
        if self._model_path is not None:
            return Path(self._model_path)

        # Default location in cache
        variant = self._config.model.variant
        precision = self._config.model.precision
        resolution = self._config.model.resolution

        # Build filename
        precision_suffix = "" if precision == Precision.FP32 else f"_{precision.value}"
        filename = f"{variant.value}_{resolution}{precision_suffix}.onnx"

        cache_dir = self._config.cache_dir / "onnx"
        return cache_dir / filename

    @property
    def name(self) -> str:
        """Human-readable engine name."""
        return f"ONNX ({self._provider_name})"

    @property
    def is_ready(self) -> bool:
        """Whether the engine is loaded and ready."""
        return self._session is not None

    def load(self) -> None:
        """Load the ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            msg = "onnxruntime is required. Install with: pip install mast3r-runtime[onnx]"
            raise ImportError(msg) from e

        model_path = self._get_model_path()
        if not model_path.exists():
            msg = (
                f"ONNX model not found: {model_path}\n"
                f"Download with: mast3r-download {self._config.model.variant.value}"
            )
            raise FileNotFoundError(msg)

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Set number of threads
        sess_options.intra_op_num_threads = self._config.runtime.num_threads
        sess_options.inter_op_num_threads = 1

        # Get execution providers
        providers = get_execution_providers(self._config.runtime.backend)

        # Create session
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Get actual provider being used
        active_providers = self._session.get_providers()
        self._provider_name = active_providers[0] if active_providers else "Unknown"

        # Get input/output names
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

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
            img: Input image [H, W, 3] RGB uint8

        Returns:
            Preprocessed tensor [1, 3, H', W'] float32
        """
        from PIL import Image

        # Resize to target resolution
        if img.shape[0] != self._resolution or img.shape[1] != self._resolution:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((self._resolution, self._resolution), Image.BILINEAR)
            img = np.array(pil_img)

        # Convert to float and normalize
        img_float = img.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD

        # Transpose to NCHW format
        img_nchw = np.transpose(img_norm, (2, 0, 1))

        # Add batch dimension
        return np.expand_dims(img_nchw, axis=0)

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
        if not self.is_ready:
            self.load()

        timing = {}

        # Preprocess
        t0 = time.perf_counter()
        tensor1 = self._preprocess(img1)
        tensor2 = self._preprocess(img2)
        timing["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        # Prepare inputs
        # True shape for padding handling
        h, w = self._resolution, self._resolution
        true_shape = np.array([[h, w]], dtype=np.int64)

        inputs = {
            "img1": tensor1,
            "img2": tensor2,
            "true_shape1": true_shape,
            "true_shape2": true_shape,
        }

        # Run inference
        t0 = time.perf_counter()
        outputs = self._session.run(self._output_names, inputs)
        timing["inference_ms"] = (time.perf_counter() - t0) * 1000

        # Parse outputs (order depends on model export)
        # Expected: pts3d_1, pts3d_2, desc_1, desc_2, conf_1, conf_2
        output_dict = dict(zip(self._output_names, outputs))

        # Extract and reshape outputs
        # Remove batch dimension, outputs are [1, H, W, C] or [1, H, W]
        pts3d_1 = output_dict["pts3d_1"][0]  # [H, W, 3]
        pts3d_2 = output_dict["pts3d_2"][0]  # [H, W, 3]
        desc_1 = output_dict["desc_1"][0]  # [H, W, D]
        desc_2 = output_dict["desc_2"][0]  # [H, W, D]
        conf_1 = output_dict["conf_1"][0]  # [H, W]
        conf_2 = output_dict["conf_2"][0]  # [H, W]

        # Optional descriptor confidence
        desc_conf_1 = output_dict.get("desc_conf_1")
        desc_conf_2 = output_dict.get("desc_conf_2")
        if desc_conf_1 is not None:
            desc_conf_1 = desc_conf_1[0]
        if desc_conf_2 is not None:
            desc_conf_2 = desc_conf_2[0]

        timing["total_ms"] = timing["preprocess_ms"] + timing["inference_ms"]

        return InferenceResult(
            pts3d_1=pts3d_1.astype(np.float32),
            pts3d_2=pts3d_2.astype(np.float32),
            desc_1=desc_1.astype(np.float32),
            desc_2=desc_2.astype(np.float32),
            conf_1=conf_1.astype(np.float32),
            conf_2=conf_2.astype(np.float32),
            desc_conf_1=desc_conf_1.astype(np.float32) if desc_conf_1 is not None else None,
            desc_conf_2=desc_conf_2.astype(np.float32) if desc_conf_2 is not None else None,
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
        """Release ONNX session."""
        if self._session is not None:
            del self._session
            self._session = None
        self._is_ready = False
