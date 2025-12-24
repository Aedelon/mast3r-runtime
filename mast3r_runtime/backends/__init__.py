"""Backend dispatcher for MASt3R runtime.

Automatically selects the optimal backend based on platform and configuration:
- ONNX Runtime (default, cross-platform)
- CoreML (Apple Silicon)
- TensorRT (NVIDIA Jetson/Linux)

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING

from ..core.config import BackendType

if TYPE_CHECKING:
    from ..core.config import MASt3RRuntimeConfig
    from ..core.engine_interface import EngineInterface


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def is_jetson() -> bool:
    """Check if running on NVIDIA Jetson."""
    try:
        with open("/etc/nv_tegra_release") as f:
            return "NVIDIA" in f.read()
    except FileNotFoundError:
        return False


def is_cuda_available() -> bool:
    """Check if CUDA is available via onnxruntime-gpu."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        return "CUDAExecutionProvider" in providers
    except ImportError:
        return False


def is_coreml_available() -> bool:
    """Check if CoreML is available."""
    if not is_apple_silicon():
        return False
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        return "CoreMLExecutionProvider" in providers
    except ImportError:
        return False


def get_available_backends() -> list[BackendType]:
    """Get list of available backends on this system."""
    available = [BackendType.ONNX]  # Always available with onnxruntime

    if is_coreml_available():
        available.append(BackendType.COREML)

    if is_cuda_available():
        available.append(BackendType.TENSORRT)

    return available


def get_runtime(config: MASt3RRuntimeConfig) -> EngineInterface:
    """Get the appropriate runtime engine based on configuration.

    This is the main entry point for getting an inference engine.

    Args:
        config: Runtime configuration

    Returns:
        Initialized engine ready for inference

    Raises:
        ImportError: If required backend dependencies are missing
        ValueError: If requested backend is not available

    Example:
        >>> from mast3r_runtime import get_runtime, MASt3RRuntimeConfig
        >>> config = MASt3RRuntimeConfig()
        >>> with get_runtime(config) as engine:
        ...     result = engine.infer(img1, img2)
    """
    backend_type = config.runtime.backend

    # Auto-select based on platform
    if backend_type == BackendType.AUTO:
        if is_apple_silicon() and is_coreml_available():
            backend_type = BackendType.COREML
        elif is_cuda_available():
            backend_type = BackendType.ONNX  # ONNX with CUDA provider
        else:
            backend_type = BackendType.ONNX

    # ONNX backend (default, handles CUDA/CoreML via execution providers)
    if backend_type in (BackendType.ONNX, BackendType.COREML, BackendType.TENSORRT):
        try:
            from .onnx import ONNXEngine

            return ONNXEngine(config)
        except ImportError as e:
            msg = f"ONNX Runtime not available: {e}\nInstall with: pip install mast3r-runtime[onnx]"
            raise ImportError(msg) from e

    # PyTorch backend (requires external mast3r package)
    if backend_type == BackendType.PYTORCH:
        msg = (
            "PyTorch backend requires MASt3R installed separately.\n"
            "This is not included in mast3r-runtime due to licensing.\n"
            "Install MASt3R from: https://github.com/naver/mast3r"
        )
        raise ImportError(msg)

    msg = f"Unknown backend type: {backend_type}"
    raise ValueError(msg)


# Convenience aliases
get_engine = get_runtime
get_backend = get_runtime


__all__ = [
    "BackendType",
    "get_available_backends",
    "get_backend",
    "get_engine",
    "get_runtime",
    "is_apple_silicon",
    "is_coreml_available",
    "is_cuda_available",
    "is_jetson",
]
