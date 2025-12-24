"""ONNX Runtime backend for MASt3R inference.

Lightweight inference backend using ONNX Runtime.
Supports CPU, CUDA, CoreML, and TensorRT execution providers.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations


def is_available() -> bool:
    """Check if ONNX Runtime backend is available."""
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def get_available_providers() -> list[str]:
    """Get list of available ONNX Runtime execution providers."""
    try:
        import onnxruntime

        return onnxruntime.get_available_providers()
    except ImportError:
        return []


# Lazy import to avoid loading onnxruntime if not needed
def get_engine():
    """Get ONNXEngine class (lazy import)."""
    from .onnx_engine import ONNXEngine

    return ONNXEngine


__all__ = ["get_available_providers", "get_engine", "is_available"]
