"""TensorRT backend for MASt3R inference (NVIDIA Jetson/Linux).

Optimized for NVIDIA GPUs with INT8/FP16 acceleration.

Status: Planned for v0.2.0
Currently uses ONNX Runtime with TensorrtExecutionProvider.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations


def is_available() -> bool:
    """Check if TensorRT backend is available."""
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = ["is_available"]
