"""CoreML backend for MASt3R inference (Apple Silicon).

Optimized for Apple Neural Engine (ANE) on M1/M2/M3/M4 chips.

Status: Planned for v0.2.0
Currently uses ONNX Runtime with CoreMLExecutionProvider.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations


def is_available() -> bool:
    """Check if CoreML backend is available."""
    try:
        import coremltools  # noqa: F401
        import platform

        return platform.system() == "Darwin" and platform.machine() == "arm64"
    except ImportError:
        return False


__all__ = ["is_available"]
