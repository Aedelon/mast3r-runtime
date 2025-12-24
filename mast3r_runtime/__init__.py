"""MASt3R Runtime - Lightweight inference for MASt3R/DUNE 3D vision models.

Cross-platform inference runtime optimized for embedded deployment:
- Apple Silicon (M1/M2/M3/M4) via CoreML
- NVIDIA Jetson (Orin) via TensorRT
- Any platform via ONNX Runtime

Supports multiple model variants:
- DUNE ViT-Small/14 (110MB) - Real-time drone (<20ms on M4)
- DUNE ViT-Base/14 (420MB) - Quality/speed balance
- MASt3R ViT-Large (1.2GB) - Maximum precision

Installation:
    pip install mast3r-runtime[onnx]     # ONNX Runtime (recommended)
    pip install mast3r-runtime[coreml]   # CoreML (Apple Silicon)
    pip install mast3r-runtime[all]      # All backends

Usage:
    from mast3r_runtime import get_runtime, MASt3RRuntimeConfig

    config = MASt3RRuntimeConfig()
    with get_runtime(config) as engine:
        result = engine.infer(img1, img2)
        matches = engine.match(result.desc_1, result.desc_2)

Note:
    This package provides the inference runtime only.
    Model weights must be downloaded separately using `mast3r-download`.
    Models are licensed under CC BY-NC-SA 4.0 by Naver Corporation.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Delanoe Pirard"
__license__ = "Apache-2.0"

# Core configuration
# Backend dispatcher
from .backends import (
    get_available_backends,
    get_backend,
    get_engine,
    get_runtime,
    is_apple_silicon,
    is_coreml_available,
    is_cuda_available,
    is_jetson,
)
from .core.config import (
    MODEL_SPECS,
    PRESET_DESKTOP_PRECISION,
    PRESET_DRONE_FAST,
    PRESET_DRONE_QUALITY,
    BackendType,
    MASt3RRuntimeConfig,
    MatchingConfig,
    ModelConfig,
    ModelVariant,
    Precision,
    RuntimeConfig,
)

# Engine interface and results
from .core.engine_interface import (
    EngineInterface,
    InferenceResult,
    MatchResult,
)

__all__ = [
    # Presets
    "MODEL_SPECS",
    "PRESET_DESKTOP_PRECISION",
    "PRESET_DRONE_FAST",
    "PRESET_DRONE_QUALITY",
    # Config enums
    "BackendType",
    # Engine interface
    "EngineInterface",
    "InferenceResult",
    # Config classes
    "MASt3RRuntimeConfig",
    "MatchResult",
    "MatchingConfig",
    "ModelConfig",
    "ModelVariant",
    "Precision",
    "RuntimeConfig",
    # Version info
    "__author__",
    "__license__",
    "__version__",
    # Runtime factory
    "get_available_backends",
    "get_backend",
    "get_engine",
    "get_runtime",
    # Platform detection
    "is_apple_silicon",
    "is_coreml_available",
    "is_cuda_available",
    "is_jetson",
]
