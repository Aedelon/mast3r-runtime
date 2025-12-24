"""MASt3R Runtime - Optimized inference for MASt3R/DUNE 3D vision models.

Cross-platform inference runtime for embedded deployment:
- Apple Silicon (M1/M2/M3/M4) via Metal
- NVIDIA GPU via CUDA
- NVIDIA Jetson (Orin) via TensorRT + DLA
- Fallback: CPU (C++) or Python (numpy)

Supports multiple model variants:
- DUNE-MASt3R ViT-Small/14 (~1.3GB) - Real-time drone
- DUNE-MASt3R ViT-Base/14 (~1.7GB) - Quality/speed balance
- MASt3R ViT-Large (~2.6GB) - Maximum precision

Usage:
    from mast3r_runtime import get_runtime, MASt3RRuntimeConfig

    config = MASt3RRuntimeConfig()
    engine = get_runtime(config)
    engine.load()
    result = engine.infer(img1, img2)
    matches = engine.match(result.desc_1, result.desc_2)

Note:
    Model weights must be downloaded separately:
    $ mast3r-runtime download dune_vit_small_14
    Models are licensed under CC BY-NC-SA 4.0 by Naver Corporation.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Delanoe Pirard"
__license__ = "Apache-2.0"

# Backend dispatcher
from .backends import (
    get_available_backends,
    get_backend_info,
    get_best_backend,
    get_runtime,
)

# Core configuration
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
    get_checkpoint_paths,
    get_checkpoint_urls,
    get_total_checkpoint_size_mb,
)

# Engine interface and results
from .core.engine_interface import (
    EngineInterface,
    InferenceResult,
    MatchResult,
)

__all__ = [
    # Version
    "__author__",
    "__license__",
    "__version__",
    # Config
    "MODEL_SPECS",
    "PRESET_DESKTOP_PRECISION",
    "PRESET_DRONE_FAST",
    "PRESET_DRONE_QUALITY",
    "BackendType",
    "MASt3RRuntimeConfig",
    "MatchingConfig",
    "ModelConfig",
    "ModelVariant",
    "Precision",
    "RuntimeConfig",
    # Checkpoint helpers
    "get_checkpoint_paths",
    "get_checkpoint_urls",
    "get_total_checkpoint_size_mb",
    # Engine
    "EngineInterface",
    "InferenceResult",
    "MatchResult",
    # Backend
    "get_available_backends",
    "get_backend_info",
    "get_best_backend",
    "get_runtime",
]
