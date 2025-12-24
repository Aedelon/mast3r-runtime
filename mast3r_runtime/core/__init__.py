"""MASt3R runtime core module.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from .config import (
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
from .engine_interface import (
    EngineInterface,
    InferenceResult,
    MatchResult,
)
from .preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NumpyPreprocessor,
    PreprocessorBase,
    prepare_image_numpy,
)

__all__ = [
    # Preprocessing
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NumpyPreprocessor",
    "PreprocessorBase",
    "prepare_image_numpy",
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
    # Engine
    "EngineInterface",
    "InferenceResult",
    "MatchResult",
]
