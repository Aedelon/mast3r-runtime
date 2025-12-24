"""MASt3R runtime core module.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from .config import (
    MODEL_SPECS,
    NAVER_CDN_BASE,
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
    "NAVER_CDN_BASE",
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
    "get_checkpoint_paths",
    "get_checkpoint_urls",
    "get_total_checkpoint_size_mb",
    # Engine
    "EngineInterface",
    "InferenceResult",
    "MatchResult",
]
