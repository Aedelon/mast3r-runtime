"""Configuration for MASt3R embedded runtime.

Supports multiple model variants:
- DUNE-MASt3R ViT-Small/14 (~1.3GB) - Fastest, for real-time drone
- DUNE-MASt3R ViT-Base/14 (~1.7GB) - Balance quality/speed
- MASt3R ViT-Large (~2.6GB) - Maximum precision

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ModelVariant(str, Enum):
    """Supported model variants."""

    DUNE_VIT_SMALL_14 = "dune_vit_small_14"
    DUNE_VIT_BASE_14 = "dune_vit_base_14"
    MAST3R_VIT_LARGE = "mast3r_vit_large"


class BackendType(str, Enum):
    """Runtime backend types."""

    AUTO = "auto"
    CPU = "cpu"  # C++ with OpenMP/BLAS
    METAL = "metal"  # Apple Silicon (Metal + MPSGraph)
    CUDA = "cuda"  # NVIDIA GPU (cuBLAS + custom kernels)
    JETSON = "jetson"  # NVIDIA Jetson (TensorRT + DLA)
    PYTHON = "python"  # Pure Python/numpy fallback


class Precision(str, Enum):
    """Inference precision."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"  # CoreML palettization


# Base URL for DUNE/MASt3R checkpoints
NAVER_CDN_BASE = "https://download.europe.naverlabs.com"

# Model specifications
# DUNE models require both encoder and decoder checkpoints
# MASt3R ViT-Large is a single unified checkpoint
MODEL_SPECS: dict[ModelVariant, dict] = {
    ModelVariant.DUNE_VIT_SMALL_14: {
        "encoder_arch": "vit_small_patch14_dinov2",
        "decoder_arch": "vit_base_decoder",
        "patch_size": 14,
        "embed_dim": 384,
        "num_heads": 6,
        "depth": 12,
        "native_resolution": 448,
        "checkpoints": {
            "encoder": {
                "url": f"{NAVER_CDN_BASE}/dune/dune_vitsmall14_448.pth",
                "filename": "dune_vitsmall14_448.pth",
                "size_mb": 109,
            },
            "decoder": {
                "url": f"{NAVER_CDN_BASE}/dune/dunemast3r_cvpr25_vitsmall.pth",
                "filename": "dunemast3r_cvpr25_vitsmall.pth",
                "size_mb": 1234,
            },
        },
    },
    ModelVariant.DUNE_VIT_BASE_14: {
        "encoder_arch": "vit_base_patch14_dinov2",
        "decoder_arch": "vit_base_decoder",
        "patch_size": 14,
        "embed_dim": 768,
        "num_heads": 12,
        "depth": 12,
        "native_resolution": 448,
        "checkpoints": {
            "encoder": {
                "url": f"{NAVER_CDN_BASE}/dune/dune_vitbase14_448.pth",
                "filename": "dune_vitbase14_448.pth",
                "size_mb": 420,
            },
            "decoder": {
                "url": f"{NAVER_CDN_BASE}/dune/dunemast3r_cvpr25_vitbase.pth",
                "filename": "dunemast3r_cvpr25_vitbase.pth",
                "size_mb": 1325,
            },
        },
    },
    ModelVariant.MAST3R_VIT_LARGE: {
        "encoder_arch": "vit_large_patch14_dinov2",
        "decoder_arch": "vit_base_decoder",
        "patch_size": 14,
        "embed_dim": 1024,
        "num_heads": 16,
        "depth": 24,
        "native_resolution": 512,
        "checkpoints": {
            "unified": {
                "url": f"{NAVER_CDN_BASE}/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                "filename": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                "size_mb": 2627,
            },
            "retrieval": {
                "url": f"{NAVER_CDN_BASE}/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
                "filename": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
                "size_mb": 8,
            },
            "codebook": {
                "url": f"{NAVER_CDN_BASE}/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
                "filename": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
                "size_mb": 256,
            },
        },
    },
}


class ModelConfig(BaseModel):
    """Model configuration."""

    variant: ModelVariant = Field(
        default=ModelVariant.DUNE_VIT_SMALL_14,
        description="Model variant to use",
    )
    resolution: int = Field(
        default=448,
        ge=224,
        le=560,  # Max 560 = 14 * 40
        description="Input resolution (must be divisible by patch_size)",
    )
    precision: Precision = Field(
        default=Precision.FP16,
        description="Inference precision",
    )

    @model_validator(mode="after")
    def validate_resolution(self) -> ModelConfig:
        """Ensure resolution is divisible by patch size."""
        spec = MODEL_SPECS[self.variant]
        patch_size = spec["patch_size"]
        if self.resolution % patch_size != 0:
            msg = f"Resolution {self.resolution} must be divisible by patch_size {patch_size}"
            raise ValueError(msg)
        return self

    @property
    def spec(self) -> dict:
        """Get model specification."""
        return MODEL_SPECS[self.variant]


class RuntimeConfig(BaseModel):
    """Runtime configuration."""

    backend: BackendType = Field(
        default=BackendType.AUTO,
        description="Backend to use (auto selects based on platform)",
    )
    use_dual_resolution: bool = Field(
        default=True,
        description="Use lower resolution for tracking, higher for keyframes",
    )
    tracking_resolution: int = Field(
        default=256,
        ge=128,
        le=512,
        description="Resolution for tracking frames (when dual_resolution enabled)",
    )
    keyframe_resolution: int = Field(
        default=336,
        ge=224,
        le=512,
        description="Resolution for keyframes (when dual_resolution enabled)",
    )
    num_threads: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of CPU threads for preprocessing",
    )
    use_gpu_preprocessing: bool = Field(
        default=True,
        description="Use GPU for image preprocessing",
    )

    @model_validator(mode="after")
    def validate_resolutions(self) -> RuntimeConfig:
        """Ensure tracking resolution <= keyframe resolution."""
        if self.use_dual_resolution and self.tracking_resolution > self.keyframe_resolution:
            msg = f"Tracking resolution ({self.tracking_resolution}) must be <= keyframe resolution ({self.keyframe_resolution})"
            raise ValueError(msg)
        return self


class MatchingConfig(BaseModel):
    """Matching configuration."""

    top_k: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Number of top matches to keep",
    )
    reciprocal: bool = Field(
        default=True,
        description="Use reciprocal matching (bidirectional consistency)",
    )
    spatial_filter: bool = Field(
        default=True,
        description="Apply spatial consistency filtering",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for matches",
    )


class MASt3RRuntimeConfig(BaseModel):
    """Complete MASt3R runtime configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)

    # Paths
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "mast3r_runtime",
        description="Directory for cached models and exports",
    )
    checkpoint_dir: Path | None = Field(
        default=None,
        description="Custom checkpoint directory (uses cache_dir/checkpoints if None)",
    )
    export_dir: Path | None = Field(
        default=None,
        description="Custom export directory (uses cache_dir/exports if None)",
    )

    @model_validator(mode="after")
    def set_default_paths(self) -> MASt3RRuntimeConfig:
        """Set default paths based on cache_dir."""
        if self.checkpoint_dir is None:
            object.__setattr__(self, "checkpoint_dir", self.cache_dir / "checkpoints")
        if self.export_dir is None:
            object.__setattr__(self, "export_dir", self.cache_dir / "exports")
        return self

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.export_dir:
            self.export_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MASt3RRuntimeConfig:
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)


# Preset configurations
PRESET_DRONE_FAST = MASt3RRuntimeConfig(
    model=ModelConfig(
        variant=ModelVariant.DUNE_VIT_SMALL_14,
        resolution=448,
        precision=Precision.FP16,
    ),
    runtime=RuntimeConfig(
        backend=BackendType.AUTO,
        use_dual_resolution=True,
        tracking_resolution=224,
        keyframe_resolution=448,
    ),
    matching=MatchingConfig(
        top_k=512,
        reciprocal=True,
    ),
)

PRESET_DRONE_QUALITY = MASt3RRuntimeConfig(
    model=ModelConfig(
        variant=ModelVariant.DUNE_VIT_BASE_14,
        resolution=448,
        precision=Precision.FP16,
    ),
    runtime=RuntimeConfig(
        backend=BackendType.AUTO,
        use_dual_resolution=True,
        tracking_resolution=224,
        keyframe_resolution=448,
    ),
    matching=MatchingConfig(
        top_k=1024,
        reciprocal=True,
    ),
)

PRESET_DESKTOP_PRECISION = MASt3RRuntimeConfig(
    model=ModelConfig(
        variant=ModelVariant.MAST3R_VIT_LARGE,
        resolution=518,  # 518 = 14 * 37 (divisible by patch_size)
        precision=Precision.FP32,
    ),
    runtime=RuntimeConfig(
        backend=BackendType.PYTHON,
        use_dual_resolution=False,
    ),
    matching=MatchingConfig(
        top_k=2048,
        reciprocal=True,
    ),
)


def get_checkpoint_paths(
    variant: ModelVariant,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """Get paths for model checkpoints.

    Args:
        variant: Model variant.
        cache_dir: Optional cache directory. Uses ~/.cache/mast3r_runtime if None.

    Returns:
        Dict mapping checkpoint type to path.
        - DUNE models: {"encoder": Path, "decoder": Path}
        - MASt3R: {"unified": Path}
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "mast3r_runtime"

    models_dir = cache_dir / "checkpoints"
    spec = MODEL_SPECS[variant]

    paths = {}
    for ckpt_type, ckpt_info in spec["checkpoints"].items():
        paths[ckpt_type] = models_dir / ckpt_info["filename"]

    return paths


def get_checkpoint_urls(variant: ModelVariant) -> dict[str, str]:
    """Get download URLs for model checkpoints.

    Args:
        variant: Model variant.

    Returns:
        Dict mapping checkpoint type to download URL.
    """
    spec = MODEL_SPECS[variant]
    return {
        ckpt_type: ckpt_info["url"]
        for ckpt_type, ckpt_info in spec["checkpoints"].items()
    }


def get_total_checkpoint_size_mb(variant: ModelVariant) -> int:
    """Get total size of all checkpoints for a variant.

    Args:
        variant: Model variant.

    Returns:
        Total size in MB.
    """
    spec = MODEL_SPECS[variant]
    return sum(ckpt_info["size_mb"] for ckpt_info in spec["checkpoints"].values())
