"""Configuration for MASt3R embedded runtime.

Supports multiple model variants:
- DUNE-MASt3R ViT-Small/14 (~280MB) - Fastest, for real-time drone
- DUNE-MASt3R ViT-Base/14 (~650MB) - Balance quality/speed
- MASt3R ViT-Large (1.2GB) - Maximum precision

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
MODEL_SPECS: dict[ModelVariant, dict] = {
    ModelVariant.DUNE_VIT_SMALL_14: {
        "encoder": "vit_small_patch14_dinov2",
        "patch_size": 14,
        "embed_dim": 384,
        "num_heads": 6,
        "depth": 12,
        "native_resolution": 448,
        "checkpoint_size_mb": 280,
        "download_url": f"{NAVER_CDN_BASE}/dune/dunemast3r_cvpr25_vitsmall.pth",
        "checkpoint_name": "dunemast3r_cvpr25_vitsmall.pth",
    },
    ModelVariant.DUNE_VIT_BASE_14: {
        "encoder": "vit_base_patch14_dinov2",
        "patch_size": 14,
        "embed_dim": 768,
        "num_heads": 12,
        "depth": 12,
        "native_resolution": 448,
        "checkpoint_size_mb": 650,
        "download_url": f"{NAVER_CDN_BASE}/dune/dunemast3r_cvpr25_vitbase.pth",
        "checkpoint_name": "dunemast3r_cvpr25_vitbase.pth",
    },
    ModelVariant.MAST3R_VIT_LARGE: {
        "encoder": "vit_large_patch14_dinov2",
        "patch_size": 14,
        "embed_dim": 1024,
        "num_heads": 16,
        "depth": 24,
        "native_resolution": 512,
        "checkpoint_size_mb": 1200,
        "download_url": "https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/main/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "checkpoint_name": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
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


def get_default_model_path(
    variant: ModelVariant,
    precision: Precision = Precision.FP16,
    cache_dir: Path | None = None,
) -> Path:
    """Get default path for model weights.

    Args:
        variant: Model variant.
        precision: Model precision.
        cache_dir: Optional cache directory. Uses ~/.cache/mast3r_runtime if None.

    Returns:
        Path to model weights file (may not exist yet).
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "mast3r_runtime"

    models_dir = cache_dir / "models"
    filename = f"{variant.value}_{precision.value}.safetensors"

    return models_dir / filename
