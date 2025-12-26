"""Unit tests for mast3r_runtime.core.config.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mast3r_runtime.core.config import (
    MODEL_SPECS,
    PRESET_DESKTOP_PRECISION,
    PRESET_DRONE_FAST,
    PRESET_DRONE_QUALITY,
    BackendType,
    DownloadSource,
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


# ==============================================================================
# ModelVariant Tests
# ==============================================================================


class TestModelVariant:
    """Tests for ModelVariant enum."""

    def test_all_variants_defined(self):
        """All expected variants are defined."""
        expected = {
            "dune_vit_small_336",
            "dune_vit_small_448",
            "dune_vit_base_336",
            "dune_vit_base_448",
            "mast3r_vit_large",
        }
        actual = {v.value for v in ModelVariant}
        assert actual == expected

    def test_variant_from_string(self):
        """Variants can be created from string."""
        variant = ModelVariant("dune_vit_small_336")
        assert variant == ModelVariant.DUNE_VIT_SMALL_336

    def test_invalid_variant_raises(self):
        """Invalid variant string raises ValueError."""
        with pytest.raises(ValueError):
            ModelVariant("invalid_model")

    def test_variant_is_str_enum(self):
        """ModelVariant is a string enum."""
        assert ModelVariant.DUNE_VIT_SMALL_336.value == "dune_vit_small_336"
        # StrEnum value accessible directly
        assert ModelVariant.DUNE_VIT_SMALL_336 == "dune_vit_small_336"


# ==============================================================================
# BackendType Tests
# ==============================================================================


class TestBackendType:
    """Tests for BackendType enum."""

    def test_all_backends_defined(self):
        """All expected backends are defined."""
        expected = {"auto", "cpu", "metal", "cuda", "jetson", "python"}
        actual = {b.value for b in BackendType}
        assert actual == expected

    def test_backend_from_string(self):
        """Backends can be created from string."""
        backend = BackendType("metal")
        assert backend == BackendType.METAL


# ==============================================================================
# Precision Tests
# ==============================================================================


class TestPrecision:
    """Tests for Precision enum."""

    def test_all_precisions_defined(self):
        """All expected precisions are defined."""
        expected = {"fp32", "fp16", "int8", "int4"}
        actual = {p.value for p in Precision}
        assert actual == expected


# ==============================================================================
# DownloadSource Tests
# ==============================================================================


class TestDownloadSource:
    """Tests for DownloadSource enum."""

    def test_all_sources_defined(self):
        """All expected sources are defined."""
        expected = {"auto", "naver", "hf"}
        actual = {s.value for s in DownloadSource}
        assert actual == expected


# ==============================================================================
# MODEL_SPECS Tests
# ==============================================================================


class TestModelSpecs:
    """Tests for MODEL_SPECS dictionary."""

    def test_all_variants_have_specs(self):
        """All variants have corresponding specs."""
        for variant in ModelVariant:
            assert variant in MODEL_SPECS

    @pytest.mark.parametrize("variant", list(ModelVariant))
    def test_spec_has_required_fields(self, variant: ModelVariant):
        """Each spec has all required fields."""
        spec = MODEL_SPECS[variant]
        required_fields = [
            "encoder_arch",
            "decoder_arch",
            "patch_size",
            "embed_dim",
            "num_heads",
            "depth",
            "native_resolution",
            "hf_repo",
            "checkpoints",
        ]
        for field in required_fields:
            assert field in spec, f"Missing field {field} for {variant}"

    @pytest.mark.parametrize("variant", list(ModelVariant))
    def test_checkpoints_have_required_fields(self, variant: ModelVariant):
        """Each checkpoint has required fields."""
        spec = MODEL_SPECS[variant]
        for ckpt_type, ckpt_info in spec["checkpoints"].items():
            required = ["url", "filename", "hf_filename", "size_mb"]
            for field in required:
                assert field in ckpt_info, f"Missing {field} in {variant}/{ckpt_type}"

    def test_dune_models_have_encoder_decoder(self):
        """DUNE models have encoder and decoder checkpoints."""
        dune_variants = [
            ModelVariant.DUNE_VIT_SMALL_336,
            ModelVariant.DUNE_VIT_SMALL_448,
            ModelVariant.DUNE_VIT_BASE_336,
            ModelVariant.DUNE_VIT_BASE_448,
        ]
        for variant in dune_variants:
            spec = MODEL_SPECS[variant]
            assert "encoder" in spec["checkpoints"]
            assert "decoder" in spec["checkpoints"]

    def test_mast3r_has_unified_checkpoint(self):
        """MASt3R ViT-Large has unified checkpoint."""
        spec = MODEL_SPECS[ModelVariant.MAST3R_VIT_LARGE]
        assert "unified" in spec["checkpoints"]


# ==============================================================================
# ModelConfig Tests
# ==============================================================================


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Default values are correct."""
        config = ModelConfig()
        assert config.variant == ModelVariant.DUNE_VIT_SMALL_336
        # Resolution defaults to variant's native resolution (336 for DUNE_VIT_SMALL_336)
        assert config.resolution == 336
        assert config.precision == Precision.FP16

    def test_custom_values(self):
        """Custom values are accepted."""
        config = ModelConfig(
            variant=ModelVariant.MAST3R_VIT_LARGE,
            resolution=480,  # 480 = 16 * 30 (divisible by MASt3R patch_size)
            precision=Precision.FP32,
        )
        assert config.variant == ModelVariant.MAST3R_VIT_LARGE
        assert config.resolution == 480
        assert config.precision == Precision.FP32

    def test_resolution_validation_divisible_by_patch(self):
        """Resolution must be divisible by patch_size."""
        # 336 is divisible by 14
        config = ModelConfig(resolution=336)
        assert config.resolution == 336

        # 337 is not divisible by 14
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(resolution=337)
        assert "divisible by patch_size" in str(exc_info.value)

    def test_resolution_bounds(self):
        """Resolution must be within bounds."""
        with pytest.raises(ValidationError):
            ModelConfig(resolution=100)  # Too small

        with pytest.raises(ValidationError):
            ModelConfig(resolution=1000)  # Too large

    def test_spec_property(self):
        """spec property returns correct spec."""
        config = ModelConfig(variant=ModelVariant.DUNE_VIT_BASE_448)
        spec = config.spec
        assert spec["native_resolution"] == 448
        assert spec["embed_dim"] == 768


# ==============================================================================
# RuntimeConfig Tests
# ==============================================================================


class TestRuntimeConfig:
    """Tests for RuntimeConfig."""

    def test_default_values(self):
        """Default values are correct."""
        config = RuntimeConfig()
        assert config.backend == BackendType.AUTO
        assert config.use_dual_resolution is True
        assert config.tracking_resolution == 256
        assert config.keyframe_resolution == 336
        assert config.num_threads == 4
        assert config.use_gpu_preprocessing is True

    def test_resolution_validation(self):
        """Tracking resolution must be <= keyframe resolution."""
        # Valid
        config = RuntimeConfig(
            use_dual_resolution=True,
            tracking_resolution=224,
            keyframe_resolution=336,
        )
        assert config.tracking_resolution == 224

        # Invalid (tracking > keyframe)
        with pytest.raises(ValidationError) as exc_info:
            RuntimeConfig(
                use_dual_resolution=True,
                tracking_resolution=512,
                keyframe_resolution=336,
            )
        assert "must be <=" in str(exc_info.value)

    def test_dual_resolution_disabled_no_validation(self):
        """No validation when dual_resolution is disabled."""
        # Should not raise even if tracking > keyframe
        config = RuntimeConfig(
            use_dual_resolution=False,
            tracking_resolution=512,
            keyframe_resolution=336,
        )
        assert config.tracking_resolution == 512

    def test_num_threads_bounds(self):
        """num_threads must be within bounds."""
        with pytest.raises(ValidationError):
            RuntimeConfig(num_threads=0)

        with pytest.raises(ValidationError):
            RuntimeConfig(num_threads=100)


# ==============================================================================
# MatchingConfig Tests
# ==============================================================================


class TestMatchingConfig:
    """Tests for MatchingConfig."""

    def test_default_values(self):
        """Default values are correct."""
        config = MatchingConfig()
        assert config.top_k == 512
        assert config.reciprocal is True
        assert config.spatial_filter is True
        assert config.confidence_threshold == 0.5

    def test_top_k_bounds(self):
        """top_k must be within bounds."""
        with pytest.raises(ValidationError):
            MatchingConfig(top_k=10)  # Too small

        with pytest.raises(ValidationError):
            MatchingConfig(top_k=10000)  # Too large

    def test_confidence_threshold_bounds(self):
        """confidence_threshold must be between 0 and 1."""
        with pytest.raises(ValidationError):
            MatchingConfig(confidence_threshold=-0.1)

        with pytest.raises(ValidationError):
            MatchingConfig(confidence_threshold=1.5)


# ==============================================================================
# MASt3RRuntimeConfig Tests
# ==============================================================================


class TestMASt3RRuntimeConfig:
    """Tests for MASt3RRuntimeConfig."""

    def test_default_values(self):
        """Default values are correct."""
        config = MASt3RRuntimeConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.runtime, RuntimeConfig)
        assert isinstance(config.matching, MatchingConfig)
        assert config.cache_dir == Path.home() / ".cache" / "mast3r_runtime"

    def test_default_paths_set(self):
        """Default checkpoint and export paths are set."""
        config = MASt3RRuntimeConfig()
        assert config.checkpoint_dir == config.cache_dir / "checkpoints"
        assert config.export_dir == config.cache_dir / "exports"

    def test_custom_paths(self):
        """Custom paths are preserved."""
        config = MASt3RRuntimeConfig(
            cache_dir=Path("/tmp/cache"),
            checkpoint_dir=Path("/tmp/checkpoints"),
            export_dir=Path("/tmp/exports"),
        )
        assert config.cache_dir == Path("/tmp/cache")
        assert config.checkpoint_dir == Path("/tmp/checkpoints")
        assert config.export_dir == Path("/tmp/exports")

    def test_ensure_dirs(self, tmp_path: Path):
        """ensure_dirs creates directories."""
        config = MASt3RRuntimeConfig(
            cache_dir=tmp_path / "cache",
            checkpoint_dir=tmp_path / "ckpts",
            export_dir=tmp_path / "exports",
        )
        config.ensure_dirs()

        assert config.cache_dir.exists()
        assert config.checkpoint_dir.exists()
        assert config.export_dir.exists()

    def test_from_yaml(self, tmp_path: Path):
        """Config can be loaded from YAML."""
        yaml_content = """
model:
  variant: mast3r_vit_large
  resolution: 480
  precision: fp32
runtime:
  backend: python
  use_dual_resolution: false
matching:
  top_k: 1024
  reciprocal: true
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        config = MASt3RRuntimeConfig.from_yaml(yaml_path)

        assert config.model.variant == ModelVariant.MAST3R_VIT_LARGE
        assert config.model.resolution == 480  # 480 = 16 * 30
        assert config.runtime.backend == BackendType.PYTHON
        assert config.matching.top_k == 1024

    def test_to_yaml(self, tmp_path: Path):
        """Config can be saved to YAML."""
        config = MASt3RRuntimeConfig(
            model=ModelConfig(variant=ModelVariant.DUNE_VIT_BASE_336),
            runtime=RuntimeConfig(backend=BackendType.METAL),
        )
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        assert yaml_path.exists()
        loaded = MASt3RRuntimeConfig.from_yaml(yaml_path)
        assert loaded.model.variant == ModelVariant.DUNE_VIT_BASE_336
        assert loaded.runtime.backend == BackendType.METAL


# ==============================================================================
# Preset Tests
# ==============================================================================


class TestPresets:
    """Tests for preset configurations."""

    def test_drone_fast_preset(self):
        """PRESET_DRONE_FAST is correct."""
        config = PRESET_DRONE_FAST
        assert config.model.variant == ModelVariant.DUNE_VIT_SMALL_336
        assert config.model.resolution == 336
        assert config.model.precision == Precision.FP16
        assert config.runtime.use_dual_resolution is True
        assert config.runtime.tracking_resolution == 224
        assert config.runtime.keyframe_resolution == 336

    def test_drone_quality_preset(self):
        """PRESET_DRONE_QUALITY is correct."""
        config = PRESET_DRONE_QUALITY
        assert config.model.variant == ModelVariant.DUNE_VIT_BASE_448
        assert config.model.resolution == 448
        assert config.matching.top_k == 1024

    def test_desktop_precision_preset(self):
        """PRESET_DESKTOP_PRECISION is correct."""
        config = PRESET_DESKTOP_PRECISION
        assert config.model.variant == ModelVariant.MAST3R_VIT_LARGE
        assert config.model.resolution == 512  # CroCoNet uses patch_size=16
        assert config.model.precision == Precision.FP32
        assert config.runtime.backend == BackendType.PYTHON
        assert config.matching.top_k == 2048


# ==============================================================================
# Checkpoint Helper Functions Tests
# ==============================================================================


class TestCheckpointHelpers:
    """Tests for checkpoint helper functions."""

    def test_get_checkpoint_paths(self, tmp_path: Path):
        """get_checkpoint_paths returns correct paths."""
        paths = get_checkpoint_paths(ModelVariant.DUNE_VIT_SMALL_336, tmp_path)
        assert "encoder" in paths
        assert "decoder" in paths
        assert paths["encoder"].suffix == ".pth"
        assert paths["decoder"].suffix == ".pth"

    def test_get_checkpoint_paths_mast3r(self, tmp_path: Path):
        """get_checkpoint_paths for MASt3R returns unified."""
        paths = get_checkpoint_paths(ModelVariant.MAST3R_VIT_LARGE, tmp_path)
        assert "unified" in paths
        assert "retrieval" in paths

    def test_get_checkpoint_urls(self):
        """get_checkpoint_urls returns valid URLs."""
        urls = get_checkpoint_urls(ModelVariant.DUNE_VIT_SMALL_336)
        assert "encoder" in urls
        assert "decoder" in urls
        assert urls["encoder"].startswith("https://")
        assert "naverlabs.com" in urls["encoder"]

    def test_get_total_checkpoint_size_mb(self):
        """get_total_checkpoint_size_mb returns sum of sizes."""
        size = get_total_checkpoint_size_mb(ModelVariant.DUNE_VIT_SMALL_336)
        # Encoder ~109 MB + Decoder ~1234 MB
        assert size > 1000
        assert size < 2000

    def test_get_total_checkpoint_size_mast3r(self):
        """get_total_checkpoint_size_mb for MASt3R."""
        size = get_total_checkpoint_size_mb(ModelVariant.MAST3R_VIT_LARGE)
        # Unified ~2627 MB + retrieval ~8 MB + codebook ~256 MB
        assert size > 2500
