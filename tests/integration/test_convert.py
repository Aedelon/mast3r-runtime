"""Integration tests for mast3r_runtime.utils.convert.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mast3r_runtime.core.config import ModelVariant
from mast3r_runtime.utils.convert import (
    _check_dependencies,
    convert_all_models,
    convert_checkpoint,
    convert_model,
    get_safetensors_paths,
    is_converted,
)

from ..conftest import skip_no_safetensors, skip_no_torch


# ==============================================================================
# _check_dependencies Tests
# ==============================================================================


class TestCheckDependencies:
    """Tests for _check_dependencies function."""

    @skip_no_torch
    @skip_no_safetensors
    def test_passes_with_deps(self):
        """Passes when dependencies are installed."""
        # Should not raise
        _check_dependencies()

    def test_raises_without_torch(self, monkeypatch):
        """Raises ImportError without torch."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError) as exc_info:
            _check_dependencies()
        assert "PyTorch" in str(exc_info.value)


# ==============================================================================
# convert_checkpoint Tests
# ==============================================================================


class TestConvertCheckpoint:
    """Tests for convert_checkpoint function."""

    @skip_no_torch
    @skip_no_safetensors
    def test_converts_pth_to_safetensors(self, tmp_path):
        """Converts .pth file to .safetensors."""
        import torch

        # Create a mock checkpoint
        pth_path = tmp_path / "model.pth"
        state_dict = {
            "layer.weight": torch.randn(10, 10),
            "layer.bias": torch.randn(10),
        }
        torch.save({"model": state_dict}, pth_path)

        # Convert
        output_path = convert_checkpoint(pth_path)

        assert output_path.exists()
        assert output_path.suffix == ".safetensors"

    @skip_no_torch
    @skip_no_safetensors
    def test_custom_output_path(self, tmp_path):
        """Respects custom output path."""
        import torch

        pth_path = tmp_path / "model.pth"
        torch.save({"model": {"w": torch.randn(5, 5)}}, pth_path)

        output_path = tmp_path / "custom" / "output.safetensors"
        result = convert_checkpoint(pth_path, output_path)

        assert result == output_path
        assert output_path.exists()

    @skip_no_torch
    @skip_no_safetensors
    def test_dtype_fp16(self, tmp_path):
        """Converts to fp16."""
        import torch
        from safetensors import safe_open

        pth_path = tmp_path / "model.pth"
        torch.save({"model": {"w": torch.randn(5, 5)}}, pth_path)

        output_path = convert_checkpoint(pth_path, dtype="fp16")

        with safe_open(output_path, framework="pt") as f:
            tensor = f.get_tensor("w")
            assert tensor.dtype == torch.float16

    @skip_no_torch
    @skip_no_safetensors
    def test_dtype_fp32(self, tmp_path):
        """Converts to fp32."""
        import torch
        from safetensors import safe_open

        pth_path = tmp_path / "model.pth"
        torch.save({"model": {"w": torch.randn(5, 5)}}, pth_path)

        output_path = convert_checkpoint(pth_path, dtype="fp32")

        with safe_open(output_path, framework="pt") as f:
            tensor = f.get_tensor("w")
            assert tensor.dtype == torch.float32

    @skip_no_torch
    @skip_no_safetensors
    def test_handles_nested_state_dict(self, tmp_path):
        """Handles checkpoints with nested state_dict."""
        import torch

        pth_path = tmp_path / "model.pth"
        torch.save({"state_dict": {"w": torch.randn(5, 5)}}, pth_path)

        output_path = convert_checkpoint(pth_path)
        assert output_path.exists()

    @skip_no_torch
    @skip_no_safetensors
    def test_handles_plain_state_dict(self, tmp_path):
        """Handles checkpoints that are just state_dict."""
        import torch

        pth_path = tmp_path / "model.pth"
        torch.save({"w": torch.randn(5, 5)}, pth_path)

        output_path = convert_checkpoint(pth_path)
        assert output_path.exists()


# ==============================================================================
# convert_model Tests
# ==============================================================================


class TestConvertModel:
    """Tests for convert_model function."""

    @skip_no_torch
    @skip_no_safetensors
    def test_missing_checkpoint(self, tmp_path, capsys):
        """Handles missing checkpoints gracefully."""
        result = convert_model(ModelVariant.DUNE_VIT_SMALL_336, cache_dir=tmp_path)

        assert result == {}  # No conversions
        captured = capsys.readouterr()
        assert "MISSING" in captured.out

    @skip_no_torch
    @skip_no_safetensors
    def test_skips_pkl_files(self, tmp_path, capsys):
        """Skips .pkl files (codebooks)."""
        import torch

        # Create checkpoints dir with .pkl file
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # MASt3R has a codebook.pkl that should be skipped
        pkl_path = (
            ckpt_dir / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
        )
        import pickle

        with open(pkl_path, "wb") as f:
            pickle.dump({}, f)

        # Also need the main checkpoint
        pth_path = ckpt_dir / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        torch.save({"model": {"w": torch.randn(5, 5)}}, pth_path)

        result = convert_model(ModelVariant.MAST3R_VIT_LARGE, cache_dir=tmp_path)

        captured = capsys.readouterr()
        assert "SKIPPED" in captured.out or "pkl" in captured.out.lower()

    @skip_no_torch
    @skip_no_safetensors
    def test_converts_all_checkpoints(self, tmp_path):
        """Converts all available checkpoints for variant."""
        import torch

        # Create checkpoints
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # DUNE needs encoder and decoder
        encoder_path = ckpt_dir / "dune_vitsmall14_336.pth"
        decoder_path = ckpt_dir / "dunemast3r_cvpr25_vitsmall.pth"

        torch.save({"model": {"w": torch.randn(5, 5)}}, encoder_path)
        torch.save({"model": {"w": torch.randn(5, 5)}}, decoder_path)

        result = convert_model(ModelVariant.DUNE_VIT_SMALL_336, cache_dir=tmp_path)

        assert "encoder" in result
        assert "decoder" in result
        assert result["encoder"].exists()
        assert result["decoder"].exists()


# ==============================================================================
# convert_all_models Tests
# ==============================================================================


class TestConvertAllModels:
    """Tests for convert_all_models function."""

    @skip_no_torch
    @skip_no_safetensors
    def test_empty_cache(self, tmp_path):
        """Returns empty dict for empty cache."""
        result = convert_all_models(cache_dir=tmp_path)
        assert result == {}

    @skip_no_torch
    @skip_no_safetensors
    def test_converts_available(self, tmp_path):
        """Converts all available models."""
        import torch

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # Create one checkpoint
        path = ckpt_dir / "dune_vitsmall14_336.pth"
        torch.save({"model": {"w": torch.randn(5, 5)}}, path)

        result = convert_all_models(cache_dir=tmp_path)

        assert ModelVariant.DUNE_VIT_SMALL_336 in result


# ==============================================================================
# get_safetensors_paths Tests
# ==============================================================================


class TestGetSafetensorsPaths:
    """Tests for get_safetensors_paths function."""

    def test_returns_dict(self, tmp_path):
        """Returns a dictionary."""
        paths = get_safetensors_paths(ModelVariant.DUNE_VIT_SMALL_336, tmp_path)
        assert isinstance(paths, dict)

    def test_dune_has_encoder_decoder(self, tmp_path):
        """DUNE variants have encoder and decoder paths."""
        paths = get_safetensors_paths(ModelVariant.DUNE_VIT_SMALL_336, tmp_path)
        assert "encoder" in paths
        assert "decoder" in paths

    def test_mast3r_has_unified(self, tmp_path):
        """MASt3R has unified path (but not codebook - it's .pkl)."""
        paths = get_safetensors_paths(ModelVariant.MAST3R_VIT_LARGE, tmp_path)
        assert "unified" in paths
        # Codebook is .pkl, should not be in safetensors paths
        assert "codebook" not in paths

    def test_paths_have_safetensors_extension(self, tmp_path):
        """All paths have .safetensors extension."""
        paths = get_safetensors_paths(ModelVariant.DUNE_VIT_SMALL_336, tmp_path)
        for path in paths.values():
            assert path.suffix == ".safetensors"


# ==============================================================================
# is_converted Tests
# ==============================================================================


class TestIsConverted:
    """Tests for is_converted function."""

    def test_false_when_missing(self, tmp_path):
        """Returns False when files don't exist."""
        result = is_converted(ModelVariant.DUNE_VIT_SMALL_336, tmp_path)
        assert result is False

    @skip_no_safetensors
    def test_true_when_complete(self, tmp_path):
        """Returns True when all files exist."""
        from safetensors.numpy import save_file

        # Create safetensors files
        st_dir = tmp_path / "safetensors" / "dune_vit_small_336"
        st_dir.mkdir(parents=True)

        tensors = {"w": np.zeros((5, 5), dtype=np.float16)}
        save_file(tensors, st_dir / "encoder.safetensors")
        save_file(tensors, st_dir / "decoder.safetensors")

        result = is_converted(ModelVariant.DUNE_VIT_SMALL_336, tmp_path)
        assert result is True

    @skip_no_safetensors
    def test_false_when_partial(self, tmp_path):
        """Returns False when only some files exist."""
        from safetensors.numpy import save_file

        st_dir = tmp_path / "safetensors" / "dune_vit_small_336"
        st_dir.mkdir(parents=True)

        tensors = {"w": np.zeros((5, 5), dtype=np.float16)}
        save_file(tensors, st_dir / "encoder.safetensors")
        # decoder missing

        result = is_converted(ModelVariant.DUNE_VIT_SMALL_336, tmp_path)
        assert result is False
