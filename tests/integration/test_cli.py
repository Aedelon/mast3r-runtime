"""Integration tests for mast3r_runtime.cli.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mast3r_runtime.cli import (
    cmd_convert,
    cmd_download,
    cmd_status,
    main,
)
from mast3r_runtime.core.config import DownloadSource, ModelVariant


# ==============================================================================
# Main Entry Point Tests
# ==============================================================================


class TestMainEntryPoint:
    """Tests for main() function."""

    def test_no_args_prints_help(self, capsys):
        """No arguments prints help and returns 0."""
        with patch("sys.argv", ["mast3r-runtime"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "mast3r-runtime" in captured.out or "usage" in captured.out.lower()

    def test_help_flag(self, capsys):
        """--help flag prints help."""
        with patch("sys.argv", ["mast3r-runtime", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_status_command(self, capsys, tmp_path):
        """status command works."""
        with patch("sys.argv", ["mast3r-runtime", "--cache-dir", str(tmp_path), "status"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Checkpoint Status" in captured.out


# ==============================================================================
# cmd_download Tests
# ==============================================================================


class TestCmdDownload:
    """Tests for cmd_download function."""

    @pytest.fixture
    def download_args(self, tmp_path):
        """Create argparse.Namespace for download command."""
        return argparse.Namespace(
            cache_dir=str(tmp_path),
            model="dune_vit_small_336",
            force=False,
            quiet=True,
            source="auto",
        )

    def test_download_with_invalid_model(self, tmp_path):
        """Invalid model raises ValueError."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="invalid_model",
            force=False,
            quiet=True,
            source="auto",
        )
        result = cmd_download(args)
        assert result == 1  # Error

    @patch("mast3r_runtime.cli.download_model")
    def test_download_single_model(self, mock_download, tmp_path):
        """Downloading single model calls download_model."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="dune_vit_small_336",
            force=False,
            quiet=True,
            source="auto",
        )
        mock_download.return_value = {}

        result = cmd_download(args)

        assert result == 0
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args.args[0] == ModelVariant.DUNE_VIT_SMALL_336

    @patch("mast3r_runtime.cli.download_all_models")
    def test_download_all_models(self, mock_download_all, tmp_path):
        """Downloading 'all' calls download_all_models."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="all",
            force=False,
            quiet=True,
            source="auto",
        )
        mock_download_all.return_value = {}

        result = cmd_download(args)

        assert result == 0
        mock_download_all.assert_called_once()

    @patch("mast3r_runtime.cli.download_model")
    def test_download_force_flag(self, mock_download, tmp_path):
        """Force flag is passed to download_model."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="dune_vit_small_336",
            force=True,
            quiet=False,
            source="auto",
        )
        mock_download.return_value = {}

        cmd_download(args)

        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["force"] is True

    @patch("mast3r_runtime.cli.download_model")
    def test_download_source_naver(self, mock_download, tmp_path):
        """Source 'naver' is passed correctly."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="dune_vit_small_336",
            force=False,
            quiet=True,
            source="naver",
        )
        mock_download.return_value = {}

        cmd_download(args)

        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["source"] == DownloadSource.NAVER

    @patch("mast3r_runtime.cli.download_model")
    def test_download_source_hf(self, mock_download, tmp_path):
        """Source 'hf' is passed correctly."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="dune_vit_small_336",
            force=False,
            quiet=True,
            source="hf",
        )
        mock_download.return_value = {}

        cmd_download(args)

        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["source"] == DownloadSource.HF


# ==============================================================================
# cmd_status Tests
# ==============================================================================


class TestCmdStatus:
    """Tests for cmd_status function."""

    def test_status_with_empty_cache(self, tmp_path, capsys):
        """Status with empty cache shows all missing."""
        args = argparse.Namespace(cache_dir=str(tmp_path))
        result = cmd_status(args)

        assert result == 0
        captured = capsys.readouterr()
        # Should show status for all variants
        assert "dune_vit_small_336" in captured.out or "DUNE" in captured.out.upper()

    def test_status_with_default_cache(self, capsys):
        """Status with default cache dir works."""
        args = argparse.Namespace(cache_dir=None)
        result = cmd_status(args)

        assert result == 0

    @patch("mast3r_runtime.cli.print_status")
    def test_status_calls_print_status(self, mock_print, tmp_path):
        """cmd_status calls print_status."""
        args = argparse.Namespace(cache_dir=str(tmp_path))
        cmd_status(args)

        mock_print.assert_called_once_with(Path(tmp_path))


# ==============================================================================
# cmd_convert Tests
# ==============================================================================


class TestCmdConvert:
    """Tests for cmd_convert function."""

    @pytest.fixture
    def convert_args(self, tmp_path):
        """Create argparse.Namespace for convert command."""
        return argparse.Namespace(
            cache_dir=str(tmp_path),
            model="dune_vit_small_336",
            output=str(tmp_path / "output"),
            dtype="fp16",
        )

    def test_convert_with_invalid_model(self, tmp_path):
        """Invalid model raises error."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="invalid_model",
            output=None,
            dtype="fp16",
        )
        result = cmd_convert(args)
        assert result == 1

    @patch("mast3r_runtime.cli.convert_model")
    def test_convert_single_model(self, mock_convert, tmp_path):
        """Converting single model calls convert_model."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="dune_vit_small_336",
            output=str(tmp_path / "output"),
            dtype="fp16",
        )
        mock_convert.return_value = {}

        result = cmd_convert(args)

        assert result == 0
        mock_convert.assert_called_once()

    @patch("mast3r_runtime.cli.convert_all_models")
    def test_convert_all_models(self, mock_convert_all, tmp_path):
        """Converting 'all' calls convert_all_models."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="all",
            output=None,
            dtype="fp16",
        )
        mock_convert_all.return_value = {}

        result = cmd_convert(args)

        assert result == 0
        mock_convert_all.assert_called_once()

    @patch("mast3r_runtime.cli.convert_model")
    def test_convert_dtype_passed(self, mock_convert, tmp_path):
        """dtype is passed to convert_model."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="dune_vit_small_336",
            output=None,
            dtype="bf16",
        )
        mock_convert.return_value = {}

        cmd_convert(args)

        call_kwargs = mock_convert.call_args.kwargs
        assert call_kwargs["dtype"] == "bf16"

    def test_convert_import_error(self, tmp_path, capsys):
        """ImportError is caught and returns 1."""
        args = argparse.Namespace(
            cache_dir=str(tmp_path),
            model="dune_vit_small_336",
            output=None,
            dtype="fp16",
        )

        with patch("mast3r_runtime.cli.convert_model") as mock:
            mock.side_effect = ImportError("torch not installed")
            result = cmd_convert(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err


# ==============================================================================
# CLI Argument Parsing Tests
# ==============================================================================


class TestCliArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_download_model_choices(self):
        """download command accepts valid model choices."""
        valid_models = ["all"] + [v.value for v in ModelVariant]

        for model in valid_models:
            with patch("sys.argv", ["mast3r-runtime", "download", model, "-q"]):
                with patch("mast3r_runtime.cli.download_model") as mock:
                    mock.return_value = {}
                    with patch("mast3r_runtime.cli.download_all_models") as mock_all:
                        mock_all.return_value = {}
                        # Should not raise
                        main()

    def test_download_source_choices(self):
        """download --source accepts valid choices."""
        valid_sources = ["auto", "naver", "hf"]

        for source in valid_sources:
            with patch("sys.argv", ["mast3r-runtime", "download", "all", "-q", "-s", source]):
                with patch("mast3r_runtime.cli.download_all_models") as mock:
                    mock.return_value = {}
                    main()

    def test_convert_dtype_choices(self):
        """convert --dtype accepts valid choices."""
        valid_dtypes = ["fp32", "fp16", "bf16"]

        for dtype in valid_dtypes:
            with patch("sys.argv", ["mast3r-runtime", "convert", "all", "--dtype", dtype]):
                with patch("mast3r_runtime.cli.convert_all_models") as mock:
                    mock.return_value = {}
                    main()

    def test_global_cache_dir_option(self, tmp_path):
        """--cache-dir is parsed correctly."""
        cache_path = str(tmp_path / "custom_cache")

        with patch("sys.argv", ["mast3r-runtime", "--cache-dir", cache_path, "status"]):
            with patch("mast3r_runtime.cli.print_status") as mock:
                main()

            mock.assert_called_with(Path(cache_path))
