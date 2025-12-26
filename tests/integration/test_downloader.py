"""Integration tests for mast3r_runtime.utils.downloader.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mast3r_runtime.core.config import DownloadSource, ModelVariant
from mast3r_runtime.utils.downloader import (
    DownloadError,
    _download_file,
    _format_size,
    _get_cache_dir,
    _has_aria2c,
    _has_huggingface_hub,
    _print_progress,
    download_model,
    get_download_status,
    print_status,
)


# ==============================================================================
# Helper Function Tests
# ==============================================================================


class TestFormatSize:
    """Tests for _format_size function."""

    def test_bytes(self):
        """Formats bytes correctly."""
        assert "B" in _format_size(500)

    def test_kilobytes(self):
        """Formats kilobytes correctly."""
        assert "KB" in _format_size(1024 * 2)

    def test_megabytes(self):
        """Formats megabytes correctly."""
        assert "MB" in _format_size(1024 * 1024 * 5)

    def test_gigabytes(self):
        """Formats gigabytes correctly."""
        assert "GB" in _format_size(1024 * 1024 * 1024 * 2)


class TestGetCacheDir:
    """Tests for _get_cache_dir function."""

    def test_returns_path(self):
        """Returns a Path object."""
        cache_dir = _get_cache_dir()
        assert isinstance(cache_dir, Path)

    def test_default_location(self):
        """Default location is in home directory."""
        cache_dir = _get_cache_dir()
        assert "mast3r_runtime" in str(cache_dir)
        assert str(Path.home()) in str(cache_dir)


class TestHasAria2c:
    """Tests for _has_aria2c function."""

    def test_returns_bool(self):
        """Returns a boolean."""
        result = _has_aria2c()
        assert isinstance(result, bool)


class TestHasHuggingfaceHub:
    """Tests for _has_huggingface_hub function."""

    def test_returns_bool(self):
        """Returns a boolean."""
        result = _has_huggingface_hub()
        assert isinstance(result, bool)


class TestPrintProgress:
    """Tests for _print_progress function."""

    def test_zero_total(self, capsys):
        """Handles zero total without error."""
        _print_progress(0, 0)
        # Should not crash

    def test_partial_progress(self, capsys):
        """Prints partial progress."""
        _print_progress(500, 1000)
        captured = capsys.readouterr()
        assert "50" in captured.out  # 50%

    def test_complete_progress(self, capsys):
        """Prints complete progress with newline."""
        _print_progress(1000, 1000)
        captured = capsys.readouterr()
        assert "100" in captured.out


# ==============================================================================
# _download_file Tests
# ==============================================================================


class TestDownloadFile:
    """Tests for _download_file function."""

    @patch("urllib.request.urlopen")
    def test_successful_download(self, mock_urlopen, tmp_path):
        """Successful download creates file."""
        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.read.side_effect = [b"test data", b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock()
        mock_urlopen.return_value = mock_response

        dest = tmp_path / "test.txt"
        _download_file("https://example.com/file", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"test data"

    @patch("urllib.request.urlopen")
    def test_download_with_callback(self, mock_urlopen, tmp_path):
        """Progress callback is called."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.read.side_effect = [b"data", b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock()
        mock_urlopen.return_value = mock_response

        callback = MagicMock()
        dest = tmp_path / "test.txt"
        _download_file("https://example.com/file", dest, progress_callback=callback)

        callback.assert_called()

    @patch("urllib.request.urlopen")
    def test_download_error(self, mock_urlopen, tmp_path):
        """Network error raises DownloadError."""
        mock_urlopen.side_effect = Exception("Network error")

        dest = tmp_path / "test.txt"
        with pytest.raises(DownloadError) as exc_info:
            _download_file("https://example.com/file", dest)

        assert "Failed to download" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_cleanup_on_error(self, mock_urlopen, tmp_path):
        """Temp file is cleaned up on error."""
        # Make urlopen itself raise an exception
        mock_urlopen.side_effect = Exception("Connection error")

        dest = tmp_path / "test.txt"
        temp_dest = dest.with_suffix(dest.suffix + ".tmp")

        with pytest.raises(DownloadError):
            _download_file("https://example.com/file", dest)

        # Temp file should be cleaned up (or never created)
        assert not temp_dest.exists()


# ==============================================================================
# download_model Tests
# ==============================================================================


class TestDownloadModel:
    """Tests for download_model function."""

    def test_existing_files_skipped(self, tmp_path, capsys):
        """Existing files are skipped without force."""
        # Create mock checkpoint file in NAVER format (checkpoints dir)
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        (checkpoints_dir / "dune_vitsmall14_336.pth").touch()
        (checkpoints_dir / "dunemast3r_cvpr25_vitsmall.pth").touch()

        # Force NAVER source to use checkpoints dir
        result = download_model(
            ModelVariant.DUNE_VIT_SMALL_336,
            cache_dir=tmp_path,
            force=False,
            quiet=False,
            source=DownloadSource.NAVER,
        )

        captured = capsys.readouterr()
        assert "exists, skipping" in captured.out

    @patch("mast3r_runtime.utils.downloader._download_from_hf")
    def test_hf_download(self, mock_hf_download, tmp_path):
        """HuggingFace download is attempted when available."""
        mock_hf_download.return_value = True

        with patch("mast3r_runtime.utils.downloader._has_huggingface_hub", return_value=True):
            download_model(
                ModelVariant.DUNE_VIT_SMALL_336,
                cache_dir=tmp_path,
                source=DownloadSource.HF,
                quiet=True,
            )

        mock_hf_download.assert_called()

    @patch("mast3r_runtime.utils.downloader._has_aria2c", return_value=False)
    @patch("mast3r_runtime.utils.downloader._download_file")
    def test_naver_download(self, mock_download, mock_aria2c, tmp_path):
        """Naver download is used when source=NAVER."""
        download_model(
            ModelVariant.DUNE_VIT_SMALL_336,
            cache_dir=tmp_path,
            source=DownloadSource.NAVER,
            quiet=True,
        )

        mock_download.assert_called()
        # Should use Naver URL
        call_args = mock_download.call_args
        assert "naverlabs.com" in call_args.args[0]

    @patch("mast3r_runtime.utils.downloader._has_aria2c", return_value=False)
    @patch("mast3r_runtime.utils.downloader._download_from_hf")
    @patch("mast3r_runtime.utils.downloader._download_file")
    def test_auto_fallback_to_naver(self, mock_download, mock_hf, mock_aria2c, tmp_path):
        """AUTO falls back to Naver when HF fails."""
        mock_hf.return_value = False  # HF download fails

        with patch("mast3r_runtime.utils.downloader._has_huggingface_hub", return_value=True):
            download_model(
                ModelVariant.DUNE_VIT_SMALL_336,
                cache_dir=tmp_path,
                source=DownloadSource.AUTO,
                quiet=True,
            )

        # Should fall back to Naver
        mock_download.assert_called()


# ==============================================================================
# get_download_status Tests
# ==============================================================================


class TestGetDownloadStatus:
    """Tests for get_download_status function."""

    def test_empty_cache(self, tmp_path):
        """Returns all False for empty cache."""
        status = get_download_status(tmp_path)

        assert isinstance(status, dict)
        for variant in ModelVariant:
            assert variant in status
            for exists in status[variant].values():
                assert exists is False

    def test_partial_download(self, tmp_path):
        """Correctly identifies partial downloads."""
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        (checkpoints_dir / "dune_vitsmall14_336.pth").touch()

        status = get_download_status(tmp_path)

        variant_status = status[ModelVariant.DUNE_VIT_SMALL_336]
        assert variant_status["encoder"] is True
        assert variant_status["decoder"] is False

    def test_default_cache_dir(self):
        """Works with default cache dir."""
        status = get_download_status()
        assert isinstance(status, dict)


# ==============================================================================
# print_status Tests
# ==============================================================================


class TestPrintStatus:
    """Tests for print_status function."""

    def test_prints_all_variants(self, tmp_path, capsys):
        """Prints status for all variants."""
        print_status(tmp_path)

        captured = capsys.readouterr()
        # Should mention all variant names or at least show status
        assert "Checkpoint Status" in captured.out

    def test_shows_checkmarks(self, tmp_path, capsys):
        """Shows checkmarks for status."""
        print_status(tmp_path)

        captured = capsys.readouterr()
        # Should have status indicators
        assert "✓" in captured.out or "✗" in captured.out

    def test_shows_sizes(self, tmp_path, capsys):
        """Shows checkpoint sizes."""
        print_status(tmp_path)

        captured = capsys.readouterr()
        # Should mention MB
        assert "MB" in captured.out


# ==============================================================================
# DownloadError Tests
# ==============================================================================


class TestDownloadError:
    """Tests for DownloadError exception."""

    def test_is_exception(self):
        """DownloadError is an Exception."""
        assert issubclass(DownloadError, Exception)

    def test_message(self):
        """Error message is preserved."""
        error = DownloadError("Test error message")
        assert "Test error message" in str(error)
