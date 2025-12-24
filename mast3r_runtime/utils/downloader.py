"""Checkpoint downloader for MASt3R runtime.

Downloads model checkpoints from Naver CDN.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Callable

from ..core.config import (
    MODEL_SPECS,
    ModelVariant,
    get_checkpoint_paths,
    get_checkpoint_urls,
    get_total_checkpoint_size_mb,
)


class DownloadError(Exception):
    """Error during download."""


def _get_cache_dir() -> Path:
    """Get default cache directory."""
    return Path.home() / ".cache" / "mast3r_runtime"


def _format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _download_file(
    url: str,
    dest: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Download a file with optional progress callback.

    Args:
        url: URL to download.
        dest: Destination path.
        progress_callback: Optional callback(downloaded_bytes, total_bytes).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192

            with open(temp_dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

        # Move temp file to final destination
        shutil.move(temp_dest, dest)

    except Exception as e:
        # Cleanup temp file on error
        if temp_dest.exists():
            temp_dest.unlink()
        raise DownloadError(f"Failed to download {url}: {e}") from e


def _print_progress(downloaded: int, total: int, prefix: str = "") -> None:
    """Print progress bar to console."""
    if total == 0:
        return

    percent = downloaded / total * 100
    bar_len = 40
    filled = int(bar_len * downloaded / total)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(
        f"\r{prefix}[{bar}] {percent:5.1f}% ({_format_size(downloaded)}/{_format_size(total)})",
        end="",
        flush=True,
    )

    if downloaded >= total:
        print()  # Newline when done


def download_model(
    variant: ModelVariant,
    cache_dir: Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> dict[str, Path]:
    """Download checkpoints for a model variant.

    Args:
        variant: Model variant to download.
        cache_dir: Cache directory. Uses ~/.cache/mast3r_runtime if None.
        force: Force re-download even if files exist.
        quiet: Suppress progress output.

    Returns:
        Dict mapping checkpoint type to local path.

    Raises:
        DownloadError: If download fails.
    """
    if cache_dir is None:
        cache_dir = _get_cache_dir()

    paths = get_checkpoint_paths(variant, cache_dir)
    urls = get_checkpoint_urls(variant)
    spec = MODEL_SPECS[variant]

    if not quiet:
        total_mb = get_total_checkpoint_size_mb(variant)
        print(f"Downloading {variant.value} (~{total_mb} MB)")

    for ckpt_type, url in urls.items():
        dest = paths[ckpt_type]
        ckpt_info = spec["checkpoints"][ckpt_type]

        if dest.exists() and not force:
            if not quiet:
                print(f"  {ckpt_type}: {dest.name} (exists, skipping)")
            continue

        if not quiet:
            print(f"  {ckpt_type}: {ckpt_info['filename']}")

            def progress(downloaded: int, total: int) -> None:
                _print_progress(downloaded, total, prefix="    ")

            _download_file(url, dest, progress_callback=progress)
        else:
            _download_file(url, dest)

    if not quiet:
        print(f"Done: {variant.value}")

    return paths


def download_all_models(
    cache_dir: Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> dict[ModelVariant, dict[str, Path]]:
    """Download all model checkpoints.

    Args:
        cache_dir: Cache directory. Uses ~/.cache/mast3r_runtime if None.
        force: Force re-download even if files exist.
        quiet: Suppress progress output.

    Returns:
        Dict mapping variant to checkpoint paths.
    """
    results = {}

    for variant in ModelVariant:
        results[variant] = download_model(variant, cache_dir=cache_dir, force=force, quiet=quiet)
        if not quiet:
            print()

    return results


def get_download_status(
    cache_dir: Path | None = None,
) -> dict[ModelVariant, dict[str, bool]]:
    """Check which checkpoints are downloaded.

    Args:
        cache_dir: Cache directory. Uses ~/.cache/mast3r_runtime if None.

    Returns:
        Dict mapping variant to dict of checkpoint type -> exists.
    """
    if cache_dir is None:
        cache_dir = _get_cache_dir()

    status = {}

    for variant in ModelVariant:
        paths = get_checkpoint_paths(variant, cache_dir)
        status[variant] = {ckpt_type: path.exists() for ckpt_type, path in paths.items()}

    return status


def print_status(cache_dir: Path | None = None) -> None:
    """Print download status for all models."""
    status = get_download_status(cache_dir)

    print("MASt3R Runtime - Checkpoint Status")
    print("=" * 50)

    for variant in ModelVariant:
        total_mb = get_total_checkpoint_size_mb(variant)
        ckpts = status[variant]
        all_present = all(ckpts.values())
        icon = "✓" if all_present else "✗"

        print(f"\n{icon} {variant.value} (~{total_mb} MB)")

        for ckpt_type, exists in ckpts.items():
            icon = "✓" if exists else "✗"
            print(f"  {icon} {ckpt_type}")
