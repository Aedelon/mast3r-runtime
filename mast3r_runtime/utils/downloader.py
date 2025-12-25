"""Checkpoint downloader for MASt3R runtime.

Downloads model checkpoints from:
- Naver CDN (official .pth files)
- HuggingFace (pre-converted .safetensors)

Uses aria2c if available for faster downloads.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable

from ..core.config import (
    MODEL_SPECS,
    DownloadSource,
    ModelVariant,
    get_checkpoint_paths,
    get_checkpoint_urls,
    get_total_checkpoint_size_mb,
)


class DownloadError(Exception):
    """Error during download."""


def _has_huggingface_hub() -> bool:
    """Check if huggingface_hub is available."""
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


def _download_from_hf(
    repo_id: str,
    filename: str,
    dest: Path,
    quiet: bool = False,
) -> bool:
    """Download a file from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "Aedelon/mast3r-vit-large-fp16").
        filename: File path in the repo.
        dest: Destination path.
        quiet: Suppress progress output.

    Returns:
        True if successful.
    """
    try:
        from huggingface_hub import hf_hub_download

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Download to HF cache, then copy/link to dest
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest.parent,
            local_dir_use_symlinks=False,
        )

        # hf_hub_download with local_dir puts file at dest.parent/filename
        downloaded = dest.parent / filename
        if downloaded != dest and downloaded.exists():
            shutil.move(str(downloaded), str(dest))

        return dest.exists()

    except Exception as e:
        if not quiet:
            print(f"    HF download failed: {e}")
        return False


def _get_cache_dir() -> Path:
    """Get default cache directory."""
    return Path.home() / ".cache" / "mast3r_runtime"


def _has_aria2c() -> bool:
    """Check if aria2c is available."""
    return shutil.which("aria2c") is not None


def _download_with_aria2c(url: str, dest: Path) -> bool:
    """Download using aria2c.

    Args:
        url: URL to download.
        dest: Destination path.

    Returns:
        True if successful.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aria2c",
        "--dir", str(dest.parent),
        "--out", dest.name,
        "--continue=true",           # Resume downloads
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--console-log-level=warn",
        url,
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


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
    source: DownloadSource = DownloadSource.AUTO,
) -> dict[str, Path]:
    """Download checkpoints for a model variant.

    Args:
        variant: Model variant to download.
        cache_dir: Cache directory. Uses ~/.cache/mast3r_runtime if None.
        force: Force re-download even if files exist.
        quiet: Suppress progress output.
        source: Download source (auto, naver, hf).
            - auto: Try HF first (safetensors), fallback to Naver
            - naver: Official Naver CDN (.pth files)
            - hf: HuggingFace (.safetensors, pre-converted)

    Returns:
        Dict mapping checkpoint type to local path.

    Raises:
        DownloadError: If download fails.
    """
    if cache_dir is None:
        cache_dir = _get_cache_dir()

    spec = MODEL_SPECS[variant]
    use_hf = source == DownloadSource.HF or (
        source == DownloadSource.AUTO and _has_huggingface_hub()
    )

    # Determine paths based on source
    if use_hf:
        # HF downloads go to safetensors directory
        output_dir = cache_dir / "safetensors" / variant.value
        paths = {
            ckpt_type: output_dir / f"{ckpt_type}.safetensors"
            if not info["hf_filename"].endswith(".pkl")
            else output_dir / f"{ckpt_type}.pkl"
            for ckpt_type, info in spec["checkpoints"].items()
        }
    else:
        # Naver downloads go to checkpoints directory
        paths = get_checkpoint_paths(variant, cache_dir)

    if not quiet:
        total_mb = get_total_checkpoint_size_mb(variant)
        if use_hf:
            src_name = f"HuggingFace ({spec['hf_repo']})"
        elif _has_aria2c():
            src_name = "Naver CDN [aria2c]"
        else:
            src_name = "Naver CDN [urllib]"
        print(f"Downloading {variant.value} (~{total_mb} MB) [{src_name}]")

    for ckpt_type, ckpt_info in spec["checkpoints"].items():
        dest = paths[ckpt_type]

        if dest.exists() and not force:
            if not quiet:
                print(f"  {ckpt_type}: {dest.name} (exists, skipping)")
            continue

        if not quiet:
            print(f"  {ckpt_type}: {ckpt_info['size_mb']} MB")

        if use_hf:
            # Download from HuggingFace
            success = _download_from_hf(
                repo_id=spec["hf_repo"],
                filename=ckpt_info["hf_filename"],
                dest=dest,
                quiet=quiet,
            )
            if not success:
                if source == DownloadSource.AUTO:
                    # Fallback to Naver
                    if not quiet:
                        print(f"    Falling back to Naver CDN...")
                    use_hf = False
                else:
                    raise DownloadError(f"HF download failed for {ckpt_info['hf_filename']}")

        if not use_hf:
            # Download from Naver CDN
            url = ckpt_info["url"]
            naver_dest = get_checkpoint_paths(variant, cache_dir)[ckpt_type]

            if _has_aria2c():
                success = _download_with_aria2c(url, naver_dest)
                if not success:
                    raise DownloadError(f"aria2c failed for {url}")
            else:
                if not quiet:
                    def progress(downloaded: int, total: int) -> None:
                        _print_progress(downloaded, total, prefix="    ")
                    _download_file(url, naver_dest, progress_callback=progress)
                else:
                    _download_file(url, naver_dest)

            # Update paths to point to Naver download location
            paths[ckpt_type] = naver_dest

    if not quiet:
        print(f"Done: {variant.value}")

    return paths


def download_all_models(
    cache_dir: Path | None = None,
    force: bool = False,
    quiet: bool = False,
    source: DownloadSource = DownloadSource.AUTO,
) -> dict[ModelVariant, dict[str, Path]]:
    """Download all model checkpoints.

    Args:
        cache_dir: Cache directory. Uses ~/.cache/mast3r_runtime if None.
        force: Force re-download even if files exist.
        quiet: Suppress progress output.
        source: Download source (auto, naver, hf).

    Returns:
        Dict mapping variant to checkpoint paths.
    """
    results = {}

    for variant in ModelVariant:
        results[variant] = download_model(
            variant, cache_dir=cache_dir, force=force, quiet=quiet, source=source
        )
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
