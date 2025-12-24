"""Download MASt3R/DUNE model weights.

Downloads pre-exported ONNX models from Hugging Face Hub.

Usage:
    mast3r-download                     # Download default (dune_vit_small_14)
    mast3r-download dune_vit_base_14    # Download specific variant
    mast3r-download --list              # List available models
    mast3r-download --all               # Download all models

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..core.config import MASt3RRuntimeConfig, ModelVariant

# Hugging Face Hub repository for ONNX models
HF_REPO_ID = "aedelon/mast3r-runtime-onnx"

# Model files to download for each variant
MODEL_FILES = {
    ModelVariant.DUNE_VIT_SMALL_14: {
        "onnx": "dune_vit_small_14_336_fp16.onnx",
        "size_mb": 220,
    },
    ModelVariant.DUNE_VIT_BASE_14: {
        "onnx": "dune_vit_base_14_336_fp16.onnx",
        "size_mb": 840,
    },
    ModelVariant.MAST3R_VIT_LARGE: {
        "onnx": "mast3r_vit_large_512_fp16.onnx",
        "size_mb": 2400,
    },
    ModelVariant.DUST3R_224_LINEAR: {
        "onnx": "dust3r_224_linear_fp16.onnx",
        "size_mb": 1600,
    },
}


def get_cache_dir() -> Path:
    """Get the cache directory for models."""
    config = MASt3RRuntimeConfig()
    return config.cache_dir / "onnx"


def list_models() -> None:
    """Print available models and their status."""
    cache_dir = get_cache_dir()

    print("\n📦 Available MASt3R Models")
    print("=" * 60)
    print(f"{'Variant':<25} {'Size':<10} {'Status':<15}")
    print("-" * 60)

    for variant in ModelVariant:
        info = MODEL_FILES.get(variant, {})
        size = info.get("size_mb", "?")
        filename = info.get("onnx", "N/A")

        # Check if downloaded
        model_path = cache_dir / filename
        status = "✅ Downloaded" if model_path.exists() else "❌ Not found"

        print(f"{variant.value:<25} {size:>6} MB  {status:<15}")

    print("-" * 60)
    print(f"Cache directory: {cache_dir}")
    print()


def download_model(variant: ModelVariant, force: bool = False) -> Path:
    """Download a model from Hugging Face Hub.

    Args:
        variant: Model variant to download
        force: Force re-download even if exists

    Returns:
        Path to downloaded model
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        msg = "huggingface-hub is required. Install with: pip install huggingface-hub"
        raise ImportError(msg) from e

    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    info = MODEL_FILES.get(variant)
    if info is None:
        msg = f"No ONNX model available for {variant.value}"
        raise ValueError(msg)

    filename = info["onnx"]
    size_mb = info["size_mb"]
    dest_path = cache_dir / filename

    # Check if already exists
    if dest_path.exists() and not force:
        print(f"✅ Model already exists: {dest_path}")
        return dest_path

    print(f"\n📥 Downloading {variant.value} ({size_mb} MB)...")
    print(f"   From: huggingface.co/{HF_REPO_ID}")
    print(f"   To: {dest_path}")
    print()

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
        print(f"\n✅ Downloaded: {downloaded_path}")
        return Path(downloaded_path)

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print()
        print("Alternative: Export manually from PyTorch checkpoint")
        print("  1. Install MASt3R: pip install git+https://github.com/naver/mast3r")
        print("  2. Export: mast3r-export --variant", variant.value)
        raise


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download MASt3R/DUNE ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mast3r-download                      # Download default model
  mast3r-download dune_vit_base_14     # Download specific variant
  mast3r-download --list               # List available models
  mast3r-download --all                # Download all models

Note:
  Models are licensed under CC BY-NC-SA 4.0 by Naver Corporation.
  See: https://github.com/naver/mast3r
""",
    )

    parser.add_argument(
        "variant",
        nargs="?",
        default="dune_vit_small_14",
        choices=[v.value for v in ModelVariant],
        help="Model variant to download (default: dune_vit_small_14)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Download all available models",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if model exists",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Custom cache directory",
    )

    args = parser.parse_args()

    # List models
    if args.list:
        list_models()
        return 0

    # Download all
    if args.all:
        print("\n📦 Downloading all models...")
        for variant in ModelVariant:
            try:
                download_model(variant, force=args.force)
            except Exception as e:
                print(f"⚠️  Failed to download {variant.value}: {e}")
        return 0

    # Download specific variant
    try:
        variant = ModelVariant(args.variant)
        download_model(variant, force=args.force)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
