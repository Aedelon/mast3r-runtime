"""Display system and MASt3R runtime information.

Usage:
    mast3r-info              # Show system info
    mast3r-info --models     # Show downloaded models
    mast3r-info --backends   # Show available backends

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import argparse
import platform
import sys


def get_system_info() -> dict[str, str]:
    """Get system information."""
    info = {
        "Platform": platform.system(),
        "Architecture": platform.machine(),
        "Python": platform.python_version(),
    }

    # macOS specific
    if platform.system() == "Darwin":
        info["macOS"] = platform.mac_ver()[0]

    # Linux specific
    if platform.system() == "Linux":
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        info["Linux"] = line.split("=")[1].strip().strip('"')
                        break
        except FileNotFoundError:
            pass

    return info


def get_package_info() -> dict[str, str]:
    """Get mast3r-runtime package information."""
    from .. import __version__

    return {
        "mast3r-runtime": __version__,
    }


def get_dependency_info() -> dict[str, str]:
    """Get dependency versions."""
    deps = {}

    # Core dependencies
    try:
        import numpy

        deps["numpy"] = numpy.__version__
    except ImportError:
        deps["numpy"] = "❌ Not installed"

    try:
        import PIL

        deps["pillow"] = PIL.__version__
    except ImportError:
        deps["pillow"] = "❌ Not installed"

    try:
        import pydantic

        deps["pydantic"] = pydantic.__version__
    except ImportError:
        deps["pydantic"] = "❌ Not installed"

    # Optional: ONNX Runtime
    try:
        import onnxruntime

        deps["onnxruntime"] = onnxruntime.__version__
        deps["onnx_providers"] = ", ".join(onnxruntime.get_available_providers())
    except ImportError:
        deps["onnxruntime"] = "❌ Not installed"

    # Optional: PyTorch
    try:
        import torch

        deps["torch"] = torch.__version__
        if torch.cuda.is_available():
            deps["cuda"] = torch.version.cuda or "Available"
        if torch.backends.mps.is_available():
            deps["mps"] = "Available"
    except ImportError:
        deps["torch"] = "Not installed (optional)"

    # Optional: CoreML Tools
    try:
        import coremltools

        deps["coremltools"] = coremltools.__version__
    except ImportError:
        if platform.system() == "Darwin":
            deps["coremltools"] = "Not installed (optional)"

    return deps


def get_backend_info() -> dict[str, str]:
    """Get available backend information."""
    from ..backends import (
        get_available_backends,
        is_apple_silicon,
        is_coreml_available,
        is_cuda_available,
        is_jetson,
    )

    info = {
        "Apple Silicon": "✅ Yes" if is_apple_silicon() else "❌ No",
        "NVIDIA Jetson": "✅ Yes" if is_jetson() else "❌ No",
        "CUDA Available": "✅ Yes" if is_cuda_available() else "❌ No",
        "CoreML Available": "✅ Yes" if is_coreml_available() else "❌ No",
    }

    backends = get_available_backends()
    info["Available Backends"] = ", ".join(b.value for b in backends)

    return info


def get_model_info() -> dict[str, str]:
    """Get information about downloaded models."""
    from ..core.config import MASt3RRuntimeConfig

    config = MASt3RRuntimeConfig()
    cache_dir = config.cache_dir / "onnx"

    info = {"Cache Directory": str(cache_dir)}

    if not cache_dir.exists():
        info["Models"] = "No models downloaded"
        return info

    # List downloaded models
    onnx_files = list(cache_dir.glob("*.onnx"))
    if onnx_files:
        for f in onnx_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            info[f.stem] = f"{size_mb:.1f} MB"
    else:
        info["Models"] = "No ONNX models found"

    return info


def print_section(title: str, data: dict[str, str]) -> None:
    """Print a section with title and key-value pairs."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    for key, value in data.items():
        print(f"  {key:<22} {value}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Display MASt3R runtime system information",
    )
    parser.add_argument(
        "--models",
        "-m",
        action="store_true",
        help="Show downloaded models",
    )
    parser.add_argument(
        "--backends",
        "-b",
        action="store_true",
        help="Show available backends",
    )
    parser.add_argument(
        "--deps",
        "-d",
        action="store_true",
        help="Show dependency versions",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Show all information",
    )

    args = parser.parse_args()

    # If no specific flag, show all
    show_all = args.all or not (args.models or args.backends or args.deps)

    print("\n🤖 MASt3R Runtime Information")
    print("=" * 50)

    if show_all or args.deps:
        print_section("📦 Package", get_package_info())
        print_section("💻 System", get_system_info())
        print_section("📚 Dependencies", get_dependency_info())

    if show_all or args.backends:
        print_section("⚙️  Backends", get_backend_info())

    if show_all or args.models:
        print_section("🗂️  Models", get_model_info())

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
