"""Backend dispatcher for MASt3R runtime.

Automatically selects the best available backend for the current platform:
- macOS arm64: Metal (Apple Silicon GPU)
- Linux x86_64: CUDA (NVIDIA GPU)
- Linux aarch64: Jetson (TensorRT + DLA)
- Fallback: CPU (C++ with OpenMP/BLAS) or Python (pure numpy)

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import BackendType, MASt3RRuntimeConfig
    from ..core.engine_interface import EngineInterface


def get_available_backends() -> dict[str, bool]:
    """Detect which compiled backends are available.

    Returns:
        Dict mapping backend name to availability status.
    """
    backends = {"python": True}  # Always available

    # Try importing compiled extensions
    try:
        from . import _cpu

        backends["cpu"] = _cpu.is_available()
    except ImportError:
        backends["cpu"] = False

    try:
        from . import _mps

        backends["metal"] = _mps.is_available()
    except ImportError:
        backends["metal"] = False

    try:
        from . import _cuda

        backends["cuda"] = _cuda.is_available()
    except ImportError:
        backends["cuda"] = False

    try:
        from . import _jetson

        backends["jetson"] = _jetson.is_available()
    except ImportError:
        backends["jetson"] = False

    return backends


def get_best_backend() -> str:
    """Get the best available backend for the current platform.

    Returns:
        Backend name string.
    """
    available = get_available_backends()
    system = platform.system()
    machine = platform.machine()

    # macOS arm64 → Metal
    if system == "Darwin" and machine == "arm64":
        if available.get("metal"):
            return "metal"

    # Linux x86_64 → CUDA
    if system == "Linux" and machine == "x86_64":
        if available.get("cuda"):
            return "cuda"

    # Linux aarch64 (Jetson) → Jetson > CUDA
    if system == "Linux" and machine == "aarch64":
        if available.get("jetson"):
            return "jetson"
        if available.get("cuda"):
            return "cuda"

    # Fallback → CPU → Python
    if available.get("cpu"):
        return "cpu"

    return "python"


def get_backend_info() -> dict:
    """Get detailed information about available backends.

    Returns:
        Dict with platform info and backend details.
    """
    available = get_available_backends()
    best = get_best_backend()

    info = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "backends": available,
        "best_backend": best,
        "details": {},
    }

    # Get backend-specific details
    if available.get("metal"):
        try:
            from . import _mps

            info["details"]["metal"] = {
                "device": _mps.get_device_name(),
                "unified_memory": _mps.has_unified_memory(),
            }
        except Exception:
            pass

    if available.get("cuda"):
        try:
            from . import _cuda

            info["details"]["cuda"] = {
                "device": _cuda.get_device_name(),
                "compute_capability": _cuda.get_compute_capability(),
            }
        except Exception:
            pass

    return info


def get_runtime(config: MASt3RRuntimeConfig) -> EngineInterface:
    """Get runtime engine based on config and platform.

    Args:
        config: Runtime configuration.

    Returns:
        EngineInterface instance for the selected backend.

    Raises:
        RuntimeError: If no suitable backend is available.
    """
    from ..core.config import BackendType

    backend = config.runtime.backend

    if backend == BackendType.AUTO:
        backend_name = get_best_backend()
    else:
        backend_name = backend.value

    # Import and return the appropriate engine
    if backend_name == "metal":
        try:
            from ._mps import MPSEngine

            return _create_native_engine(MPSEngine, config)
        except ImportError as e:
            raise RuntimeError(f"MPS backend not available: {e}") from e

    elif backend_name == "cuda":
        try:
            from ._cuda import CUDAEngine

            return _create_native_engine(CUDAEngine, config)
        except ImportError as e:
            raise RuntimeError(f"CUDA backend not available: {e}") from e

    elif backend_name == "jetson":
        try:
            from ._jetson import JetsonEngine

            return _create_native_engine(JetsonEngine, config)
        except ImportError as e:
            raise RuntimeError(f"Jetson backend not available: {e}") from e

    elif backend_name == "cpu":
        try:
            from ._cpu import CPUEngine

            return _create_native_engine(CPUEngine, config)
        except ImportError as e:
            raise RuntimeError(f"CPU backend not available: {e}") from e

    else:
        # Python fallback
        from .python.python_engine import PythonEngine

        return PythonEngine(config)


def _create_native_engine(engine_class, config: MASt3RRuntimeConfig):
    """Create a native engine from pybind11 class with Python wrapper."""
    from .native_wrapper import NativeEngineWrapper

    native = engine_class(
        variant=config.model.variant.value,
        resolution=config.model.resolution,
        precision=config.model.precision.value,
        num_threads=config.runtime.num_threads,
    )

    return NativeEngineWrapper(native, config)


__all__ = [
    "get_available_backends",
    "get_backend_info",
    "get_best_backend",
    "get_runtime",
]
