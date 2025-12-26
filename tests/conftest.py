"""Pytest configuration and fixtures for mast3r_runtime tests.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ==============================================================================
# Skip conditions
# ==============================================================================


def has_metal_backend() -> bool:
    """Check if Metal backend is available."""
    if platform.system() != "Darwin":
        return False
    try:
        from mast3r_runtime import _metal

        return _metal.is_available()
    except ImportError:
        return False


def has_cuda_backend() -> bool:
    """Check if CUDA backend is available."""
    try:
        from mast3r_runtime import _cuda

        return _cuda.is_available()
    except ImportError:
        return False


def has_cpu_backend() -> bool:
    """Check if CPU backend is available."""
    try:
        from mast3r_runtime import _cpu

        return _cpu.is_available()
    except ImportError:
        return False


def has_torch() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def has_safetensors() -> bool:
    """Check if safetensors is available."""
    try:
        import safetensors  # noqa: F401

        return True
    except ImportError:
        return False


def has_huggingface_hub() -> bool:
    """Check if huggingface_hub is available."""
    try:
        import huggingface_hub  # noqa: F401

        return True
    except ImportError:
        return False


# Skip markers
skip_no_metal = pytest.mark.skipif(not has_metal_backend(), reason="Metal backend not available")
skip_no_cuda = pytest.mark.skipif(not has_cuda_backend(), reason="CUDA backend not available")
skip_no_cpu = pytest.mark.skipif(not has_cpu_backend(), reason="CPU backend not available")
skip_no_torch = pytest.mark.skipif(not has_torch(), reason="PyTorch not installed")
skip_no_safetensors = pytest.mark.skipif(not has_safetensors(), reason="safetensors not installed")
skip_no_hf_hub = pytest.mark.skipif(
    not has_huggingface_hub(), reason="huggingface_hub not installed"
)
skip_not_macos = pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only test")
skip_not_linux = pytest.mark.skipif(platform.system() != "Linux", reason="Linux only test")


# ==============================================================================
# Fixtures - Images
# ==============================================================================


@pytest.fixture
def random_rgb_image() -> NDArray[np.uint8]:
    """Generate a random 336x336 RGB image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (336, 336, 3), dtype=np.uint8)


@pytest.fixture
def random_rgb_image_pair(random_rgb_image) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Generate a pair of random RGB images."""
    rng = np.random.default_rng(43)
    img2 = rng.integers(0, 255, (336, 336, 3), dtype=np.uint8)
    return random_rgb_image, img2


@pytest.fixture
def small_rgb_image() -> NDArray[np.uint8]:
    """Generate a small 224x224 RGB image for fast tests."""
    rng = np.random.default_rng(44)
    return rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def gradient_image() -> NDArray[np.uint8]:
    """Generate a gradient test image."""
    h, w = 336, 336
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = x[np.newaxis, :]  # Red horizontal gradient
    img[:, :, 1] = y[:, np.newaxis]  # Green vertical gradient
    img[:, :, 2] = 128  # Blue constant
    return img


# ==============================================================================
# Fixtures - Descriptors
# ==============================================================================


@pytest.fixture
def random_descriptors() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate random descriptor maps for matching tests."""
    rng = np.random.default_rng(45)
    H, W, D = 24, 24, 256  # Typical resolution for 336x336 with patch_size=14
    desc_1 = rng.random((H, W, D), dtype=np.float32)
    desc_2 = rng.random((H, W, D), dtype=np.float32)
    return desc_1, desc_2


@pytest.fixture
def matching_descriptors() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate descriptor maps with known matches for testing."""
    rng = np.random.default_rng(46)
    H, W, D = 24, 24, 256
    desc_1 = rng.random((H, W, D), dtype=np.float32)
    # Create desc_2 with some exact matches (shifted by 1)
    desc_2 = np.roll(desc_1, 1, axis=0)
    return desc_1, desc_2


@pytest.fixture
def confidence_maps() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate confidence maps for matching tests."""
    rng = np.random.default_rng(47)
    H, W = 24, 24
    conf_1 = rng.random((H, W), dtype=np.float32)
    conf_2 = rng.random((H, W), dtype=np.float32)
    return conf_1, conf_2


@pytest.fixture
def pts3d_maps() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate 3D point maps for matching tests."""
    rng = np.random.default_rng(48)
    H, W = 24, 24
    pts3d_1 = rng.random((H, W, 3), dtype=np.float32) * 10  # Scale to ~10m range
    pts3d_2 = rng.random((H, W, 3), dtype=np.float32) * 10
    return pts3d_1, pts3d_2


# ==============================================================================
# Fixtures - Configuration
# ==============================================================================


@pytest.fixture
def default_config():
    """Create default MASt3RRuntimeConfig."""
    from mast3r_runtime import MASt3RRuntimeConfig

    return MASt3RRuntimeConfig()


@pytest.fixture
def drone_fast_config():
    """Create drone fast preset config."""
    from mast3r_runtime import PRESET_DRONE_FAST

    return PRESET_DRONE_FAST


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "mast3r_cache"
    cache_dir.mkdir()
    return cache_dir


# ==============================================================================
# Fixtures - Mocks
# ==============================================================================


@pytest.fixture
def mock_checkpoint_file(tmp_path: Path) -> Path:
    """Create a mock checkpoint file."""
    ckpt_path = tmp_path / "mock_checkpoint.pth"
    # Create minimal pickle file (empty dict)
    import pickle

    with open(ckpt_path, "wb") as f:
        pickle.dump({"model": {}}, f)
    return ckpt_path


@pytest.fixture
def mock_safetensors_file(tmp_path: Path) -> Path:
    """Create a mock safetensors file."""
    if not has_safetensors():
        pytest.skip("safetensors not installed")

    from safetensors.numpy import save_file

    st_path = tmp_path / "mock.safetensors"
    tensors = {"weight": np.zeros((10, 10), dtype=np.float16)}
    save_file(tensors, st_path)
    return st_path
