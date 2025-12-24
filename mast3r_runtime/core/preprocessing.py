"""Image preprocessing interface for MASt3R runtime.

Each backend implements its own optimized preprocessing:
- ONNX: numpy (CPU) or integrated in model
- CoreML: Metal shaders
- TensorRT: CUDA kernels
- PyTorch: torch tensors (GPU)

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ImageNet normalization (used by DINOv2/DUNE)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PreprocessorBase(ABC):
    """Abstract base class for backend-specific preprocessors."""

    def __init__(
        self,
        resolution: int = 336,
        patch_size: int = 14,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        if resolution % patch_size != 0:
            msg = f"Resolution {resolution} must be divisible by patch_size {patch_size}"
            raise ValueError(msg)

        self.resolution = resolution
        self.patch_size = patch_size
        self.mean = mean
        self.std = std

    @abstractmethod
    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Preprocess image for inference.

        Args:
            image: Input image [H, W, 3] RGB uint8

        Returns:
            Preprocessed array [1, 3, resolution, resolution] float32
        """
        ...


class NumpyPreprocessor(PreprocessorBase):
    """CPU-based numpy preprocessor (fallback/debug).

    Use for development or when GPU preprocessing is unavailable.
    Not recommended for production due to CPU bottleneck.
    """

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Preprocess image using numpy/PIL."""
        # Convert to PIL for resize
        if image.ndim != 3 or image.shape[2] != 3:
            msg = f"Expected [H, W, 3] array, got {image.shape}"
            raise ValueError(msg)

        pil_image = Image.fromarray(image)

        # Resize with aspect ratio preservation
        h, w = image.shape[:2]
        target = self.resolution
        scale = max(target / h, target / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        pil_image = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Center crop
        left = (new_w - target) // 2
        top = (new_h - target) // 2
        pil_image = pil_image.crop((left, top, left + target, top + target))

        # Convert to float32 [0, 1]
        arr = np.array(pil_image, dtype=np.float32) / 255.0

        # HWC -> CHW
        arr = arr.transpose(2, 0, 1)

        # Normalize with ImageNet stats
        mean = np.array(self.mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(self.std, dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / std

        # Add batch dimension
        return arr[np.newaxis, ...]


def prepare_image_numpy(
    image: NDArray[np.uint8],
    resolution: int = 336,
    patch_size: int = 14,
) -> NDArray[np.float32]:
    """Convenience function for numpy preprocessing.

    Args:
        image: Input image [H, W, 3] RGB uint8
        resolution: Target resolution
        patch_size: ViT patch size

    Returns:
        Preprocessed array [1, 3, resolution, resolution] float32
    """
    prep = NumpyPreprocessor(resolution=resolution, patch_size=patch_size)
    return prep(image)
