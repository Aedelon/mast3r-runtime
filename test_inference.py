#!/usr/bin/env python3
"""Test MASt3R inference with Metal backend."""

import numpy as np
from pathlib import Path


def test_inference():
    """Run inference test with MASt3R model."""
    print("=" * 60)
    print("MASt3R Runtime - Inference Test")
    print("=" * 60)

    # Check Metal directly first
    print("\n[1] Checking Metal backend...")
    try:
        from mast3r_runtime.backends import _metal
        metal_available = _metal.is_available()
        print(f"Metal available: {metal_available}")
        if metal_available:
            print(f"Device: {_metal.get_device_name()}")
    except ImportError as e:
        print(f"Metal import error: {e}")
        metal_available = False

    if not metal_available:
        print("\n[ERROR] Metal backend not available!")
        return False

    # Load model
    print("\n[2] Loading MASt3R model...")
    from mast3r_runtime.core.config import (
        MASt3RRuntimeConfig,
        ModelConfig,
        RuntimeConfig,
        ModelVariant,
        Precision,
        BackendType,
    )
    from mast3r_runtime.backends import get_runtime

    config = MASt3RRuntimeConfig(
        model=ModelConfig(
            variant=ModelVariant.MAST3R_VIT_LARGE,
            resolution=512,
            precision=Precision.FP16,
        ),
        runtime=RuntimeConfig(
            backend=BackendType.METAL,
        ),
    )

    runtime = get_runtime(config)
    print(f"Runtime created: {type(runtime).__name__}")

    # Load weights
    print("\n[3] Loading weights...")
    runtime.load()
    print("Weights loaded successfully!")

    # Create test images
    print("\n[4] Creating test images...")
    H, W = 512, 512
    img1 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    print(f"Image shape: {img1.shape}")

    # Run inference
    print("\n[5] Running inference...")
    result = runtime.infer(img1, img2)

    # Check results
    print("\n[6] Results:")
    print(
        f"  pts3d_1 shape: {result.pts3d_1.shape}, range: [{result.pts3d_1.min():.3f}, {result.pts3d_1.max():.3f}]"
    )
    print(
        f"  pts3d_2 shape: {result.pts3d_2.shape}, range: [{result.pts3d_2.min():.3f}, {result.pts3d_2.max():.3f}]"
    )
    print(
        f"  desc_1 shape: {result.desc_1.shape}, range: [{result.desc_1.min():.3f}, {result.desc_1.max():.3f}]"
    )
    print(
        f"  desc_2 shape: {result.desc_2.shape}, range: [{result.desc_2.min():.3f}, {result.desc_2.max():.3f}]"
    )
    print(
        f"  conf_1 shape: {result.conf_1.shape}, range: [{result.conf_1.min():.3f}, {result.conf_1.max():.3f}]"
    )
    print(
        f"  conf_2 shape: {result.conf_2.shape}, range: [{result.conf_2.min():.3f}, {result.conf_2.max():.3f}]"
    )

    # Validate outputs
    print("\n[7] Validation:")

    # Check for zeros/NaN
    pts3d_ok = not (np.all(result.pts3d_1 == 0) or np.isnan(result.pts3d_1).any())
    desc_ok = not (np.all(result.desc_1 == 0) or np.isnan(result.desc_1).any())
    conf_ok = not (np.all(result.conf_1 == 0.5) or np.isnan(result.conf_1).any())

    print(f"  pts3d valid (not all zeros/NaN): {'✓' if pts3d_ok else '✗'}")
    print(f"  desc valid (not all zeros/NaN): {'✓' if desc_ok else '✗'}")
    print(f"  conf valid (not all 0.5/NaN): {'✓' if conf_ok else '✗'}")

    # Timing
    print(f"\n[8] Timing:")
    for key, val in result.timing_ms.items():
        print(f"  {key}: {val:.1f} ms")

    success = pts3d_ok and desc_ok and conf_ok
    print("\n" + "=" * 60)
    print(f"Test {'PASSED ✓' if success else 'FAILED ✗'}")
    print("=" * 60)

    return success


if __name__ == "__main__":
    import sys

    success = test_inference()
    sys.exit(0 if success else 1)
