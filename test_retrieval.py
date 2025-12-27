#!/usr/bin/env python3
"""Test retrieval API for MPS backend.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

Usage:
    uv run python test_retrieval.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def test_retrieval():
    """Test the retrieval API."""
    print("=" * 60)
    print("MPS Backend - Retrieval Test")
    print("=" * 60)

    # Check MPS availability
    try:
        from mast3r_runtime.backends import _mps
    except ImportError as e:
        print(f"MPS backend not available: {e}")
        return

    if not _mps.is_available():
        print("MPS requires macOS 15+")
        return

    # Check model paths
    from mast3r_runtime.core.config import ModelVariant, Precision, get_default_model_path

    model_path = get_default_model_path(ModelVariant.MAST3R_VIT_LARGE, Precision.FP16)
    retrieval_path = model_path.parent / "retrieval.safetensors"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: uv run mast3r-runtime download mast3r_vit_large")
        return

    if not retrieval_path.exists():
        print(f"Retrieval weights not found: {retrieval_path}")
        print("Please ensure retrieval.safetensors is in the model directory")
        return

    print(f"Model: {model_path}")
    print(f"Retrieval: {retrieval_path}")

    # Create engine
    print("\nCreating MPS engine...")
    engine = _mps.MPSEngine(
        variant="mast3r_vit_large",
        resolution=512,
        precision="fp16",
        num_threads=4,
    )

    # Load main model
    print("Loading main model...")
    t0 = time.perf_counter()
    engine.load(str(model_path))
    print(f"Model loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")

    # Load retrieval weights
    print("Loading retrieval weights...")
    t0 = time.perf_counter()
    engine.load_retrieval(str(retrieval_path))
    print(f"Retrieval loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(f"Retrieval ready: {engine.is_retrieval_ready()}")

    # Create test image (512x682 = 4:3 aspect, HWC format)
    img = np.random.randint(0, 255, (512, 682, 3), dtype=np.uint8)

    # Warmup
    print("\nWarmup...")
    for _ in range(3):
        _ = engine.encode_retrieval(img)

    # Benchmark
    print("Benchmarking (10 iterations)...")
    times = []
    for i in range(10):
        t0 = time.perf_counter()
        result = engine.encode_retrieval(img)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Iter {i + 1:2d}: {elapsed:6.1f} ms")

    avg = np.mean(times)
    std = np.std(times)
    print(f"\nAverage: {avg:.1f} ms ± {std:.1f} ms")

    # Check output
    print("\nOutput:")
    print(f"  features: {result.features.shape}")
    print(f"  attention: {result.attention.shape}")
    print(f"  attention sum: {result.attention.sum():.6f}")  # Should be ~1.0
    print(f"  features norm (mean): {np.linalg.norm(result.features, axis=1).mean():.4f}")

    # Timing breakdown
    print("\nTiming:")
    print(f"  Preprocess: {result.timing.get('preprocess_ms', 0):.1f} ms")
    print(f"  Encoder:    {result.timing.get('encoder_ms', 0):.1f} ms")
    print(f"  Whitening:  {result.timing.get('whiten_ms', 0):.1f} ms")
    print(f"  Total:      {result.timing.get('total_ms', 0):.1f} ms")

    # Top-k attention (select most salient patches)
    top_k = 300
    top_indices = np.argsort(result.attention)[-top_k:][::-1]
    print(f"\nTop-{top_k} attention patches:")
    print(f"  Indices: {top_indices[:10]}... (showing first 10)")
    print(f"  Attention values: {result.attention[top_indices[:10]]}")

    print("\nRetrieval test passed!")


if __name__ == "__main__":
    test_retrieval()
