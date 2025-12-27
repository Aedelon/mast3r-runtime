#!/usr/bin/env python3
"""Benchmark MPS backend vs Python backend.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

Usage:
    uv run python benchmark_mpsgraph.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def benchmark_python_backend():
    """Benchmark pure Python/NumPy backend."""
    from safetensors.numpy import load_file as load_safetensors

    from mast3r_runtime.backends.python.mast3r_model import MASt3RModel
    from mast3r_runtime.core.config import ModelVariant, Precision, get_default_model_path

    print("=" * 60)
    print("Python Backend Benchmark")
    print("=" * 60)

    model_path = get_default_model_path(ModelVariant.MAST3R_VIT_LARGE, Precision.FP16)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: uv run mast3r-runtime download mast3r_vit_large")
        return None

    print(f"Loading model from: {model_path}")
    weights = load_safetensors(str(model_path))
    model = MASt3RModel(weights)
    print(f"Model loaded ({len(weights)} tensors)")

    # Test images (512x682 = 4:3 aspect) in BCHW format
    img1 = np.random.randint(0, 255, (1, 3, 512, 682), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (1, 3, 512, 682), dtype=np.uint8)

    # Warmup
    print("Warmup...")
    _ = model.forward(img1, img2)

    # Benchmark
    print("Benchmarking (5 iterations)...")
    times = []
    for i in range(5):
        start = time.perf_counter()
        result = model.forward(img1, img2)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Iter {i + 1}: {elapsed:.1f} ms")

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.1f} ms")
    print(f"Output: pts3d={result['pts3d_1'].shape}, desc={result['desc_1'].shape}")
    return avg


def benchmark_mps_backend():
    """Benchmark MPS backend (requires compiled _mps module)."""
    print("\n" + "=" * 60)
    print("MPS Backend Benchmark")
    print("=" * 60)

    try:
        from mast3r_runtime.backends import _mps
    except ImportError as e:
        print(f"MPS backend not available: {e}")
        print("Build with: uv pip install -e '.[dev]'")
        return None

    if not _mps.is_available():
        print("MPS not available (requires macOS 15+)")
        return None

    from mast3r_runtime.core.config import ModelVariant, Precision, get_default_model_path

    model_path = get_default_model_path(ModelVariant.MAST3R_VIT_LARGE, Precision.FP16)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    print(f"Creating MPS engine...")
    engine = _mps.MPSEngine(
        variant="mast3r_vit_large",
        resolution=512,
        precision="fp32",
        num_threads=4,
    )

    print(f"Loading weights from: {model_path}")
    engine.load(str(model_path))
    print(f"Engine: {engine.name()}")

    # Test images (512x682 = 4:3 aspect)
    img1 = np.random.randint(0, 255, (512, 682, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (512, 682, 3), dtype=np.uint8)

    # Warmup
    print("Warmup (3 iterations)...")
    engine.warmup(3)

    # Benchmark
    print("Benchmarking (10 iterations)...")
    times = []
    for i in range(10):
        start = time.perf_counter()
        result = engine.infer(img1, img2)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Iter {i + 1}: {elapsed:.1f} ms")

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.1f} ms")
    print(f"Timing breakdown:")
    print(f"  Preprocess: {result.timing.get('preprocess_ms', 0):.1f} ms")
    print(f"  Inference:  {result.timing.get('inference_ms', 0):.1f} ms")
    print(f"  Total:      {result.timing.get('total_ms', 0):.1f} ms")
    print(f"Output: pts3d_1={result.pts3d_1.shape}, desc_1={result.desc_1.shape}")
    return avg


def main():
    print("\n" + "=" * 60)
    print("  MASt3R Runtime Benchmark - MPS vs Python")
    print("=" * 60 + "\n")

    # Run benchmarks
    python_ms = benchmark_python_backend()
    mps_ms = benchmark_mps_backend()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if python_ms:
        print(f"Python backend: {python_ms:.1f} ms/pair")
    else:
        print("Python backend: Not tested")

    if mps_ms:
        print(f"MPS backend:    {mps_ms:.1f} ms/pair")
    else:
        print("MPS backend:    Not available")

    if python_ms and mps_ms:
        speedup = python_ms / mps_ms
        print(f"\nSpeedup: {speedup:.1f}x faster with MPS")


if __name__ == "__main__":
    main()
