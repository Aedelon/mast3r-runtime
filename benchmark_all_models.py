#!/usr/bin/env python3
"""Benchmark all available models on MPS backend.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

Usage:
    uv run python benchmark_all_models.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def benchmark_model(variant: str, resolution: int, model_path: Path | None) -> dict | None:
    """Benchmark a single model variant."""
    from mast3r_runtime.backends import _mps

    if model_path is None or not model_path.exists():
        return None

    print(f"\n{'─' * 50}")
    print(f"Model: {variant} @ {resolution}px")
    print(f"Path: {model_path}")
    print(f"{'─' * 50}")

    try:
        # Create engine
        engine = _mps.MPSEngine(
            variant=variant,
            resolution=resolution,
            precision="fp16",
            num_threads=4,
        )

        # Load weights
        t0 = time.perf_counter()
        engine.load(str(model_path))
        load_time = (time.perf_counter() - t0) * 1000
        print(f"Load time: {load_time:.0f} ms")

        # Create test images (HWC format for MPSGraph)
        img_w = int(resolution * 4 / 3)  # 4:3 aspect
        img1 = np.random.randint(0, 255, (resolution, img_w, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (resolution, img_w, 3), dtype=np.uint8)

        # Warmup
        print("Warmup (3 iterations)...")
        engine.warmup(3)

        # Benchmark
        print("Benchmarking (10 iterations)...")
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            result = engine.infer(img1, img2)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            print(f"  Iter {i + 1:2d}: {elapsed:6.1f} ms")

        avg = np.mean(times)
        std = np.std(times)
        min_t = np.min(times)
        max_t = np.max(times)

        print(f"\nResults:")
        print(f"  Average: {avg:.1f} ms ± {std:.1f} ms")
        print(f"  Min/Max: {min_t:.1f} / {max_t:.1f} ms")
        print(f"  FPS: {1000 / avg:.1f}")
        print(f"  Output: pts3d={result.pts3d_1.shape}, desc={result.desc_1.shape}")

        return {
            "variant": variant,
            "resolution": resolution,
            "load_ms": load_time,
            "avg_ms": avg,
            "std_ms": std,
            "min_ms": min_t,
            "max_ms": max_t,
            "fps": 1000 / avg,
            "output_shape": result.pts3d_1.shape,
        }

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("\n" + "=" * 60)
    print("  MASt3R Runtime - Full Model Benchmark (MPS)")
    print("=" * 60)

    # Check MPS availability
    try:
        from mast3r_runtime.backends import _mps
    except ImportError as e:
        print(f"MPS not available: {e}")
        return

    if not _mps.is_available():
        print("MPS requires macOS 15+")
        return

    # Show device info
    info = _mps.get_context_info()
    print(f"\nDevice: {info['device_name']}")
    print(f"Working Set: {format_size(info['recommended_working_set_size'])}")
    print(f"Max Buffer: {format_size(info['max_buffer_length'])}")

    # Define all model variants
    from mast3r_runtime.core.config import ModelVariant, Precision, get_default_model_path

    models = [
        # DUNE models
        ("dune_vit_small_336", 336, ModelVariant.DUNE_VIT_SMALL_336),
        ("dune_vit_small_448", 448, ModelVariant.DUNE_VIT_SMALL_448),
        ("dune_vit_base_336", 336, ModelVariant.DUNE_VIT_BASE_336),
        ("dune_vit_base_448", 448, ModelVariant.DUNE_VIT_BASE_448),
        # MASt3R model (native resolution 512px)
        ("mast3r_vit_large", 512, ModelVariant.MAST3R_VIT_LARGE),
    ]

    # Check which models are available
    print("\n" + "=" * 60)
    print("Checking available models...")
    print("=" * 60)

    available = []
    for variant_str, resolution, variant_enum in models:
        try:
            path = get_default_model_path(variant_enum, Precision.FP16)
            if path.exists():
                size = path.stat().st_size
                print(f"  ✓ {variant_str}: {format_size(size)}")
                available.append((variant_str, resolution, path))
            else:
                print(f"  ✗ {variant_str}: not found")
        except Exception as e:
            print(f"  ✗ {variant_str}: {e}")

    if not available:
        print("\nNo models found. Download with:")
        print("  uv run mast3r-runtime download dune_vit_small_336")
        print("  uv run mast3r-runtime download mast3r_vit_large")
        return

    # Benchmark each available model
    print("\n" + "=" * 60)
    print("Running benchmarks...")
    print("=" * 60)

    results = []
    for variant_str, resolution, path in available:
        result = benchmark_model(variant_str, resolution, path)
        if result:
            results.append(result)

    # Summary table
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\n{'Model':<25} {'Res':>5} {'Load':>8} {'Avg':>8} {'FPS':>6}")
        print("-" * 60)
        for r in results:
            print(
                f"{r['variant']:<25} {r['resolution']:>5} "
                f"{r['load_ms']:>7.0f}ms {r['avg_ms']:>7.1f}ms {r['fps']:>6.1f}"
            )
        print("-" * 60)

        # Find fastest
        fastest = min(results, key=lambda x: x["avg_ms"])
        print(f"\nFastest: {fastest['variant']} @ {fastest['fps']:.1f} FPS")

    # Show final buffer pool stats
    info = _mps.get_context_info()
    print(
        f"\nBuffer pool: {info['buffer_pool_count']} buffers, {format_size(info['buffer_pool_bytes'])}"
    )


if __name__ == "__main__":
    main()
