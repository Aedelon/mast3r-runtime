#!/usr/bin/env python3
"""Benchmark FP16 vs FP32 compute on MPS backend.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

Usage:
    uv run python benchmark_fp16.py
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


def benchmark_precision(
    variant: str, resolution: int, model_path: Path, precision: str, iterations: int = 10
) -> dict:
    """Benchmark a single precision."""
    from mast3r_runtime.backends import _mps

    print(f"\n  Testing {precision.upper()}...")

    # Create engine
    engine = _mps.MPSEngine(
        variant=variant,
        resolution=resolution,
        precision=precision,
        num_threads=4,
    )

    # Load weights
    t0 = time.perf_counter()
    engine.load(str(model_path))
    load_time = (time.perf_counter() - t0) * 1000

    # Create test images (HWC format for MPSGraph)
    img_w = int(resolution * 4 / 3)  # 4:3 aspect
    img1 = np.random.randint(0, 255, (resolution, img_w, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (resolution, img_w, 3), dtype=np.uint8)

    # Warmup
    engine.warmup(3)

    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = engine.infer(img1, img2)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    avg = np.mean(times)
    std = np.std(times)
    min_t = np.min(times)

    print(
        f"    Load: {load_time:.0f}ms, Inference: {avg:.1f}ms +/- {std:.1f}ms (min: {min_t:.1f}ms)"
    )

    return {
        "precision": precision,
        "load_ms": load_time,
        "avg_ms": avg,
        "std_ms": std,
        "min_ms": min_t,
        "fps": 1000 / avg,
        "pts3d_sample": result.pts3d_1[0, 0, :3],  # First pixel XYZ
        "desc_sample": result.desc_1[0, 0, :3],  # First pixel descriptor
    }


def benchmark_model(variant: str, resolution: int, model_path: Path) -> dict | None:
    """Benchmark FP16 vs FP32 for a single model."""
    print(f"\n{'─' * 60}")
    print(f"Model: {variant} @ {resolution}px")
    print(f"{'─' * 60}")

    results = []
    for precision in ["fp32", "fp16"]:
        try:
            result = benchmark_precision(variant, resolution, model_path, precision, iterations=10)
            results.append(result)
        except Exception as e:
            print(f"    Error with {precision}: {e}")
            return None

    if len(results) != 2:
        return None

    fp32, fp16 = results
    speedup = fp32["avg_ms"] / fp16["avg_ms"]

    print(f"\n  {'Metric':<16} {'FP32':>10} {'FP16':>10} {'Speedup':>8}")
    print(f"  {'-' * 50}")
    print(f"  {'Load (ms)':<16} {fp32['load_ms']:>10.0f} {fp16['load_ms']:>10.0f}")
    print(f"  {'Inference (ms)':<16} {fp32['avg_ms']:>10.1f} {fp16['avg_ms']:>10.1f} {speedup:>7.2f}x")
    print(f"  {'FPS':<16} {fp32['fps']:>10.1f} {fp16['fps']:>10.1f}")

    # Numerical accuracy
    pts_diff = np.max(np.abs(fp32["pts3d_sample"] - fp16["pts3d_sample"]))
    desc_diff = np.max(np.abs(fp32["desc_sample"] - fp16["desc_sample"]))
    print(f"  {'pts3d max diff':<16} {pts_diff:>10.4f}")
    print(f"  {'desc max diff':<16} {desc_diff:>10.6f}")

    return {
        "variant": variant,
        "resolution": resolution,
        "fp32_ms": fp32["avg_ms"],
        "fp16_ms": fp16["avg_ms"],
        "speedup": speedup,
        "fp32_load": fp32["load_ms"],
        "fp16_load": fp16["load_ms"],
        "pts_diff": pts_diff,
        "desc_diff": desc_diff,
    }


def main():
    print("\n" + "=" * 60)
    print("  MASt3R Runtime - FP16 vs FP32 Benchmark (All Models)")
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

    from mast3r_runtime.core.config import ModelVariant, Precision, get_default_model_path

    # All model variants
    models = [
        ("dune_vit_small_336", 336, ModelVariant.DUNE_VIT_SMALL_336),
        ("dune_vit_small_448", 448, ModelVariant.DUNE_VIT_SMALL_448),
        ("dune_vit_base_336", 336, ModelVariant.DUNE_VIT_BASE_336),
        ("dune_vit_base_448", 448, ModelVariant.DUNE_VIT_BASE_448),
        ("mast3r_vit_large", 512, ModelVariant.MAST3R_VIT_LARGE),
    ]

    # Find available models
    available = []
    print("\nChecking available models...")
    for variant_str, resolution, variant_enum in models:
        try:
            path = get_default_model_path(variant_enum, Precision.FP16)
            if path.exists():
                print(f"  ✓ {variant_str}")
                available.append((variant_str, resolution, path))
            else:
                print(f"  ✗ {variant_str}: not found")
        except Exception as e:
            print(f"  ✗ {variant_str}: {e}")

    if not available:
        print("\nNo models found. Download with:")
        print("  uv run mast3r-runtime download dune_vit_small_336")
        return

    # Benchmark each model
    all_results = []
    for variant_str, resolution, path in available:
        result = benchmark_model(variant_str, resolution, path)
        if result:
            all_results.append(result)

    # Summary table
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY - FP16 vs FP32")
        print("=" * 60)
        print(f"\n{'Model':<22} {'Res':>4} {'FP32':>8} {'FP16':>8} {'Speed':>7} {'pts diff':>9} {'desc diff':>10}")
        print("-" * 75)
        for r in all_results:
            print(
                f"{r['variant']:<22} {r['resolution']:>4} "
                f"{r['fp32_ms']:>7.1f}ms {r['fp16_ms']:>7.1f}ms "
                f"{r['speedup']:>6.2f}x {r['pts_diff']:>9.3f} {r['desc_diff']:>10.6f}"
            )
        print("-" * 75)

        # Average speedup
        avg_speedup = np.mean([r["speedup"] for r in all_results])
        avg_load_speedup = np.mean([r["fp32_load"] / r["fp16_load"] for r in all_results])
        print(f"\nAverage inference speedup: {avg_speedup:.2f}x")
        print(f"Average load speedup: {avg_load_speedup:.2f}x")


if __name__ == "__main__":
    main()
