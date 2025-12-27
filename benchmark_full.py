#!/usr/bin/env python3
"""Comprehensive benchmark for MASt3R Runtime.

Tests all models, all modes (inference, retrieval, batch pipelined).
Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

Usage:
    uv run python benchmark_full.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Model configurations
MODELS = [
    {
        "name": "DUNE ViT-Small 336",
        "variant": "dune_vit_small_336",
        "resolution": 336,
        "path": "dune_vit_small_336",
        "has_retrieval": False,
    },
    {
        "name": "DUNE ViT-Small 448",
        "variant": "dune_vit_small_448",
        "resolution": 448,
        "path": "dune_vit_small_448",
        "has_retrieval": False,
    },
    {
        "name": "DUNE ViT-Base 336",
        "variant": "dune_vit_base_336",
        "resolution": 336,
        "path": "dune_vit_base_336",
        "has_retrieval": False,
    },
    {
        "name": "DUNE ViT-Base 448",
        "variant": "dune_vit_base_448",
        "resolution": 448,
        "path": "dune_vit_base_448",
        "has_retrieval": False,
    },
    {
        "name": "MASt3R ViT-Large 512",
        "variant": "mast3r_vit_large",
        "resolution": 512,
        "path": "mast3r_vit_large",
        "has_retrieval": True,
    },
]


@dataclass
class BenchmarkResult:
    model: str
    mode: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput: float  # images/sec


def get_model_path(model_info: dict) -> Path | None:
    """Get model path from cache."""
    base = Path.home() / ".cache/mast3r_runtime/safetensors" / model_info["path"]

    # Check for unified.safetensors (MASt3R)
    unified = base / "unified.safetensors"
    if unified.exists():
        return unified

    # DUNE models use split files - return directory path
    # The C++ code handles directory loading with encoder.safetensors + decoder.safetensors
    if (base / "encoder.safetensors").exists() and (base / "decoder.safetensors").exists():
        return base

    return None


def benchmark_inference(
    engine, img1, img2, warmup: int = 3, iterations: int = 10
) -> BenchmarkResult:
    """Benchmark standard inference (image pair)."""
    # Warmup
    for _ in range(warmup):
        _ = engine.infer_gpu(img1, img2)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.infer_gpu(img1, img2)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return BenchmarkResult(
        model="",
        mode="infer_gpu (pair)",
        iterations=iterations,
        mean_ms=float(np.mean(times)),
        std_ms=float(np.std(times)),
        min_ms=float(np.min(times)),
        max_ms=float(np.max(times)),
        throughput=2000.0 / float(np.mean(times)),  # 2 images per call
    )


def benchmark_batch_pipelined(
    engine, images: list, warmup: int = 2, iterations: int = 5
) -> BenchmarkResult:
    """Benchmark batch pipelined inference."""
    n_images = len(images)

    # Warmup
    for _ in range(warmup):
        _ = engine.infer_batch_pipelined(images)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.infer_batch_pipelined(images)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    per_image = times / n_images

    return BenchmarkResult(
        model="",
        mode=f"batch_pipelined (x{n_images})",
        iterations=iterations,
        mean_ms=float(np.mean(per_image)),
        std_ms=float(np.std(per_image)),
        min_ms=float(np.min(per_image)),
        max_ms=float(np.max(per_image)),
        throughput=n_images * 1000.0 / float(np.mean(times)),
    )


def benchmark_retrieval(engine, img, warmup: int = 3, iterations: int = 10) -> BenchmarkResult:
    """Benchmark retrieval encoding (single image)."""
    # Warmup
    for _ in range(warmup):
        _ = engine.encode_retrieval(img)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.encode_retrieval(img)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return BenchmarkResult(
        model="",
        mode="encode_retrieval",
        iterations=iterations,
        mean_ms=float(np.mean(times)),
        std_ms=float(np.std(times)),
        min_ms=float(np.min(times)),
        max_ms=float(np.max(times)),
        throughput=1000.0 / float(np.mean(times)),
    )


def benchmark_retrieval_batch(
    engine, images: list, warmup: int = 2, iterations: int = 5
) -> BenchmarkResult:
    """Benchmark batch retrieval encoding."""
    n_images = len(images)

    # Warmup
    for _ in range(warmup):
        _ = engine.encode_retrieval_batch(images)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.encode_retrieval_batch(images)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    per_image = times / n_images

    return BenchmarkResult(
        model="",
        mode=f"retrieval_batch (x{n_images})",
        iterations=iterations,
        mean_ms=float(np.mean(per_image)),
        std_ms=float(np.std(per_image)),
        min_ms=float(np.min(per_image)),
        max_ms=float(np.max(per_image)),
        throughput=n_images * 1000.0 / float(np.mean(times)),
    )


def benchmark_retrieval_standalone(model_info: dict, _mps) -> list[BenchmarkResult]:
    """Benchmark retrieval in standalone mode (encoder-only, no decoder loaded)."""
    results = []

    model_path = get_model_path(model_info)
    if model_path is None:
        return results

    retrieval_path = (
        model_path.parent / "retrieval.safetensors"
        if model_path.is_file()
        else model_path / "retrieval.safetensors"
    )

    if not retrieval_path.exists():
        return results

    # Create engine WITHOUT loading main model
    engine = _mps.MPSEngine(
        variant=model_info["variant"],
        resolution=model_info["resolution"],
        precision="fp16",
        num_threads=4,
    )

    # Load retrieval in standalone mode
    engine.load_retrieval_standalone(str(model_path), str(retrieval_path))

    # Create test image
    h = model_info["resolution"]
    w = int(h * 4 / 3)
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        _ = engine.encode_retrieval(img)

    # Benchmark single
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = engine.encode_retrieval(img)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    results.append(
        BenchmarkResult(
            model=model_info["name"],
            mode="retrieval_standalone",
            iterations=10,
            mean_ms=float(np.mean(times)),
            std_ms=float(np.std(times)),
            min_ms=float(np.min(times)),
            max_ms=float(np.max(times)),
            throughput=1000.0 / float(np.mean(times)),
        )
    )

    # Benchmark batch standalone
    images = [img, img.copy(), img.copy(), img.copy(), img.copy()]  # 5 images

    # Warmup
    for _ in range(2):
        _ = engine.encode_retrieval_batch(images)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = engine.encode_retrieval_batch(images)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    per_image = times / 5
    results.append(
        BenchmarkResult(
            model=model_info["name"],
            mode="retrieval_standalone_batch (x5)",
            iterations=5,
            mean_ms=float(np.mean(per_image)),
            std_ms=float(np.std(per_image)),
            min_ms=float(np.min(per_image)),
            max_ms=float(np.max(per_image)),
            throughput=5 * 1000.0 / float(np.mean(times)),
        )
    )

    return results


def print_results(results: list[BenchmarkResult]):
    """Print results in a nice table."""
    print("\n" + "=" * 100)
    print(
        f"{'Model':<25} {'Mode':<30} {'Mean (ms)':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'Throughput':<12}"
    )
    print("=" * 100)

    current_model = None
    for r in results:
        if r.model != current_model:
            if current_model is not None:
                print("-" * 100)
            current_model = r.model
            print(f"\n{r.model}")

        print(
            f"{'':25} {r.mode:<30} {r.mean_ms:>10.1f}  {r.std_ms:>8.1f}  {r.min_ms:>8.1f}  {r.max_ms:>8.1f}  {r.throughput:>8.2f} img/s"
        )

    print("=" * 100)


def main():
    print("=" * 80)
    print("MASt3R Runtime - Comprehensive Benchmark")
    print("=" * 80)

    # Check MPS availability
    try:
        from mast3r_runtime.backends import _mps
    except ImportError as e:
        print(f"MPS backend not available: {e}")
        return

    if not _mps.is_available():
        print("MPS requires macOS 15+")
        return

    print(f"\nDevice: {_mps.get_device_name()}")
    print(f"Context info: {_mps.get_context_info()}")

    all_results = []

    for model_info in MODELS:
        model_path = get_model_path(model_info)

        if model_path is None:
            print(f"\n⚠️  Skipping {model_info['name']}: not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {model_info['name']}")
        print(f"{'=' * 60}")

        # Create engine
        engine = _mps.MPSEngine(
            variant=model_info["variant"],
            resolution=model_info["resolution"],
            precision="fp16",
            num_threads=4,
        )

        # Load model
        print(f"Loading model from {model_path}...")
        t0 = time.perf_counter()
        engine.load(str(model_path))
        load_time = (time.perf_counter() - t0) * 1000
        print(f"Model loaded in {load_time:.0f} ms")

        # Create test images
        h = model_info["resolution"]
        w = int(h * 4 / 3)
        img1 = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Warmup engine
        print("Warmup...")
        engine.warmup(3)

        # Benchmark inference (pair)
        print("Benchmarking infer_gpu (pair)...")
        result = benchmark_inference(engine, img1, img2)
        result.model = model_info["name"]
        all_results.append(result)
        print(f"  → {result.mean_ms:.1f} ± {result.std_ms:.1f} ms ({result.throughput:.2f} img/s)")

        # Benchmark batch pipelined (various sizes)
        for batch_size in [2, 4, 8]:
            images = [
                np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(batch_size)
            ]
            print(f"Benchmarking batch_pipelined (x{batch_size})...")
            result = benchmark_batch_pipelined(engine, images)
            result.model = model_info["name"]
            all_results.append(result)
            print(
                f"  → {result.mean_ms:.1f} ± {result.std_ms:.1f} ms/img ({result.throughput:.2f} img/s)"
            )

        # Benchmark retrieval if available
        if model_info["has_retrieval"]:
            retrieval_path = (
                model_path.parent / "retrieval.safetensors"
                if model_path.is_file()
                else model_path / "retrieval.safetensors"
            )

            if retrieval_path.exists():
                print("Loading retrieval weights (weight sharing mode)...")
                engine.load_retrieval(str(retrieval_path))

                # Single retrieval
                print("Benchmarking encode_retrieval...")
                result = benchmark_retrieval(engine, img1)
                result.model = model_info["name"]
                all_results.append(result)
                print(
                    f"  → {result.mean_ms:.1f} ± {result.std_ms:.1f} ms ({result.throughput:.2f} img/s)"
                )

                # Batch retrieval
                for batch_size in [4, 8, 16]:
                    images = [
                        np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                        for _ in range(batch_size)
                    ]
                    print(f"Benchmarking retrieval_batch (x{batch_size})...")
                    result = benchmark_retrieval_batch(engine, images)
                    result.model = model_info["name"]
                    all_results.append(result)
                    print(
                        f"  → {result.mean_ms:.1f} ± {result.std_ms:.1f} ms/img ({result.throughput:.2f} img/s)"
                    )

        # Clear engine
        del engine
        _mps.clear_buffer_pool()

    # Benchmark retrieval standalone mode
    print(f"\n{'=' * 60}")
    print("Benchmarking: Retrieval Standalone Mode")
    print(f"{'=' * 60}")

    for model_info in MODELS:
        if model_info["has_retrieval"]:
            print(f"\n{model_info['name']} (standalone)...")
            standalone_results = benchmark_retrieval_standalone(model_info, _mps)
            all_results.extend(standalone_results)
            for r in standalone_results:
                print(f"  {r.mode}: {r.mean_ms:.1f} ± {r.std_ms:.1f} ms ({r.throughput:.2f} img/s)")

    # Print summary
    print_results(all_results)

    # Memory usage
    print("\nMemory Usage:")
    ctx_info = _mps.get_context_info()
    print(
        f"  Buffer pool: {ctx_info.get('buffer_pool_count', 0)} buffers, {ctx_info.get('buffer_pool_bytes', 0) / 1024 / 1024:.1f} MB"
    )

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
