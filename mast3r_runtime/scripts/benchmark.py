"""Benchmark MASt3R runtime performance.

Usage:
    mast3r-benchmark                    # Benchmark default model
    mast3r-benchmark --iterations 100   # Custom iterations
    mast3r-benchmark --resolution 512   # Custom resolution

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import argparse
import sys

from ..backends import get_runtime
from ..core.config import MASt3RRuntimeConfig, ModelVariant


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark MASt3R runtime performance",
    )
    parser.add_argument(
        "--variant",
        "-v",
        default="dune_vit_small_14",
        choices=[v.value for v in ModelVariant],
        help="Model variant to benchmark",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=336,
        help="Input resolution (default: 336)",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=100,
        help="Number of iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )

    args = parser.parse_args()

    print("\n🚀 MASt3R Runtime Benchmark")
    print("=" * 50)
    print(f"  Model: {args.variant}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Warmup: {args.warmup}")
    print()

    # Create config
    config = MASt3RRuntimeConfig()
    config.model.variant = ModelVariant(args.variant)
    config.model.resolution = args.resolution

    try:
        with get_runtime(config) as engine:
            print(f"  Backend: {engine.name}")
            print()
            print("⏱️  Running benchmark...")

            results = engine.benchmark(
                num_iterations=args.iterations,
                warmup_iterations=args.warmup,
                resolution=args.resolution,
            )

            print()
            print("📊 Results")
            print("-" * 50)
            print(f"  Mean:   {results['mean_ms']:>8.2f} ms")
            print(f"  Std:    {results['std_ms']:>8.2f} ms")
            print(f"  Min:    {results['min_ms']:>8.2f} ms")
            print(f"  Max:    {results['max_ms']:>8.2f} ms")
            print(f"  P50:    {results['p50_ms']:>8.2f} ms")
            print(f"  P95:    {results['p95_ms']:>8.2f} ms")
            print(f"  P99:    {results['p99_ms']:>8.2f} ms")
            print("-" * 50)
            print(f"  FPS:    {results['fps']:>8.1f}")
            print()

            # Check if meets real-time target
            if results["mean_ms"] < 33.3:
                print("✅ Meets 30 FPS real-time target!")
            elif results["mean_ms"] < 66.6:
                print("⚠️  Meets 15 FPS target")
            else:
                print("❌ Below real-time performance")

            print()
            return 0

    except FileNotFoundError as e:
        print(f"❌ Model not found: {e}")
        print("   Download with: mast3r-download", args.variant)
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
