#!/usr/bin/env python3
"""Test all MASt3R/DUNE models with MPS backend.

Uses subprocess isolation to avoid MPS memory issues between model loads.
"""

import subprocess
import sys
from typing import NamedTuple


class TestResult(NamedTuple):
    name: str
    passed: bool
    desc_nan: int
    time_ms: float
    error: str = ""


def test_model(variant_name: str, resolution: int) -> TestResult:
    """Test a single model variant in a subprocess."""
    code = f'''
import numpy as np
import time
from mast3r_runtime.core.config import (
    MASt3RRuntimeConfig, ModelConfig, RuntimeConfig,
    Precision, BackendType, ModelVariant
)
from mast3r_runtime.backends import get_runtime

variant = getattr(ModelVariant, "{variant_name}")
config = MASt3RRuntimeConfig(
    model=ModelConfig(variant=variant, resolution={resolution}, precision=Precision.FP16),
    runtime=RuntimeConfig(backend=BackendType.METAL),
)
runtime = get_runtime(config)
runtime.load()

img1 = np.random.randint(0, 255, ({resolution}, {resolution}, 3), dtype=np.uint8)
img2 = np.random.randint(0, 255, ({resolution}, {resolution}, 3), dtype=np.uint8)

start = time.perf_counter()
result = runtime.infer(img1, img2)
elapsed_ms = (time.perf_counter() - start) * 1000

desc_nan = int(np.isnan(result.desc_1).sum() + np.isnan(result.desc_2).sum())
conf_nan = int(np.isnan(result.conf_1).sum() + np.isnan(result.conf_2).sum())
passed = desc_nan == 0 and conf_nan == 0

desc_range = (
    float(min(np.nanmin(result.desc_1), np.nanmin(result.desc_2))),
    float(max(np.nanmax(result.desc_1), np.nanmax(result.desc_2))),
)

print(f"RESULT:{{passed}}:{{desc_nan}}:{{elapsed_ms:.0f}}:{{desc_range[0]:.3f}}:{{desc_range[1]:.3f}}")
'''

    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
        )

        # Parse output
        for line in result.stdout.split("\n"):
            if line.startswith("RESULT:"):
                parts = line.split(":")
                passed = parts[1] == "True"
                desc_nan = int(parts[2])
                time_ms = float(parts[3])
                return TestResult(
                    name=variant_name.lower(),
                    passed=passed,
                    desc_nan=desc_nan,
                    time_ms=time_ms,
                )

        # No result found - check stderr
        error = result.stderr.split("\n")[-5:]
        return TestResult(
            name=variant_name.lower(),
            passed=False,
            desc_nan=-1,
            time_ms=0.0,
            error="\n".join(error),
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            name=variant_name.lower(),
            passed=False,
            desc_nan=-1,
            time_ms=0.0,
            error="Timeout",
        )
    except Exception as e:
        return TestResult(
            name=variant_name.lower(),
            passed=False,
            desc_nan=-1,
            time_ms=0.0,
            error=str(e),
        )


def main():
    """Run all model tests."""
    print("=" * 70)
    print("MASt3R Runtime - All Models Test")
    print("=" * 70)

    # Check MPS availability
    try:
        from mast3r_runtime.backends import _mps

        if not _mps.is_available():
            print("ERROR: MPS backend not available!")
            return 1
        print(f"MPS device: {_mps.get_device_name()}\n")
    except ImportError as e:
        print(f"ERROR: Cannot import MPS backend: {e}")
        return 1

    # Define models to test
    models = [
        ("DUNE_VIT_SMALL_336", 336),
        ("DUNE_VIT_SMALL_448", 448),
        ("DUNE_VIT_BASE_336", 336),
        ("DUNE_VIT_BASE_448", 448),
        ("MAST3R_VIT_LARGE", 512),
    ]

    results = []

    for variant_name, resolution in models:
        name = variant_name.lower()
        print(f"Testing {name} @ {resolution}x{resolution}...", end=" ", flush=True)

        result = test_model(variant_name, resolution)
        results.append(result)

        if result.error:
            print(f"ERROR: {result.error[:50]}")
        elif result.passed:
            print(f"PASSED ({result.time_ms:.0f}ms)")
        else:
            print(f"FAILED (desc_nan={result.desc_nan})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\n{'Model':<25} {'Status':<10} {'Desc NaN':<12} {'Time':<10}")
    print("-" * 60)

    for r in results:
        status = "PASS" if r.passed else ("ERROR" if r.error else "FAIL")
        desc_nan = str(r.desc_nan) if r.desc_nan >= 0 else "N/A"
        time_str = f"{r.time_ms:.0f}ms" if r.time_ms > 0 else "N/A"
        print(f"{r.name:<25} {status:<10} {desc_nan:<12} {time_str:<10}")

    print("-" * 60)
    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print(f"\n{total - passed} TEST(S) FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
