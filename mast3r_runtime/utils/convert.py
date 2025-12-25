"""Convert PyTorch checkpoints to safetensors format.

Requires: torch, safetensors

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

from pathlib import Path

from ..core.config import MODEL_SPECS, ModelVariant, get_checkpoint_paths


def _check_dependencies() -> None:
    """Check that required dependencies are installed."""
    try:
        import safetensors  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        msg = (
            "Conversion requires PyTorch and safetensors. "
            "Install with: pip install torch safetensors"
        )
        raise ImportError(msg) from e


def convert_checkpoint(
    pth_path: Path,
    output_path: Path | None = None,
    dtype: str = "fp16",
) -> Path:
    """Convert a PyTorch checkpoint to safetensors.

    Args:
        pth_path: Path to .pth file.
        output_path: Output path. Defaults to same name with .safetensors.
        dtype: Target dtype ("fp32", "fp16", "bf16").

    Returns:
        Path to output file.
    """
    _check_dependencies()

    import torch
    from safetensors.torch import save_file

    print(f"Loading {pth_path.name}...")
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

    # Convert dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, torch.float16)

    converted = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            if tensor.is_floating_point():
                converted[key] = tensor.to(target_dtype).contiguous()
            else:
                converted[key] = tensor.contiguous()

    # Output path
    if output_path is None:
        output_path = pth_path.with_suffix(".safetensors")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {output_path.name} ({len(converted)} tensors, {dtype})...")
    save_file(converted, output_path)

    # File sizes
    pth_size = pth_path.stat().st_size / 1024 / 1024
    st_size = output_path.stat().st_size / 1024 / 1024
    print(f"  {pth_size:.1f} MB -> {st_size:.1f} MB ({st_size / pth_size * 100:.0f}%)")

    return output_path


def convert_model(
    variant: ModelVariant,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    dtype: str = "fp16",
) -> dict[str, Path]:
    """Convert all checkpoints for a model variant.

    Args:
        variant: Model variant.
        cache_dir: Cache directory with downloaded checkpoints.
        output_dir: Output directory. Defaults to cache_dir/safetensors.
        dtype: Target dtype.

    Returns:
        Dict mapping checkpoint type to converted path.
    """
    _check_dependencies()

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "mast3r_runtime"

    if output_dir is None:
        output_dir = cache_dir / "safetensors" / variant.value

    paths = get_checkpoint_paths(variant, cache_dir)
    results = {}

    print(f"\n{'=' * 50}")
    print(f"Converting {variant.value}")
    print(f"{'=' * 50}")

    for ckpt_type, pth_path in paths.items():
        if not pth_path.exists():
            print(f"\n[{ckpt_type}] MISSING - download first")
            continue

        # Skip non-tensor files (pickle codebooks, etc.)
        if pth_path.suffix == ".pkl":
            print(f"\n[{ckpt_type}] SKIPPED - .pkl files cannot be converted to safetensors")
            continue

        output_path = output_dir / f"{ckpt_type}.safetensors"

        print(f"\n[{ckpt_type}]")
        results[ckpt_type] = convert_checkpoint(pth_path, output_path, dtype=dtype)

    print(f"\nOutput: {output_dir}")
    return results


def convert_all_models(
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    dtype: str = "fp16",
) -> dict[ModelVariant, dict[str, Path]]:
    """Convert all downloaded models."""
    results = {}

    for variant in ModelVariant:
        paths = get_checkpoint_paths(variant, cache_dir)
        if any(p.exists() for p in paths.values()):
            var_output = output_dir / variant.value if output_dir else None
            results[variant] = convert_model(variant, cache_dir, var_output, dtype)

    return results


def get_safetensors_paths(
    variant: ModelVariant,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """Get paths to converted safetensors files."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "mast3r_runtime"

    output_dir = cache_dir / "safetensors" / variant.value
    spec = MODEL_SPECS[variant]

    # Only include .pth files (skip .pkl codebooks)
    return {
        ckpt_type: output_dir / f"{ckpt_type}.safetensors"
        for ckpt_type, info in spec["checkpoints"].items()
        if info["filename"].endswith(".pth")
    }


def is_converted(variant: ModelVariant, cache_dir: Path | None = None) -> bool:
    """Check if a model variant has been converted."""
    paths = get_safetensors_paths(variant, cache_dir)
    return all(p.exists() for p in paths.values())
