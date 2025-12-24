"""Export MASt3R models to ONNX format.

Requires PyTorch and MASt3R to be installed:
    pip install mast3r-runtime[export]
    pip install git+https://github.com/naver/mast3r

Usage:
    mast3r-export                           # Export default model
    mast3r-export --variant dune_vit_base_14
    mast3r-export --resolution 448
    mast3r-export --output ./my_model.onnx

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..core.config import MODEL_SPECS, MASt3RRuntimeConfig, ModelVariant, Precision


def check_dependencies() -> None:
    """Check that required dependencies are installed."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        msg = "PyTorch is required for export. Install with: pip install torch"
        raise ImportError(msg) from e

    try:
        import onnx  # noqa: F401
    except ImportError as e:
        msg = "ONNX is required for export. Install with: pip install onnx"
        raise ImportError(msg) from e


def get_output_path(
    variant: ModelVariant,
    resolution: int,
    precision: Precision,
    output_dir: Path,
) -> Path:
    """Generate output path for exported model."""
    precision_suffix = "" if precision == Precision.FP32 else f"_{precision.value}"
    filename = f"{variant.value}_{resolution}{precision_suffix}.onnx"
    return output_dir / filename


def export_model(
    variant: ModelVariant,
    resolution: int,
    precision: Precision,
    output_path: Path,
    opset_version: int = 17,
) -> Path:
    """Export MASt3R model to ONNX format.

    Args:
        variant: Model variant to export
        resolution: Input resolution
        precision: Export precision (FP32 or FP16)
        output_path: Output path for ONNX model
        opset_version: ONNX opset version

    Returns:
        Path to exported model
    """
    import torch

    check_dependencies()

    print(f"\n📦 Exporting {variant.value}")
    print(f"   Resolution: {resolution}x{resolution}")
    print(f"   Precision: {precision.value}")
    print(f"   Output: {output_path}")
    print()

    # Load MASt3R model
    print("⏳ Loading MASt3R model...")
    try:
        from mast3r.model import AsymmetricMASt3R

        spec = MODEL_SPECS[variant]
        model = AsymmetricMASt3R.from_pretrained(spec["hf_repo"])
        model.eval()
    except ImportError as e:
        msg = (
            "MASt3R is required for export.\n"
            "Install with: pip install git+https://github.com/naver/mast3r"
        )
        raise ImportError(msg) from e

    # Move to appropriate dtype
    if precision == Precision.FP16:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Create dummy inputs
    print("⏳ Creating dummy inputs...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dummy_img1 = torch.randn(1, 3, resolution, resolution, dtype=dtype, device=device)
    dummy_img2 = torch.randn(1, 3, resolution, resolution, dtype=dtype, device=device)
    dummy_shape = torch.tensor([[resolution, resolution]], dtype=torch.int64, device=device)

    # Create input dict for MASt3R
    view1 = {"img": dummy_img1, "true_shape": dummy_shape}
    view2 = {"img": dummy_img2, "true_shape": dummy_shape}

    # Export to ONNX
    print("⏳ Exporting to ONNX...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap model for export
    class MASt3RWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, img1, img2, true_shape1, true_shape2):
            view1 = {"img": img1, "true_shape": true_shape1}
            view2 = {"img": img2, "true_shape": true_shape2}
            res1, res2 = self.model(view1, view2)

            return (
                res1["pts3d"],
                res2["pts3d_in_other_view"],
                res1["desc"],
                res2["desc"],
                res1["conf"],
                res2["conf"],
            )

    wrapper = MASt3RWrapper(model)

    torch.onnx.export(
        wrapper,
        (dummy_img1, dummy_img2, dummy_shape, dummy_shape),
        str(output_path),
        input_names=["img1", "img2", "true_shape1", "true_shape2"],
        output_names=["pts3d_1", "pts3d_2", "desc_1", "desc_2", "conf_1", "conf_2"],
        dynamic_axes={
            "img1": {0: "batch"},
            "img2": {0: "batch"},
            "true_shape1": {0: "batch"},
            "true_shape2": {0: "batch"},
            "pts3d_1": {0: "batch"},
            "pts3d_2": {0: "batch"},
            "desc_1": {0: "batch"},
            "desc_2": {0: "batch"},
            "conf_1": {0: "batch"},
            "conf_2": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Verify export
    print("⏳ Verifying ONNX model...")
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Export complete: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")

    return output_path


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export MASt3R models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mast3r-export                                    # Export default
  mast3r-export --variant dune_vit_base_14         # Export specific variant
  mast3r-export --resolution 448 --precision fp16  # Custom settings
  mast3r-export --output ./models/my_model.onnx    # Custom output path

Note:
  Requires MASt3R to be installed:
  pip install git+https://github.com/naver/mast3r
""",
    )

    parser.add_argument(
        "--variant",
        "-v",
        default="dune_vit_small_14",
        choices=[v.value for v in ModelVariant],
        help="Model variant to export (default: dune_vit_small_14)",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=336,
        help="Input resolution (default: 336, must be divisible by 14)",
    )
    parser.add_argument(
        "--precision",
        "-p",
        default="fp16",
        choices=["fp32", "fp16"],
        help="Export precision (default: fp16)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path (default: ~/.cache/mast3r_runtime/onnx/<model>.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )

    args = parser.parse_args()

    # Validate resolution
    if args.resolution % 14 != 0:
        print(f"❌ Resolution {args.resolution} must be divisible by 14")
        return 1

    variant = ModelVariant(args.variant)
    precision = Precision(args.precision)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        config = MASt3RRuntimeConfig()
        output_dir = config.cache_dir / "onnx"
        output_path = get_output_path(variant, args.resolution, precision, output_dir)

    try:
        export_model(
            variant=variant,
            resolution=args.resolution,
            precision=precision,
            output_path=output_path,
            opset_version=args.opset,
        )
        return 0
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return 1
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
