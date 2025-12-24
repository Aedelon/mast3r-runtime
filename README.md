# MASt3R Runtime

Lightweight inference runtime for [MASt3R](https://github.com/naver/mast3r) and [DUNE](https://github.com/naver/dune) 3D vision models.

Optimized for embedded deployment on:
- **Apple Silicon** (M1/M2/M3/M4) via CoreML
- **NVIDIA Jetson** (Orin) via TensorRT
- **Any platform** via ONNX Runtime

## Features

- 🚀 **Fast**: <20ms inference on M4 Pro with DUNE ViT-Small
- 📦 **Lightweight**: No PyTorch dependency for inference
- 🔌 **Cross-platform**: Single API, multiple backends
- 🎯 **Production-ready**: Typed, tested, documented

## Installation

```bash
# Basic installation with ONNX Runtime (CPU)
pip install mast3r-runtime[onnx]

# NVIDIA GPU (CUDA)
pip install mast3r-runtime[cuda]

# Apple Silicon (CoreML + ANE)
pip install mast3r-runtime[coreml]

# NVIDIA Jetson (TensorRT)
pip install mast3r-runtime[tensorrt]

# All backends
pip install mast3r-runtime[all]
```

## Quick Start

```python
from mast3r_runtime import get_runtime, MASt3RRuntimeConfig
import numpy as np

# Load runtime
config = MASt3RRuntimeConfig()
with get_runtime(config) as engine:
    # Load images (H, W, 3) RGB uint8
    img1 = np.array(Image.open("image1.jpg"))
    img2 = np.array(Image.open("image2.jpg"))

    # Run inference
    result = engine.infer(img1, img2)

    # Get 3D points and matches
    pts3d_1 = result.pts3d_1  # [H, W, 3]
    pts3d_2 = result.pts3d_2  # [H, W, 3]

    # Match descriptors
    matches = engine.match(result.desc_1, result.desc_2)
    print(f"Found {matches.num_matches} matches")
```

## Download Models

Models must be downloaded separately (licensed by Naver under CC BY-NC-SA 4.0):

```bash
# Download default model (DUNE ViT-Small, 220MB)
mast3r-download

# List available models
mast3r-download --list

# Download specific variant
mast3r-download dune_vit_base_14
```

## Supported Models

| Model | Size | Resolution | M4 Pro | Jetson Orin | Use Case |
|-------|------|------------|--------|-------------|----------|
| DUNE ViT-Small/14 | 220 MB | 336 | ~15ms | ~25ms | Real-time drone |
| DUNE ViT-Base/14 | 840 MB | 336 | ~25ms | ~40ms | Quality balance |
| MASt3R ViT-Large | 2.4 GB | 512 | ~60ms | ~120ms | Max precision |

## Configuration

```python
from mast3r_runtime import (
    MASt3RRuntimeConfig,
    ModelVariant,
    BackendType,
    Precision,
)

config = MASt3RRuntimeConfig(
    model=ModelConfig(
        variant=ModelVariant.DUNE_VIT_SMALL_14,
        resolution=336,
        precision=Precision.FP16,
    ),
    runtime=RuntimeConfig(
        backend=BackendType.AUTO,  # Auto-select best backend
    ),
    matching=MatchingConfig(
        top_k=512,
        reciprocal=True,
    ),
)
```

Or use YAML:

```yaml
# config.yaml
model:
  variant: dune_vit_small_14
  resolution: 336
  precision: fp16

runtime:
  backend: auto

matching:
  top_k: 512
  reciprocal: true
```

```python
config = MASt3RRuntimeConfig.from_yaml("config.yaml")
```

## CLI Tools

```bash
# System information
mast3r-info

# Benchmark performance
mast3r-benchmark --variant dune_vit_small_14 --iterations 100

# Download models
mast3r-download --list
```

## License

This runtime is licensed under **Apache 2.0**.

**Important**: The model weights are licensed under **CC BY-NC-SA 4.0** by Naver Corporation.
See [MASt3R](https://github.com/naver/mast3r) for details.

## Citation

If you use this in research, please cite:

```bibtex
@inproceedings{leroy2024mast3r,
  title={Grounding Image Matching in 3D with MASt3R},
  author={Leroy, Vincent and Cabon, Yohann and Revaud, J{\'e}r{\^o}me},
  booktitle={ECCV},
  year={2024}
}

@inproceedings{wang2024dune,
  title={DUNE: Dataset for Unified 3D Estimation},
  author={Wang, Shuai and others},
  booktitle={CVPR},
  year={2025}
}
```

## Links

- [MASt3R (Naver)](https://github.com/naver/mast3r)
- [DUNE (Naver)](https://github.com/naver/dune)
- [Issues](https://github.com/aedelon/mast3r-runtime/issues)
