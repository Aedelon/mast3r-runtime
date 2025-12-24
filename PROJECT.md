# MASt3R Runtime - Documentation Technique

## Vue d'ensemble

**mast3r-runtime** est un package Python léger pour l'inférence des modèles de vision 3D [MASt3R](https://github.com/naver/mast3r) et [DUNE](https://github.com/naver/dune) développés par Naver.

### Objectif

Déployer MASt3R/DUNE sur plateformes embarquées avec latence < 33ms (30 FPS) :
- **Apple Silicon** (M1/M2/M3/M4) via CoreML + ANE
- **NVIDIA Jetson** (Orin) via TensorRT
- **Tout GPU** via ONNX Runtime + CUDA

### Philosophie

| Aspect | Choix | Justification |
|--------|-------|---------------|
| **Dépendances** | Minimales (numpy, PIL, pydantic) | Léger, déploiement facile |
| **PyTorch** | Optionnel | Évite 2GB+ de dépendances |
| **Preprocessing** | Par backend | GPU-native, zero-copy |
| **Licence** | Apache 2.0 (runtime) | PyPI compatible |
| **Modèles** | Téléchargement séparé | CC BY-NC-SA 4.0 (Naver) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        mast3r-runtime                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Public API                            │   │
│  │  get_runtime(config) → EngineInterface                  │   │
│  │  MASt3RRuntimeConfig, ModelVariant, BackendType         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Backend Dispatcher                       │   │
│  │  Auto-détection : CoreML (macOS) → TensorRT (Linux) →   │   │
│  │                   CUDA → ONNX CPU                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│          ┌───────────────────┼───────────────────┐             │
│          ▼                   ▼                   ▼             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │   CoreML    │     │  TensorRT   │     │    ONNX     │      │
│  │  (M1-M4)    │     │  (Jetson)   │     │  (Fallback) │      │
│  │             │     │             │     │             │      │
│  │ Metal prep  │     │ CUDA prep   │     │ Numpy prep  │      │
│  │ ANE accel   │     │ INT8/FP16   │     │ CPU/CUDA    │      │
│  └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Structure des fichiers

```
mast3r_runtime/
├── pyproject.toml              # Configuration PyPI
├── LICENSE                     # Apache 2.0
├── NOTICE                      # Attribution Naver (CC BY-NC-SA 4.0)
├── README.md                   # Documentation utilisateur
├── PROJECT.md                  # Ce fichier
│
└── src/mast3r_runtime/
    ├── __init__.py             # Exports publics
    ├── py.typed                # PEP 561 marker
    │
    ├── core/
    │   ├── __init__.py
    │   ├── config.py           # Configuration Pydantic
    │   ├── engine_interface.py # Interface abstraite
    │   └── preprocessing.py    # Interface preprocessing
    │
    ├── backends/
    │   ├── __init__.py         # get_runtime() dispatcher
    │   │
    │   ├── onnx/
    │   │   ├── __init__.py     # is_available(), get_engine()
    │   │   └── onnx_engine.py  # ONNXEngine implementation
    │   │
    │   ├── coreml/
    │   │   └── __init__.py     # Stub (v0.2.0)
    │   │
    │   ├── tensorrt/
    │   │   └── __init__.py     # Stub (v0.2.0)
    │   │
    │   └── pytorch/
    │       ├── __init__.py
    │       └── torch_engine.py # PyTorchEngine (fallback)
    │
    └── scripts/
        ├── download.py         # mast3r-download CLI
        ├── export.py           # mast3r-export CLI
        ├── benchmark.py        # mast3r-benchmark CLI
        └── info.py             # mast3r-info CLI
```

---

## Modèles supportés

| Modèle | Taille | Résolution | M4 Pro | Jetson Orin | Usage |
|--------|--------|------------|--------|-------------|-------|
| **DUNE ViT-Small/14** | 220 MB | 336 | ~15ms | ~25ms | Drone temps-réel |
| **DUNE ViT-Base/14** | 840 MB | 336 | ~25ms | ~40ms | Qualité/vitesse |
| **MASt3R ViT-Large** | 2.4 GB | 512 | ~60ms | ~120ms | Précision max |
| **DUSt3R 224 Linear** | 1.8 GB | 224 | ~40ms | ~80ms | Compatibilité |

### Presets

```python
# Drone temps-réel (< 20ms)
PRESET_DRONE_FAST = MASt3RRuntimeConfig(
    model=ModelConfig(variant=ModelVariant.DUNE_VIT_SMALL_14, resolution=336),
    runtime=RuntimeConfig(backend=BackendType.AUTO),
)

# Qualité/vitesse équilibré
PRESET_DRONE_QUALITY = MASt3RRuntimeConfig(
    model=ModelConfig(variant=ModelVariant.DUNE_VIT_BASE_14, resolution=336),
)

# Précision maximale (desktop)
PRESET_DESKTOP_PRECISION = MASt3RRuntimeConfig(
    model=ModelConfig(variant=ModelVariant.MAST3R_VIT_LARGE, resolution=518),
)
```

---

## Configuration

### Pydantic Models

```python
class ModelConfig(BaseModel):
    variant: ModelVariant = ModelVariant.DUNE_VIT_SMALL_14
    resolution: int = 336  # Must be divisible by patch_size (14)
    precision: Precision = Precision.FP16

class RuntimeConfig(BaseModel):
    backend: BackendType = BackendType.AUTO
    num_threads: int = 4
    use_gpu: bool = True

class MatchingConfig(BaseModel):
    top_k: int = 512
    reciprocal: bool = True
    confidence_threshold: float = 0.0

class MASt3RRuntimeConfig(BaseModel):
    model: ModelConfig
    runtime: RuntimeConfig
    matching: MatchingConfig
    cache_dir: Path  # ~/.cache/mast3r_runtime
```

### YAML

```yaml
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

---

## Interface Engine

### EngineInterface (ABC)

```python
class EngineInterface(ABC):
    """Interface abstraite pour tous les backends."""

    @abstractmethod
    def infer(self, img1: NDArray, img2: NDArray) -> InferenceResult:
        """Inférence sur paire stéréo."""
        ...

    @abstractmethod
    def match(self, desc_1: NDArray, desc_2: NDArray) -> MatchResult:
        """Matching de descripteurs."""
        ...

    def benchmark(self, num_iterations: int = 100) -> dict:
        """Benchmark de performance."""
        ...
```

### Résultats (numpy-based)

```python
@dataclass
class InferenceResult:
    pts3d_1: NDArray[np.float32]      # [H, W, 3] points 3D view1
    pts3d_2: NDArray[np.float32]      # [H, W, 3] points 3D view2
    desc_1: NDArray[np.float32]       # [H, W, D] descripteurs view1
    desc_2: NDArray[np.float32]       # [H, W, D] descripteurs view2
    conf_1: NDArray[np.float32]       # [H, W] confiance view1
    conf_2: NDArray[np.float32]       # [H, W] confiance view2
    timing_ms: dict[str, float]       # Timing par étape

@dataclass
class MatchResult:
    idx_1: NDArray[np.int64]          # Indices matches view1
    idx_2: NDArray[np.int64]          # Indices matches view2
    pts2d_1: NDArray[np.float32]      # [N, 2] coords 2D view1
    pts2d_2: NDArray[np.float32]      # [N, 2] coords 2D view2
    confidence: NDArray[np.float32]   # [N] scores
```

---

## Preprocessing par Backend

### Philosophie

Le preprocessing (resize, normalize) doit être GPU-native pour éviter les bottlenecks CPU→GPU.

| Backend | Preprocessing | Technologie |
|---------|---------------|-------------|
| ONNX | Intégré au modèle ou numpy | ONNX ops / PIL |
| CoreML | Metal shaders | MLShaderCompiler |
| TensorRT | CUDA kernels | nvJPEG / custom |
| PyTorch | torch tensors | torchvision |

### Interface

```python
class PreprocessorBase(ABC):
    """Interface pour preprocessing backend-specific."""

    @abstractmethod
    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Preprocess image [H,W,3] RGB → [1,3,res,res] float32."""
        ...

class NumpyPreprocessor(PreprocessorBase):
    """Fallback CPU (dev/debug only)."""
    ...
```

---

## CLI Tools

### mast3r-download

```bash
# Télécharger modèle par défaut
mast3r-download

# Lister modèles disponibles
mast3r-download --list

# Télécharger variante spécifique
mast3r-download --variant dune_vit_base_14
```

### mast3r-export

```bash
# Exporter vers ONNX (nécessite PyTorch + MASt3R)
mast3r-export --variant dune_vit_small_14 --resolution 336
mast3r-export --precision fp16 --output ./model.onnx
```

### mast3r-benchmark

```bash
# Benchmark performance
mast3r-benchmark --variant dune_vit_small_14 --iterations 100
mast3r-benchmark --backend onnx --device cuda
```

### mast3r-info

```bash
# Informations système
mast3r-info
# → Platform, backends disponibles, GPU détecté, etc.
```

---

## Installation

### PyPI

```bash
# CPU only
pip install mast3r-runtime

# ONNX Runtime (recommandé)
pip install mast3r-runtime[onnx]

# Apple Silicon (CoreML)
pip install mast3r-runtime[coreml]

# NVIDIA GPU (CUDA)
pip install mast3r-runtime[cuda]

# Tous les backends
pip install mast3r-runtime[all]

# Export (nécessite PyTorch)
pip install mast3r-runtime[export]
```

### Dépendances

```toml
# Core (toujours installé)
dependencies = [
    "numpy>=1.24",
    "pillow>=10.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "tqdm>=4.60",
    "huggingface-hub>=0.20",
]

# Optionnelles
[project.optional-dependencies]
onnx = ["onnxruntime>=1.17"]
cuda = ["torch>=2.0", "onnxruntime-gpu>=1.17"]
coreml = ["coremltools>=7.0; sys_platform == 'darwin'"]
tensorrt = ["tensorrt>=8.6; sys_platform == 'linux'"]
```

---

## Usage

### Basic

```python
from mast3r_runtime import get_runtime, MASt3RRuntimeConfig
import numpy as np
from PIL import Image

# Charger config
config = MASt3RRuntimeConfig()

# Créer runtime (auto-sélection backend)
with get_runtime(config) as engine:
    # Charger images
    img1 = np.array(Image.open("view1.jpg"))
    img2 = np.array(Image.open("view2.jpg"))

    # Inférence
    result = engine.infer(img1, img2)

    # Points 3D
    pts3d = result.pts3d_1  # [H, W, 3]

    # Matching
    matches = engine.match(result.desc_1, result.desc_2)
    print(f"Found {len(matches.idx_1)} matches")
```

### Presets

```python
from mast3r_runtime import PRESET_DRONE_FAST, get_runtime

# Config optimisée drone
with get_runtime(PRESET_DRONE_FAST) as engine:
    result = engine.infer(img1, img2)
```

### YAML Config

```python
config = MASt3RRuntimeConfig.from_yaml("config.yaml")
```

---

## Licences

| Composant | Licence | Note |
|-----------|---------|------|
| **mast3r-runtime** (ce code) | Apache 2.0 | Libre, commercial OK |
| **Modèles MASt3R/DUNE** | CC BY-NC-SA 4.0 | Non-commercial, Naver |
| **ONNX Runtime** | MIT | Libre |
| **CoreML Tools** | BSD-3 | Libre |

> **Important** : Les modèles sont téléchargés séparément et restent sous licence Naver.
> Ce runtime ne redistribue pas les poids, uniquement le code d'inférence.

---

## Roadmap

### v0.1.0 (actuel)
- [x] Structure package PyPI
- [x] Backend ONNX Runtime
- [x] CLI tools (download, export, benchmark, info)
- [x] Configuration Pydantic
- [x] Stubs CoreML/TensorRT

### v0.2.0 (prévu)
- [ ] Backend CoreML natif (Metal preprocessing)
- [ ] Backend TensorRT natif (CUDA preprocessing)
- [ ] INT8 quantization pour Jetson
- [ ] Dual-resolution (tracking 256, keyframes 336)

### v0.3.0 (futur)
- [ ] Intégration SLAM
- [ ] Streaming video pipeline
- [ ] ROS2 node

---

## Références

- [MASt3R (Naver)](https://github.com/naver/mast3r) - Grounding Image Matching in 3D
- [DUNE (Naver)](https://github.com/naver/dune) - Dataset for Unified 3D Estimation
- [DUSt3R (Naver)](https://github.com/naver/dust3r) - Dense Unconstrained Stereo 3D Reconstruction

### Citations

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

---

## Auteur

**Delanoe Pirard / Aedelon**

- Email: delanoe@aedelon.com
- GitHub: [github.com/aedelon](https://github.com/aedelon)

Copyright 2024. Apache 2.0.