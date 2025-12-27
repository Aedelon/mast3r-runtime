# MPS/Metal Backend Documentation

Backend d'inference haute performance pour Apple Silicon (M1/M2/M3/M4).

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Prérequis](#prérequis)
- [Architecture](#architecture)
- [API Python](#api-python)
- [Performance](#performance)
- [API C++](#api-c)
- [Internals](#internals)

---

## Vue d'ensemble

Le backend MPS utilise **MPSGraph** avec **Scaled Dot-Product Attention (SDPA)** natif d'Apple pour atteindre des performances optimales sur Apple Silicon.

### Caractéristiques

| Feature | Description |
|---------|-------------|
| **SDPA natif** | Utilise `scaledDotProductAttention` (WWDC 2024) |
| **Zero-copy** | Données GPU → CPU uniquement à la demande |
| **Pipelining** | Encoder[N+1] \|\| Decoder[N] pour meilleur throughput |
| **Weight sharing** | Retrieval partage l'encoder avec l'inference |
| **Buffer pooling** | Réutilisation des buffers GPU (O(1) allocation) |
| **ANE placement** | `OptimizationLevel1` pour routage ANE+GPU |

### Performance typique (M4 Pro)

| Mode | Latence | Throughput |
|------|---------|------------|
| `infer_gpu()` | ~194 ms/paire | 10.3 img/s |
| `infer_batch_pipelined()` | ~139 ms/paire | 14.4 img/s |
| `encode_retrieval()` | ~97 ms/img | 10.3 img/s |

---

## Prérequis

- **macOS 15.0+** (Sequoia) - requis pour SDPA
- **Apple Silicon** (M1/M2/M3/M4)
- **Python 3.10+**

```bash
# Installation
pip install mast3r-runtime

# Ou depuis les sources
git clone https://github.com/aedelon/mast3r-runtime.git
cd mast3r-runtime
uv pip install -e .
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       MPSGraphEngine                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ EncoderGraph│───▶│DecoderGraph │───▶│ GPUInferenceResult  │  │
│  │ (ViT)       │    │ (DPT Head)  │    │ (lazy-copy tensors) │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐                                             │
│  │ WhiteningGraph  │───▶ RetrievalResult                         │
│  │ (prewhiten.m/p) │                                             │
│  └─────────────────┘                                             │
├─────────────────────────────────────────────────────────────────┤
│                      MPSGraphContext                             │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐                 │
│  │MTLDevice │  │MTLCommandQueue│  │ BufferPool │                 │
│  └──────────┘  └──────────────┘  └────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

### Composants

| Composant | Rôle |
|-----------|------|
| `MPSGraphContext` | Singleton partagé (device, queue, buffer pool) |
| `EncoderGraph` | ViT encoder (patch_embed → transformer → norm) |
| `DecoderGraph` | Decoder + DPT head + local features |
| `WhiteningGraph` | PCA whitening pour retrieval |
| `GPUTensor` | Handle lazy-copy vers CPU |

---

## API Python

### Import

```python
from mast3r_runtime.backends import _mps

# Vérifier disponibilité
if _mps.is_available():
    print(f"Backend: {_mps.get_name()}")
```

### Initialisation

```python
engine = _mps.MPSEngine(
    variant="mast3r_vit_large",  # ou "dune_vit_small_14", "dune_vit_base_14"
    resolution=512,              # hauteur d'image
    precision="fp16",            # "fp16" ou "fp32"
    num_threads=4                # threads CPU (preprocessing)
)

# Charger les poids
engine.load("/path/to/unified.safetensors")

# Warmup (compilation JIT)
engine.warmup(num_iterations=3)
```

### Inference (paire d'images)

```python
import numpy as np

# Images: [H, W, 3] uint8 RGB
img1 = np.array(Image.open("view1.jpg"))
img2 = np.array(Image.open("view2.jpg"))

# Inference GPU (données restent sur GPU)
result = engine.infer_gpu(img1, img2)

# Accès aux résultats (copie GPU → CPU à la demande)
pts3d_1 = result.pts3d_1.numpy()  # [H, W, 3] float32
pts3d_2 = result.pts3d_2.numpy()  # [H, W, 3] float32
conf_1 = result.conf_1.numpy()    # [H, W] float32
conf_2 = result.conf_2.numpy()    # [H, W] float32
desc_1 = result.desc_1.numpy()    # [H, W, 24] float32
desc_2 = result.desc_2.numpy()    # [H, W, 24] float32

# Timing
print(f"Inference: {result.timing['inference_ms']:.1f} ms")
```

### Batch pipeliné (haut throughput)

```python
# Liste d'images consécutives
images = [np.array(Image.open(f"frame_{i}.jpg")) for i in range(10)]

# Inference pipelinée: encoder[N+1] || decoder[N]
# N images → N-1 paires de résultats
results = engine.infer_batch_pipelined(images)

for i, result in enumerate(results):
    print(f"Pair {i}-{i+1}: pts3d_1 shape = {result.pts3d_1.shape}")
```

### Retrieval (embeddings pour recherche)

```python
# Charger les poids de retrieval
engine.load_retrieval("/path/to/retrieval.safetensors")

# Encoder une seule image
result = engine.encode_retrieval(img)
features = result.features.numpy()   # [N, D] embeddings
attention = result.attention.numpy() # [N] attention scores

# Batch retrieval (plus efficace)
results = engine.encode_retrieval_batch(images)
```

### Feature Matching

```python
# Match entre deux sets de descripteurs
match_result = engine.match(
    desc_1=desc_1,      # [H, W, D] float32
    desc_2=desc_2,      # [H, W, D] float32
    top_k=512,          # max correspondances
    reciprocal=True,    # matching bidirectionnel
    confidence_threshold=0.5
)

# Résultats
idx_1 = match_result.idx_1        # indices image 1
idx_2 = match_result.idx_2        # indices image 2
pts2d_1 = match_result.pts2d_1    # coordonnées 2D image 1
pts2d_2 = match_result.pts2d_2    # coordonnées 2D image 2
pts3d_1 = match_result.pts3d_1    # points 3D image 1
pts3d_2 = match_result.pts3d_2    # points 3D image 2
confidence = match_result.confidence
```

---

## Performance

### Optimisations clés

1. **SDPA natif** - 21x plus rapide que les kernels Metal manuels
2. **Lazy-copy** - Élimine ~420ms de copie GPU→CPU par inference
3. **Pipelining** - Overlap encoder/decoder pour +40% throughput
4. **Buffer pooling** - O(1) allocation avec classes de taille
5. **ANE placement** - `OptimizationLevel1` pour routage intelligent

### Profiling

```bash
# Profiling avec timing détaillé
uv run python profile_metal.py

# Avec Instruments.app (Metal System Trace)
uv run python profile_metal.py --wait
# → Attacher Instruments puis appuyer sur Entrée
```

### Bottlenecks typiques

| Bottleneck | Symptôme | Solution |
|------------|----------|----------|
| IOSurface creation | ~225ms overhead ANE | Attendre Metal 4 (macOS 26) |
| Memory bandwidth | Limitée par 273 GB/s | Déjà optimisé (memory-bound) |
| Compilation JIT | Premier run lent | Utiliser `warmup()` |

---

## API C++

### Headers principaux

```cpp
#include "csrc/mps/mpsgraph_engine.hpp"
#include "csrc/mps/mpsgraph_context.hpp"
#include "csrc/mps/gpu_tensor.hpp"
```

### Utilisation

```cpp
#include "mpsgraph_engine.hpp"

using namespace mast3r::mpsgraph;

// Configuration
RuntimeConfig config;
config.variant = "mast3r_vit_large";
config.resolution = 512;
config.precision = "fp16";

// Créer l'engine
MPSGraphEngine engine(config);
engine.load("/path/to/model.safetensors");
engine.warmup(3);

// Inference
ImageView img1{data1, height, width, 3};
ImageView img2{data2, height, width, 3};
GPUInferenceResult result = engine.infer_gpu(img1, img2);

// Accès aux données (lazy-copy)
float* pts3d = result.pts3d_1.to_cpu();  // Alloue et copie
// ou
result.pts3d_1.copy_to(pre_allocated_buffer);  // Zero-alloc
```

### Classes principales

#### MPSGraphContext

```cpp
// Singleton partagé (recommandé)
auto ctx = MPSGraphContext::shared();

// Ou contexte isolé (tests)
auto ctx = MPSGraphContext::create();

// Accesseurs
ctx->device();       // id<MTLDevice>
ctx->queue();        // id<MTLCommandQueue>
ctx->buffer_pool();  // BufferPool&
ctx->device_name();  // "Apple M4 Pro"
```

#### GPUTensor

```cpp
// Handle lazy-copy
GPUTensor tensor = ...;

tensor.shape();   // std::vector<int64_t>
tensor.numel();   // nombre d'éléments
tensor.nbytes();  // taille en bytes
tensor.is_valid();

// Copie GPU → CPU
float* data = tensor.to_cpu();        // Alloue nouveau buffer
tensor.copy_to(existing_buffer);      // Copie dans buffer existant
```

#### BufferPool

```cpp
auto& pool = ctx->buffer_pool();

// Acquérir un buffer (O(1) avec classes de taille)
id<MTLBuffer> buf = pool.acquire(size);

// Rendre au pool
pool.release(buf);

// RAII pattern
{
    PooledBuffer buf(ctx, size);
    // utiliser buf.buffer(), buf.contents()
}  // auto-release
```

---

## Internals

### Flux de données

```
Input: [2, H, W, 3] uint8
    │
    ▼ (GPU preprocessing: normalize, transpose)
[2, 3, H, W] FP16
    │
    ▼ (Patch embedding: Conv2D 16x16)
[2, N, enc_dim] FP16
    │
    ▼ (Encoder: 24 transformer blocks)
[2, N, enc_dim] FP16
    │
    ▼ (Decoder: 12 blocks + cross-attention)
[2, N, dec_dim] FP16
    │
    ├──▶ DPT Head ──▶ pts3d [2, H, W, 3] + conf [2, H, W]
    │
    └──▶ Local Features ──▶ desc [2, H, W, 24]
```

### Compilation MPSGraph

```objc
// Descriptor de compilation
MPSGraphCompilationDescriptor* desc = [[MPSGraphCompilationDescriptor alloc] init];
desc.optimizationLevel = MPSGraphOptimizationLevel1;  // ANE + GPU + CPU
desc.waitForCompilationCompletion = YES;

// Exécution avec descriptor
MPSGraphExecutionDescriptor* execDesc = [[MPSGraphExecutionDescriptor alloc] init];
execDesc.compilationDescriptor = desc;
execDesc.waitUntilCompleted = YES;

[graph runAsyncWithMTLCommandQueue:queue
                             feeds:inputs
                     targetTensors:outputs
                  targetOperations:nil
               executionDescriptor:execDesc];
```

### Fichiers sources

```
csrc/mps/
├── mpsgraph_engine.hpp/mm   # Engine principal
├── mpsgraph_context.hpp/mm  # Context singleton + buffer pool
├── encoder_graph.hpp/mm     # ViT encoder partitionné
├── decoder_graph.hpp/mm     # Decoder + DPT partitionné
├── whitening_graph.hpp/mm   # PCA whitening (retrieval)
├── graph_builder.hpp        # Helper pour construire les ops
├── gpu_tensor.hpp/mm        # Handle lazy-copy
└── bindings.mm              # pybind11 Python bindings
```

---

## Troubleshooting

### Erreur "MPSGraph SDPA not available"

```python
if not _mps.is_available():
    # Vérifier macOS version
    import platform
    print(platform.mac_ver()[0])  # Doit être >= 15.0
```

### Performance dégradée

```bash
# Vérifier l'utilisation GPU
sudo powermetrics --samplers gpu_power -i 1000

# Vérifier que l'ANE n'est pas saturé
sudo powermetrics --samplers ane_power -i 1000
```

### Mémoire insuffisante

```python
# Vérifier la mémoire recommandée
info = _mps.get_context_info()
print(f"Recommended: {info['recommended_working_set_size'] / 1e9:.1f} GB")
print(f"Buffer pool: {info['buffer_pool_bytes'] / 1e6:.1f} MB")
```

---

## Références

- [WWDC 2024: MPSGraph Scaled Dot-Product Attention](https://developer.apple.com/videos/play/wwdc2024/10152/)
- [Metal Performance Shaders Graph Documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [Apple Silicon GPU Architecture](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

---

*Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.*