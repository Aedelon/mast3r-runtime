// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
// MPSGraph-based ViT encoder using native SDPA (macOS 15+)

#pragma once

#include "../common/types.hpp"
#include <memory>
#include <vector>
#include <string>

namespace mast3r {

/**
 * MPSGraph-based inference engine for MASt3R/DUNE models.
 * Uses Apple's native Scaled Dot-Product Attention (SDPA) from macOS 15.
 *
 * Advantages over raw Metal kernels:
 * - Native SDPA optimized by Apple
 * - Automatic kernel fusion
 * - KV-cache support
 * - Better memory management
 */
class MPSGraphEngine {
public:
    MPSGraphEngine();
    ~MPSGraphEngine();

    // Initialize with model specification
    bool initialize(const ModelSpec& spec);

    // Load weights from safetensors file
    bool load_weights(const std::string& path);

    // Run inference on a pair of images
    // Returns: pointmaps [2, H, W, 3], confidence [2, H, W], descriptors [2, H, W, D]
    InferenceResult infer(const uint8_t* img1, const uint8_t* img2,
                          int width, int height);

    // Check if MPSGraph SDPA is available (macOS 15+)
    static bool is_available();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mast3r
