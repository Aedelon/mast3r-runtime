// MASt3R Runtime - Metal Engine
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/types.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace mast3r {
namespace metal {

/**
 * Metal-based inference engine for Apple Silicon.
 *
 * Uses Metal compute shaders for preprocessing and matching.
 * Uses MPSGraph for neural network inference (optimized for ANE).
 */
class MetalEngine {
public:
    explicit MetalEngine(const RuntimeConfig& config);
    ~MetalEngine();

    // Non-copyable
    MetalEngine(const MetalEngine&) = delete;
    MetalEngine& operator=(const MetalEngine&) = delete;

    // Load model
    void load(const std::string& model_path);

    // Check if ready
    bool is_ready() const { return is_loaded_; }

    // Engine name
    std::string name() const;

    // Warmup
    void warmup(int num_iterations = 3);

    // Inference
    InferenceResult infer(const ImageView& img1, const ImageView& img2);

    // Matching
    MatchResult match(
        const float* desc_1, const float* desc_2,
        int height, int width, int desc_dim,
        const MatchingConfig& config
    );

private:
    RuntimeConfig config_;
    bool is_loaded_ = false;

#ifdef __OBJC__
    // Metal buffers
    id<MTLBuffer> buffer_img1_ = nil;
    id<MTLBuffer> buffer_img2_ = nil;
    id<MTLBuffer> buffer_preprocessed1_ = nil;
    id<MTLBuffer> buffer_preprocessed2_ = nil;
    id<MTLBuffer> buffer_pts3d_1_ = nil;
    id<MTLBuffer> buffer_pts3d_2_ = nil;
    id<MTLBuffer> buffer_desc_1_ = nil;
    id<MTLBuffer> buffer_desc_2_ = nil;
    id<MTLBuffer> buffer_conf_1_ = nil;
    id<MTLBuffer> buffer_conf_2_ = nil;

    // MPSGraph for inference
    // MPSGraph* graph_ = nil;
#endif

    void allocate_buffers();
    void preprocess_gpu(const ImageView& img, void* output_buffer);
};

}  // namespace metal
}  // namespace mast3r