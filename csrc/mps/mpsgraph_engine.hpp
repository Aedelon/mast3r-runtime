// MASt3R Runtime - MPSGraph Engine
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
// Uses MPSGraph with native SDPA (macOS 15+) for ~21x speedup.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../common/types.hpp"
#include "mpsgraph_context.hpp"

namespace mast3r {
namespace mpsgraph {

/**
 * MPSGraph-based inference engine for Apple Silicon.
 *
 * Uses Apple's native scaledDotProductAttention (WWDC 2024) for
 * maximum performance on M-series chips. Achieves ~194ms per image
 * vs ~8000ms with manual Metal kernels (21x speedup).
 *
 * Requires macOS 15.0+ (Sequoia) for SDPA support.
 *
 * Uses shared MPSGraphContext for device/queue management.
 */
class API_AVAILABLE(macos(15.0)) MPSGraphEngine {
public:
    // Use shared context (recommended)
    explicit MPSGraphEngine(const RuntimeConfig& config);

    // Use provided context (for isolated testing)
    MPSGraphEngine(const RuntimeConfig& config, std::shared_ptr<MPSGraphContext> ctx);

    ~MPSGraphEngine();

    // Non-copyable, movable
    MPSGraphEngine(const MPSGraphEngine&) = delete;
    MPSGraphEngine& operator=(const MPSGraphEngine&) = delete;
    MPSGraphEngine(MPSGraphEngine&&) = default;
    MPSGraphEngine& operator=(MPSGraphEngine&&) = default;

    // Load model from safetensors
    void load(const std::string& model_path);

    // Check if ready
    bool is_ready() const { return is_loaded_; }

    // Engine name
    std::string name() const;

    // Warmup
    void warmup(int num_iterations = 3);

    // Inference on image pair
    InferenceResult infer(const ImageView& img1, const ImageView& img2);

    // Feature matching
    MatchResult match(
        const float* desc_1, const float* desc_2,
        int height, int width, int desc_dim,
        const MatchingConfig& config
    );

    // Load retrieval weights (prewhiten.m, prewhiten.p)
    // If main model is loaded: uses weight sharing (encoder from main graph)
    // If main model is NOT loaded: builds standalone encoder + whitening graph
    void load_retrieval(const std::string& model_path, const std::string& retrieval_path);

    // Convenience overload: retrieval_path in same directory as model
    void load_retrieval(const std::string& retrieval_path);

    // Encoder-only mode for retrieval (returns whitened features + attention)
    RetrievalResult encode_retrieval(const ImageView& img);

    // Check if retrieval is ready
    bool is_retrieval_ready() const { return is_retrieval_loaded_; }

    // Check if using standalone retrieval (no main model)
    bool is_retrieval_standalone() const { return is_retrieval_standalone_; }

    // Access context
    std::shared_ptr<MPSGraphContext> context() const { return ctx_; }

    // Check if MPSGraph SDPA is available
    static bool is_available();

private:
    RuntimeConfig config_;
    ModelSpec spec_;
    bool is_loaded_ = false;
    bool is_retrieval_loaded_ = false;
    bool is_retrieval_standalone_ = false;  // true if retrieval loaded without main model

    // Shared context
    std::shared_ptr<MPSGraphContext> ctx_;

    // Graph (not shared - each engine has its own compiled graph)
    MPSGraph* graph_ = nil;

    // Placeholders (uint8 input for GPU preprocessing)
    MPSGraphTensor* input_placeholder_ = nil;  // [1, H, W, 3] uint8

    // Output tensors
    MPSGraphTensor* output_pts3d_conf_ = nil;
    MPSGraphTensor* output_descriptors_ = nil;
    MPSGraphTensor* output_enc_features_ = nil;  // Encoder output [N, D] for retrieval

    // Whitening graph (small, reuses encoder from main graph when available)
    MPSGraph* whitening_graph_ = nil;
    MPSGraphTensor* whitening_input_ = nil;     // [N, D] encoder features
    MPSGraphTensor* whitening_output_ = nil;    // [N, D] whitened features
    MPSGraphTensor* whitening_attention_ = nil; // [N] L2 attention scores

    // Standalone retrieval graph (encoder + whitening, used when main model not loaded)
    MPSGraph* retrieval_graph_ = nil;
    MPSGraphTensor* retrieval_input_ = nil;     // [1, H, W, 3] uint8
    MPSGraphTensor* retrieval_features_ = nil;  // [N, D] whitened features
    MPSGraphTensor* retrieval_attention_ = nil; // [N] L2 attention scores

    // Build the complete graph
    void build_graph();

    // Preprocessing
    void preprocess(const ImageView& img, float* output);
};

}  // namespace mpsgraph
}  // namespace mast3r
