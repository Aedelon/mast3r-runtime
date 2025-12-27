// MASt3R Runtime - Whitening Graph (Retrieval)
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
// Whitening transformation for retrieval features.
// Takes encoder output, produces whitened features + attention scores.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "graph_builder.hpp"
#include "mpsgraph_context.hpp"
#include <memory>
#include <string>

namespace mast3r {
namespace mpsgraph {

/**
 * Whitening output structure.
 * Features: [N, D] whitened descriptor features
 * Attention: [N] L2-based attention scores (normalized to sum=1)
 */
struct WhiteningOutput {
    MPSGraphTensorData* features;   // [N, enc_dim] FP32
    MPSGraphTensorData* attention;  // [N] FP32
};

/**
 * Pre-allocated whitening output buffers (zero-copy pattern).
 */
struct WhiteningOutputBuffers {
    id<MTLBuffer> buf_features = nil;
    id<MTLBuffer> buf_attention = nil;
    MPSGraphTensorData* td_features = nil;
    MPSGraphTensorData* td_attention = nil;
    float* ptr_features = nullptr;
    float* ptr_attention = nullptr;
};

/**
 * Whitening Graph for retrieval.
 *
 * Input: encoder features [N, enc_dim] FP16
 * Output: whitened features [N, D] + attention [N]
 *
 * Whitening transformation:
 * 1. Center: x = x - mean (prewhiten.m)
 * 2. Project: x = x @ P (prewhiten.p)
 * 3. Attention: L2 norm of each patch, normalized
 *
 * Used in retrieval pipeline:
 * - Encoder → Whitening → Top-K selection → Matching
 */
class API_AVAILABLE(macos(15.0)) WhiteningGraph {
public:
    explicit WhiteningGraph(std::shared_ptr<MPSGraphContext> ctx, const RuntimeConfig& config);
    ~WhiteningGraph() = default;

    // Load whitening weights from retrieval.safetensors
    // Expects: prewhiten.m [enc_dim], prewhiten.p [enc_dim, enc_dim]
    void load(const std::string& retrieval_path);

    // Load from already-opened file
    void load(safetensors::SafetensorsFile& file);

    // Compile graph for faster execution
    void compile();

    // Run whitening - returns GPU tensors (lazy copy)
    WhiteningOutput run(MPSGraphTensorData* enc_out);

    // Run async - returns immediately, use completion handler
    void run_async(MPSGraphTensorData* enc_out,
                   void (^completion)(WhiteningOutput output));

    // Allocate output buffers for zero-copy pattern
    WhiteningOutputBuffers allocate_output_buffers();

    // Run with pre-allocated buffers (zero allocation per inference)
    void run_into(MPSGraphTensorData* enc_out, WhiteningOutputBuffers& buffers);

    // Run async with pre-allocated buffers
    void run_async_into(MPSGraphTensorData* enc_out, WhiteningOutputBuffers& buffers,
                        void (^completion)(void));

    // Accessors
    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input_placeholder() const { return input_; }
    MPSGraphTensor* output_features_tensor() const { return output_features_; }
    MPSGraphTensor* output_attention_tensor() const { return output_attention_; }
    MPSGraphExecutable* executable() const { return executable_; }

    int num_patches() const { return num_patches_; }
    int enc_dim() const { return enc_dim_; }

    bool is_loaded() const { return is_loaded_; }
    bool is_compiled() const { return executable_ != nil; }

private:
    std::shared_ptr<MPSGraphContext> ctx_;
    RuntimeConfig config_;
    ModelConfig cfg_;

    // Graph
    MPSGraph* graph_ = nil;
    MPSGraphExecutable* executable_ = nil;
    bool is_loaded_ = false;

    // Placeholders
    MPSGraphTensor* input_ = nil;  // [N, enc_dim] FP16

    // Output tensors
    MPSGraphTensor* output_features_ = nil;   // [N, enc_dim] FP32
    MPSGraphTensor* output_attention_ = nil;  // [N] FP32

    // Dimensions
    int num_patches_ = 0;
    int enc_dim_ = 0;
};

}  // namespace mpsgraph
}  // namespace mast3r
