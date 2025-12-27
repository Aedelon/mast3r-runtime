// MASt3R Runtime - Encoder Graph (Partitioned)
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
// Standalone encoder graph: patch_embed → transformer blocks → norm.
// Shared between inference (→ DecoderDPT) and retrieval (→ Whitening).

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "graph_builder.hpp"
#include "mpsgraph_context.hpp"
#include <memory>
#include <string>

namespace mast3r {
namespace mpsgraph {

/**
 * Pre-allocated encoder output buffer (zero-copy pattern).
 */
struct EncoderOutputBuffer {
    id<MTLBuffer> buf = nil;
    MPSGraphTensorData* td = nil;
    void* ptr = nullptr;  // Direct pointer (FP16 data)
};

/**
 * Partitioned Encoder Graph for pipelining.
 *
 * Input: uint8 image [1, H, W, 3]
 * Output: encoder features [N, enc_dim] in FP16
 *
 * This graph is shared between:
 * - Inference pipeline: Encoder → DecoderDPT
 * - Retrieval pipeline: Encoder → Whitening
 *
 * Pipelining benefit:
 * - Encoder[N+1] can run async while DecoderDPT[N] processes
 */
class API_AVAILABLE(macos(15.0)) EncoderGraph {
public:
    explicit EncoderGraph(std::shared_ptr<MPSGraphContext> ctx, const RuntimeConfig& config);
    ~EncoderGraph() = default;

    // Load weights from safetensors
    void load(safetensors::MultiSafetensorsFile& files);

    // Compile graph for faster execution
    void compile();

    // Run encoder - returns GPU tensor (lazy copy)
    // Output: [N, enc_dim] FP16 tensor on GPU
    MPSGraphTensorData* run(MPSGraphTensorData* input);

    // Run async - returns immediately, use completion handler
    void run_async(MPSGraphTensorData* input,
                   void (^completion)(MPSGraphTensorData* output));

    // Allocate output buffer for zero-copy pattern
    EncoderOutputBuffer allocate_output_buffer();

    // Run with pre-allocated buffer (zero allocation per inference)
    void run_into(MPSGraphTensorData* input, EncoderOutputBuffer& buffer);

    // Run async with pre-allocated buffer
    void run_async_into(MPSGraphTensorData* input, EncoderOutputBuffer& buffer,
                        void (^completion)(void));

    // Accessors
    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input_placeholder() const { return input_; }
    MPSGraphTensor* output_tensor() const { return output_; }
    MPSGraphExecutable* executable() const { return executable_; }

    int num_patches() const { return num_patches_; }
    int enc_dim() const { return cfg_.enc_dim; }
    int height() const { return img_h_; }
    int width() const { return img_w_; }

    bool is_compiled() const { return executable_ != nil; }

private:
    std::shared_ptr<MPSGraphContext> ctx_;
    RuntimeConfig config_;
    ModelConfig cfg_;

    // Graph
    MPSGraph* graph_ = nil;
    MPSGraphExecutable* executable_ = nil;

    // Placeholders
    MPSGraphTensor* input_ = nil;   // [1, H, W, 3] uint8
    MPSGraphTensor* output_ = nil;  // [N, enc_dim] FP16

    // Dimensions
    int img_h_ = 0;
    int img_w_ = 0;
    int patch_h_ = 0;
    int patch_w_ = 0;
    int num_patches_ = 0;
};

}  // namespace mpsgraph
}  // namespace mast3r
