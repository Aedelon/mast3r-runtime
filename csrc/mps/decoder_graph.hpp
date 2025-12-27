// MASt3R Runtime - Decoder+DPT Graph (Partitioned)
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
// Decoder with cross-attention + DPT head + Local Features.
// Takes encoder output, produces pts3d, conf, descriptors.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "graph_builder.hpp"
#include "mpsgraph_context.hpp"
#include <memory>
#include <string>

namespace mast3r {
namespace mpsgraph {

/**
 * Decoder output structure.
 * All tensors are GPU-resident (lazy copy pattern).
 */
struct DecoderOutput {
    MPSGraphTensorData* pts3d;   // [1, H, W, 3] FP32
    MPSGraphTensorData* conf;    // [1, H, W] FP32
    MPSGraphTensorData* desc;    // [1, H, W, desc_dim] FP32
};

/**
 * Pre-allocated decoder output buffers (zero-copy pattern).
 * Allocated once, reused across inferences.
 */
struct DecoderOutputBuffers {
    id<MTLBuffer> buf_pts3d = nil;
    id<MTLBuffer> buf_conf = nil;
    id<MTLBuffer> buf_desc = nil;
    MPSGraphTensorData* td_pts3d = nil;
    MPSGraphTensorData* td_conf = nil;
    MPSGraphTensorData* td_desc = nil;
    float* ptr_pts3d = nullptr;  // Direct pointer for zero-copy access
    float* ptr_conf = nullptr;
    float* ptr_desc = nullptr;
};

/**
 * Partitioned Decoder+DPT Graph for pipelining.
 *
 * Input: encoder features [N, enc_dim] FP16
 * Output: pts3d [1,H,W,3], conf [1,H,W], desc [1,H,W,D]
 *
 * Contains:
 * - Decoder embed (enc_dim → dec_dim)
 * - Decoder blocks with self-attention + cross-attention
 * - DPT head (multi-scale feature fusion)
 * - Local features (descriptors)
 *
 * Pipelining benefit:
 * - Decoder[N] can run while Encoder[N+1] processes next image
 */
class API_AVAILABLE(macos(15.0)) DecoderGraph {
public:
    explicit DecoderGraph(std::shared_ptr<MPSGraphContext> ctx, const RuntimeConfig& config);
    ~DecoderGraph() = default;

    // Load weights from safetensors
    void load(safetensors::MultiSafetensorsFile& files);

    // Compile graph for faster execution
    void compile();

    // Run decoder - returns GPU tensors (lazy copy)
    DecoderOutput run(MPSGraphTensorData* enc_out);

    // Run async - returns immediately, use completion handler
    void run_async(MPSGraphTensorData* enc_out,
                   void (^completion)(DecoderOutput output));

    // Allocate output buffers for zero-copy pattern
    DecoderOutputBuffers allocate_output_buffers();

    // Run with pre-allocated buffers (zero allocation per inference)
    void run_into(MPSGraphTensorData* enc_out, DecoderOutputBuffers& buffers);

    // Run async with pre-allocated buffers
    void run_async_into(MPSGraphTensorData* enc_out, DecoderOutputBuffers& buffers,
                        void (^completion)(void));

    // Accessors
    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input_placeholder() const { return input_; }
    MPSGraphTensor* output_pts3d_tensor() const { return output_pts3d_; }
    MPSGraphTensor* output_conf_tensor() const { return output_conf_; }
    MPSGraphTensor* output_desc_tensor() const { return output_desc_; }
    MPSGraphExecutable* executable() const { return executable_; }

    int height() const { return img_h_; }
    int width() const { return img_w_; }
    int desc_dim() const { return cfg_.desc_dim; }

    bool is_compiled() const { return executable_ != nil; }

private:
    std::shared_ptr<MPSGraphContext> ctx_;
    RuntimeConfig config_;
    ModelConfig cfg_;

    // Graph
    MPSGraph* graph_ = nil;
    MPSGraphExecutable* executable_ = nil;

    // Placeholders
    MPSGraphTensor* input_ = nil;  // [N, enc_dim] FP16

    // Output tensors
    MPSGraphTensor* output_pts3d_ = nil;  // [1, H, W, 3] FP32
    MPSGraphTensor* output_conf_ = nil;   // [1, H, W] FP32
    MPSGraphTensor* output_desc_ = nil;   // [1, H, W, desc_dim] FP32

    // Dimensions
    int img_h_ = 0;
    int img_w_ = 0;
    int patch_h_ = 0;
    int patch_w_ = 0;
    int num_patches_ = 0;
};

}  // namespace mpsgraph
}  // namespace mast3r
