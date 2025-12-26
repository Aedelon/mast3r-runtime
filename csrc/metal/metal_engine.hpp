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

// Forward declaration
class ViTBuffers;

/**
 * Metal-based inference engine for Apple Silicon.
 *
 * Uses Metal compute shaders for ViT encoder/decoder inference.
 * Custom kernels for patch embedding, attention, and MLP.
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
    ModelSpec spec_;
    bool is_loaded_ = false;
    std::unique_ptr<ModelWeights> weights_;
    std::unique_ptr<ViTBuffers> vit_buffers_;

#ifdef __OBJC__
    // I/O buffers
    id<MTLBuffer> buffer_img1_ = nil;
    id<MTLBuffer> buffer_img2_ = nil;
    id<MTLBuffer> buffer_preprocessed1_ = nil;
    id<MTLBuffer> buffer_preprocessed2_ = nil;
    id<MTLBuffer> buffer_pts3d_1_ = nil;
    id<MTLBuffer> buffer_pts3d_2_ = nil;
    id<MTLBuffer> buffer_desc_1_ = nil;
    id<MTLBuffer> buffer_desc_2_ = nil;
    id<MTLBuffer> buffer_desc_half_1_ = nil;  // Half precision intermediate
    id<MTLBuffer> buffer_desc_half_2_ = nil;  // Half precision intermediate
    id<MTLBuffer> buffer_conf_1_ = nil;
    id<MTLBuffer> buffer_conf_2_ = nil;

    // Intermediate encoder buffers
    id<MTLBuffer> buffer_patches1_ = nil;    // [num_patches, embed_dim]
    id<MTLBuffer> buffer_patches2_ = nil;
    id<MTLBuffer> buffer_encoder1_ = nil;    // Encoder output
    id<MTLBuffer> buffer_encoder2_ = nil;
    id<MTLBuffer> buffer_decoder1_ = nil;    // Decoder output
    id<MTLBuffer> buffer_decoder2_ = nil;

    // Scratch buffers for attention
    id<MTLBuffer> buffer_qkv_ = nil;         // [num_patches, 3 * embed_dim]
    id<MTLBuffer> buffer_q_ = nil;           // [num_patches, num_heads, head_dim]
    id<MTLBuffer> buffer_k_ = nil;           // [num_patches, num_heads, head_dim]
    id<MTLBuffer> buffer_v_ = nil;           // [num_patches, num_heads, head_dim]
    id<MTLBuffer> buffer_attn_out_ = nil;    // [num_patches, embed_dim]
    id<MTLBuffer> buffer_attn_ = nil;        // [num_heads, num_patches, num_patches] (optional)
    id<MTLBuffer> buffer_mlp_ = nil;         // [num_patches, 4 * embed_dim]
    id<MTLBuffer> buffer_residual_ = nil;    // [num_patches, embed_dim] for residual

    // Cross-attention buffers (decoder)
    id<MTLBuffer> buffer_cross_q_ = nil;     // [num_patches, num_heads, head_dim]
    id<MTLBuffer> buffer_cross_k_ = nil;     // [num_patches, num_heads, head_dim]
    id<MTLBuffer> buffer_cross_v_ = nil;     // [num_patches, num_heads, head_dim]

    // RoPE 2D positions (for MASt3R)
    id<MTLBuffer> buffer_rope_positions_ = nil;  // [num_patches, 2]

    // Intermediate outputs at patch level (before upsampling)
    id<MTLBuffer> buffer_pts3d_patch1_ = nil;  // [num_patches, 3]
    id<MTLBuffer> buffer_pts3d_patch2_ = nil;
    id<MTLBuffer> buffer_desc_patch1_ = nil;   // [num_patches, desc_dim]
    id<MTLBuffer> buffer_desc_patch2_ = nil;
    id<MTLBuffer> buffer_conf_patch1_ = nil;   // [num_patches]
    id<MTLBuffer> buffer_conf_patch2_ = nil;

    // DPT intermediate buffers (for full DPT architecture)
    // Hooks from decoder at layers [0, 6, 9, 12]
    static constexpr int DPT_NUM_HOOKS = 4;
    id<MTLBuffer> buffer_dpt_hooks_[DPT_NUM_HOOKS] = {nil, nil, nil, nil};

    // After act_postprocess (different sizes due to up/downsampling)
    // All reshaped to same spatial size after processing
    id<MTLBuffer> buffer_dpt_postprocess_[DPT_NUM_HOOKS] = {nil, nil, nil, nil};

    // After layer_rn projection (all 256 channels)
    id<MTLBuffer> buffer_dpt_layer_rn_[DPT_NUM_HOOKS] = {nil, nil, nil, nil};

    // RefineNet outputs (256 channels each)
    id<MTLBuffer> buffer_dpt_refinenet_[DPT_NUM_HOOKS] = {nil, nil, nil, nil};

    // Scratch buffers for DPT convolutions
    id<MTLBuffer> buffer_dpt_scratch1_ = nil;  // [H, W, 256]
    id<MTLBuffer> buffer_dpt_scratch2_ = nil;  // [H, W, 256]
    id<MTLBuffer> buffer_dpt_head_temp_ = nil; // [H*2, W*2, 128] after upsample

    // Local features MLP intermediate
    id<MTLBuffer> buffer_local_hidden_ = nil;  // [num_patches, in_dim] concat input
    id<MTLBuffer> buffer_fc1_out_ = nil;       // [num_patches, hidden_dim] fc1 output
    id<MTLBuffer> buffer_local_shuffle_ = nil; // [pH*r, pW*r, local_feat_dim] after pixel shuffle

    // Run encoder forward pass on one image
    void run_encoder(id<MTLBuffer> input, id<MTLBuffer> output,
                     id<MTLBuffer> scratch_qkv, id<MTLBuffer> scratch_mlp);

    // Run decoder forward pass with hook outputs
    // use_decoder2: if true, use dec_blocks2 weights (for image 2 in MASt3R)
    void run_decoder(id<MTLBuffer> decoder_tokens,
                     id<MTLBuffer> encoder_output,
                     id<MTLBuffer> output,
                     id<MTLBuffer> scratch_qkv, id<MTLBuffer> scratch_mlp,
                     id<MTLBuffer> __unsafe_unretained * hook_outputs = nullptr,
                     bool use_decoder2 = false);

    // Run full DPT pipeline
    void run_dpt(id<MTLBuffer> encoder_output,
                 id<MTLBuffer> __unsafe_unretained * hook_outputs,  // [4] from decoder
                 id<MTLBuffer> pts3d_out,
                 id<MTLBuffer> conf_out);

    // Run local features MLP for descriptors
    void run_local_features_mlp(id<MTLBuffer> encoder_output,
                                id<MTLBuffer> decoder_output,
                                id<MTLBuffer> desc_out);

    // Run DPT heads (pts3d, desc, conf projections at patch level)
    void run_dpt_heads(id<MTLBuffer> decoder_output,
                       id<MTLBuffer> pts3d_out,
                       id<MTLBuffer> desc_out,
                       id<MTLBuffer> conf_out);

    // Upsample from patch level to pixel level
    void run_upsample(id<MTLBuffer> patch_pts3d, id<MTLBuffer> patch_desc, id<MTLBuffer> patch_conf,
                      id<MTLBuffer> pixel_pts3d, id<MTLBuffer> pixel_desc, id<MTLBuffer> pixel_conf);
#endif

    void allocate_buffers();
    void preprocess_gpu(const ImageView& img, void* output_buffer);
};

}  // namespace metal
}  // namespace mast3r