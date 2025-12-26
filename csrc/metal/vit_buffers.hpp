// MASt3R Runtime - ViT GPU Buffers
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#pragma once

#import <Metal/Metal.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.hpp"

namespace mast3r {
namespace metal {

/**
 * GPU buffer storage for ViT encoder/decoder weights.
 *
 * Organizes weights by layer for efficient access during forward pass.
 * Supports both DUNE (DINOv2) and MASt3R (CroCoNet) architectures.
 */
class ViTBuffers {
public:
    ViTBuffers(id<MTLDevice> device, const ModelSpec& spec);
    ~ViTBuffers();

    // Non-copyable
    ViTBuffers(const ViTBuffers&) = delete;
    ViTBuffers& operator=(const ViTBuffers&) = delete;

    // Move semantics
    ViTBuffers(ViTBuffers&&) noexcept;
    ViTBuffers& operator=(ViTBuffers&&) noexcept;

    /**
     * Upload weights from CPU to GPU buffers.
     * Handles dtype conversion (BF16 -> F16 for Metal).
     */
    void upload(const ModelWeights& weights);

    // Patch embedding
    id<MTLBuffer> patch_embed_weight() const { return patch_embed_weight_; }
    id<MTLBuffer> patch_embed_bias() const { return patch_embed_bias_; }

    // Position embedding (DUNE only)
    id<MTLBuffer> pos_embed() const { return pos_embed_; }

    // CLS token (if present)
    id<MTLBuffer> cls_token() const { return cls_token_; }

    // Encoder layer i
    id<MTLBuffer> encoder_norm1_weight(int layer) const;
    id<MTLBuffer> encoder_norm1_bias(int layer) const;
    id<MTLBuffer> encoder_norm2_weight(int layer) const;
    id<MTLBuffer> encoder_norm2_bias(int layer) const;
    id<MTLBuffer> encoder_qkv_weight(int layer) const;
    id<MTLBuffer> encoder_qkv_bias(int layer) const;
    id<MTLBuffer> encoder_proj_weight(int layer) const;
    id<MTLBuffer> encoder_proj_bias(int layer) const;
    id<MTLBuffer> encoder_mlp_fc1_weight(int layer) const;
    id<MTLBuffer> encoder_mlp_fc1_bias(int layer) const;
    id<MTLBuffer> encoder_mlp_fc2_weight(int layer) const;
    id<MTLBuffer> encoder_mlp_fc2_bias(int layer) const;

    // Decoder layer i (dec_blocks - for image 1)
    id<MTLBuffer> decoder_norm1_weight(int layer) const;
    id<MTLBuffer> decoder_norm1_bias(int layer) const;
    id<MTLBuffer> decoder_norm2_weight(int layer) const;
    id<MTLBuffer> decoder_norm2_bias(int layer) const;
    id<MTLBuffer> decoder_qkv_weight(int layer) const;
    id<MTLBuffer> decoder_qkv_bias(int layer) const;
    id<MTLBuffer> decoder_proj_weight(int layer) const;
    id<MTLBuffer> decoder_proj_bias(int layer) const;
    id<MTLBuffer> decoder_mlp_fc1_weight(int layer) const;
    id<MTLBuffer> decoder_mlp_fc1_bias(int layer) const;
    id<MTLBuffer> decoder_mlp_fc2_weight(int layer) const;
    id<MTLBuffer> decoder_mlp_fc2_bias(int layer) const;

    // Cross-attention (decoder to encoder)
    id<MTLBuffer> decoder_cross_norm_weight(int layer) const;
    id<MTLBuffer> decoder_cross_norm_bias(int layer) const;
    id<MTLBuffer> decoder_cross_q_weight(int layer) const;
    id<MTLBuffer> decoder_cross_q_bias(int layer) const;
    id<MTLBuffer> decoder_cross_kv_weight(int layer) const;
    id<MTLBuffer> decoder_cross_kv_bias(int layer) const;
    id<MTLBuffer> decoder_cross_proj_weight(int layer) const;
    id<MTLBuffer> decoder_cross_proj_bias(int layer) const;

    // Decoder2 layer i (dec_blocks2 - for image 2, MASt3R only)
    id<MTLBuffer> decoder2_norm1_weight(int layer) const;
    id<MTLBuffer> decoder2_norm1_bias(int layer) const;
    id<MTLBuffer> decoder2_norm2_weight(int layer) const;
    id<MTLBuffer> decoder2_norm2_bias(int layer) const;
    id<MTLBuffer> decoder2_qkv_weight(int layer) const;
    id<MTLBuffer> decoder2_qkv_bias(int layer) const;
    id<MTLBuffer> decoder2_proj_weight(int layer) const;
    id<MTLBuffer> decoder2_proj_bias(int layer) const;
    id<MTLBuffer> decoder2_mlp_fc1_weight(int layer) const;
    id<MTLBuffer> decoder2_mlp_fc1_bias(int layer) const;
    id<MTLBuffer> decoder2_mlp_fc2_weight(int layer) const;
    id<MTLBuffer> decoder2_mlp_fc2_bias(int layer) const;
    id<MTLBuffer> decoder2_cross_norm_weight(int layer) const;
    id<MTLBuffer> decoder2_cross_norm_bias(int layer) const;
    id<MTLBuffer> decoder2_cross_q_weight(int layer) const;
    id<MTLBuffer> decoder2_cross_q_bias(int layer) const;
    id<MTLBuffer> decoder2_cross_kv_weight(int layer) const;
    id<MTLBuffer> decoder2_cross_kv_bias(int layer) const;
    id<MTLBuffer> decoder2_cross_proj_weight(int layer) const;
    id<MTLBuffer> decoder2_cross_proj_bias(int layer) const;

    // Check if dec_blocks2 is available (MASt3R only)
    bool has_decoder2() const { return !decoder_layers2_.empty(); }

    // =========================================================================
    // DPT (Dense Prediction Transformer) weights
    // =========================================================================

    // act_postprocess[i] - channel projection + scale adjustment
    // [0]: Conv1x1 + ConvTranspose4x4 (4x upsample)
    // [1]: Conv1x1 + ConvTranspose2x2 (2x upsample)
    // [2]: Conv1x1 only (same scale)
    // [3]: Conv1x1 + Conv3x3 stride2 (0.5x downsample)
    struct ActPostprocess {
        id<MTLBuffer> conv1_weight = nil;  // Conv1x1
        id<MTLBuffer> conv1_bias = nil;
        id<MTLBuffer> conv2_weight = nil;  // ConvTranspose or Conv3x3
        id<MTLBuffer> conv2_bias = nil;
    };
    const ActPostprocess& dpt_act_postprocess(int idx) const { return dpt_act_postprocess_[idx]; }

    // scratch.layer_rn[i] - Conv3x3 projection to feature_dim (256)
    id<MTLBuffer> dpt_layer_rn_weight(int idx) const { return dpt_layer_rn_[idx]; }

    // refinenet[i] - FeatureFusionBlock
    struct RefineNet {
        // resConfUnit1: 2x Conv3x3
        id<MTLBuffer> rcu1_conv1_weight = nil;
        id<MTLBuffer> rcu1_conv1_bias = nil;
        id<MTLBuffer> rcu1_conv2_weight = nil;
        id<MTLBuffer> rcu1_conv2_bias = nil;
        // resConfUnit2: 2x Conv3x3
        id<MTLBuffer> rcu2_conv1_weight = nil;
        id<MTLBuffer> rcu2_conv1_bias = nil;
        id<MTLBuffer> rcu2_conv2_weight = nil;
        id<MTLBuffer> rcu2_conv2_bias = nil;
        // out_conv: Conv1x1
        id<MTLBuffer> out_conv_weight = nil;
        id<MTLBuffer> out_conv_bias = nil;
    };
    const RefineNet& dpt_refinenet(int idx) const { return dpt_refinenet_[idx]; }

    // DPT head (regression): Conv3x3 -> Upsample -> Conv3x3 -> ReLU -> Conv1x1
    struct DPTHead {
        id<MTLBuffer> conv1_weight = nil;  // [128, 256, 3, 3]
        id<MTLBuffer> conv1_bias = nil;
        id<MTLBuffer> conv2_weight = nil;  // [128, 128, 3, 3]
        id<MTLBuffer> conv2_bias = nil;
        id<MTLBuffer> conv3_weight = nil;  // [4, 128, 1, 1] - outputs pts3d(3) + conf(1)
        id<MTLBuffer> conv3_bias = nil;
    };
    const DPTHead& dpt_head() const { return dpt_head_; }

    // Local features MLP: fc1 -> GELU -> fc2 -> pixel_shuffle
    struct LocalFeaturesMLP {
        id<MTLBuffer> fc1_weight = nil;  // [hidden, in_dim]
        id<MTLBuffer> fc1_bias = nil;
        id<MTLBuffer> fc2_weight = nil;  // [out_dim, hidden]
        id<MTLBuffer> fc2_bias = nil;
    };
    const LocalFeaturesMLP& local_features_mlp() const { return local_features_mlp_; }

    // RoPE frequencies (MASt3R only)
    id<MTLBuffer> rope_freqs() const { return rope_freqs_; }

    bool is_uploaded() const { return is_uploaded_; }

    // DPT hooks configuration
    static constexpr int DPT_NUM_HOOKS = 4;
    const int* dpt_hooks() const { return dpt_hooks_; }

    // Weight precision (detected from model)
    bool is_fp16() const { return weights_fp16_; }

private:
    id<MTLDevice> device_;
    ModelSpec spec_;
    bool is_uploaded_ = false;
    bool weights_fp16_ = false;  // true if weights are FP16, false if FP32

    // DPT hook indices (which decoder layers to tap)
    int dpt_hooks_[4] = {0, 6, 9, 12};  // For decoder_depth=12

    // Helper to create buffer from tensor
    id<MTLBuffer> create_buffer_from_tensor(const Tensor& tensor);

    // Patch embedding
    id<MTLBuffer> patch_embed_weight_ = nil;
    id<MTLBuffer> patch_embed_bias_ = nil;
    id<MTLBuffer> pos_embed_ = nil;
    id<MTLBuffer> cls_token_ = nil;

    // Encoder layers [depth]
    struct LayerBuffers {
        id<MTLBuffer> norm1_weight = nil;
        id<MTLBuffer> norm1_bias = nil;
        id<MTLBuffer> norm2_weight = nil;
        id<MTLBuffer> norm2_bias = nil;
        id<MTLBuffer> qkv_weight = nil;
        id<MTLBuffer> qkv_bias = nil;
        id<MTLBuffer> proj_weight = nil;
        id<MTLBuffer> proj_bias = nil;
        id<MTLBuffer> mlp_fc1_weight = nil;
        id<MTLBuffer> mlp_fc1_bias = nil;
        id<MTLBuffer> mlp_fc2_weight = nil;
        id<MTLBuffer> mlp_fc2_bias = nil;
    };

    struct DecoderLayerBuffers : LayerBuffers {
        // Cross-attention (DUNE has separate projk/projv)
        id<MTLBuffer> cross_norm_weight = nil;
        id<MTLBuffer> cross_norm_bias = nil;
        id<MTLBuffer> cross_q_weight = nil;
        id<MTLBuffer> cross_q_bias = nil;
        id<MTLBuffer> cross_kv_weight = nil;   // projk for DUNE
        id<MTLBuffer> cross_kv_bias = nil;
        id<MTLBuffer> cross_v_weight = nil;    // projv for DUNE (separate)
        id<MTLBuffer> cross_v_bias = nil;
        id<MTLBuffer> cross_proj_weight = nil;
        id<MTLBuffer> cross_proj_bias = nil;
    };

    std::vector<LayerBuffers> encoder_layers_;
    std::vector<DecoderLayerBuffers> decoder_layers_;
    std::vector<DecoderLayerBuffers> decoder_layers2_;  // dec_blocks2 for MASt3R

    // DPT weights
    ActPostprocess dpt_act_postprocess_[4];
    id<MTLBuffer> dpt_layer_rn_[4] = {nil, nil, nil, nil};  // No bias (groups=1)
    RefineNet dpt_refinenet_[4];
    DPTHead dpt_head_;
    LocalFeaturesMLP local_features_mlp_;

    // RoPE
    id<MTLBuffer> rope_freqs_ = nil;
};

}  // namespace metal
}  // namespace mast3r
