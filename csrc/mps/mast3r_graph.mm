// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
// Complete MASt3R model using MPSGraph with native SDPA

#import "mast3r_graph.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include "../common/safetensors.hpp"
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace mast3r {
namespace mpsgraph {

// ============================================================================
// Constants
// ============================================================================

// Encoder (ViT-Large)
constexpr int ENC_EMBED_DIM = 1024;
constexpr int ENC_NUM_HEADS = 16;
constexpr int ENC_HEAD_DIM = 64;
constexpr int ENC_MLP_DIM = 4096;
constexpr int ENC_DEPTH = 24;

// Decoder
constexpr int DEC_EMBED_DIM = 768;
constexpr int DEC_NUM_HEADS = 12;
constexpr int DEC_HEAD_DIM = 64;
constexpr int DEC_MLP_DIM = 3072;
constexpr int DEC_DEPTH = 12;

// Common
constexpr int PATCH_SIZE = 16;
constexpr float LN_EPS = 1e-6f;

// ============================================================================
// Helper macros
// ============================================================================

#define TENSOR_FROM_DATA(data, shape) \
    [graph_ constantWithData:[NSData dataWithBytes:(data).data() \
                                            length:(data).size() * sizeof(float)] \
                       shape:shape \
                    dataType:MPSDataTypeFloat32]

// ============================================================================
// Implementation
// ============================================================================

class API_AVAILABLE(macos(15.0)) MASt3RGraph::Impl {
public:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraph* graph_ = nil;

    ModelSpec spec_;
    bool initialized_ = false;
    bool weights_loaded_ = false;

    // Timing
    std::unordered_map<std::string, double> timings_;

    // ========================================================================
    // Encoder weights
    // ========================================================================
    struct EncoderWeights {
        MPSGraphTensor* patch_proj_weight = nil;  // [D, 3, P, P]
        MPSGraphTensor* patch_proj_bias = nil;
        MPSGraphTensor* enc_norm_weight = nil;
        MPSGraphTensor* enc_norm_bias = nil;

        struct Layer {
            MPSGraphTensor* norm1_w = nil;
            MPSGraphTensor* norm1_b = nil;
            MPSGraphTensor* qkv_w = nil;
            MPSGraphTensor* qkv_b = nil;
            MPSGraphTensor* proj_w = nil;
            MPSGraphTensor* proj_b = nil;
            MPSGraphTensor* norm2_w = nil;
            MPSGraphTensor* norm2_b = nil;
            MPSGraphTensor* fc1_w = nil;
            MPSGraphTensor* fc1_b = nil;
            MPSGraphTensor* fc2_w = nil;
            MPSGraphTensor* fc2_b = nil;
        };
        std::vector<Layer> layers;
    } enc_weights_;

    // ========================================================================
    // Decoder weights
    // ========================================================================
    struct DecoderWeights {
        MPSGraphTensor* dec_norm_weight = nil;
        MPSGraphTensor* dec_norm_bias = nil;

        struct Layer {
            // Self-attention
            MPSGraphTensor* norm1_w = nil;
            MPSGraphTensor* norm1_b = nil;
            MPSGraphTensor* qkv_w = nil;
            MPSGraphTensor* qkv_b = nil;
            MPSGraphTensor* proj_w = nil;
            MPSGraphTensor* proj_b = nil;

            // Cross-attention
            MPSGraphTensor* norm_y_w = nil;  // For encoder features
            MPSGraphTensor* norm_y_b = nil;
            MPSGraphTensor* norm3_w = nil;   // For decoder query
            MPSGraphTensor* norm3_b = nil;
            MPSGraphTensor* cross_q_w = nil;
            MPSGraphTensor* cross_q_b = nil;
            MPSGraphTensor* cross_k_w = nil;
            MPSGraphTensor* cross_k_b = nil;
            MPSGraphTensor* cross_v_w = nil;
            MPSGraphTensor* cross_v_b = nil;
            MPSGraphTensor* cross_proj_w = nil;
            MPSGraphTensor* cross_proj_b = nil;

            // MLP
            MPSGraphTensor* norm2_w = nil;
            MPSGraphTensor* norm2_b = nil;
            MPSGraphTensor* fc1_w = nil;
            MPSGraphTensor* fc1_b = nil;
            MPSGraphTensor* fc2_w = nil;
            MPSGraphTensor* fc2_b = nil;
        };
        std::vector<Layer> layers;
    } dec_weights_, dec2_weights_;

    // Graph I/O tensors
    MPSGraphTensor* input_tensor_ = nil;  // [B, H, W, 3]
    MPSGraphTensor* output_pts3d_ = nil;
    MPSGraphTensor* output_conf_ = nil;
    MPSGraphTensor* output_desc_ = nil;

    // ========================================================================
    // Initialization
    // ========================================================================

    Impl() {
        device_ = MTLCreateSystemDefaultDevice();
        if (device_) {
            queue_ = [device_ newCommandQueue];
            graph_ = [[MPSGraph alloc] init];
        }
    }

    bool initialize(const ModelSpec& spec) {
        if (!device_ || !graph_) {
            NSLog(@"[mpsgraph] No Metal device");
            return false;
        }

        spec_ = spec;
        enc_weights_.layers.resize(ENC_DEPTH);
        dec_weights_.layers.resize(DEC_DEPTH);
        dec2_weights_.layers.resize(DEC_DEPTH);

        initialized_ = true;
        NSLog(@"[mpsgraph] Initialized MASt3RGraph for %s", spec_.name.c_str());
        return true;
    }

    // ========================================================================
    // Weight loading
    // ========================================================================

    bool load_weights(const std::string& path) {
        if (!initialized_) return false;

        try {
            safetensors::SafetensorsFile file(path);
            NSLog(@"[mpsgraph] Loading %zu tensors from %s",
                  file.num_tensors(), path.c_str());

            load_encoder_weights(file);
            load_decoder_weights(file, "dec_blocks", dec_weights_);
            load_decoder_weights(file, "dec_blocks2", dec2_weights_);

            weights_loaded_ = true;
            NSLog(@"[mpsgraph] Weights loaded successfully");
            return true;
        } catch (const std::exception& e) {
            NSLog(@"[mpsgraph] Failed to load weights: %s", e.what());
            return false;
        }
    }

    void load_encoder_weights(const safetensors::SafetensorsFile& file) {
        // Patch embedding
        auto patch_w = file.load_tensor_f32("patch_embed.proj.weight");
        enc_weights_.patch_proj_weight = TENSOR_FROM_DATA(patch_w,
            (@[@(ENC_EMBED_DIM), @3, @(PATCH_SIZE), @(PATCH_SIZE)]));
        enc_weights_.patch_proj_bias = TENSOR_FROM_DATA(
            file.load_tensor_f32("patch_embed.proj.bias"), (@[@(ENC_EMBED_DIM)]));

        // Final norm
        enc_weights_.enc_norm_weight = TENSOR_FROM_DATA(
            file.load_tensor_f32("enc_norm.weight"), (@[@(ENC_EMBED_DIM)]));
        enc_weights_.enc_norm_bias = TENSOR_FROM_DATA(
            file.load_tensor_f32("enc_norm.bias"), (@[@(ENC_EMBED_DIM)]));

        // Encoder layers
        for (int i = 0; i < ENC_DEPTH; i++) {
            std::string p = "enc_blocks." + std::to_string(i) + ".";
            auto& L = enc_weights_.layers[i];

            L.norm1_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm1.weight"),
                                         (@[@(ENC_EMBED_DIM)]));
            L.norm1_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm1.bias"),
                                         (@[@(ENC_EMBED_DIM)]));
            L.qkv_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "attn.qkv.weight"),
                                       (@[@(3 * ENC_EMBED_DIM), @(ENC_EMBED_DIM)]));
            L.qkv_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "attn.qkv.bias"),
                                       (@[@(3 * ENC_EMBED_DIM)]));
            L.proj_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "attn.proj.weight"),
                                        (@[@(ENC_EMBED_DIM), @(ENC_EMBED_DIM)]));
            L.proj_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "attn.proj.bias"),
                                        (@[@(ENC_EMBED_DIM)]));
            L.norm2_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm2.weight"),
                                         (@[@(ENC_EMBED_DIM)]));
            L.norm2_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm2.bias"),
                                         (@[@(ENC_EMBED_DIM)]));
            L.fc1_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "mlp.fc1.weight"),
                                       (@[@(ENC_MLP_DIM), @(ENC_EMBED_DIM)]));
            L.fc1_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "mlp.fc1.bias"),
                                       (@[@(ENC_MLP_DIM)]));
            L.fc2_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "mlp.fc2.weight"),
                                       (@[@(ENC_EMBED_DIM), @(ENC_MLP_DIM)]));
            L.fc2_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "mlp.fc2.bias"),
                                       (@[@(ENC_EMBED_DIM)]));
        }
    }

    void load_decoder_weights(const safetensors::SafetensorsFile& file,
                              const std::string& prefix,
                              DecoderWeights& weights) {
        // Check if this decoder exists
        std::string test_key = prefix + ".0.norm1.weight";
        if (!file.has_tensor(test_key)) {
            NSLog(@"[mpsgraph] Decoder %s not found, skipping", prefix.c_str());
            return;
        }

        for (int i = 0; i < DEC_DEPTH; i++) {
            std::string p = prefix + "." + std::to_string(i) + ".";
            auto& L = weights.layers[i];

            // Self-attention
            L.norm1_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm1.weight"),
                                         (@[@(DEC_EMBED_DIM)]));
            L.norm1_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm1.bias"),
                                         (@[@(DEC_EMBED_DIM)]));
            L.qkv_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "attn.qkv.weight"),
                                       (@[@(3 * DEC_EMBED_DIM), @(DEC_EMBED_DIM)]));
            L.qkv_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "attn.qkv.bias"),
                                       (@[@(3 * DEC_EMBED_DIM)]));
            L.proj_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "attn.proj.weight"),
                                        (@[@(DEC_EMBED_DIM), @(DEC_EMBED_DIM)]));
            L.proj_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "attn.proj.bias"),
                                        (@[@(DEC_EMBED_DIM)]));

            // Cross-attention
            L.norm_y_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm_y.weight"),
                                          (@[@(DEC_EMBED_DIM)]));
            L.norm_y_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm_y.bias"),
                                          (@[@(DEC_EMBED_DIM)]));
            L.norm3_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm3.weight"),
                                         (@[@(DEC_EMBED_DIM)]));
            L.norm3_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm3.bias"),
                                         (@[@(DEC_EMBED_DIM)]));
            L.cross_q_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "cross_attn.projq.weight"),
                                           (@[@(DEC_EMBED_DIM), @(DEC_EMBED_DIM)]));
            L.cross_q_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "cross_attn.projq.bias"),
                                           (@[@(DEC_EMBED_DIM)]));
            L.cross_k_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "cross_attn.projk.weight"),
                                           (@[@(DEC_EMBED_DIM), @(DEC_EMBED_DIM)]));
            L.cross_k_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "cross_attn.projk.bias"),
                                           (@[@(DEC_EMBED_DIM)]));
            L.cross_v_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "cross_attn.projv.weight"),
                                           (@[@(DEC_EMBED_DIM), @(DEC_EMBED_DIM)]));
            L.cross_v_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "cross_attn.projv.bias"),
                                           (@[@(DEC_EMBED_DIM)]));
            L.cross_proj_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "cross_attn.proj.weight"),
                                              (@[@(DEC_EMBED_DIM), @(DEC_EMBED_DIM)]));
            L.cross_proj_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "cross_attn.proj.bias"),
                                              (@[@(DEC_EMBED_DIM)]));

            // MLP
            L.norm2_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm2.weight"),
                                         (@[@(DEC_EMBED_DIM)]));
            L.norm2_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "norm2.bias"),
                                         (@[@(DEC_EMBED_DIM)]));
            L.fc1_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "mlp.fc1.weight"),
                                       (@[@(DEC_MLP_DIM), @(DEC_EMBED_DIM)]));
            L.fc1_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "mlp.fc1.bias"),
                                       (@[@(DEC_MLP_DIM)]));
            L.fc2_w = TENSOR_FROM_DATA(file.load_tensor_f32(p + "mlp.fc2.weight"),
                                       (@[@(DEC_EMBED_DIM), @(DEC_MLP_DIM)]));
            L.fc2_b = TENSOR_FROM_DATA(file.load_tensor_f32(p + "mlp.fc2.bias"),
                                       (@[@(DEC_EMBED_DIM)]));
        }
    }

    // ========================================================================
    // Graph operations
    // ========================================================================

    // LayerNorm
    MPSGraphTensor* layer_norm(MPSGraphTensor* x,
                               MPSGraphTensor* weight,
                               MPSGraphTensor* bias) {
        MPSGraphTensor* mean = [graph_ meanOfTensor:x axes:@[@(-1)] name:nil];
        MPSGraphTensor* centered = [graph_ subtractionWithPrimaryTensor:x
                                                        secondaryTensor:mean
                                                                   name:nil];
        MPSGraphTensor* sq = [graph_ squareWithTensor:centered name:nil];
        MPSGraphTensor* var = [graph_ meanOfTensor:sq axes:@[@(-1)] name:nil];

        MPSGraphTensor* eps = [graph_ constantWithScalar:LN_EPS
                                                   shape:@[@1]
                                                dataType:MPSDataTypeFloat32];
        MPSGraphTensor* std = [graph_ squareRootWithTensor:
                               [graph_ additionWithPrimaryTensor:var
                                                 secondaryTensor:eps
                                                            name:nil]
                                                      name:nil];

        MPSGraphTensor* normalized = [graph_ divisionWithPrimaryTensor:centered
                                                       secondaryTensor:std
                                                                  name:nil];
        normalized = [graph_ multiplicationWithPrimaryTensor:normalized
                                             secondaryTensor:weight
                                                        name:nil];
        return [graph_ additionWithPrimaryTensor:normalized
                                 secondaryTensor:bias
                                            name:nil];
    }

    // GELU activation
    MPSGraphTensor* gelu(MPSGraphTensor* x) {
        MPSGraphTensor* inv_sqrt2 = [graph_ constantWithScalar:0.7071067811865475
                                                         shape:@[@1]
                                                      dataType:MPSDataTypeFloat32];
        MPSGraphTensor* half = [graph_ constantWithScalar:0.5
                                                    shape:@[@1]
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensor* one = [graph_ constantWithScalar:1.0
                                                   shape:@[@1]
                                                dataType:MPSDataTypeFloat32];

        MPSGraphTensor* x_scaled = [graph_ multiplicationWithPrimaryTensor:x
                                                           secondaryTensor:inv_sqrt2
                                                                      name:nil];
        MPSGraphTensor* erf_val = [graph_ erfWithTensor:x_scaled name:nil];
        MPSGraphTensor* one_plus_erf = [graph_ additionWithPrimaryTensor:one
                                                         secondaryTensor:erf_val
                                                                    name:nil];
        MPSGraphTensor* cdf = [graph_ multiplicationWithPrimaryTensor:half
                                                      secondaryTensor:one_plus_erf
                                                                 name:nil];
        return [graph_ multiplicationWithPrimaryTensor:x secondaryTensor:cdf name:nil];
    }

    // Linear projection (matmul + bias)
    MPSGraphTensor* linear(MPSGraphTensor* x,
                           MPSGraphTensor* weight,
                           MPSGraphTensor* bias) {
        // weight is [out, in], need transpose for x @ W^T
        MPSGraphTensor* wt = [graph_ transposeTensor:weight
                                           dimension:0
                                       withDimension:1
                                                name:nil];
        MPSGraphTensor* out = [graph_ matrixMultiplicationWithPrimaryTensor:x
                                                            secondaryTensor:wt
                                                                       name:nil];
        return [graph_ additionWithPrimaryTensor:out secondaryTensor:bias name:nil];
    }

    // Self-attention with SDPA
    MPSGraphTensor* self_attention(MPSGraphTensor* x,
                                   MPSGraphTensor* qkv_w,
                                   MPSGraphTensor* qkv_b,
                                   MPSGraphTensor* proj_w,
                                   MPSGraphTensor* proj_b,
                                   int num_heads,
                                   int head_dim) {
        const float scale = 1.0f / sqrtf((float)head_dim);
        int embed_dim = num_heads * head_dim;

        // QKV projection
        MPSGraphTensor* qkv = linear(x, qkv_w, qkv_b);

        // Split Q, K, V
        NSArray<MPSGraphTensor*>* splits = [graph_ splitTensor:qkv
                                                     numSplits:3
                                                          axis:-1
                                                          name:nil];

        // Reshape for attention: [N, D] -> [1, H, N, head_dim]
        NSArray<NSNumber*>* attn_shape = @[@1, @(num_heads), @(-1), @(head_dim)];
        MPSGraphTensor* Q = [graph_ reshapeTensor:splits[0] withShape:attn_shape name:nil];
        MPSGraphTensor* K = [graph_ reshapeTensor:splits[1] withShape:attn_shape name:nil];
        MPSGraphTensor* V = [graph_ reshapeTensor:splits[2] withShape:attn_shape name:nil];

        // SDPA
        MPSGraphTensor* attn_out = [graph_ scaledDotProductAttentionWithQueryTensor:Q
                                                                          keyTensor:K
                                                                        valueTensor:V
                                                                         maskTensor:nil
                                                                              scale:scale
                                                                               name:nil];

        // Reshape back: [1, H, N, head_dim] -> [N, D]
        attn_out = [graph_ reshapeTensor:attn_out
                               withShape:@[@(-1), @(embed_dim)]
                                    name:nil];

        // Output projection
        return linear(attn_out, proj_w, proj_b);
    }

    // Cross-attention with SDPA
    MPSGraphTensor* cross_attention(MPSGraphTensor* x,      // Decoder query
                                    MPSGraphTensor* y,      // Encoder memory
                                    MPSGraphTensor* q_w, MPSGraphTensor* q_b,
                                    MPSGraphTensor* k_w, MPSGraphTensor* k_b,
                                    MPSGraphTensor* v_w, MPSGraphTensor* v_b,
                                    MPSGraphTensor* proj_w, MPSGraphTensor* proj_b,
                                    int num_heads,
                                    int head_dim) {
        const float scale = 1.0f / sqrtf((float)head_dim);
        int embed_dim = num_heads * head_dim;

        // Separate Q (from decoder), K/V (from encoder)
        MPSGraphTensor* Q_flat = linear(x, q_w, q_b);
        MPSGraphTensor* K_flat = linear(y, k_w, k_b);
        MPSGraphTensor* V_flat = linear(y, v_w, v_b);

        // Reshape for attention
        NSArray<NSNumber*>* attn_shape = @[@1, @(num_heads), @(-1), @(head_dim)];
        MPSGraphTensor* Q = [graph_ reshapeTensor:Q_flat withShape:attn_shape name:nil];
        MPSGraphTensor* K = [graph_ reshapeTensor:K_flat withShape:attn_shape name:nil];
        MPSGraphTensor* V = [graph_ reshapeTensor:V_flat withShape:attn_shape name:nil];

        // SDPA
        MPSGraphTensor* attn_out = [graph_ scaledDotProductAttentionWithQueryTensor:Q
                                                                          keyTensor:K
                                                                        valueTensor:V
                                                                         maskTensor:nil
                                                                              scale:scale
                                                                               name:nil];

        // Reshape back
        attn_out = [graph_ reshapeTensor:attn_out
                               withShape:@[@(-1), @(embed_dim)]
                                    name:nil];

        return linear(attn_out, proj_w, proj_b);
    }

    // ========================================================================
    // Encoder
    // ========================================================================

    MPSGraphTensor* encoder_forward(MPSGraphTensor* patches) {
        // patches: [N, D] where N = num_patches, D = ENC_EMBED_DIM
        MPSGraphTensor* x = patches;

        for (int i = 0; i < ENC_DEPTH; i++) {
            auto& L = enc_weights_.layers[i];

            // Self-attention
            MPSGraphTensor* residual = x;
            x = layer_norm(x, L.norm1_w, L.norm1_b);
            x = self_attention(x, L.qkv_w, L.qkv_b, L.proj_w, L.proj_b,
                               ENC_NUM_HEADS, ENC_HEAD_DIM);
            x = [graph_ additionWithPrimaryTensor:x secondaryTensor:residual name:nil];

            // MLP
            residual = x;
            x = layer_norm(x, L.norm2_w, L.norm2_b);
            x = linear(x, L.fc1_w, L.fc1_b);
            x = gelu(x);
            x = linear(x, L.fc2_w, L.fc2_b);
            x = [graph_ additionWithPrimaryTensor:x secondaryTensor:residual name:nil];
        }

        // Final norm
        x = layer_norm(x, enc_weights_.enc_norm_weight, enc_weights_.enc_norm_bias);
        return x;
    }

    // ========================================================================
    // Decoder
    // ========================================================================

    MPSGraphTensor* decoder_forward(MPSGraphTensor* x,
                                    MPSGraphTensor* encoder_out,
                                    DecoderWeights& weights) {
        // x: [N, D_dec] - decoder input (projected from encoder)
        // encoder_out: [N, D_enc] - encoder output (need projection to D_dec)

        // Project encoder output to decoder dim
        // Note: In MASt3R, this is done via the cross-attention K/V projections

        for (int i = 0; i < DEC_DEPTH; i++) {
            auto& L = weights.layers[i];

            // Self-attention
            MPSGraphTensor* residual = x;
            x = layer_norm(x, L.norm1_w, L.norm1_b);
            x = self_attention(x, L.qkv_w, L.qkv_b, L.proj_w, L.proj_b,
                               DEC_NUM_HEADS, DEC_HEAD_DIM);
            x = [graph_ additionWithPrimaryTensor:x secondaryTensor:residual name:nil];

            // Cross-attention
            residual = x;
            MPSGraphTensor* x_normed = layer_norm(x, L.norm3_w, L.norm3_b);
            MPSGraphTensor* y_normed = layer_norm(encoder_out, L.norm_y_w, L.norm_y_b);
            MPSGraphTensor* cross_out = cross_attention(x_normed, y_normed,
                                                        L.cross_q_w, L.cross_q_b,
                                                        L.cross_k_w, L.cross_k_b,
                                                        L.cross_v_w, L.cross_v_b,
                                                        L.cross_proj_w, L.cross_proj_b,
                                                        DEC_NUM_HEADS, DEC_HEAD_DIM);
            x = [graph_ additionWithPrimaryTensor:cross_out secondaryTensor:residual name:nil];

            // MLP
            residual = x;
            x = layer_norm(x, L.norm2_w, L.norm2_b);
            x = linear(x, L.fc1_w, L.fc1_b);
            x = gelu(x);
            x = linear(x, L.fc2_w, L.fc2_b);
            x = [graph_ additionWithPrimaryTensor:x secondaryTensor:residual name:nil];
        }

        return x;
    }

    // ========================================================================
    // Inference
    // ========================================================================

    InferenceResult infer(const uint8_t* img1, const uint8_t* img2,
                          int width, int height) {
        // TODO: Implement image preprocessing
        // For now, just return empty result
        InferenceResult result;
        NSLog(@"[mpsgraph] Full inference not yet implemented");
        return result;
    }
};

// ============================================================================
// Public API
// ============================================================================

MASt3RGraph::MASt3RGraph() {
    if (@available(macOS 15.0, *)) {
        impl_ = std::make_unique<Impl>();
    }
}

MASt3RGraph::~MASt3RGraph() = default;

bool MASt3RGraph::initialize(const ModelSpec& spec) {
    if (@available(macOS 15.0, *)) {
        return impl_ ? impl_->initialize(spec) : false;
    }
    return false;
}

bool MASt3RGraph::load_weights(const std::string& path) {
    if (@available(macOS 15.0, *)) {
        return impl_ ? impl_->load_weights(path) : false;
    }
    return false;
}

InferenceResult MASt3RGraph::infer(const uint8_t* img1, const uint8_t* img2,
                                    int width, int height) {
    if (@available(macOS 15.0, *)) {
        if (impl_) return impl_->infer(img1, img2, width, height);
    }
    return InferenceResult{};
}

InferenceResult MASt3RGraph::infer_normalized(const float* img1, const float* img2,
                                               int width, int height) {
    // TODO: Implement
    return InferenceResult{};
}

bool MASt3RGraph::is_available() {
    if (@available(macOS 15.0, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
    return false;
}

std::unordered_map<std::string, double> MASt3RGraph::get_timings() const {
    if (@available(macOS 15.0, *)) {
        return impl_ ? impl_->timings_ : std::unordered_map<std::string, double>{};
    }
    return {};
}

}  // namespace mpsgraph
}  // namespace mast3r
