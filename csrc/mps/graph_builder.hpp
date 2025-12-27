// MASt3R Runtime - MPSGraph Builder Utilities
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
// Shared utilities for building MPSGraph operations.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "../common/safetensors.hpp"
#include "../common/types.hpp"
#include <cmath>
#include <string>

namespace mast3r {
namespace mpsgraph {

// ============================================================================
// Model Configuration
// ============================================================================

struct ModelConfig {
    int patch_size;
    bool is_dune;

    int enc_dim;
    int enc_heads;
    int enc_head_dim;
    int enc_mlp;
    int enc_depth;

    int dec_dim;
    int dec_heads;
    int dec_head_dim;
    int dec_mlp;
    int dec_depth;

    int desc_dim;
    int lf_hidden;

    std::string enc_block_key(int i) const {
        return is_dune
            ? "encoder.blocks.0." + std::to_string(i) + "."
            : "enc_blocks." + std::to_string(i) + ".";
    }

    std::string dec_block_key(int i) const {
        return is_dune
            ? "mast3r.dec_blocks." + std::to_string(i) + "."
            : "dec_blocks." + std::to_string(i) + ".";
    }

    std::string patch_embed_key() const {
        return is_dune ? "encoder.patch_embed.proj." : "patch_embed.proj.";
    }

    std::string enc_norm_key() const {
        return is_dune ? "encoder.norm." : "enc_norm.";
    }

    std::string decoder_embed_key() const {
        return is_dune ? "mast3r.decoder_embed." : "decoder_embed.";
    }

    std::string head_key() const {
        return is_dune ? "mast3r.downstream_head1." : "downstream_head1.";
    }

    static ModelConfig mast3r_vit_large() {
        return {16, false, 1024, 16, 64, 4096, 24, 768, 12, 64, 3072, 12, 24, 7168};
    }

    static ModelConfig dune_vit_small() {
        return {14, true, 384, 6, 64, 1536, 12, 768, 12, 64, 3072, 12, 24, 4096};
    }

    static ModelConfig dune_vit_base() {
        return {14, true, 768, 12, 64, 3072, 12, 768, 12, 64, 3072, 12, 24, 4096};
    }

    static ModelConfig from_variant(ModelVariant variant) {
        switch (variant) {
            case ModelVariant::MAST3R_VIT_LARGE: return mast3r_vit_large();
            case ModelVariant::DUNE_VIT_SMALL_336:
            case ModelVariant::DUNE_VIT_SMALL_448: return dune_vit_small();
            case ModelVariant::DUNE_VIT_BASE_336:
            case ModelVariant::DUNE_VIT_BASE_448: return dune_vit_base();
            default: return dune_vit_small();
        }
    }
};

constexpr float LN_EPS = 1e-6f;

// ============================================================================
// Layer Weight Structures
// ============================================================================

struct EncoderLayer {
    MPSGraphTensor* n1w; MPSGraphTensor* n1b;
    MPSGraphTensor* n2w; MPSGraphTensor* n2b;
    MPSGraphTensor* qkvw; MPSGraphTensor* qkvb;
    MPSGraphTensor* pw; MPSGraphTensor* pb;
    MPSGraphTensor* f1w; MPSGraphTensor* f1b;
    MPSGraphTensor* f2w; MPSGraphTensor* f2b;
};

struct DecoderLayer {
    MPSGraphTensor* n1w; MPSGraphTensor* n1b;
    MPSGraphTensor* n2w; MPSGraphTensor* n2b;
    MPSGraphTensor* n3w; MPSGraphTensor* n3b;
    MPSGraphTensor* nyw; MPSGraphTensor* nyb;
    MPSGraphTensor* qkvw; MPSGraphTensor* qkvb;
    MPSGraphTensor* pw; MPSGraphTensor* pb;
    MPSGraphTensor* cqw; MPSGraphTensor* cqb;
    MPSGraphTensor* ckw; MPSGraphTensor* ckb;
    MPSGraphTensor* cvw; MPSGraphTensor* cvb;
    MPSGraphTensor* cpw; MPSGraphTensor* cpb;
    MPSGraphTensor* f1w; MPSGraphTensor* f1b;
    MPSGraphTensor* f2w; MPSGraphTensor* f2b;
};

// ============================================================================
// Graph Builder Helper
// ============================================================================

class API_AVAILABLE(macos(15.0)) GraphBuilder {
public:
    MPSGraph* graph_;
    id<MTLDevice> device_;
    safetensors::MultiSafetensorsFile* files_;
    MPSDataType dtype_;

    GraphBuilder(MPSGraph* graph, id<MTLDevice> device, safetensors::MultiSafetensorsFile* files,
                 bool use_fp16 = false)
        : graph_(graph), device_(device), files_(files),
          dtype_(use_fp16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32) {}

    MPSGraphTensor* load(const std::string& name, NSArray<NSNumber*>* shape) {
        auto data = files_->load_tensor_f32(name);
        NSData* nsdata = [NSData dataWithBytes:data.data() length:data.size() * sizeof(float)];
        MPSGraphTensor* t = [graph_ constantWithData:nsdata shape:shape dataType:MPSDataTypeFloat32];
        if (dtype_ == MPSDataTypeFloat16) {
            t = [graph_ castTensor:t toType:MPSDataTypeFloat16 name:nil];
        }
        return t;
    }

    bool has_weight(const std::string& name) { return files_->has_tensor(name); }

    MPSGraphTensor* layer_norm(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b) {
        auto mean = [graph_ meanOfTensor:x axes:@[@(-1)] name:nil];
        auto centered = [graph_ subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
        auto var = [graph_ meanOfTensor:[graph_ squareWithTensor:centered name:nil] axes:@[@(-1)] name:nil];
        auto eps = [graph_ constantWithScalar:LN_EPS shape:@[@1] dataType:dtype_];
        auto std = [graph_ squareRootWithTensor:[graph_ additionWithPrimaryTensor:var secondaryTensor:eps name:nil] name:nil];
        auto norm = [graph_ divisionWithPrimaryTensor:centered secondaryTensor:std name:nil];
        norm = [graph_ multiplicationWithPrimaryTensor:norm secondaryTensor:w name:nil];
        return [graph_ additionWithPrimaryTensor:norm secondaryTensor:b name:nil];
    }

    MPSGraphTensor* gelu(MPSGraphTensor* x) {
        auto inv_sqrt2 = [graph_ constantWithScalar:0.7071067811865475 shape:@[@1] dataType:dtype_];
        auto half = [graph_ constantWithScalar:0.5 shape:@[@1] dataType:dtype_];
        auto one = [graph_ constantWithScalar:1.0 shape:@[@1] dataType:dtype_];
        auto cdf = [graph_ multiplicationWithPrimaryTensor:half
                       secondaryTensor:[graph_ additionWithPrimaryTensor:one
                           secondaryTensor:[graph_ erfWithTensor:
                               [graph_ multiplicationWithPrimaryTensor:x secondaryTensor:inv_sqrt2 name:nil] name:nil] name:nil] name:nil];
        return [graph_ multiplicationWithPrimaryTensor:x secondaryTensor:cdf name:nil];
    }

    MPSGraphTensor* linear(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b) {
        auto wt = [graph_ transposeTensor:w dimension:0 withDimension:1 name:nil];
        auto out = [graph_ matrixMultiplicationWithPrimaryTensor:x secondaryTensor:wt name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
    }

    MPSGraphTensor* self_attention(MPSGraphTensor* x, MPSGraphTensor* qkv_w, MPSGraphTensor* qkv_b,
                                   MPSGraphTensor* proj_w, MPSGraphTensor* proj_b, int heads, int head_dim) {
        int dim = heads * head_dim;
        float scale = 1.0f / sqrtf((float)head_dim);
        auto qkv = linear(x, qkv_w, qkv_b);
        auto splits = [graph_ splitTensor:qkv numSplits:3 axis:-1 name:nil];
        NSArray<NSNumber*>* shape = @[@1, @(heads), @(-1), @(head_dim)];
        auto Q = [graph_ reshapeTensor:splits[0] withShape:shape name:nil];
        auto K = [graph_ reshapeTensor:splits[1] withShape:shape name:nil];
        auto V = [graph_ reshapeTensor:splits[2] withShape:shape name:nil];
        auto attn = [graph_ scaledDotProductAttentionWithQueryTensor:Q keyTensor:K valueTensor:V maskTensor:nil scale:scale name:nil];
        attn = [graph_ reshapeTensor:attn withShape:@[@(-1), @(dim)] name:nil];
        return linear(attn, proj_w, proj_b);
    }

    // ============================================================================
    // Batched operations for [B, N, D] tensors (e.g., batch=2 for image pairs)
    // ============================================================================

    MPSGraphTensor* layer_norm_batched(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b) {
        // x: [B, N, D], w: [D], b: [D]
        // Mean/var on last axis broadcasts correctly for batched input
        auto mean = [graph_ meanOfTensor:x axes:@[@(-1)] name:nil];
        auto centered = [graph_ subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
        auto var = [graph_ meanOfTensor:[graph_ squareWithTensor:centered name:nil] axes:@[@(-1)] name:nil];
        auto eps = [graph_ constantWithScalar:LN_EPS shape:@[@1] dataType:dtype_];
        auto std = [graph_ squareRootWithTensor:[graph_ additionWithPrimaryTensor:var secondaryTensor:eps name:nil] name:nil];
        auto norm = [graph_ divisionWithPrimaryTensor:centered secondaryTensor:std name:nil];
        norm = [graph_ multiplicationWithPrimaryTensor:norm secondaryTensor:w name:nil];
        return [graph_ additionWithPrimaryTensor:norm secondaryTensor:b name:nil];
    }

    MPSGraphTensor* linear_batched(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b) {
        // x: [B, N, D], w: [out, D], b: [out]
        // matmul broadcasts: [B, N, D] @ [D, out] -> [B, N, out]
        auto wt = [graph_ transposeTensor:w dimension:0 withDimension:1 name:nil];
        auto out = [graph_ matrixMultiplicationWithPrimaryTensor:x secondaryTensor:wt name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
    }

    MPSGraphTensor* self_attention_batched(MPSGraphTensor* x, MPSGraphTensor* qkv_w, MPSGraphTensor* qkv_b,
                                           MPSGraphTensor* proj_w, MPSGraphTensor* proj_b, int heads, int head_dim,
                                           int batch_size = 2, int num_patches = -1) {
        // x: [B, N, D] where B=batch_size (e.g., 2 for image pairs)
        int dim = heads * head_dim;
        float scale = 1.0f / sqrtf((float)head_dim);

        // QKV projection: [B, N, D] -> [B, N, 3*D]
        auto qkv = linear_batched(x, qkv_w, qkv_b);

        // Split into Q, K, V: [B, N, D] each
        auto splits = [graph_ splitTensor:qkv numSplits:3 axis:-1 name:nil];

        // Reshape for SDPA: [B, N, D] -> [B, heads, N, head_dim]
        NSArray<NSNumber*>* shape = @[@(batch_size), @(heads), @(-1), @(head_dim)];
        auto Q = [graph_ reshapeTensor:splits[0] withShape:shape name:nil];
        auto K = [graph_ reshapeTensor:splits[1] withShape:shape name:nil];
        auto V = [graph_ reshapeTensor:splits[2] withShape:shape name:nil];

        // SDPA: [B, heads, N, head_dim] -> [B, heads, N, head_dim]
        auto attn = [graph_ scaledDotProductAttentionWithQueryTensor:Q keyTensor:K valueTensor:V maskTensor:nil scale:scale name:nil];

        // Reshape back: [B, heads, N, head_dim] -> [B, N, D]
        attn = [graph_ reshapeTensor:attn withShape:@[@(batch_size), @(-1), @(dim)] name:nil];

        // Output projection: [B, N, D] -> [B, N, D]
        return linear_batched(attn, proj_w, proj_b);
    }

    MPSGraphTensor* cross_attention(MPSGraphTensor* x, MPSGraphTensor* y,
                                    MPSGraphTensor* q_w, MPSGraphTensor* q_b,
                                    MPSGraphTensor* k_w, MPSGraphTensor* k_b,
                                    MPSGraphTensor* v_w, MPSGraphTensor* v_b,
                                    MPSGraphTensor* proj_w, MPSGraphTensor* proj_b, int heads, int head_dim) {
        int dim = heads * head_dim;
        float scale = 1.0f / sqrtf((float)head_dim);
        auto Q = linear(x, q_w, q_b);
        auto K = linear(y, k_w, k_b);
        auto V = linear(y, v_w, v_b);
        NSArray<NSNumber*>* shape = @[@1, @(heads), @(-1), @(head_dim)];
        Q = [graph_ reshapeTensor:Q withShape:shape name:nil];
        K = [graph_ reshapeTensor:K withShape:shape name:nil];
        V = [graph_ reshapeTensor:V withShape:shape name:nil];
        auto attn = [graph_ scaledDotProductAttentionWithQueryTensor:Q keyTensor:K valueTensor:V maskTensor:nil scale:scale name:nil];
        attn = [graph_ reshapeTensor:attn withShape:@[@(-1), @(dim)] name:nil];
        return linear(attn, proj_w, proj_b);
    }

    MPSGraphTensor* conv2d(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b, int stride=1, int pad=0) {
        auto desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride dilationRateInX:1 dilationRateInY:1 groups:1
                       paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        desc.paddingLeft = pad; desc.paddingRight = pad; desc.paddingTop = pad; desc.paddingBottom = pad;
        auto out = [graph_ convolution2DWithSourceTensor:x weightsTensor:w descriptor:desc name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
    }

    MPSGraphTensor* conv_transpose2d(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b, int stride) {
        auto desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride dilationRateInX:1 dilationRateInY:1 groups:1
                       paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        int pad = (stride - 1) / 2;
        desc.paddingLeft = pad; desc.paddingRight = pad; desc.paddingTop = pad; desc.paddingBottom = pad;
        NSArray<NSNumber*>* x_shape = [x shape];
        int in_h = [x_shape[1] intValue], in_w = [x_shape[2] intValue], out_c = [[w shape][1] intValue];
        NSArray<NSNumber*>* out_shape = @[@1, @(in_h * stride), @(in_w * stride), @(out_c)];
        auto wt = [graph_ transposeTensor:w dimension:0 withDimension:1 name:nil];
        auto out = [graph_ convolutionTranspose2DWithSourceTensor:x weightsTensor:wt outputShape:out_shape descriptor:desc name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
    }

    MPSGraphTensor* upsample(MPSGraphTensor* x, int scale) {
        auto shape = [x shape];
        int h = [shape[1] intValue], w = [shape[2] intValue];
        return [graph_ resizeTensor:x size:@[@(h*scale), @(w*scale)] mode:MPSGraphResizeBilinear
                       centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNHWC name:nil];
    }

    MPSGraphTensor* relu(MPSGraphTensor* x) { return [graph_ reLUWithTensor:x name:nil]; }

    MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }

    MPSGraphTensor* pixel_shuffle(MPSGraphTensor* x, int r) {
        auto shape = [x shape];
        int pH = [shape[1] intValue], pW = [shape[2] intValue], C_rr = [shape[3] intValue];
        int C = C_rr / (r * r);
        auto reshaped = [graph_ reshapeTensor:x withShape:@[@1, @(pH), @(pW), @(r), @(r), @(C)] name:nil];
        reshaped = [graph_ transposeTensor:reshaped dimension:2 withDimension:3 name:nil];
        return [graph_ reshapeTensor:reshaped withShape:@[@1, @(pH*r), @(pW*r), @(C)] name:nil];
    }

    MPSGraphTensor* l2_norm(MPSGraphTensor* x) {
        auto sq = [graph_ squareWithTensor:x name:nil];
        auto sum = [graph_ reductionSumWithTensor:sq axis:-1 name:nil];
        auto eps = [graph_ constantWithScalar:1e-8 shape:@[@1] dataType:dtype_];
        auto norm = [graph_ squareRootWithTensor:[graph_ additionWithPrimaryTensor:sum secondaryTensor:eps name:nil] name:nil];
        return [graph_ divisionWithPrimaryTensor:x secondaryTensor:norm name:nil];
    }

    MPSGraphTensor* imagenet_normalize(MPSGraphTensor* x) {
        auto xf = [graph_ castTensor:x toType:MPSDataTypeFloat32 name:nil];
        auto inv255 = [graph_ constantWithScalar:1.0f/255.0f shape:@[@1] dataType:MPSDataTypeFloat32];
        xf = [graph_ multiplicationWithPrimaryTensor:xf secondaryTensor:inv255 name:nil];

        float mean_vals[3] = {0.485f, 0.456f, 0.406f};
        float inv_std_vals[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
        NSData* mean_data = [NSData dataWithBytes:mean_vals length:3 * sizeof(float)];
        NSData* inv_std_data = [NSData dataWithBytes:inv_std_vals length:3 * sizeof(float)];

        auto mean = [graph_ constantWithData:mean_data shape:@[@1, @1, @1, @3] dataType:MPSDataTypeFloat32];
        auto inv_std = [graph_ constantWithData:inv_std_data shape:@[@1, @1, @1, @3] dataType:MPSDataTypeFloat32];

        xf = [graph_ subtractionWithPrimaryTensor:xf secondaryTensor:mean name:nil];
        xf = [graph_ multiplicationWithPrimaryTensor:xf secondaryTensor:inv_std name:nil];

        if (dtype_ == MPSDataTypeFloat16) {
            xf = [graph_ castTensor:xf toType:MPSDataTypeFloat16 name:nil];
        }
        return xf;
    }
};

}  // namespace mpsgraph
}  // namespace mast3r
