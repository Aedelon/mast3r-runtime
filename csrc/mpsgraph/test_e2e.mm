// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
// End-to-end MASt3R inference with MPSGraph

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <random>

#include "../common/safetensors.hpp"

// ============================================================================
// Configuration
// ============================================================================

constexpr int IMG_W = 512;
constexpr int IMG_H = 384;
constexpr int PATCH_SIZE = 16;
constexpr int PATCH_W = IMG_W / PATCH_SIZE;  // 32
constexpr int PATCH_H = IMG_H / PATCH_SIZE;  // 24
constexpr int NUM_PATCHES = PATCH_W * PATCH_H;  // 768

constexpr int ENC_DIM = 1024;
constexpr int ENC_HEADS = 16;
constexpr int ENC_HEAD_DIM = 64;
constexpr int ENC_MLP = 4096;
constexpr int ENC_DEPTH = 24;

constexpr int DEC_DIM = 768;
constexpr int DEC_HEADS = 12;
constexpr int DEC_HEAD_DIM = 64;
constexpr int DEC_MLP = 3072;
constexpr int DEC_DEPTH = 12;

constexpr int DPT_DIM = 256;
constexpr int DESC_DIM = 24;
constexpr float LN_EPS = 1e-6f;

// ============================================================================
// Complete MASt3R Graph Builder
// ============================================================================

class API_AVAILABLE(macos(15.0)) MASt3RGraphBuilder {
public:
    MPSGraph* graph_;
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;

    MASt3RGraphBuilder() {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newCommandQueue];
        graph_ = [[MPSGraph alloc] init];
    }

    // ========================================================================
    // Basic operations
    // ========================================================================

    MPSGraphTensor* tensor(const std::vector<float>& data, NSArray<NSNumber*>* shape) {
        NSData* nsdata = [NSData dataWithBytes:data.data() length:data.size() * sizeof(float)];
        return [graph_ constantWithData:nsdata shape:shape dataType:MPSDataTypeFloat32];
    }

    MPSGraphTensor* layer_norm(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b) {
        MPSGraphTensor* mean = [graph_ meanOfTensor:x axes:@[@(-1)] name:nil];
        MPSGraphTensor* centered = [graph_ subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
        MPSGraphTensor* var = [graph_ meanOfTensor:[graph_ squareWithTensor:centered name:nil] axes:@[@(-1)] name:nil];
        MPSGraphTensor* eps = [graph_ constantWithScalar:LN_EPS shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* std = [graph_ squareRootWithTensor:[graph_ additionWithPrimaryTensor:var secondaryTensor:eps name:nil] name:nil];
        MPSGraphTensor* norm = [graph_ divisionWithPrimaryTensor:centered secondaryTensor:std name:nil];
        norm = [graph_ multiplicationWithPrimaryTensor:norm secondaryTensor:w name:nil];
        return [graph_ additionWithPrimaryTensor:norm secondaryTensor:b name:nil];
    }

    MPSGraphTensor* gelu(MPSGraphTensor* x) {
        MPSGraphTensor* inv_sqrt2 = [graph_ constantWithScalar:0.7071067811865475 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* half = [graph_ constantWithScalar:0.5 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* one = [graph_ constantWithScalar:1.0 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* cdf = [graph_ multiplicationWithPrimaryTensor:half
                                  secondaryTensor:[graph_ additionWithPrimaryTensor:one
                                                      secondaryTensor:[graph_ erfWithTensor:
                                                          [graph_ multiplicationWithPrimaryTensor:x secondaryTensor:inv_sqrt2 name:nil] name:nil] name:nil] name:nil];
        return [graph_ multiplicationWithPrimaryTensor:x secondaryTensor:cdf name:nil];
    }

    MPSGraphTensor* linear(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b) {
        MPSGraphTensor* wt = [graph_ transposeTensor:w dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* out = [graph_ matrixMultiplicationWithPrimaryTensor:x secondaryTensor:wt name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
    }

    MPSGraphTensor* self_attention(MPSGraphTensor* x, MPSGraphTensor* qkv_w, MPSGraphTensor* qkv_b,
                                   MPSGraphTensor* proj_w, MPSGraphTensor* proj_b, int heads, int head_dim) {
        int dim = heads * head_dim;
        float scale = 1.0f / sqrtf((float)head_dim);
        MPSGraphTensor* qkv = linear(x, qkv_w, qkv_b);
        NSArray<MPSGraphTensor*>* splits = [graph_ splitTensor:qkv numSplits:3 axis:-1 name:nil];
        NSArray<NSNumber*>* shape = @[@1, @(heads), @(-1), @(head_dim)];
        MPSGraphTensor* Q = [graph_ reshapeTensor:splits[0] withShape:shape name:nil];
        MPSGraphTensor* K = [graph_ reshapeTensor:splits[1] withShape:shape name:nil];
        MPSGraphTensor* V = [graph_ reshapeTensor:splits[2] withShape:shape name:nil];
        MPSGraphTensor* attn = [graph_ scaledDotProductAttentionWithQueryTensor:Q keyTensor:K valueTensor:V maskTensor:nil scale:scale name:nil];
        attn = [graph_ reshapeTensor:attn withShape:@[@(-1), @(dim)] name:nil];
        return linear(attn, proj_w, proj_b);
    }

    MPSGraphTensor* cross_attention(MPSGraphTensor* x, MPSGraphTensor* y,
                                    MPSGraphTensor* q_w, MPSGraphTensor* q_b,
                                    MPSGraphTensor* k_w, MPSGraphTensor* k_b,
                                    MPSGraphTensor* v_w, MPSGraphTensor* v_b,
                                    MPSGraphTensor* proj_w, MPSGraphTensor* proj_b, int heads, int head_dim) {
        int dim = heads * head_dim;
        float scale = 1.0f / sqrtf((float)head_dim);
        MPSGraphTensor* Q = linear(x, q_w, q_b);
        MPSGraphTensor* K = linear(y, k_w, k_b);
        MPSGraphTensor* V = linear(y, v_w, v_b);
        NSArray<NSNumber*>* shape = @[@1, @(heads), @(-1), @(head_dim)];
        Q = [graph_ reshapeTensor:Q withShape:shape name:nil];
        K = [graph_ reshapeTensor:K withShape:shape name:nil];
        V = [graph_ reshapeTensor:V withShape:shape name:nil];
        MPSGraphTensor* attn = [graph_ scaledDotProductAttentionWithQueryTensor:Q keyTensor:K valueTensor:V maskTensor:nil scale:scale name:nil];
        attn = [graph_ reshapeTensor:attn withShape:@[@(-1), @(dim)] name:nil];
        return linear(attn, proj_w, proj_b);
    }

    MPSGraphTensor* conv2d(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b, int stride=1, int pad=0) {
        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride dilationRateInX:1 dilationRateInY:1 groups:1
                       paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        desc.paddingLeft = pad; desc.paddingRight = pad; desc.paddingTop = pad; desc.paddingBottom = pad;
        MPSGraphTensor* out = [graph_ convolution2DWithSourceTensor:x weightsTensor:w descriptor:desc name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
    }

    MPSGraphTensor* conv_transpose2d(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b, int stride) {
        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride dilationRateInX:1 dilationRateInY:1 groups:1
                       paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        int pad = (stride - 1) / 2;
        desc.paddingLeft = pad; desc.paddingRight = pad; desc.paddingTop = pad; desc.paddingBottom = pad;
        NSArray<NSNumber*>* x_shape = [x shape];
        int in_h = [x_shape[1] intValue], in_w = [x_shape[2] intValue], out_c = [[w shape][1] intValue];
        NSArray<NSNumber*>* out_shape = @[@1, @(in_h * stride), @(in_w * stride), @(out_c)];
        MPSGraphTensor* wt = [graph_ transposeTensor:w dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* out = [graph_ convolutionTranspose2DWithSourceTensor:x weightsTensor:wt outputShape:out_shape descriptor:desc name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
    }

    MPSGraphTensor* upsample(MPSGraphTensor* x, int scale) {
        NSArray<NSNumber*>* shape = [x shape];
        int h = [shape[1] intValue], w = [shape[2] intValue];
        return [graph_ resizeTensor:x size:@[@(h*scale), @(w*scale)] mode:MPSGraphResizeBilinear
                       centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNHWC name:nil];
    }

    MPSGraphTensor* relu(MPSGraphTensor* x) { return [graph_ reLUWithTensor:x name:nil]; }

    MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }

    MPSGraphTensor* pixel_shuffle(MPSGraphTensor* x, int r) {
        NSArray<NSNumber*>* shape = [x shape];
        int pH = [shape[1] intValue], pW = [shape[2] intValue], C_rr = [shape[3] intValue];
        int C = C_rr / (r * r);
        MPSGraphTensor* reshaped = [graph_ reshapeTensor:x withShape:@[@1, @(pH), @(pW), @(r), @(r), @(C)] name:nil];
        reshaped = [graph_ transposeTensor:reshaped dimension:2 withDimension:3 name:nil];
        return [graph_ reshapeTensor:reshaped withShape:@[@1, @(pH*r), @(pW*r), @(C)] name:nil];
    }

    MPSGraphTensor* l2_norm(MPSGraphTensor* x) {
        MPSGraphTensor* sq = [graph_ squareWithTensor:x name:nil];
        MPSGraphTensor* sum = [graph_ reductionSumWithTensor:sq axis:-1 name:nil];
        MPSGraphTensor* eps = [graph_ constantWithScalar:1e-8 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* norm = [graph_ squareRootWithTensor:[graph_ additionWithPrimaryTensor:sum secondaryTensor:eps name:nil] name:nil];
        return [graph_ divisionWithPrimaryTensor:x secondaryTensor:norm name:nil];
    }
};

// ============================================================================
// End-to-End Test
// ============================================================================

bool test_e2e(const std::string& weights_path) {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            printf("╔══════════════════════════════════════════════════════════════╗\n");
            printf("║     MASt3R End-to-End MPSGraph Inference Benchmark           ║\n");
            printf("╚══════════════════════════════════════════════════════════════╝\n\n");

            printf("[1/6] Loading weights...\n");
            mast3r::safetensors::SafetensorsFile file(weights_path);
            printf("      Loaded %zu tensors\n", file.num_tensors());

            MASt3RGraphBuilder gb;
            printf("      Device: %s\n\n", [[gb.device_ name] UTF8String]);

            // ================================================================
            // Load all weights
            // ================================================================
            printf("[2/6] Loading model weights...\n");

            // Patch embedding
            auto patch_w = gb.tensor(file.load_tensor_f32("patch_embed.proj.weight"), @[@(ENC_DIM), @3, @(PATCH_SIZE), @(PATCH_SIZE)]);
            auto patch_b = gb.tensor(file.load_tensor_f32("patch_embed.proj.bias"), @[@(ENC_DIM)]);

            // Encoder layers
            struct EncL { MPSGraphTensor *n1w,*n1b,*qkvw,*qkvb,*pw,*pb,*n2w,*n2b,*f1w,*f1b,*f2w,*f2b; };
            std::vector<EncL> enc(ENC_DEPTH);
            auto enc_nw = gb.tensor(file.load_tensor_f32("enc_norm.weight"), @[@(ENC_DIM)]);
            auto enc_nb = gb.tensor(file.load_tensor_f32("enc_norm.bias"), @[@(ENC_DIM)]);
            for (int i = 0; i < ENC_DEPTH; i++) {
                std::string p = "enc_blocks." + std::to_string(i) + ".";
                enc[i] = {
                    gb.tensor(file.load_tensor_f32(p+"norm1.weight"), @[@(ENC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"norm1.bias"), @[@(ENC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"attn.qkv.weight"), @[@(3*ENC_DIM), @(ENC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"attn.qkv.bias"), @[@(3*ENC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"attn.proj.weight"), @[@(ENC_DIM), @(ENC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"attn.proj.bias"), @[@(ENC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"norm2.weight"), @[@(ENC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"norm2.bias"), @[@(ENC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"mlp.fc1.weight"), @[@(ENC_MLP), @(ENC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"mlp.fc1.bias"), @[@(ENC_MLP)]),
                    gb.tensor(file.load_tensor_f32(p+"mlp.fc2.weight"), @[@(ENC_DIM), @(ENC_MLP)]),
                    gb.tensor(file.load_tensor_f32(p+"mlp.fc2.bias"), @[@(ENC_DIM)])
                };
            }

            // Decoder layers
            struct DecL {
                MPSGraphTensor *n1w,*n1b,*qkvw,*qkvb,*pw,*pb;
                MPSGraphTensor *nyw,*nyb,*n3w,*n3b,*cqw,*cqb,*ckw,*ckb,*cvw,*cvb,*cpw,*cpb;
                MPSGraphTensor *n2w,*n2b,*f1w,*f1b,*f2w,*f2b;
            };
            std::vector<DecL> dec(DEC_DEPTH);
            auto e2d_w = gb.tensor(file.load_tensor_f32("decoder_embed.weight"), @[@(DEC_DIM), @(ENC_DIM)]);
            auto e2d_b = gb.tensor(file.load_tensor_f32("decoder_embed.bias"), @[@(DEC_DIM)]);
            for (int i = 0; i < DEC_DEPTH; i++) {
                std::string p = "dec_blocks." + std::to_string(i) + ".";
                dec[i] = {
                    gb.tensor(file.load_tensor_f32(p+"norm1.weight"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"norm1.bias"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"attn.qkv.weight"), @[@(3*DEC_DIM), @(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"attn.qkv.bias"), @[@(3*DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"attn.proj.weight"), @[@(DEC_DIM), @(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"attn.proj.bias"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"norm_y.weight"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"norm_y.bias"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"norm3.weight"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"norm3.bias"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"cross_attn.projq.weight"), @[@(DEC_DIM), @(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"cross_attn.projq.bias"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"cross_attn.projk.weight"), @[@(DEC_DIM), @(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"cross_attn.projk.bias"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"cross_attn.projv.weight"), @[@(DEC_DIM), @(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"cross_attn.projv.bias"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"cross_attn.proj.weight"), @[@(DEC_DIM), @(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"cross_attn.proj.bias"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"norm2.weight"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"norm2.bias"), @[@(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"mlp.fc1.weight"), @[@(DEC_MLP), @(DEC_DIM)]),
                    gb.tensor(file.load_tensor_f32(p+"mlp.fc1.bias"), @[@(DEC_MLP)]),
                    gb.tensor(file.load_tensor_f32(p+"mlp.fc2.weight"), @[@(DEC_DIM), @(DEC_MLP)]),
                    gb.tensor(file.load_tensor_f32(p+"mlp.fc2.bias"), @[@(DEC_DIM)])
                };
            }

            // DPT weights
            std::string dp = "downstream_head1.dpt.";
            auto ap0_1w = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.0.0.weight"), @[@96, @1024, @1, @1]);
            auto ap0_1b = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.0.0.bias"), @[@96]);
            auto ap0_2w = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.0.1.weight"), @[@96, @96, @4, @4]);
            auto ap0_2b = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.0.1.bias"), @[@96]);
            auto ap1_1w = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.1.0.weight"), @[@192, @768, @1, @1]);
            auto ap1_1b = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.1.0.bias"), @[@192]);
            auto ap1_2w = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.1.1.weight"), @[@192, @192, @2, @2]);
            auto ap1_2b = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.1.1.bias"), @[@192]);
            auto ap2_1w = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.2.0.weight"), @[@384, @768, @1, @1]);
            auto ap2_1b = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.2.0.bias"), @[@384]);
            auto ap3_1w = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.3.0.weight"), @[@768, @768, @1, @1]);
            auto ap3_1b = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.3.0.bias"), @[@768]);
            auto ap3_2w = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.3.1.weight"), @[@768, @768, @3, @3]);
            auto ap3_2b = gb.tensor(file.load_tensor_f32(dp+"act_postprocess.3.1.bias"), @[@768]);
            auto lr0w = gb.tensor(file.load_tensor_f32(dp+"scratch.layer_rn.0.weight"), @[@256, @96, @3, @3]);
            auto lr1w = gb.tensor(file.load_tensor_f32(dp+"scratch.layer_rn.1.weight"), @[@256, @192, @3, @3]);
            auto lr2w = gb.tensor(file.load_tensor_f32(dp+"scratch.layer_rn.2.weight"), @[@256, @384, @3, @3]);
            auto lr3w = gb.tensor(file.load_tensor_f32(dp+"scratch.layer_rn.3.weight"), @[@256, @768, @3, @3]);
            auto hd0w = gb.tensor(file.load_tensor_f32(dp+"head.0.weight"), @[@128, @256, @3, @3]);
            auto hd0b = gb.tensor(file.load_tensor_f32(dp+"head.0.bias"), @[@128]);
            auto hd2w = gb.tensor(file.load_tensor_f32(dp+"head.2.weight"), @[@128, @128, @3, @3]);
            auto hd2b = gb.tensor(file.load_tensor_f32(dp+"head.2.bias"), @[@128]);
            auto hd4w = gb.tensor(file.load_tensor_f32(dp+"head.4.weight"), @[@4, @128, @1, @1]);
            auto hd4b = gb.tensor(file.load_tensor_f32(dp+"head.4.bias"), @[@4]);

            // Local features
            std::string lf = "downstream_head1.head_local_features.";
            auto lf1w = gb.tensor(file.load_tensor_f32(lf+"fc1.weight"), @[@7168, @1792]);
            auto lf1b = gb.tensor(file.load_tensor_f32(lf+"fc1.bias"), @[@7168]);
            auto lf2w = gb.tensor(file.load_tensor_f32(lf+"fc2.weight"), @[@6400, @7168]);
            auto lf2b = gb.tensor(file.load_tensor_f32(lf+"fc2.bias"), @[@6400]);

            printf("      All weights loaded\n\n");

            // ================================================================
            // Build graph
            // ================================================================
            printf("[3/6] Building inference graph...\n");

            // Input: normalized image [1, H, W, 3]
            MPSGraphTensor* img_input = [gb.graph_ placeholderWithShape:@[@1, @(IMG_H), @(IMG_W), @3]
                                                               dataType:MPSDataTypeFloat32 name:@"image"];

            // Patch embedding: [1, H, W, 3] -> [1, pH, pW, D]
            MPSGraphTensor* patches = gb.conv2d(img_input, patch_w, patch_b, PATCH_SIZE, 0);
            // Flatten: [1, pH, pW, D] -> [N, D]
            patches = [gb.graph_ reshapeTensor:patches withShape:@[@(NUM_PATCHES), @(ENC_DIM)] name:nil];

            // Encoder
            MPSGraphTensor* x = patches;
            for (int i = 0; i < ENC_DEPTH; i++) {
                auto& L = enc[i];
                MPSGraphTensor* res = x;
                x = gb.layer_norm(x, L.n1w, L.n1b);
                x = gb.self_attention(x, L.qkvw, L.qkvb, L.pw, L.pb, ENC_HEADS, ENC_HEAD_DIM);
                x = gb.add(x, res);
                res = x;
                x = gb.layer_norm(x, L.n2w, L.n2b);
                x = gb.linear(x, L.f1w, L.f1b);
                x = gb.gelu(x);
                x = gb.linear(x, L.f2w, L.f2b);
                x = gb.add(x, res);
            }
            x = gb.layer_norm(x, enc_nw, enc_nb);
            MPSGraphTensor* enc_out = x;

            // Decoder
            MPSGraphTensor* dec_in = gb.linear(enc_out, e2d_w, e2d_b);
            MPSGraphTensor* enc_proj = dec_in;
            x = dec_in;
            MPSGraphTensor* hooks[4];
            hooks[0] = enc_out;
            for (int i = 0; i < DEC_DEPTH; i++) {
                auto& L = dec[i];
                MPSGraphTensor* res = x;
                x = gb.layer_norm(x, L.n1w, L.n1b);
                x = gb.self_attention(x, L.qkvw, L.qkvb, L.pw, L.pb, DEC_HEADS, DEC_HEAD_DIM);
                x = gb.add(x, res);
                res = x;
                MPSGraphTensor* xn = gb.layer_norm(x, L.n3w, L.n3b);
                MPSGraphTensor* yn = gb.layer_norm(enc_proj, L.nyw, L.nyb);
                x = gb.add(gb.cross_attention(xn, yn, L.cqw, L.cqb, L.ckw, L.ckb, L.cvw, L.cvb, L.cpw, L.cpb, DEC_HEADS, DEC_HEAD_DIM), res);
                res = x;
                x = gb.layer_norm(x, L.n2w, L.n2b);
                x = gb.linear(x, L.f1w, L.f1b);
                x = gb.gelu(x);
                x = gb.linear(x, L.f2w, L.f2b);
                x = gb.add(x, res);
                if (i == 5) hooks[1] = x;
                if (i == 8) hooks[2] = x;
                if (i == 11) hooks[3] = x;
            }
            MPSGraphTensor* dec_out = x;

            // DPT: reshape hooks to spatial
            auto to_spatial = [&](MPSGraphTensor* t, int dim) {
                return [gb.graph_ reshapeTensor:t withShape:@[@1, @(PATCH_H), @(PATCH_W), @(dim)] name:nil];
            };
            MPSGraphTensor* h0 = to_spatial(hooks[0], ENC_DIM);
            MPSGraphTensor* h1 = to_spatial(hooks[1], DEC_DIM);
            MPSGraphTensor* h2 = to_spatial(hooks[2], DEC_DIM);
            MPSGraphTensor* h3 = to_spatial(hooks[3], DEC_DIM);

            // act_postprocess
            MPSGraphTensor* f0 = gb.conv_transpose2d(gb.conv2d(h0, ap0_1w, ap0_1b), ap0_2w, ap0_2b, 4);
            MPSGraphTensor* f1 = gb.conv_transpose2d(gb.conv2d(h1, ap1_1w, ap1_1b), ap1_2w, ap1_2b, 2);
            MPSGraphTensor* f2 = gb.conv2d(h2, ap2_1w, ap2_1b);
            MPSGraphTensor* f3 = gb.conv2d(gb.conv2d(h3, ap3_1w, ap3_1b), ap3_2w, ap3_2b, 2, 1);

            // layer_rn
            f0 = gb.conv2d(f0, lr0w, nil, 1, 1);
            f1 = gb.conv2d(f1, lr1w, nil, 1, 1);
            f2 = gb.conv2d(f2, lr2w, nil, 1, 1);
            f3 = gb.conv2d(f3, lr3w, nil, 1, 1);

            // Upsample and fuse
            f1 = gb.upsample(f1, 2);
            f2 = gb.upsample(f2, 4);
            f3 = gb.upsample(f3, 8);
            MPSGraphTensor* fused = gb.add(gb.add(gb.add(f0, f1), f2), f3);

            // Head
            MPSGraphTensor* head = gb.upsample(fused, 2);
            head = gb.relu(gb.conv2d(head, hd0w, hd0b, 1, 1));
            head = gb.upsample(head, 2);
            head = gb.relu(gb.conv2d(head, hd2w, hd2b, 1, 1));
            MPSGraphTensor* pts3d_conf = gb.conv2d(head, hd4w, hd4b);  // [1, H, W, 4]

            // Local features
            MPSGraphTensor* concat = [gb.graph_ concatTensors:@[enc_out, dec_out] dimension:-1 name:nil];
            MPSGraphTensor* lf_out = gb.gelu(gb.linear(concat, lf1w, lf1b));
            lf_out = gb.linear(lf_out, lf2w, lf2b);
            lf_out = [gb.graph_ reshapeTensor:lf_out withShape:@[@1, @(PATCH_H), @(PATCH_W), @6400] name:nil];
            lf_out = gb.pixel_shuffle(lf_out, PATCH_SIZE);
            NSArray<MPSGraphTensor*>* desc_split = [gb.graph_ splitTensor:lf_out splitSizes:@[@(DESC_DIM), @1] axis:-1 name:nil];
            MPSGraphTensor* descriptors = gb.l2_norm(desc_split[0]);
            MPSGraphTensor* desc_conf = desc_split[1];

            printf("      Graph built successfully\n\n");

            // ================================================================
            // Create input and run
            // ================================================================
            printf("[4/6] Creating test input...\n");

            std::vector<float> img_data(IMG_H * IMG_W * 3);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto& v : img_data) v = dist(rng);

            MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                                                shape:@[@1, @(IMG_H), @(IMG_W), @3]];
            MPSNDArray* img_arr = [[MPSNDArray alloc] initWithDevice:gb.device_ descriptor:desc];
            [img_arr writeBytes:(void*)img_data.data() strideBytes:nil];
            MPSGraphTensorData* img_td = [[MPSGraphTensorData alloc] initWithMPSNDArray:img_arr];
            NSDictionary* feeds = @{img_input: img_td};

            printf("      Input shape: [1, %d, %d, 3]\n\n", IMG_H, IMG_W);

            // Warmup
            printf("[5/6] Warmup (3 iterations)...\n");
            for (int i = 0; i < 3; i++) {
                [gb.graph_ runWithMTLCommandQueue:gb.queue_ feeds:feeds
                                    targetTensors:@[pts3d_conf, descriptors] targetOperations:nil];
            }
            printf("      Done\n\n");

            // Benchmark
            printf("[6/6] Benchmarking (10 iterations)...\n");
            const int iters = 10;
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {
                [gb.graph_ runWithMTLCommandQueue:gb.queue_ feeds:feeds
                                    targetTensors:@[pts3d_conf, descriptors] targetOperations:nil];
            }
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;

            printf("\n");
            printf("╔══════════════════════════════════════════════════════════════╗\n");
            printf("║                        RESULTS                               ║\n");
            printf("╠══════════════════════════════════════════════════════════════╣\n");
            printf("║  Single image inference:     %6.1f ms                       ║\n", ms);
            printf("║  Two images (MASt3R):        %6.1f ms                       ║\n", ms * 2);
            printf("║                                                              ║\n");
            printf("║  vs Metal current (~8000ms): %5.0fx speedup                  ║\n", 8000.0 / (ms * 2));
            printf("╚══════════════════════════════════════════════════════════════╝\n\n");

            // Verify
            NSDictionary* results = [gb.graph_ runWithMTLCommandQueue:gb.queue_ feeds:feeds
                                                        targetTensors:@[pts3d_conf, descriptors] targetOperations:nil];

            MPSGraphTensorData* pts_data = results[pts3d_conf];
            MPSGraphTensorData* desc_data = results[descriptors];

            NSArray<NSNumber*>* pts_shape = [pts_data shape];
            NSArray<NSNumber*>* desc_shape = [desc_data shape];

            printf("Output shapes:\n");
            printf("  pts3d_conf: [%d, %d, %d, %d]\n",
                   [pts_shape[0] intValue], [pts_shape[1] intValue],
                   [pts_shape[2] intValue], [pts_shape[3] intValue]);
            printf("  descriptors: [%d, %d, %d, %d]\n",
                   [desc_shape[0] intValue], [desc_shape[1] intValue],
                   [desc_shape[2] intValue], [desc_shape[3] intValue]);

            // Check for NaN
            size_t pts_size = IMG_H * IMG_W * 4;
            std::vector<float> pts_cpu(pts_size);
            [[pts_data mpsndarray] readBytes:pts_cpu.data() strideBytes:nil];

            int nan_count = 0;
            for (auto v : pts_cpu) if (std::isnan(v) || std::isinf(v)) nan_count++;

            printf("\nValidation: %s\n", nan_count == 0 ? "✓ PASSED (no NaN)" : "✗ FAILED (NaN detected)");

            return nan_count == 0;
        }
    }
    return false;
}

int main() {
    @autoreleasepool {
        const char* home = getenv("HOME");
        std::string path = std::string(home) +
            "/.cache/mast3r_runtime/safetensors/mast3r_vit_large/unified.safetensors";
        test_e2e(path);
        return 0;
    }
}
