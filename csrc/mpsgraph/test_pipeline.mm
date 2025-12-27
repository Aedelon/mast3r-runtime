// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
// Test complete MASt3R inference pipeline with MPSGraph

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <functional>

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

// DPT hooks: decoder layers to capture
constexpr int DPT_HOOK_LAYERS[3] = {5, 8, 11};  // 0-indexed, layer 6, 9, 12

// ============================================================================
// Graph Builder with all operations
// ============================================================================

class API_AVAILABLE(macos(15.0)) PipelineBuilder {
public:
    MPSGraph* graph_;
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;

    PipelineBuilder() {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newCommandQueue];
        graph_ = [[MPSGraph alloc] init];
    }

    MPSGraphTensor* tensor(const std::vector<float>& data, NSArray<NSNumber*>* shape) {
        NSData* nsdata = [NSData dataWithBytes:data.data() length:data.size() * sizeof(float)];
        return [graph_ constantWithData:nsdata shape:shape dataType:MPSDataTypeFloat32];
    }

    MPSGraphTensor* layer_norm(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b) {
        MPSGraphTensor* mean = [graph_ meanOfTensor:x axes:@[@(-1)] name:nil];
        MPSGraphTensor* centered = [graph_ subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
        MPSGraphTensor* sq = [graph_ squareWithTensor:centered name:nil];
        MPSGraphTensor* var = [graph_ meanOfTensor:sq axes:@[@(-1)] name:nil];
        MPSGraphTensor* eps = [graph_ constantWithScalar:LN_EPS shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* std = [graph_ squareRootWithTensor:
                               [graph_ additionWithPrimaryTensor:var secondaryTensor:eps name:nil] name:nil];
        MPSGraphTensor* normalized = [graph_ divisionWithPrimaryTensor:centered secondaryTensor:std name:nil];
        normalized = [graph_ multiplicationWithPrimaryTensor:normalized secondaryTensor:w name:nil];
        return [graph_ additionWithPrimaryTensor:normalized secondaryTensor:b name:nil];
    }

    MPSGraphTensor* gelu(MPSGraphTensor* x) {
        MPSGraphTensor* inv_sqrt2 = [graph_ constantWithScalar:0.7071067811865475 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* half = [graph_ constantWithScalar:0.5 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* one = [graph_ constantWithScalar:1.0 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* x_scaled = [graph_ multiplicationWithPrimaryTensor:x secondaryTensor:inv_sqrt2 name:nil];
        MPSGraphTensor* erf_val = [graph_ erfWithTensor:x_scaled name:nil];
        MPSGraphTensor* cdf = [graph_ multiplicationWithPrimaryTensor:half
                                  secondaryTensor:[graph_ additionWithPrimaryTensor:one secondaryTensor:erf_val name:nil]
                                             name:nil];
        return [graph_ multiplicationWithPrimaryTensor:x secondaryTensor:cdf name:nil];
    }

    MPSGraphTensor* linear(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b) {
        MPSGraphTensor* wt = [graph_ transposeTensor:w dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* out = [graph_ matrixMultiplicationWithPrimaryTensor:x secondaryTensor:wt name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
    }

    MPSGraphTensor* self_attention(MPSGraphTensor* x,
                                   MPSGraphTensor* qkv_w, MPSGraphTensor* qkv_b,
                                   MPSGraphTensor* proj_w, MPSGraphTensor* proj_b,
                                   int num_heads, int head_dim) {
        int embed_dim = num_heads * head_dim;
        float scale = 1.0f / sqrtf((float)head_dim);

        MPSGraphTensor* qkv = linear(x, qkv_w, qkv_b);
        NSArray<MPSGraphTensor*>* splits = [graph_ splitTensor:qkv numSplits:3 axis:-1 name:nil];

        NSArray<NSNumber*>* attn_shape = @[@1, @(num_heads), @(-1), @(head_dim)];
        MPSGraphTensor* Q = [graph_ reshapeTensor:splits[0] withShape:attn_shape name:nil];
        MPSGraphTensor* K = [graph_ reshapeTensor:splits[1] withShape:attn_shape name:nil];
        MPSGraphTensor* V = [graph_ reshapeTensor:splits[2] withShape:attn_shape name:nil];

        MPSGraphTensor* attn_out = [graph_ scaledDotProductAttentionWithQueryTensor:Q keyTensor:K valueTensor:V
                                                                         maskTensor:nil scale:scale name:nil];
        attn_out = [graph_ reshapeTensor:attn_out withShape:@[@(-1), @(embed_dim)] name:nil];
        return linear(attn_out, proj_w, proj_b);
    }

    MPSGraphTensor* cross_attention(MPSGraphTensor* x, MPSGraphTensor* y,
                                    MPSGraphTensor* q_w, MPSGraphTensor* q_b,
                                    MPSGraphTensor* k_w, MPSGraphTensor* k_b,
                                    MPSGraphTensor* v_w, MPSGraphTensor* v_b,
                                    MPSGraphTensor* proj_w, MPSGraphTensor* proj_b,
                                    int num_heads, int head_dim) {
        int embed_dim = num_heads * head_dim;
        float scale = 1.0f / sqrtf((float)head_dim);

        MPSGraphTensor* Q_flat = linear(x, q_w, q_b);
        MPSGraphTensor* K_flat = linear(y, k_w, k_b);
        MPSGraphTensor* V_flat = linear(y, v_w, v_b);

        NSArray<NSNumber*>* attn_shape = @[@1, @(num_heads), @(-1), @(head_dim)];
        MPSGraphTensor* Q = [graph_ reshapeTensor:Q_flat withShape:attn_shape name:nil];
        MPSGraphTensor* K = [graph_ reshapeTensor:K_flat withShape:attn_shape name:nil];
        MPSGraphTensor* V = [graph_ reshapeTensor:V_flat withShape:attn_shape name:nil];

        MPSGraphTensor* attn_out = [graph_ scaledDotProductAttentionWithQueryTensor:Q keyTensor:K valueTensor:V
                                                                         maskTensor:nil scale:scale name:nil];
        attn_out = [graph_ reshapeTensor:attn_out withShape:@[@(-1), @(embed_dim)] name:nil];
        return linear(attn_out, proj_w, proj_b);
    }

    MPSGraphTensor* conv2d(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b, int stride=1, int padding=0) {
        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride dilationRateInX:1 dilationRateInY:1
                             groups:1 paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        desc.paddingLeft = padding; desc.paddingRight = padding;
        desc.paddingTop = padding; desc.paddingBottom = padding;
        MPSGraphTensor* out = [graph_ convolution2DWithSourceTensor:x weightsTensor:w descriptor:desc name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
    }

    MPSGraphTensor* conv_transpose2d(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b, int stride) {
        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride dilationRateInX:1 dilationRateInY:1
                             groups:1 paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        int kernel_size = stride;
        int padding = (kernel_size - 1) / 2;
        desc.paddingLeft = padding; desc.paddingRight = padding;
        desc.paddingTop = padding; desc.paddingBottom = padding;

        NSArray<NSNumber*>* x_shape = [x shape];
        int in_h = [x_shape[1] intValue], in_w = [x_shape[2] intValue];
        int out_c = [[w shape][1] intValue];
        NSArray<NSNumber*>* output_shape = @[@1, @(in_h * stride), @(in_w * stride), @(out_c)];

        MPSGraphTensor* weight_t = [graph_ transposeTensor:w dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* out = [graph_ convolutionTranspose2DWithSourceTensor:x weightsTensor:weight_t
                                                                 outputShape:output_shape descriptor:desc name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
    }

    MPSGraphTensor* upsample_bilinear(MPSGraphTensor* x, int scale) {
        NSArray<NSNumber*>* x_shape = [x shape];
        int in_h = [x_shape[1] intValue], in_w = [x_shape[2] intValue];
        return [graph_ resizeTensor:x size:@[@(in_h * scale), @(in_w * scale)] mode:MPSGraphResizeBilinear
                       centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNHWC name:nil];
    }

    MPSGraphTensor* relu(MPSGraphTensor* x) { return [graph_ reLUWithTensor:x name:nil]; }

    MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
};

// ============================================================================
// Test: Benchmark full pipeline
// ============================================================================

bool test_full_pipeline(const std::string& weights_path) {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            printf("[test] Full MASt3R pipeline benchmark\n");
            printf("[test] Loading weights from: %s\n", weights_path.c_str());
            mast3r::safetensors::SafetensorsFile file(weights_path);

            PipelineBuilder gb;
            printf("[test] Device: %s\n\n", [[gb.device_ name] UTF8String]);

            // We'll time each component separately

            // ================================================================
            // 1. ENCODER (24 layers)
            // ================================================================
            printf("[1] Building Encoder (24 layers)...\n");

            // Load encoder weights
            struct EncLayer {
                MPSGraphTensor *n1w, *n1b, *qkvw, *qkvb, *pw, *pb, *n2w, *n2b, *f1w, *f1b, *f2w, *f2b;
            };
            std::vector<EncLayer> enc_layers(ENC_DEPTH);
            auto enc_norm_w = gb.tensor(file.load_tensor_f32("enc_norm.weight"), @[@(ENC_DIM)]);
            auto enc_norm_b = gb.tensor(file.load_tensor_f32("enc_norm.bias"), @[@(ENC_DIM)]);

            for (int i = 0; i < ENC_DEPTH; i++) {
                std::string p = "enc_blocks." + std::to_string(i) + ".";
                auto& L = enc_layers[i];
                L.n1w = gb.tensor(file.load_tensor_f32(p + "norm1.weight"), @[@(ENC_DIM)]);
                L.n1b = gb.tensor(file.load_tensor_f32(p + "norm1.bias"), @[@(ENC_DIM)]);
                L.qkvw = gb.tensor(file.load_tensor_f32(p + "attn.qkv.weight"), @[@(3*ENC_DIM), @(ENC_DIM)]);
                L.qkvb = gb.tensor(file.load_tensor_f32(p + "attn.qkv.bias"), @[@(3*ENC_DIM)]);
                L.pw = gb.tensor(file.load_tensor_f32(p + "attn.proj.weight"), @[@(ENC_DIM), @(ENC_DIM)]);
                L.pb = gb.tensor(file.load_tensor_f32(p + "attn.proj.bias"), @[@(ENC_DIM)]);
                L.n2w = gb.tensor(file.load_tensor_f32(p + "norm2.weight"), @[@(ENC_DIM)]);
                L.n2b = gb.tensor(file.load_tensor_f32(p + "norm2.bias"), @[@(ENC_DIM)]);
                L.f1w = gb.tensor(file.load_tensor_f32(p + "mlp.fc1.weight"), @[@(ENC_MLP), @(ENC_DIM)]);
                L.f1b = gb.tensor(file.load_tensor_f32(p + "mlp.fc1.bias"), @[@(ENC_MLP)]);
                L.f2w = gb.tensor(file.load_tensor_f32(p + "mlp.fc2.weight"), @[@(ENC_DIM), @(ENC_MLP)]);
                L.f2b = gb.tensor(file.load_tensor_f32(p + "mlp.fc2.bias"), @[@(ENC_DIM)]);
            }

            // Encoder input
            MPSGraphTensor* enc_input = [gb.graph_ placeholderWithShape:@[@(NUM_PATCHES), @(ENC_DIM)]
                                                               dataType:MPSDataTypeFloat32 name:@"enc_input"];

            // Encoder forward
            MPSGraphTensor* x = enc_input;
            for (int i = 0; i < ENC_DEPTH; i++) {
                auto& L = enc_layers[i];
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
            x = gb.layer_norm(x, enc_norm_w, enc_norm_b);
            MPSGraphTensor* encoder_out = x;

            // ================================================================
            // 2. DECODER (12 layers) with hook capture
            // ================================================================
            printf("[2] Building Decoder (12 layers)...\n");

            // Load decoder weights
            struct DecLayer {
                MPSGraphTensor *n1w, *n1b, *qkvw, *qkvb, *pw, *pb;
                MPSGraphTensor *nyw, *nyb, *n3w, *n3b;
                MPSGraphTensor *cqw, *cqb, *ckw, *ckb, *cvw, *cvb, *cpw, *cpb;
                MPSGraphTensor *n2w, *n2b, *f1w, *f1b, *f2w, *f2b;
            };
            std::vector<DecLayer> dec_layers(DEC_DEPTH);

            // Enc->Dec projection
            auto enc2dec_w = gb.tensor(file.load_tensor_f32("decoder_embed.weight"), @[@(DEC_DIM), @(ENC_DIM)]);
            auto enc2dec_b = gb.tensor(file.load_tensor_f32("decoder_embed.bias"), @[@(DEC_DIM)]);

            for (int i = 0; i < DEC_DEPTH; i++) {
                std::string p = "dec_blocks." + std::to_string(i) + ".";
                auto& L = dec_layers[i];
                L.n1w = gb.tensor(file.load_tensor_f32(p + "norm1.weight"), @[@(DEC_DIM)]);
                L.n1b = gb.tensor(file.load_tensor_f32(p + "norm1.bias"), @[@(DEC_DIM)]);
                L.qkvw = gb.tensor(file.load_tensor_f32(p + "attn.qkv.weight"), @[@(3*DEC_DIM), @(DEC_DIM)]);
                L.qkvb = gb.tensor(file.load_tensor_f32(p + "attn.qkv.bias"), @[@(3*DEC_DIM)]);
                L.pw = gb.tensor(file.load_tensor_f32(p + "attn.proj.weight"), @[@(DEC_DIM), @(DEC_DIM)]);
                L.pb = gb.tensor(file.load_tensor_f32(p + "attn.proj.bias"), @[@(DEC_DIM)]);

                L.nyw = gb.tensor(file.load_tensor_f32(p + "norm_y.weight"), @[@(DEC_DIM)]);
                L.nyb = gb.tensor(file.load_tensor_f32(p + "norm_y.bias"), @[@(DEC_DIM)]);
                L.n3w = gb.tensor(file.load_tensor_f32(p + "norm3.weight"), @[@(DEC_DIM)]);
                L.n3b = gb.tensor(file.load_tensor_f32(p + "norm3.bias"), @[@(DEC_DIM)]);
                L.cqw = gb.tensor(file.load_tensor_f32(p + "cross_attn.projq.weight"), @[@(DEC_DIM), @(DEC_DIM)]);
                L.cqb = gb.tensor(file.load_tensor_f32(p + "cross_attn.projq.bias"), @[@(DEC_DIM)]);
                L.ckw = gb.tensor(file.load_tensor_f32(p + "cross_attn.projk.weight"), @[@(DEC_DIM), @(DEC_DIM)]);
                L.ckb = gb.tensor(file.load_tensor_f32(p + "cross_attn.projk.bias"), @[@(DEC_DIM)]);
                L.cvw = gb.tensor(file.load_tensor_f32(p + "cross_attn.projv.weight"), @[@(DEC_DIM), @(DEC_DIM)]);
                L.cvb = gb.tensor(file.load_tensor_f32(p + "cross_attn.projv.bias"), @[@(DEC_DIM)]);
                L.cpw = gb.tensor(file.load_tensor_f32(p + "cross_attn.proj.weight"), @[@(DEC_DIM), @(DEC_DIM)]);
                L.cpb = gb.tensor(file.load_tensor_f32(p + "cross_attn.proj.bias"), @[@(DEC_DIM)]);

                L.n2w = gb.tensor(file.load_tensor_f32(p + "norm2.weight"), @[@(DEC_DIM)]);
                L.n2b = gb.tensor(file.load_tensor_f32(p + "norm2.bias"), @[@(DEC_DIM)]);
                L.f1w = gb.tensor(file.load_tensor_f32(p + "mlp.fc1.weight"), @[@(DEC_MLP), @(DEC_DIM)]);
                L.f1b = gb.tensor(file.load_tensor_f32(p + "mlp.fc1.bias"), @[@(DEC_MLP)]);
                L.f2w = gb.tensor(file.load_tensor_f32(p + "mlp.fc2.weight"), @[@(DEC_DIM), @(DEC_MLP)]);
                L.f2b = gb.tensor(file.load_tensor_f32(p + "mlp.fc2.bias"), @[@(DEC_DIM)]);
            }

            // Project encoder output
            MPSGraphTensor* dec_input = gb.linear(encoder_out, enc2dec_w, enc2dec_b);
            MPSGraphTensor* enc_proj = dec_input;

            // Decoder forward with hook capture
            x = dec_input;
            MPSGraphTensor* hooks[4];  // [encoder, dec@6, dec@9, dec@12]
            hooks[0] = encoder_out;  // First hook is encoder output

            for (int i = 0; i < DEC_DEPTH; i++) {
                auto& L = dec_layers[i];

                // Self-attention
                MPSGraphTensor* res = x;
                x = gb.layer_norm(x, L.n1w, L.n1b);
                x = gb.self_attention(x, L.qkvw, L.qkvb, L.pw, L.pb, DEC_HEADS, DEC_HEAD_DIM);
                x = gb.add(x, res);

                // Cross-attention
                res = x;
                MPSGraphTensor* x_norm = gb.layer_norm(x, L.n3w, L.n3b);
                MPSGraphTensor* y_norm = gb.layer_norm(enc_proj, L.nyw, L.nyb);
                MPSGraphTensor* cross = gb.cross_attention(x_norm, y_norm,
                    L.cqw, L.cqb, L.ckw, L.ckb, L.cvw, L.cvb, L.cpw, L.cpb, DEC_HEADS, DEC_HEAD_DIM);
                x = gb.add(cross, res);

                // MLP
                res = x;
                x = gb.layer_norm(x, L.n2w, L.n2b);
                x = gb.linear(x, L.f1w, L.f1b);
                x = gb.gelu(x);
                x = gb.linear(x, L.f2w, L.f2b);
                x = gb.add(x, res);

                // Capture hooks at layers 5, 8, 11 (0-indexed)
                if (i == 5) hooks[1] = x;
                if (i == 8) hooks[2] = x;
                if (i == 11) hooks[3] = x;
            }
            MPSGraphTensor* decoder_out = x;

            printf("[3] Graph built: Encoder + Decoder\n\n");

            // ================================================================
            // Benchmark
            // ================================================================

            // Create random input
            std::vector<float> input_data(NUM_PATCHES * ENC_DIM);
            for (auto& v : input_data) v = (float)(rand() % 1000) / 1000.0f - 0.5f;

            MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                                                shape:@[@(NUM_PATCHES), @(ENC_DIM)]];
            MPSNDArray* input_arr = [[MPSNDArray alloc] initWithDevice:gb.device_ descriptor:desc];
            [input_arr writeBytes:(void*)input_data.data() strideBytes:nil];
            MPSGraphTensorData* input_td = [[MPSGraphTensorData alloc] initWithMPSNDArray:input_arr];

            NSDictionary* feeds = @{enc_input: input_td};

            // Warmup
            printf("[test] Warmup (3 iterations)...\n");
            for (int i = 0; i < 3; i++) {
                [gb.graph_ runWithMTLCommandQueue:gb.queue_ feeds:feeds
                                    targetTensors:@[decoder_out] targetOperations:nil];
            }

            // Benchmark
            printf("[test] Benchmarking (10 iterations)...\n");
            const int iters = 10;
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iters; i++) {
                [gb.graph_ runWithMTLCommandQueue:gb.queue_ feeds:feeds
                                    targetTensors:@[decoder_out] targetOperations:nil];
            }

            auto end = std::chrono::high_resolution_clock::now();
            double ms_per_iter = std::chrono::duration<double, std::milli>(end - start).count() / iters;

            printf("\n=== RESULTS ===\n");
            printf("Encoder (24L) + Decoder (12L): %.2f ms\n", ms_per_iter);
            printf("\nFor full MASt3R (2 images, estimated):\n");
            printf("  2x (Enc + Dec):    %.0f ms\n", ms_per_iter * 2);
            printf("  2x DPT heads:      ~36 ms\n");
            printf("  Local features:    ~20 ms\n");
            printf("  --------------------------\n");
            printf("  TOTAL:            ~%.0f ms\n", ms_per_iter * 2 + 56);
            printf("\n  vs Metal current: ~8000 ms\n");
            printf("  Speedup:          ~%.0fx\n\n", 8000.0 / (ms_per_iter * 2 + 56));

            // Verify output
            NSDictionary* results = [gb.graph_ runWithMTLCommandQueue:gb.queue_ feeds:feeds
                                                        targetTensors:@[decoder_out] targetOperations:nil];
            MPSGraphTensorData* out_data = results[decoder_out];
            std::vector<float> out_cpu(NUM_PATCHES * DEC_DIM);
            [[out_data mpsndarray] readBytes:out_cpu.data() strideBytes:nil];

            int nan_count = 0;
            for (auto v : out_cpu) if (std::isnan(v) || std::isinf(v)) nan_count++;
            printf("[test] Output: nan=%d/%zu\n", nan_count, out_cpu.size());
            printf("[test] %s\n", nan_count == 0 ? "PASSED!" : "FAILED - NaN detected");

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

        printf("=== MASt3R MPSGraph Pipeline Benchmark ===\n\n");
        test_full_pipeline(path);
        return 0;
    }
}
