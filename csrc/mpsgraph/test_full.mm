// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
// Test complete encoder + decoder with MPSGraph SDPA

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

#include "../common/safetensors.hpp"

// ============================================================================
// Configuration
// ============================================================================

constexpr int ENC_EMBED_DIM = 1024;
constexpr int ENC_NUM_HEADS = 16;
constexpr int ENC_HEAD_DIM = 64;
constexpr int ENC_MLP_DIM = 4096;
constexpr int ENC_DEPTH = 24;

constexpr int DEC_EMBED_DIM = 768;
constexpr int DEC_NUM_HEADS = 12;
constexpr int DEC_HEAD_DIM = 64;
constexpr int DEC_MLP_DIM = 3072;
constexpr int DEC_DEPTH = 12;

constexpr int PATCH_SIZE = 16;
constexpr float LN_EPS = 1e-6f;

// ============================================================================
// MPSGraph Builder
// ============================================================================

class API_AVAILABLE(macos(15.0)) GraphBuilder {
public:
    MPSGraph* graph_;
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;

    GraphBuilder() {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newCommandQueue];
        graph_ = [[MPSGraph alloc] init];
    }

    // Create tensor from data
    MPSGraphTensor* tensor(const std::vector<float>& data, NSArray<NSNumber*>* shape) {
        NSData* nsdata = [NSData dataWithBytes:data.data() length:data.size() * sizeof(float)];
        return [graph_ constantWithData:nsdata shape:shape dataType:MPSDataTypeFloat32];
    }

    // LayerNorm
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

    // GELU
    MPSGraphTensor* gelu(MPSGraphTensor* x) {
        MPSGraphTensor* inv_sqrt2 = [graph_ constantWithScalar:0.7071067811865475 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* half = [graph_ constantWithScalar:0.5 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* one = [graph_ constantWithScalar:1.0 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* x_scaled = [graph_ multiplicationWithPrimaryTensor:x secondaryTensor:inv_sqrt2 name:nil];
        MPSGraphTensor* erf_val = [graph_ erfWithTensor:x_scaled name:nil];
        MPSGraphTensor* one_plus_erf = [graph_ additionWithPrimaryTensor:one secondaryTensor:erf_val name:nil];
        MPSGraphTensor* cdf = [graph_ multiplicationWithPrimaryTensor:half secondaryTensor:one_plus_erf name:nil];
        return [graph_ multiplicationWithPrimaryTensor:x secondaryTensor:cdf name:nil];
    }

    // Linear
    MPSGraphTensor* linear(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b) {
        MPSGraphTensor* wt = [graph_ transposeTensor:w dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* out = [graph_ matrixMultiplicationWithPrimaryTensor:x secondaryTensor:wt name:nil];
        return [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
    }

    // Self-attention with SDPA
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

        MPSGraphTensor* attn_out = [graph_ scaledDotProductAttentionWithQueryTensor:Q
                                                                          keyTensor:K
                                                                        valueTensor:V
                                                                         maskTensor:nil
                                                                              scale:scale
                                                                               name:nil];

        attn_out = [graph_ reshapeTensor:attn_out withShape:@[@(-1), @(embed_dim)] name:nil];
        return linear(attn_out, proj_w, proj_b);
    }

    // Cross-attention with SDPA
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

        MPSGraphTensor* attn_out = [graph_ scaledDotProductAttentionWithQueryTensor:Q
                                                                          keyTensor:K
                                                                        valueTensor:V
                                                                         maskTensor:nil
                                                                              scale:scale
                                                                               name:nil];

        attn_out = [graph_ reshapeTensor:attn_out withShape:@[@(-1), @(embed_dim)] name:nil];
        return linear(attn_out, proj_w, proj_b);
    }
};

// ============================================================================
// Weight storage
// ============================================================================

struct EncoderLayer {
    MPSGraphTensor *norm1_w, *norm1_b;
    MPSGraphTensor *qkv_w, *qkv_b;
    MPSGraphTensor *proj_w, *proj_b;
    MPSGraphTensor *norm2_w, *norm2_b;
    MPSGraphTensor *fc1_w, *fc1_b;
    MPSGraphTensor *fc2_w, *fc2_b;
};

struct DecoderLayer {
    // Self-attention
    MPSGraphTensor *norm1_w, *norm1_b;
    MPSGraphTensor *qkv_w, *qkv_b;
    MPSGraphTensor *proj_w, *proj_b;
    // Cross-attention
    MPSGraphTensor *norm_y_w, *norm_y_b;
    MPSGraphTensor *norm3_w, *norm3_b;
    MPSGraphTensor *cross_q_w, *cross_q_b;
    MPSGraphTensor *cross_k_w, *cross_k_b;
    MPSGraphTensor *cross_v_w, *cross_v_b;
    MPSGraphTensor *cross_proj_w, *cross_proj_b;
    // MLP
    MPSGraphTensor *norm2_w, *norm2_b;
    MPSGraphTensor *fc1_w, *fc1_b;
    MPSGraphTensor *fc2_w, *fc2_b;
};

// ============================================================================
// Test
// ============================================================================

bool test_encoder_decoder(const std::string& weights_path) {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            printf("[test] Loading weights from: %s\n", weights_path.c_str());
            mast3r::safetensors::SafetensorsFile file(weights_path);
            printf("[test] Loaded %zu tensors\n", file.num_tensors());

            GraphBuilder gb;
            printf("[test] Metal device: %s\n", [[gb.device_ name] UTF8String]);

            // Load encoder weights
            printf("[test] Loading encoder weights (24 layers)...\n");
            std::vector<EncoderLayer> enc_layers(ENC_DEPTH);

            auto enc_norm_w = gb.tensor(file.load_tensor_f32("enc_norm.weight"), @[@(ENC_EMBED_DIM)]);
            auto enc_norm_b = gb.tensor(file.load_tensor_f32("enc_norm.bias"), @[@(ENC_EMBED_DIM)]);

            for (int i = 0; i < ENC_DEPTH; i++) {
                std::string p = "enc_blocks." + std::to_string(i) + ".";
                auto& L = enc_layers[i];
                L.norm1_w = gb.tensor(file.load_tensor_f32(p + "norm1.weight"), @[@(ENC_EMBED_DIM)]);
                L.norm1_b = gb.tensor(file.load_tensor_f32(p + "norm1.bias"), @[@(ENC_EMBED_DIM)]);
                L.qkv_w = gb.tensor(file.load_tensor_f32(p + "attn.qkv.weight"), @[@(3*ENC_EMBED_DIM), @(ENC_EMBED_DIM)]);
                L.qkv_b = gb.tensor(file.load_tensor_f32(p + "attn.qkv.bias"), @[@(3*ENC_EMBED_DIM)]);
                L.proj_w = gb.tensor(file.load_tensor_f32(p + "attn.proj.weight"), @[@(ENC_EMBED_DIM), @(ENC_EMBED_DIM)]);
                L.proj_b = gb.tensor(file.load_tensor_f32(p + "attn.proj.bias"), @[@(ENC_EMBED_DIM)]);
                L.norm2_w = gb.tensor(file.load_tensor_f32(p + "norm2.weight"), @[@(ENC_EMBED_DIM)]);
                L.norm2_b = gb.tensor(file.load_tensor_f32(p + "norm2.bias"), @[@(ENC_EMBED_DIM)]);
                L.fc1_w = gb.tensor(file.load_tensor_f32(p + "mlp.fc1.weight"), @[@(ENC_MLP_DIM), @(ENC_EMBED_DIM)]);
                L.fc1_b = gb.tensor(file.load_tensor_f32(p + "mlp.fc1.bias"), @[@(ENC_MLP_DIM)]);
                L.fc2_w = gb.tensor(file.load_tensor_f32(p + "mlp.fc2.weight"), @[@(ENC_EMBED_DIM), @(ENC_MLP_DIM)]);
                L.fc2_b = gb.tensor(file.load_tensor_f32(p + "mlp.fc2.bias"), @[@(ENC_EMBED_DIM)]);
            }

            // Load decoder weights
            printf("[test] Loading decoder weights (12 layers)...\n");
            std::vector<DecoderLayer> dec_layers(DEC_DEPTH);

            for (int i = 0; i < DEC_DEPTH; i++) {
                std::string p = "dec_blocks." + std::to_string(i) + ".";
                auto& L = dec_layers[i];
                L.norm1_w = gb.tensor(file.load_tensor_f32(p + "norm1.weight"), @[@(DEC_EMBED_DIM)]);
                L.norm1_b = gb.tensor(file.load_tensor_f32(p + "norm1.bias"), @[@(DEC_EMBED_DIM)]);
                L.qkv_w = gb.tensor(file.load_tensor_f32(p + "attn.qkv.weight"), @[@(3*DEC_EMBED_DIM), @(DEC_EMBED_DIM)]);
                L.qkv_b = gb.tensor(file.load_tensor_f32(p + "attn.qkv.bias"), @[@(3*DEC_EMBED_DIM)]);
                L.proj_w = gb.tensor(file.load_tensor_f32(p + "attn.proj.weight"), @[@(DEC_EMBED_DIM), @(DEC_EMBED_DIM)]);
                L.proj_b = gb.tensor(file.load_tensor_f32(p + "attn.proj.bias"), @[@(DEC_EMBED_DIM)]);

                L.norm_y_w = gb.tensor(file.load_tensor_f32(p + "norm_y.weight"), @[@(DEC_EMBED_DIM)]);
                L.norm_y_b = gb.tensor(file.load_tensor_f32(p + "norm_y.bias"), @[@(DEC_EMBED_DIM)]);
                L.norm3_w = gb.tensor(file.load_tensor_f32(p + "norm3.weight"), @[@(DEC_EMBED_DIM)]);
                L.norm3_b = gb.tensor(file.load_tensor_f32(p + "norm3.bias"), @[@(DEC_EMBED_DIM)]);

                // Cross-attention: projk/projv take encoder features (D=1024 -> D=768)
                // But actually in MASt3R, the encoder output is projected to 768 first
                // For now, we'll handle this by using decoder-dim projections
                L.cross_q_w = gb.tensor(file.load_tensor_f32(p + "cross_attn.projq.weight"), @[@(DEC_EMBED_DIM), @(DEC_EMBED_DIM)]);
                L.cross_q_b = gb.tensor(file.load_tensor_f32(p + "cross_attn.projq.bias"), @[@(DEC_EMBED_DIM)]);
                L.cross_k_w = gb.tensor(file.load_tensor_f32(p + "cross_attn.projk.weight"), @[@(DEC_EMBED_DIM), @(DEC_EMBED_DIM)]);
                L.cross_k_b = gb.tensor(file.load_tensor_f32(p + "cross_attn.projk.bias"), @[@(DEC_EMBED_DIM)]);
                L.cross_v_w = gb.tensor(file.load_tensor_f32(p + "cross_attn.projv.weight"), @[@(DEC_EMBED_DIM), @(DEC_EMBED_DIM)]);
                L.cross_v_b = gb.tensor(file.load_tensor_f32(p + "cross_attn.projv.bias"), @[@(DEC_EMBED_DIM)]);
                L.cross_proj_w = gb.tensor(file.load_tensor_f32(p + "cross_attn.proj.weight"), @[@(DEC_EMBED_DIM), @(DEC_EMBED_DIM)]);
                L.cross_proj_b = gb.tensor(file.load_tensor_f32(p + "cross_attn.proj.bias"), @[@(DEC_EMBED_DIM)]);

                L.norm2_w = gb.tensor(file.load_tensor_f32(p + "norm2.weight"), @[@(DEC_EMBED_DIM)]);
                L.norm2_b = gb.tensor(file.load_tensor_f32(p + "norm2.bias"), @[@(DEC_EMBED_DIM)]);
                L.fc1_w = gb.tensor(file.load_tensor_f32(p + "mlp.fc1.weight"), @[@(DEC_MLP_DIM), @(DEC_EMBED_DIM)]);
                L.fc1_b = gb.tensor(file.load_tensor_f32(p + "mlp.fc1.bias"), @[@(DEC_MLP_DIM)]);
                L.fc2_w = gb.tensor(file.load_tensor_f32(p + "mlp.fc2.weight"), @[@(DEC_EMBED_DIM), @(DEC_MLP_DIM)]);
                L.fc2_b = gb.tensor(file.load_tensor_f32(p + "mlp.fc2.bias"), @[@(DEC_EMBED_DIM)]);
            }

            // Encoder-to-decoder projection (from 1024 to 768)
            // In MASt3R this is done via a linear projection
            auto enc2dec_w = gb.tensor(file.load_tensor_f32("decoder_embed.weight"), @[@(DEC_EMBED_DIM), @(ENC_EMBED_DIM)]);
            auto enc2dec_b = gb.tensor(file.load_tensor_f32("decoder_embed.bias"), @[@(DEC_EMBED_DIM)]);

            printf("[test] Weights loaded, building graph...\n");

            // Create input placeholder
            const int num_patches = 768;  // 512x384 / 16x16
            MPSGraphTensor* input = [gb.graph_ placeholderWithShape:@[@(num_patches), @(ENC_EMBED_DIM)]
                                                           dataType:MPSDataTypeFloat32
                                                               name:@"input"];

            // Build encoder
            MPSGraphTensor* x = input;
            for (int i = 0; i < ENC_DEPTH; i++) {
                auto& L = enc_layers[i];
                MPSGraphTensor* residual = x;
                x = gb.layer_norm(x, L.norm1_w, L.norm1_b);
                x = gb.self_attention(x, L.qkv_w, L.qkv_b, L.proj_w, L.proj_b, ENC_NUM_HEADS, ENC_HEAD_DIM);
                x = [gb.graph_ additionWithPrimaryTensor:x secondaryTensor:residual name:nil];

                residual = x;
                x = gb.layer_norm(x, L.norm2_w, L.norm2_b);
                x = gb.linear(x, L.fc1_w, L.fc1_b);
                x = gb.gelu(x);
                x = gb.linear(x, L.fc2_w, L.fc2_b);
                x = [gb.graph_ additionWithPrimaryTensor:x secondaryTensor:residual name:nil];
            }
            x = gb.layer_norm(x, enc_norm_w, enc_norm_b);
            MPSGraphTensor* encoder_out = x;

            // Project encoder output to decoder dim
            MPSGraphTensor* dec_input = gb.linear(encoder_out, enc2dec_w, enc2dec_b);
            // Also project encoder_out for cross-attention K/V
            MPSGraphTensor* encoder_proj = dec_input;  // Same projection

            // Build decoder
            x = dec_input;
            for (int i = 0; i < DEC_DEPTH; i++) {
                auto& L = dec_layers[i];

                // Self-attention
                MPSGraphTensor* residual = x;
                x = gb.layer_norm(x, L.norm1_w, L.norm1_b);
                x = gb.self_attention(x, L.qkv_w, L.qkv_b, L.proj_w, L.proj_b, DEC_NUM_HEADS, DEC_HEAD_DIM);
                x = [gb.graph_ additionWithPrimaryTensor:x secondaryTensor:residual name:nil];

                // Cross-attention
                residual = x;
                MPSGraphTensor* x_normed = gb.layer_norm(x, L.norm3_w, L.norm3_b);
                MPSGraphTensor* y_normed = gb.layer_norm(encoder_proj, L.norm_y_w, L.norm_y_b);
                MPSGraphTensor* cross_out = gb.cross_attention(x_normed, y_normed,
                    L.cross_q_w, L.cross_q_b, L.cross_k_w, L.cross_k_b,
                    L.cross_v_w, L.cross_v_b, L.cross_proj_w, L.cross_proj_b,
                    DEC_NUM_HEADS, DEC_HEAD_DIM);
                x = [gb.graph_ additionWithPrimaryTensor:cross_out secondaryTensor:residual name:nil];

                // MLP
                residual = x;
                x = gb.layer_norm(x, L.norm2_w, L.norm2_b);
                x = gb.linear(x, L.fc1_w, L.fc1_b);
                x = gb.gelu(x);
                x = gb.linear(x, L.fc2_w, L.fc2_b);
                x = [gb.graph_ additionWithPrimaryTensor:x secondaryTensor:residual name:nil];
            }
            MPSGraphTensor* decoder_out = x;

            printf("[test] Graph built (encoder + decoder)\n");

            // Create random input
            std::vector<float> input_data(num_patches * ENC_EMBED_DIM);
            for (size_t i = 0; i < input_data.size(); i++) {
                input_data[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
            }

            MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor
                descriptorWithDataType:MPSDataTypeFloat32
                                 shape:@[@(num_patches), @(ENC_EMBED_DIM)]];
            MPSNDArray* input_ndarray = [[MPSNDArray alloc] initWithDevice:gb.device_ descriptor:desc];
            [input_ndarray writeBytes:input_data.data() strideBytes:nil];
            MPSGraphTensorData* input_td = [[MPSGraphTensorData alloc] initWithMPSNDArray:input_ndarray];

            NSDictionary* feeds = @{input: input_td};

            // Warmup
            printf("[test] Warmup...\n");
            for (int i = 0; i < 3; i++) {
                [gb.graph_ runWithMTLCommandQueue:gb.queue_
                                            feeds:feeds
                                    targetTensors:@[decoder_out]
                                 targetOperations:nil];
            }

            // Benchmark
            printf("[test] Benchmarking...\n");
            const int iters = 10;
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iters; i++) {
                [gb.graph_ runWithMTLCommandQueue:gb.queue_
                                            feeds:feeds
                                    targetTensors:@[decoder_out]
                                 targetOperations:nil];
            }

            auto end = std::chrono::high_resolution_clock::now();
            double ms_total = std::chrono::duration<double, std::milli>(end - start).count();
            double ms_per_iter = ms_total / iters;

            printf("[test] Encoder(24L) + Decoder(12L): %.2f ms/iter\n", ms_per_iter);

            // Verify output
            NSDictionary* results = [gb.graph_ runWithMTLCommandQueue:gb.queue_
                                                                feeds:feeds
                                                        targetTensors:@[decoder_out]
                                                     targetOperations:nil];

            MPSGraphTensorData* output_data = results[decoder_out];
            std::vector<float> output_cpu(num_patches * DEC_EMBED_DIM);
            MPSNDArray* output_ndarray = [output_data mpsndarray];
            [output_ndarray readBytes:output_cpu.data() strideBytes:nil];

            int nan_count = 0;
            float min_val = INFINITY, max_val = -INFINITY;
            for (size_t i = 0; i < output_cpu.size(); i++) {
                float v = output_cpu[i];
                if (std::isnan(v) || std::isinf(v)) nan_count++;
                else { min_val = std::min(min_val, v); max_val = std::max(max_val, v); }
            }

            printf("[test] Output: nan=%d/%zu, range=[%.3f, %.3f]\n",
                   nan_count, output_cpu.size(), min_val, max_val);

            if (nan_count > 0) {
                printf("[test] FAILED - NaN in output\n");
                return false;
            }

            printf("[test] PASSED!\n");
            return true;
        }
    }
    return false;
}

int main() {
    @autoreleasepool {
        const char* home = getenv("HOME");
        std::string path = std::string(home) +
            "/.cache/mast3r_runtime/safetensors/mast3r_vit_large/unified.safetensors";

        printf("=== MPSGraph Encoder + Decoder Test ===\n\n");
        test_encoder_decoder(path);
        return 0;
    }
}
