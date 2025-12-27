// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
// Test MPSGraph ViT encoder with real weights

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

#include "../common/safetensors.hpp"

namespace mast3r_test {

// Configuration
constexpr int EMBED_DIM = 1024;
constexpr int NUM_HEADS = 16;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;  // 64
constexpr int MLP_DIM = EMBED_DIM * 4;  // 4096
constexpr int PATCH_SIZE = 16;
constexpr int NUM_LAYERS = 24;
constexpr float LN_EPS = 1e-6f;

// Single encoder layer using SDPA
class API_AVAILABLE(macos(15.0)) EncoderLayer {
public:
    MPSGraph* graph_;

    // Weights as graph constants
    MPSGraphTensor* norm1_weight = nil;
    MPSGraphTensor* norm1_bias = nil;
    MPSGraphTensor* qkv_weight = nil;
    MPSGraphTensor* qkv_bias = nil;
    MPSGraphTensor* proj_weight = nil;
    MPSGraphTensor* proj_bias = nil;
    MPSGraphTensor* norm2_weight = nil;
    MPSGraphTensor* norm2_bias = nil;
    MPSGraphTensor* fc1_weight = nil;
    MPSGraphTensor* fc1_bias = nil;
    MPSGraphTensor* fc2_weight = nil;
    MPSGraphTensor* fc2_bias = nil;

    EncoderLayer(MPSGraph* graph) : graph_(graph) {}

    void load_weights(const mast3r::safetensors::SafetensorsFile& file, int layer_idx) {
        std::string prefix = "enc_blocks." + std::to_string(layer_idx) + ".";

        // Load and convert weights
        norm1_weight = create_tensor(file.load_tensor_f32(prefix + "norm1.weight"), @[@(EMBED_DIM)]);
        norm1_bias = create_tensor(file.load_tensor_f32(prefix + "norm1.bias"), @[@(EMBED_DIM)]);

        // QKV: [3*D, D] -> need transpose for matmul
        auto qkv_w = file.load_tensor_f32(prefix + "attn.qkv.weight");
        qkv_weight = create_tensor(qkv_w, @[@(3 * EMBED_DIM), @(EMBED_DIM)]);
        qkv_bias = create_tensor(file.load_tensor_f32(prefix + "attn.qkv.bias"), @[@(3 * EMBED_DIM)]);

        // Proj: [D, D]
        auto proj_w = file.load_tensor_f32(prefix + "attn.proj.weight");
        proj_weight = create_tensor(proj_w, @[@(EMBED_DIM), @(EMBED_DIM)]);
        proj_bias = create_tensor(file.load_tensor_f32(prefix + "attn.proj.bias"), @[@(EMBED_DIM)]);

        // LayerNorm 2
        norm2_weight = create_tensor(file.load_tensor_f32(prefix + "norm2.weight"), @[@(EMBED_DIM)]);
        norm2_bias = create_tensor(file.load_tensor_f32(prefix + "norm2.bias"), @[@(EMBED_DIM)]);

        // MLP FC1: [4D, D]
        auto fc1_w = file.load_tensor_f32(prefix + "mlp.fc1.weight");
        fc1_weight = create_tensor(fc1_w, @[@(MLP_DIM), @(EMBED_DIM)]);
        fc1_bias = create_tensor(file.load_tensor_f32(prefix + "mlp.fc1.bias"), @[@(MLP_DIM)]);

        // MLP FC2: [D, 4D]
        auto fc2_w = file.load_tensor_f32(prefix + "mlp.fc2.weight");
        fc2_weight = create_tensor(fc2_w, @[@(EMBED_DIM), @(MLP_DIM)]);
        fc2_bias = create_tensor(file.load_tensor_f32(prefix + "mlp.fc2.bias"), @[@(EMBED_DIM)]);
    }

    MPSGraphTensor* forward(MPSGraphTensor* x) {
        // x: [N, D] where N = num_patches
        const float scale = 1.0f / sqrtf((float)HEAD_DIM);

        // === Self-Attention ===
        MPSGraphTensor* residual = x;

        // LayerNorm 1
        x = layer_norm(x, norm1_weight, norm1_bias);

        // QKV projection: [N, D] @ [D, 3D] = [N, 3D]
        // Note: weight is [3D, D], so we need transpose
        MPSGraphTensor* qkv_wt = [graph_ transposeTensor:qkv_weight
                                               dimension:0
                                           withDimension:1
                                                    name:nil];
        MPSGraphTensor* qkv = [graph_ matrixMultiplicationWithPrimaryTensor:x
                                                            secondaryTensor:qkv_wt
                                                                       name:nil];
        qkv = [graph_ additionWithPrimaryTensor:qkv secondaryTensor:qkv_bias name:nil];

        // Split into Q, K, V: each [N, D]
        NSArray<MPSGraphTensor*>* qkv_split = [graph_ splitTensor:qkv
                                                        numSplits:3
                                                             axis:-1
                                                             name:nil];

        // Reshape for multi-head: [N, D] -> [1, H, N, head_dim]
        NSArray<NSNumber*>* attn_shape = @[@1, @(NUM_HEADS), @(-1), @(HEAD_DIM)];
        MPSGraphTensor* Q = [graph_ reshapeTensor:qkv_split[0] withShape:attn_shape name:nil];
        MPSGraphTensor* K = [graph_ reshapeTensor:qkv_split[1] withShape:attn_shape name:nil];
        MPSGraphTensor* V = [graph_ reshapeTensor:qkv_split[2] withShape:attn_shape name:nil];

        // SDPA - native Apple implementation
        MPSGraphTensor* attn_out = [graph_ scaledDotProductAttentionWithQueryTensor:Q
                                                                          keyTensor:K
                                                                        valueTensor:V
                                                                         maskTensor:nil
                                                                              scale:scale
                                                                               name:nil];

        // Reshape back: [1, H, N, head_dim] -> [N, D]
        attn_out = [graph_ reshapeTensor:attn_out withShape:@[@(-1), @(EMBED_DIM)] name:nil];

        // Output projection: [N, D] @ [D, D] = [N, D]
        MPSGraphTensor* proj_wt = [graph_ transposeTensor:proj_weight
                                                dimension:0
                                            withDimension:1
                                                     name:nil];
        attn_out = [graph_ matrixMultiplicationWithPrimaryTensor:attn_out
                                                 secondaryTensor:proj_wt
                                                            name:nil];
        attn_out = [graph_ additionWithPrimaryTensor:attn_out secondaryTensor:proj_bias name:nil];

        // Residual
        x = [graph_ additionWithPrimaryTensor:attn_out secondaryTensor:residual name:nil];

        // === MLP ===
        residual = x;

        // LayerNorm 2
        x = layer_norm(x, norm2_weight, norm2_bias);

        // FC1: [N, D] @ [D, 4D] = [N, 4D]
        MPSGraphTensor* fc1_wt = [graph_ transposeTensor:fc1_weight
                                               dimension:0
                                           withDimension:1
                                                    name:nil];
        x = [graph_ matrixMultiplicationWithPrimaryTensor:x secondaryTensor:fc1_wt name:nil];
        x = [graph_ additionWithPrimaryTensor:x secondaryTensor:fc1_bias name:nil];

        // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        x = gelu(x);

        // FC2: [N, 4D] @ [4D, D] = [N, D]
        MPSGraphTensor* fc2_wt = [graph_ transposeTensor:fc2_weight
                                               dimension:0
                                           withDimension:1
                                                    name:nil];
        x = [graph_ matrixMultiplicationWithPrimaryTensor:x secondaryTensor:fc2_wt name:nil];
        x = [graph_ additionWithPrimaryTensor:x secondaryTensor:fc2_bias name:nil];

        // Residual
        x = [graph_ additionWithPrimaryTensor:x secondaryTensor:residual name:nil];

        return x;
    }

private:
    MPSGraphTensor* create_tensor(const std::vector<float>& data, NSArray<NSNumber*>* shape) {
        NSData* nsdata = [NSData dataWithBytes:data.data() length:data.size() * sizeof(float)];
        return [graph_ constantWithData:nsdata shape:shape dataType:MPSDataTypeFloat32];
    }

    // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    MPSGraphTensor* gelu(MPSGraphTensor* x) {
        // inv_sqrt2 = 1/sqrt(2) ≈ 0.7071067811865475
        MPSGraphTensor* inv_sqrt2 = [graph_ constantWithScalar:0.7071067811865475
                                                         shape:@[@1]
                                                      dataType:MPSDataTypeFloat32];
        MPSGraphTensor* half = [graph_ constantWithScalar:0.5
                                                    shape:@[@1]
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensor* one = [graph_ constantWithScalar:1.0
                                                   shape:@[@1]
                                                dataType:MPSDataTypeFloat32];

        // x / sqrt(2)
        MPSGraphTensor* x_scaled = [graph_ multiplicationWithPrimaryTensor:x
                                                           secondaryTensor:inv_sqrt2
                                                                      name:nil];
        // erf(x / sqrt(2))
        MPSGraphTensor* erf_val = [graph_ erfWithTensor:x_scaled name:nil];

        // 1 + erf(...)
        MPSGraphTensor* one_plus_erf = [graph_ additionWithPrimaryTensor:one
                                                         secondaryTensor:erf_val
                                                                    name:nil];
        // 0.5 * (1 + erf(...))
        MPSGraphTensor* cdf = [graph_ multiplicationWithPrimaryTensor:half
                                                      secondaryTensor:one_plus_erf
                                                                 name:nil];
        // x * 0.5 * (1 + erf(...))
        return [graph_ multiplicationWithPrimaryTensor:x secondaryTensor:cdf name:nil];
    }

    MPSGraphTensor* layer_norm(MPSGraphTensor* x,
                               MPSGraphTensor* weight,
                               MPSGraphTensor* bias) {
        // Mean
        MPSGraphTensor* mean = [graph_ meanOfTensor:x axes:@[@(-1)] name:nil];

        // Variance: E[(x - mean)^2]
        MPSGraphTensor* centered = [graph_ subtractionWithPrimaryTensor:x
                                                        secondaryTensor:mean
                                                                   name:nil];
        MPSGraphTensor* sq = [graph_ squareWithTensor:centered name:nil];
        MPSGraphTensor* var = [graph_ meanOfTensor:sq axes:@[@(-1)] name:nil];

        // Normalize: (x - mean) / sqrt(var + eps)
        MPSGraphTensor* eps = [graph_ constantWithScalar:LN_EPS shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* var_eps = [graph_ additionWithPrimaryTensor:var secondaryTensor:eps name:nil];
        MPSGraphTensor* std = [graph_ squareRootWithTensor:var_eps name:nil];
        MPSGraphTensor* normalized = [graph_ divisionWithPrimaryTensor:centered
                                                       secondaryTensor:std
                                                                  name:nil];

        // Scale and shift
        normalized = [graph_ multiplicationWithPrimaryTensor:normalized
                                             secondaryTensor:weight
                                                        name:nil];
        normalized = [graph_ additionWithPrimaryTensor:normalized
                                       secondaryTensor:bias
                                                  name:nil];

        return normalized;
    }
};

// Full encoder test
bool test_encoder(const std::string& weights_path, int num_layers = 1) {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            printf("[test] Loading weights from: %s\n", weights_path.c_str());

            mast3r::safetensors::SafetensorsFile file(weights_path);
            printf("[test] Loaded safetensors with %zu tensors\n", file.num_tensors());

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                printf("[test] No Metal device\n");
                return false;
            }
            printf("[test] Metal device: %s\n", [[device name] UTF8String]);

            id<MTLCommandQueue> queue = [device newCommandQueue];
            MPSGraph* graph = [[MPSGraph alloc] init];

            // Create encoder layers
            printf("[test] Building graph for %d layers...\n", num_layers);
            std::vector<EncoderLayer> layers;
            for (int i = 0; i < num_layers; i++) {
                layers.emplace_back(graph);
                layers.back().load_weights(file, i);
            }

            // Create input placeholder: [N, D] where N = num_patches
            // For 512x384 image: N = (512/16) * (384/16) = 32 * 24 = 768
            const int num_patches = 768;

            MPSGraphTensor* input = [graph placeholderWithShape:@[@(num_patches), @(EMBED_DIM)]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"input"];

            // Build forward pass
            MPSGraphTensor* x = input;
            for (int i = 0; i < num_layers; i++) {
                x = layers[i].forward(x);
            }
            MPSGraphTensor* output = x;

            printf("[test] Graph built successfully\n");

            // Create random input data
            std::vector<float> input_data(num_patches * EMBED_DIM);
            for (size_t i = 0; i < input_data.size(); i++) {
                input_data[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
            }

            // Create input tensor data
            MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor
                descriptorWithDataType:MPSDataTypeFloat32
                                 shape:@[@(num_patches), @(EMBED_DIM)]];
            MPSNDArray* input_ndarray = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];
            [input_ndarray writeBytes:input_data.data() strideBytes:nil];
            MPSGraphTensorData* input_td = [[MPSGraphTensorData alloc] initWithMPSNDArray:input_ndarray];

            NSDictionary* feeds = @{input: input_td};

            // Warmup
            printf("[test] Warmup...\n");
            for (int i = 0; i < 3; i++) {
                [graph runWithMTLCommandQueue:queue
                                        feeds:feeds
                                targetTensors:@[output]
                             targetOperations:nil];
            }

            // Benchmark
            printf("[test] Benchmarking...\n");
            const int iters = 20;
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iters; i++) {
                [graph runWithMTLCommandQueue:queue
                                        feeds:feeds
                                targetTensors:@[output]
                             targetOperations:nil];
            }

            auto end = std::chrono::high_resolution_clock::now();
            double ms_total = std::chrono::duration<double, std::milli>(end - start).count();
            double ms_per_iter = ms_total / iters;

            printf("[test] %d layers: %.2f ms/iter (%.2f ms per layer)\n",
                   num_layers, ms_per_iter, ms_per_iter / num_layers);

            // Verify output
            NSDictionary* results = [graph runWithMTLCommandQueue:queue
                                                            feeds:feeds
                                                    targetTensors:@[output]
                                                 targetOperations:nil];

            MPSGraphTensorData* output_data = results[output];
            std::vector<float> output_cpu(num_patches * EMBED_DIM);
            MPSNDArray* output_ndarray = [output_data mpsndarray];
            [output_ndarray readBytes:output_cpu.data() strideBytes:nil];

            // Check for NaN/Inf
            int nan_count = 0;
            float min_val = INFINITY, max_val = -INFINITY;
            for (size_t i = 0; i < output_cpu.size(); i++) {
                float v = output_cpu[i];
                if (std::isnan(v) || std::isinf(v)) {
                    nan_count++;
                } else {
                    min_val = std::min(min_val, v);
                    max_val = std::max(max_val, v);
                }
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
    } else {
        printf("[test] macOS 15+ required\n");
        return false;
    }
}

} // namespace mast3r_test

int main(int argc, char* argv[]) {
    @autoreleasepool {
        // Get weights path from env or use default
        const char* env_path = getenv("MAST3R_WEIGHTS");
        std::string weights_path;

        if (env_path) {
            weights_path = env_path;
        } else {
            // Default path
            const char* home = getenv("HOME");
            weights_path = std::string(home) +
                "/.cache/mast3r_runtime/safetensors/mast3r_vit_large/unified.safetensors";
        }

        printf("=== MPSGraph Encoder Test ===\n\n");

        // Test with different numbers of layers
        int test_layers[] = {1, 4, 12, 24};
        for (int num_layers : test_layers) {
            printf("\n--- Testing %d layers ---\n", num_layers);
            if (!mast3r_test::test_encoder(weights_path, num_layers)) {
                printf("Test failed for %d layers\n", num_layers);
            }
        }

        return 0;
    }
}
