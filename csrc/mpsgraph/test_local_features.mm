// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
// Test Local Features MLP with pixel shuffle

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

constexpr int IMG_W = 512;
constexpr int IMG_H = 384;
constexpr int PATCH_SIZE = 16;
constexpr int PATCH_W = IMG_W / PATCH_SIZE;  // 32
constexpr int PATCH_H = IMG_H / PATCH_SIZE;  // 24
constexpr int NUM_PATCHES = PATCH_W * PATCH_H;  // 768

constexpr int ENC_DIM = 1024;
constexpr int DEC_DIM = 768;
constexpr int DESC_DIM = 24;  // Output descriptor dimension

// Local features MLP dimensions
constexpr int LF_IN_DIM = ENC_DIM + DEC_DIM;  // 1792
constexpr int LF_HIDDEN = 7168;  // 4 * 1792
constexpr int LF_OUT_DIM = (DESC_DIM + 1) * PATCH_SIZE * PATCH_SIZE;  // 25 * 256 = 6400

// ============================================================================
// Graph Builder
// ============================================================================

class API_AVAILABLE(macos(15.0)) LocalFeaturesBuilder {
public:
    MPSGraph* graph_;
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;

    LocalFeaturesBuilder() {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newCommandQueue];
        graph_ = [[MPSGraph alloc] init];
    }

    MPSGraphTensor* tensor(const std::vector<float>& data, NSArray<NSNumber*>* shape) {
        NSData* nsdata = [NSData dataWithBytes:data.data() length:data.size() * sizeof(float)];
        return [graph_ constantWithData:nsdata shape:shape dataType:MPSDataTypeFloat32];
    }

    MPSGraphTensor* linear(MPSGraphTensor* x, MPSGraphTensor* w, MPSGraphTensor* b) {
        MPSGraphTensor* wt = [graph_ transposeTensor:w dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* out = [graph_ matrixMultiplicationWithPrimaryTensor:x secondaryTensor:wt name:nil];
        if (b) out = [graph_ additionWithPrimaryTensor:out secondaryTensor:b name:nil];
        return out;
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

    // Pixel shuffle: [pH, pW, C*r*r] -> [H, W, C] where H=pH*r, W=pW*r
    // In MASt3R: [24, 32, 6400] -> [384, 512, 25] with r=16
    MPSGraphTensor* pixel_shuffle(MPSGraphTensor* x, int upscale_factor) {
        // x shape: [1, pH, pW, C*r*r]
        // output: [1, H, W, C] where H=pH*r, W=pW*r

        NSArray<NSNumber*>* x_shape = [x shape];
        int pH = [x_shape[1] intValue];
        int pW = [x_shape[2] intValue];
        int C_rr = [x_shape[3] intValue];
        int r = upscale_factor;
        int C = C_rr / (r * r);
        int H = pH * r;
        int W = pW * r;

        // Reshape: [1, pH, pW, C*r*r] -> [1, pH, pW, r, r, C]
        MPSGraphTensor* reshaped = [graph_ reshapeTensor:x
                                               withShape:@[@1, @(pH), @(pW), @(r), @(r), @(C)]
                                                    name:nil];

        // Permute: [1, pH, pW, r, r, C] -> [1, pH, r, pW, r, C]
        reshaped = [graph_ transposeTensor:reshaped dimension:2 withDimension:3 name:nil];

        // Reshape: [1, pH, r, pW, r, C] -> [1, H, W, C]
        MPSGraphTensor* output = [graph_ reshapeTensor:reshaped
                                             withShape:@[@1, @(H), @(W), @(C)]
                                                  name:nil];

        return output;
    }

    // L2 normalize along last axis
    MPSGraphTensor* l2_normalize(MPSGraphTensor* x) {
        MPSGraphTensor* sq = [graph_ squareWithTensor:x name:nil];
        MPSGraphTensor* sum_sq = [graph_ reductionSumWithTensor:sq axis:-1 name:nil];
        MPSGraphTensor* eps = [graph_ constantWithScalar:1e-8 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* norm = [graph_ squareRootWithTensor:
                                [graph_ additionWithPrimaryTensor:sum_sq secondaryTensor:eps name:nil]
                                                       name:nil];
        return [graph_ divisionWithPrimaryTensor:x secondaryTensor:norm name:nil];
    }
};

// ============================================================================
// Test
// ============================================================================

bool test_local_features(const std::string& weights_path) {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            printf("[test] Local Features MLP test\n");
            printf("[test] Loading weights from: %s\n", weights_path.c_str());
            mast3r::safetensors::SafetensorsFile file(weights_path);

            LocalFeaturesBuilder gb;
            printf("[test] Device: %s\n", [[gb.device_ name] UTF8String]);

            // Load weights
            std::string p = "downstream_head1.head_local_features.";
            auto fc1_w = gb.tensor(file.load_tensor_f32(p + "fc1.weight"), @[@(LF_HIDDEN), @(LF_IN_DIM)]);
            auto fc1_b = gb.tensor(file.load_tensor_f32(p + "fc1.bias"), @[@(LF_HIDDEN)]);
            auto fc2_w = gb.tensor(file.load_tensor_f32(p + "fc2.weight"), @[@(LF_OUT_DIM), @(LF_HIDDEN)]);
            auto fc2_b = gb.tensor(file.load_tensor_f32(p + "fc2.bias"), @[@(LF_OUT_DIM)]);

            printf("[test] Weights loaded\n");
            printf("[test] FC1: [%d, %d], FC2: [%d, %d]\n", LF_HIDDEN, LF_IN_DIM, LF_OUT_DIM, LF_HIDDEN);

            // Input placeholders
            MPSGraphTensor* enc_input = [gb.graph_ placeholderWithShape:@[@(NUM_PATCHES), @(ENC_DIM)]
                                                               dataType:MPSDataTypeFloat32
                                                                   name:@"encoder"];
            MPSGraphTensor* dec_input = [gb.graph_ placeholderWithShape:@[@(NUM_PATCHES), @(DEC_DIM)]
                                                               dataType:MPSDataTypeFloat32
                                                                   name:@"decoder"];

            // Concatenate encoder and decoder features: [N, 1024] + [N, 768] = [N, 1792]
            MPSGraphTensor* concat = [gb.graph_ concatTensors:@[enc_input, dec_input] dimension:-1 name:nil];

            // MLP: FC1 -> GELU -> FC2
            MPSGraphTensor* x = gb.linear(concat, fc1_w, fc1_b);
            x = gb.gelu(x);
            x = gb.linear(x, fc2_w, fc2_b);
            // x shape: [N, 6400] = [768, 6400]

            // Reshape to spatial: [768, 6400] -> [1, 24, 32, 6400]
            x = [gb.graph_ reshapeTensor:x withShape:@[@1, @(PATCH_H), @(PATCH_W), @(LF_OUT_DIM)] name:nil];

            // Pixel shuffle: [1, 24, 32, 6400] -> [1, 384, 512, 25]
            x = gb.pixel_shuffle(x, PATCH_SIZE);

            // Split descriptors (24 channels) and confidence (1 channel)
            NSArray<MPSGraphTensor*>* splits = [gb.graph_ splitTensor:x
                                                            splitSizes:@[@(DESC_DIM), @1]
                                                                  axis:-1
                                                                  name:nil];
            MPSGraphTensor* desc = splits[0];  // [1, H, W, 24]
            MPSGraphTensor* conf = splits[1];  // [1, H, W, 1]

            // L2 normalize descriptors
            desc = gb.l2_normalize(desc);

            printf("[test] Graph built\n");
            printf("[test] Expected output: desc=[1, %d, %d, %d], conf=[1, %d, %d, 1]\n",
                   IMG_H, IMG_W, DESC_DIM, IMG_H, IMG_W);

            // Create random input data
            std::vector<float> enc_data(NUM_PATCHES * ENC_DIM);
            std::vector<float> dec_data(NUM_PATCHES * DEC_DIM);
            for (auto& v : enc_data) v = (float)(rand() % 1000) / 1000.0f - 0.5f;
            for (auto& v : dec_data) v = (float)(rand() % 1000) / 1000.0f - 0.5f;

            auto make_td = [&](const std::vector<float>& data, NSArray<NSNumber*>* shape) {
                MPSNDArrayDescriptor* d = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:shape];
                MPSNDArray* arr = [[MPSNDArray alloc] initWithDevice:gb.device_ descriptor:d];
                [arr writeBytes:(void*)data.data() strideBytes:nil];
                return [[MPSGraphTensorData alloc] initWithMPSNDArray:arr];
            };

            MPSGraphTensorData* enc_td = make_td(enc_data, @[@(NUM_PATCHES), @(ENC_DIM)]);
            MPSGraphTensorData* dec_td = make_td(dec_data, @[@(NUM_PATCHES), @(DEC_DIM)]);
            NSDictionary* feeds = @{enc_input: enc_td, dec_input: dec_td};

            // Warmup
            printf("[test] Warmup...\n");
            for (int i = 0; i < 3; i++) {
                [gb.graph_ runWithMTLCommandQueue:gb.queue_ feeds:feeds
                                    targetTensors:@[desc, conf] targetOperations:nil];
            }

            // Benchmark
            printf("[test] Benchmarking...\n");
            const int iters = 20;
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iters; i++) {
                [gb.graph_ runWithMTLCommandQueue:gb.queue_ feeds:feeds
                                    targetTensors:@[desc, conf] targetOperations:nil];
            }

            auto end = std::chrono::high_resolution_clock::now();
            double ms_per_iter = std::chrono::duration<double, std::milli>(end - start).count() / iters;

            printf("[test] Local Features MLP: %.2f ms/iter\n", ms_per_iter);

            // Verify output
            NSDictionary* results = [gb.graph_ runWithMTLCommandQueue:gb.queue_ feeds:feeds
                                                        targetTensors:@[desc, conf] targetOperations:nil];

            MPSGraphTensorData* desc_data = results[desc];
            MPSGraphTensorData* conf_data = results[conf];

            NSArray<NSNumber*>* desc_shape = [desc_data shape];
            NSArray<NSNumber*>* conf_shape = [conf_data shape];

            printf("[test] Desc shape: [%d, %d, %d, %d]\n",
                   [desc_shape[0] intValue], [desc_shape[1] intValue],
                   [desc_shape[2] intValue], [desc_shape[3] intValue]);
            printf("[test] Conf shape: [%d, %d, %d, %d]\n",
                   [conf_shape[0] intValue], [conf_shape[1] intValue],
                   [conf_shape[2] intValue], [conf_shape[3] intValue]);

            // Check for NaN
            size_t desc_size = IMG_H * IMG_W * DESC_DIM;
            std::vector<float> desc_cpu(desc_size);
            [[desc_data mpsndarray] readBytes:desc_cpu.data() strideBytes:nil];

            int nan_count = 0;
            float min_val = INFINITY, max_val = -INFINITY;
            for (auto v : desc_cpu) {
                if (std::isnan(v) || std::isinf(v)) nan_count++;
                else { min_val = std::min(min_val, v); max_val = std::max(max_val, v); }
            }

            printf("[test] Desc output: nan=%d/%zu, range=[%.3f, %.3f]\n",
                   nan_count, desc_size, min_val, max_val);

            // Check L2 normalization (should have values in [-1, 1])
            if (max_val > 1.1f || min_val < -1.1f) {
                printf("[test] WARNING: L2 norm may not be correct\n");
            }

            printf("[test] %s\n", nan_count == 0 ? "PASSED!" : "FAILED!");
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

        printf("=== MPSGraph Local Features Test ===\n\n");
        test_local_features(path);
        return 0;
    }
}
