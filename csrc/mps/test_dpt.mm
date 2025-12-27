// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
// Test DPT head with MPSGraph convolutions

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

// Image: 512x384 -> patches: 32x24
constexpr int IMG_W = 512;
constexpr int IMG_H = 384;
constexpr int PATCH_W = IMG_W / 16;  // 32
constexpr int PATCH_H = IMG_H / 16;  // 24
constexpr int NUM_PATCHES = PATCH_W * PATCH_H;  // 768

// Feature dimensions
constexpr int ENC_DIM = 1024;
constexpr int DEC_DIM = 768;
constexpr int DPT_DIM = 256;  // Output of layer_rn

// DPT hooks: which decoder layers to tap
constexpr int DPT_HOOKS[4] = {0, 6, 9, 12};  // 0=encoder, rest=decoder

// ============================================================================
// MPSGraph Convolution Helpers
// ============================================================================

class API_AVAILABLE(macos(15.0)) DPTBuilder {
public:
    MPSGraph* graph_;
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;

    DPTBuilder() {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newCommandQueue];
        graph_ = [[MPSGraph alloc] init];
    }

    MPSGraphTensor* tensor(const std::vector<float>& data, NSArray<NSNumber*>* shape) {
        NSData* nsdata = [NSData dataWithBytes:data.data() length:data.size() * sizeof(float)];
        return [graph_ constantWithData:nsdata shape:shape dataType:MPSDataTypeFloat32];
    }

    // Conv2D: x=[N,H,W,C], weight=[Cout,Cin,Kh,Kw], output=[N,H',W',Cout]
    MPSGraphTensor* conv2d(MPSGraphTensor* x,
                           MPSGraphTensor* weight,
                           MPSGraphTensor* bias,
                           int stride = 1,
                           int padding = 0) {
        // MPSGraph expects NHWC input and OIHW weights
        // Transpose weight from [O,I,H,W] to [O,H,W,I] for depthwise-style
        // Actually MPSGraph conv expects [O,Kh,Kw,I/groups] for grouped convs
        // For regular conv: weight shape [Cout, Cin, Kh, Kw]

        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride
                          strideInY:stride
                    dilationRateInX:1
                    dilationRateInY:1
                             groups:1
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        desc.paddingLeft = padding;
        desc.paddingRight = padding;
        desc.paddingTop = padding;
        desc.paddingBottom = padding;

        MPSGraphTensor* out = [graph_ convolution2DWithSourceTensor:x
                                                      weightsTensor:weight
                                                         descriptor:desc
                                                               name:nil];

        if (bias) {
            out = [graph_ additionWithPrimaryTensor:out secondaryTensor:bias name:nil];
        }
        return out;
    }

    // ConvTranspose2D for upsampling
    MPSGraphTensor* conv_transpose2d(MPSGraphTensor* x,
                                     MPSGraphTensor* weight,
                                     MPSGraphTensor* bias,
                                     int stride,
                                     int output_padding = 0) {
        // For transposed conv, stride is the upsampling factor
        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride
                          strideInY:stride
                    dilationRateInX:1
                    dilationRateInY:1
                             groups:1
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        // Padding for transposed conv
        int kernel_size = stride;  // Assuming kernel = stride for upsampling
        int padding = (kernel_size - 1) / 2;
        desc.paddingLeft = padding;
        desc.paddingRight = padding;
        desc.paddingTop = padding;
        desc.paddingBottom = padding;

        // Get output shape
        // For transpose conv: output_size = (input_size - 1) * stride - 2*padding + kernel_size + output_padding
        NSArray<NSNumber*>* x_shape = [x shape];
        int in_h = [x_shape[1] intValue];
        int in_w = [x_shape[2] intValue];
        int out_h = in_h * stride;
        int out_w = in_w * stride;
        int out_c = [[weight shape][1] intValue];  // For IOHW layout after transpose

        NSArray<NSNumber*>* output_shape = @[@1, @(out_h), @(out_w), @(out_c)];

        // Transpose weight for convTranspose: [O,I,H,W] -> [I,O,H,W]
        MPSGraphTensor* weight_t = [graph_ transposeTensor:weight
                                                 dimension:0
                                             withDimension:1
                                                      name:nil];

        MPSGraphTensor* out = [graph_ convolutionTranspose2DWithSourceTensor:x
                                                               weightsTensor:weight_t
                                                                 outputShape:output_shape
                                                                  descriptor:desc
                                                                        name:nil];

        if (bias) {
            out = [graph_ additionWithPrimaryTensor:out secondaryTensor:bias name:nil];
        }
        return out;
    }

    // Bilinear upsampling
    MPSGraphTensor* upsample_bilinear(MPSGraphTensor* x, int scale) {
        // Resize using bilinear interpolation
        NSArray<NSNumber*>* x_shape = [x shape];
        int in_h = [x_shape[1] intValue];
        int in_w = [x_shape[2] intValue];

        return [graph_ resizeTensor:x
                               size:@[@(in_h * scale), @(in_w * scale)]
                               mode:MPSGraphResizeBilinear
                       centerResult:YES
                        alignCorners:NO
                              layout:MPSGraphTensorNamedDataLayoutNHWC
                                name:nil];
    }

    // ReLU
    MPSGraphTensor* relu(MPSGraphTensor* x) {
        return [graph_ reLUWithTensor:x name:nil];
    }
};

// ============================================================================
// DPT Weights
// ============================================================================

struct DPTWeights {
    // act_postprocess[0]: Conv1x1(1024->96) + ConvT4x4(96->96)
    MPSGraphTensor *ap0_conv1_w, *ap0_conv1_b;
    MPSGraphTensor *ap0_conv2_w, *ap0_conv2_b;

    // act_postprocess[1]: Conv1x1(768->192) + ConvT2x2(192->192)
    MPSGraphTensor *ap1_conv1_w, *ap1_conv1_b;
    MPSGraphTensor *ap1_conv2_w, *ap1_conv2_b;

    // act_postprocess[2]: Conv1x1(768->384)
    MPSGraphTensor *ap2_conv1_w, *ap2_conv1_b;

    // act_postprocess[3]: Conv1x1(768->768) + Conv3x3(768->768, stride=2)
    MPSGraphTensor *ap3_conv1_w, *ap3_conv1_b;
    MPSGraphTensor *ap3_conv2_w, *ap3_conv2_b;

    // layer_rn[0-3]: Conv3x3 -> 256 (no bias)
    MPSGraphTensor *layer_rn_w[4];

    // head: Conv3x3(256->128) + Conv3x3(128->128) + Conv1x1(128->4)
    MPSGraphTensor *head_conv1_w, *head_conv1_b;
    MPSGraphTensor *head_conv2_w, *head_conv2_b;
    MPSGraphTensor *head_conv3_w, *head_conv3_b;
};

// ============================================================================
// Test
// ============================================================================

bool test_dpt(const std::string& weights_path) {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            printf("[test] Loading weights from: %s\n", weights_path.c_str());
            mast3r::safetensors::SafetensorsFile file(weights_path);

            DPTBuilder gb;
            printf("[test] Metal device: %s\n", [[gb.device_ name] UTF8String]);

            // Load DPT weights
            printf("[test] Loading DPT weights...\n");
            DPTWeights w;
            std::string p = "downstream_head1.dpt.";

            // act_postprocess
            w.ap0_conv1_w = gb.tensor(file.load_tensor_f32(p + "act_postprocess.0.0.weight"), @[@96, @1024, @1, @1]);
            w.ap0_conv1_b = gb.tensor(file.load_tensor_f32(p + "act_postprocess.0.0.bias"), @[@96]);
            w.ap0_conv2_w = gb.tensor(file.load_tensor_f32(p + "act_postprocess.0.1.weight"), @[@96, @96, @4, @4]);
            w.ap0_conv2_b = gb.tensor(file.load_tensor_f32(p + "act_postprocess.0.1.bias"), @[@96]);

            w.ap1_conv1_w = gb.tensor(file.load_tensor_f32(p + "act_postprocess.1.0.weight"), @[@192, @768, @1, @1]);
            w.ap1_conv1_b = gb.tensor(file.load_tensor_f32(p + "act_postprocess.1.0.bias"), @[@192]);
            w.ap1_conv2_w = gb.tensor(file.load_tensor_f32(p + "act_postprocess.1.1.weight"), @[@192, @192, @2, @2]);
            w.ap1_conv2_b = gb.tensor(file.load_tensor_f32(p + "act_postprocess.1.1.bias"), @[@192]);

            w.ap2_conv1_w = gb.tensor(file.load_tensor_f32(p + "act_postprocess.2.0.weight"), @[@384, @768, @1, @1]);
            w.ap2_conv1_b = gb.tensor(file.load_tensor_f32(p + "act_postprocess.2.0.bias"), @[@384]);

            w.ap3_conv1_w = gb.tensor(file.load_tensor_f32(p + "act_postprocess.3.0.weight"), @[@768, @768, @1, @1]);
            w.ap3_conv1_b = gb.tensor(file.load_tensor_f32(p + "act_postprocess.3.0.bias"), @[@768]);
            w.ap3_conv2_w = gb.tensor(file.load_tensor_f32(p + "act_postprocess.3.1.weight"), @[@768, @768, @3, @3]);
            w.ap3_conv2_b = gb.tensor(file.load_tensor_f32(p + "act_postprocess.3.1.bias"), @[@768]);

            // layer_rn (no bias)
            w.layer_rn_w[0] = gb.tensor(file.load_tensor_f32(p + "scratch.layer_rn.0.weight"), @[@256, @96, @3, @3]);
            w.layer_rn_w[1] = gb.tensor(file.load_tensor_f32(p + "scratch.layer_rn.1.weight"), @[@256, @192, @3, @3]);
            w.layer_rn_w[2] = gb.tensor(file.load_tensor_f32(p + "scratch.layer_rn.2.weight"), @[@256, @384, @3, @3]);
            w.layer_rn_w[3] = gb.tensor(file.load_tensor_f32(p + "scratch.layer_rn.3.weight"), @[@256, @768, @3, @3]);

            // head
            w.head_conv1_w = gb.tensor(file.load_tensor_f32(p + "head.0.weight"), @[@128, @256, @3, @3]);
            w.head_conv1_b = gb.tensor(file.load_tensor_f32(p + "head.0.bias"), @[@128]);
            w.head_conv2_w = gb.tensor(file.load_tensor_f32(p + "head.2.weight"), @[@128, @128, @3, @3]);
            w.head_conv2_b = gb.tensor(file.load_tensor_f32(p + "head.2.bias"), @[@128]);
            w.head_conv3_w = gb.tensor(file.load_tensor_f32(p + "head.4.weight"), @[@4, @128, @1, @1]);
            w.head_conv3_b = gb.tensor(file.load_tensor_f32(p + "head.4.bias"), @[@4]);

            printf("[test] Weights loaded, building DPT graph...\n");

            // Create input placeholders for hook features
            // hook0: encoder output [1, H, W, 1024] - H=24, W=32
            // hook1-3: decoder outputs [1, H, W, 768]
            MPSGraphTensor* hook0 = [gb.graph_ placeholderWithShape:@[@1, @(PATCH_H), @(PATCH_W), @(ENC_DIM)]
                                                           dataType:MPSDataTypeFloat32
                                                               name:@"hook0"];
            MPSGraphTensor* hook1 = [gb.graph_ placeholderWithShape:@[@1, @(PATCH_H), @(PATCH_W), @(DEC_DIM)]
                                                           dataType:MPSDataTypeFloat32
                                                               name:@"hook1"];
            MPSGraphTensor* hook2 = [gb.graph_ placeholderWithShape:@[@1, @(PATCH_H), @(PATCH_W), @(DEC_DIM)]
                                                           dataType:MPSDataTypeFloat32
                                                               name:@"hook2"];
            MPSGraphTensor* hook3 = [gb.graph_ placeholderWithShape:@[@1, @(PATCH_H), @(PATCH_W), @(DEC_DIM)]
                                                           dataType:MPSDataTypeFloat32
                                                               name:@"hook3"];

            // === act_postprocess ===
            // [0]: Conv1x1 + ConvT4x4 -> 4x upsample (24x32 -> 96x128)
            MPSGraphTensor* f0 = gb.conv2d(hook0, w.ap0_conv1_w, w.ap0_conv1_b);
            f0 = gb.conv_transpose2d(f0, w.ap0_conv2_w, w.ap0_conv2_b, 4);

            // [1]: Conv1x1 + ConvT2x2 -> 2x upsample (24x32 -> 48x64)
            MPSGraphTensor* f1 = gb.conv2d(hook1, w.ap1_conv1_w, w.ap1_conv1_b);
            f1 = gb.conv_transpose2d(f1, w.ap1_conv2_w, w.ap1_conv2_b, 2);

            // [2]: Conv1x1 only (24x32 -> 24x32)
            MPSGraphTensor* f2 = gb.conv2d(hook2, w.ap2_conv1_w, w.ap2_conv1_b);

            // [3]: Conv1x1 + Conv3x3 stride=2 -> 0.5x (24x32 -> 12x16)
            MPSGraphTensor* f3 = gb.conv2d(hook3, w.ap3_conv1_w, w.ap3_conv1_b);
            f3 = gb.conv2d(f3, w.ap3_conv2_w, w.ap3_conv2_b, 2, 1);  // stride=2, pad=1

            printf("[test] act_postprocess built\n");

            // === layer_rn: project to 256 channels ===
            f0 = gb.conv2d(f0, w.layer_rn_w[0], nil, 1, 1);  // pad=1 for 3x3
            f1 = gb.conv2d(f1, w.layer_rn_w[1], nil, 1, 1);
            f2 = gb.conv2d(f2, w.layer_rn_w[2], nil, 1, 1);
            f3 = gb.conv2d(f3, w.layer_rn_w[3], nil, 1, 1);

            printf("[test] layer_rn built\n");

            // === Progressive upsampling and fusion ===
            // All features to same resolution via bilinear upsampling
            // Target: 96x128 (same as f0 after 4x upsample)
            // f1: 48x64 -> 96x128 (2x)
            // f2: 24x32 -> 96x128 (4x)
            // f3: 12x16 -> 96x128 (8x)
            f1 = gb.upsample_bilinear(f1, 2);
            f2 = gb.upsample_bilinear(f2, 4);
            f3 = gb.upsample_bilinear(f3, 8);

            // Add all features
            MPSGraphTensor* fused = [gb.graph_ additionWithPrimaryTensor:f0 secondaryTensor:f1 name:nil];
            fused = [gb.graph_ additionWithPrimaryTensor:fused secondaryTensor:f2 name:nil];
            fused = [gb.graph_ additionWithPrimaryTensor:fused secondaryTensor:f3 name:nil];

            printf("[test] Feature fusion built\n");

            // === Head: final predictions ===
            // Upsample to 2x -> 192x256
            MPSGraphTensor* head = gb.upsample_bilinear(fused, 2);
            head = gb.conv2d(head, w.head_conv1_w, w.head_conv1_b, 1, 1);
            head = gb.relu(head);

            // Upsample to 2x -> 384x512
            head = gb.upsample_bilinear(head, 2);
            head = gb.conv2d(head, w.head_conv2_w, w.head_conv2_b, 1, 1);
            head = gb.relu(head);

            // Final 1x1 conv -> [1, 384, 512, 4]
            head = gb.conv2d(head, w.head_conv3_w, w.head_conv3_b);

            MPSGraphTensor* output = head;

            printf("[test] DPT head built, output shape: [1, %d, %d, 4]\n", IMG_H, IMG_W);

            // Create random input data
            std::vector<float> hook0_data(PATCH_H * PATCH_W * ENC_DIM);
            std::vector<float> hook1_data(PATCH_H * PATCH_W * DEC_DIM);
            std::vector<float> hook2_data(PATCH_H * PATCH_W * DEC_DIM);
            std::vector<float> hook3_data(PATCH_H * PATCH_W * DEC_DIM);

            for (auto& v : hook0_data) v = (float)(rand() % 1000) / 1000.0f - 0.5f;
            for (auto& v : hook1_data) v = (float)(rand() % 1000) / 1000.0f - 0.5f;
            for (auto& v : hook2_data) v = (float)(rand() % 1000) / 1000.0f - 0.5f;
            for (auto& v : hook3_data) v = (float)(rand() % 1000) / 1000.0f - 0.5f;

            // Create tensor data
            auto make_td = [&](const std::vector<float>& data, NSArray<NSNumber*>* shape) {
                MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:shape];
                MPSNDArray* arr = [[MPSNDArray alloc] initWithDevice:gb.device_ descriptor:desc];
                [arr writeBytes:(void*)data.data() strideBytes:nil];
                return [[MPSGraphTensorData alloc] initWithMPSNDArray:arr];
            };

            MPSGraphTensorData* td0 = make_td(hook0_data, @[@1, @(PATCH_H), @(PATCH_W), @(ENC_DIM)]);
            MPSGraphTensorData* td1 = make_td(hook1_data, @[@1, @(PATCH_H), @(PATCH_W), @(DEC_DIM)]);
            MPSGraphTensorData* td2 = make_td(hook2_data, @[@1, @(PATCH_H), @(PATCH_W), @(DEC_DIM)]);
            MPSGraphTensorData* td3 = make_td(hook3_data, @[@1, @(PATCH_H), @(PATCH_W), @(DEC_DIM)]);

            NSDictionary* feeds = @{hook0: td0, hook1: td1, hook2: td2, hook3: td3};

            // Warmup
            printf("[test] Warmup...\n");
            for (int i = 0; i < 3; i++) {
                [gb.graph_ runWithMTLCommandQueue:gb.queue_
                                            feeds:feeds
                                    targetTensors:@[output]
                                 targetOperations:nil];
            }

            // Benchmark
            printf("[test] Benchmarking...\n");
            const int iters = 20;
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iters; i++) {
                [gb.graph_ runWithMTLCommandQueue:gb.queue_
                                            feeds:feeds
                                    targetTensors:@[output]
                                 targetOperations:nil];
            }

            auto end = std::chrono::high_resolution_clock::now();
            double ms_total = std::chrono::duration<double, std::milli>(end - start).count();
            double ms_per_iter = ms_total / iters;

            printf("[test] DPT head: %.2f ms/iter\n", ms_per_iter);

            // Verify output
            NSDictionary* results = [gb.graph_ runWithMTLCommandQueue:gb.queue_
                                                                feeds:feeds
                                                        targetTensors:@[output]
                                                     targetOperations:nil];

            MPSGraphTensorData* output_data = results[output];
            NSArray<NSNumber*>* out_shape = [output_data shape];
            printf("[test] Output shape: [%d, %d, %d, %d]\n",
                   [out_shape[0] intValue], [out_shape[1] intValue],
                   [out_shape[2] intValue], [out_shape[3] intValue]);

            size_t out_size = IMG_H * IMG_W * 4;
            std::vector<float> output_cpu(out_size);
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

        printf("=== MPSGraph DPT Test ===\n\n");
        test_dpt(path);
        return 0;
    }
}
