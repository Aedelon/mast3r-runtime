// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
// Test MPSGraph SDPA availability and basic functionality

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <vector>

// Simple SDPA test
bool test_sdpa_basic() {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                printf("[test] No Metal device\n");
                return false;
            }

            printf("[test] Metal device: %s\n", [[device name] UTF8String]);

            id<MTLCommandQueue> queue = [device newCommandQueue];
            MPSGraph* graph = [[MPSGraph alloc] init];

            // Test dimensions: batch=1, heads=12, seq=576, head_dim=64
            const int batch = 1;
            const int heads = 12;
            const int seq_len = 576;
            const int head_dim = 64;
            const float scale = 1.0f / sqrtf((float)head_dim);

            // Create random Q, K, V data
            size_t qkv_size = batch * heads * seq_len * head_dim;
            std::vector<float> q_data(qkv_size);
            std::vector<float> k_data(qkv_size);
            std::vector<float> v_data(qkv_size);

            for (size_t i = 0; i < qkv_size; i++) {
                q_data[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
                k_data[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
                v_data[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
            }

            // Create input tensors
            NSArray<NSNumber*>* shape = @[@(batch), @(heads), @(seq_len), @(head_dim)];

            MPSGraphTensor* Q = [graph placeholderWithShape:shape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"Q"];
            MPSGraphTensor* K = [graph placeholderWithShape:shape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"K"];
            MPSGraphTensor* V = [graph placeholderWithShape:shape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"V"];

            // Call SDPA
            printf("[test] Calling scaledDotProductAttention...\n");
            MPSGraphTensor* output = [graph scaledDotProductAttentionWithQueryTensor:Q
                                                                           keyTensor:K
                                                                         valueTensor:V
                                                                          maskTensor:nil
                                                                               scale:scale
                                                                                name:@"sdpa"];

            printf("[test] SDPA graph node created successfully\n");

            // Create Metal buffers for input data
            size_t buffer_size = qkv_size * sizeof(float);
            id<MTLBuffer> q_buffer = [device newBufferWithBytes:q_data.data()
                                                         length:buffer_size
                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> k_buffer = [device newBufferWithBytes:k_data.data()
                                                         length:buffer_size
                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> v_buffer = [device newBufferWithBytes:v_data.data()
                                                         length:buffer_size
                                                        options:MTLResourceStorageModeShared];

            // Create MPSNDArray descriptors
            MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                                                shape:shape];

            // Create MPSNDArrays from buffers
            MPSNDArray* q_ndarray = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];
            MPSNDArray* k_ndarray = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];
            MPSNDArray* v_ndarray = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];

            [q_ndarray writeBytes:q_data.data() strideBytes:nil];
            [k_ndarray writeBytes:k_data.data() strideBytes:nil];
            [v_ndarray writeBytes:v_data.data() strideBytes:nil];

            // Create MPSGraphTensorData from MPSNDArrays
            MPSGraphTensorData* q_tensor_data = [[MPSGraphTensorData alloc] initWithMPSNDArray:q_ndarray];
            MPSGraphTensorData* k_tensor_data = [[MPSGraphTensorData alloc] initWithMPSNDArray:k_ndarray];
            MPSGraphTensorData* v_tensor_data = [[MPSGraphTensorData alloc] initWithMPSNDArray:v_ndarray];

            // Execute graph
            printf("[test] Executing SDPA graph...\n");

            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
                Q: q_tensor_data,
                K: k_tensor_data,
                V: v_tensor_data
            };

            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
                [graph runWithMTLCommandQueue:queue
                                        feeds:feeds
                                targetTensors:@[output]
                             targetOperations:nil];

            MPSGraphTensorData* output_data = results[output];
            if (!output_data) {
                printf("[test] SDPA execution failed - no output\n");
                return false;
            }

            // Check output shape
            NSArray<NSNumber*>* output_shape = [output_data shape];
            printf("[test] Output shape: [%d, %d, %d, %d]\n",
                   [output_shape[0] intValue],
                   [output_shape[1] intValue],
                   [output_shape[2] intValue],
                   [output_shape[3] intValue]);

            // Read output data - copy to CPU
            std::vector<float> output_cpu(qkv_size);
            MPSNDArray* ndarray = [output_data mpsndarray];
            [ndarray readBytes:output_cpu.data() strideBytes:nil];

            // Check for NaN/Inf
            int nan_count = 0;
            float min_val = INFINITY, max_val = -INFINITY;
            for (size_t i = 0; i < qkv_size; i++) {
                float v = output_cpu[i];
                if (std::isnan(v) || std::isinf(v)) {
                    nan_count++;
                } else {
                    min_val = std::min(min_val, v);
                    max_val = std::max(max_val, v);
                }
            }

            printf("[test] Output: nan=%d/%zu, range=[%.3f, %.3f]\n",
                   nan_count, qkv_size, min_val, max_val);

            if (nan_count > 0) {
                printf("[test] FAILED - NaN in output\n");
                return false;
            }

            printf("[test] SDPA test PASSED!\n");
            return true;
        }
    } else {
        printf("[test] macOS 15+ required for SDPA\n");
        return false;
    }
}

// Benchmark SDPA vs manual attention
void benchmark_sdpa() {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            MPSGraph* graph = [[MPSGraph alloc] init];

            const int batch = 1;
            const int heads = 12;
            const int seq_len = 576;
            const int head_dim = 64;
            const float scale = 1.0f / sqrtf((float)head_dim);

            size_t qkv_size = batch * heads * seq_len * head_dim;
            std::vector<float> q_data(qkv_size, 0.1f);
            std::vector<float> k_data(qkv_size, 0.1f);
            std::vector<float> v_data(qkv_size, 0.1f);

            NSArray<NSNumber*>* shape = @[@(batch), @(heads), @(seq_len), @(head_dim)];

            MPSGraphTensor* Q = [graph placeholderWithShape:shape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"Q"];
            MPSGraphTensor* K = [graph placeholderWithShape:shape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"K"];
            MPSGraphTensor* V = [graph placeholderWithShape:shape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"V"];

            MPSGraphTensor* output = [graph scaledDotProductAttentionWithQueryTensor:Q
                                                                           keyTensor:K
                                                                         valueTensor:V
                                                                          maskTensor:nil
                                                                               scale:scale
                                                                                name:@"sdpa"];

            MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                                                shape:shape];
            MPSNDArray* q_ndarray = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];
            MPSNDArray* k_ndarray = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];
            MPSNDArray* v_ndarray = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];

            [q_ndarray writeBytes:q_data.data() strideBytes:nil];
            [k_ndarray writeBytes:k_data.data() strideBytes:nil];
            [v_ndarray writeBytes:v_data.data() strideBytes:nil];

            MPSGraphTensorData* q_td = [[MPSGraphTensorData alloc] initWithMPSNDArray:q_ndarray];
            MPSGraphTensorData* k_td = [[MPSGraphTensorData alloc] initWithMPSNDArray:k_ndarray];
            MPSGraphTensorData* v_td = [[MPSGraphTensorData alloc] initWithMPSNDArray:v_ndarray];

            NSDictionary* feeds = @{Q: q_td, K: k_td, V: v_td};

            // Warmup
            for (int i = 0; i < 5; i++) {
                [graph runWithMTLCommandQueue:queue
                                        feeds:feeds
                                targetTensors:@[output]
                             targetOperations:nil];
            }

            // Benchmark
            const int iters = 100;
            CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();

            for (int i = 0; i < iters; i++) {
                [graph runWithMTLCommandQueue:queue
                                        feeds:feeds
                                targetTensors:@[output]
                             targetOperations:nil];
            }

            CFAbsoluteTime end = CFAbsoluteTimeGetCurrent();
            double ms_per_iter = (end - start) * 1000.0 / iters;

            printf("[bench] SDPA: %.2f ms/iter (seq=%d, heads=%d, head_dim=%d)\n",
                   ms_per_iter, seq_len, heads, head_dim);
        }
    }
}

int main(int argc, char* argv[]) {
    @autoreleasepool {
        printf("=== MPSGraph SDPA Test ===\n\n");

        if (test_sdpa_basic()) {
            printf("\n=== Benchmark ===\n");
            benchmark_sdpa();
        }

        return 0;
    }
}
