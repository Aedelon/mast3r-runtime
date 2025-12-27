// MASt3R Runtime - Whitening Graph Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#import "whitening_graph.hpp"
#include <stdexcept>

namespace mast3r {
namespace mpsgraph {

WhiteningGraph::WhiteningGraph(std::shared_ptr<MPSGraphContext> ctx, const RuntimeConfig& config)
    : ctx_(std::move(ctx)), config_(config) {
    if (@available(macOS 15.0, *)) {
        graph_ = ctx_->create_graph();
        cfg_ = ModelConfig::from_variant(config.variant);

        // Calculate dimensions from config resolution (4:3 aspect ratio)
        int img_h = config.resolution;
        int img_w = static_cast<int>(config.resolution * 4.0f / 3.0f);
        int patch_h = img_h / cfg_.patch_size;
        int patch_w = img_w / cfg_.patch_size;
        num_patches_ = patch_h * patch_w;
        enc_dim_ = cfg_.enc_dim;
    } else {
        throw std::runtime_error("WhiteningGraph requires macOS 15.0+");
    }
}

void WhiteningGraph::load(const std::string& retrieval_path) {
    safetensors::SafetensorsFile file(retrieval_path);
    load(file);
}

void WhiteningGraph::load(safetensors::SafetensorsFile& file) {
    if (@available(macOS 15.0, *)) {
        // Check required weights
        if (!file.has_tensor("prewhiten.m") || !file.has_tensor("prewhiten.p")) {
            throw std::runtime_error("Missing prewhiten.m or prewhiten.p in retrieval weights");
        }

        // Load whitening weights
        auto m_data = file.load_tensor_f32("prewhiten.m");  // [enc_dim]
        auto p_data = file.load_tensor_f32("prewhiten.p");  // [enc_dim, enc_dim]

        bool use_fp16 = (config_.precision == Precision::FP16);
        MPSDataType dtype = use_fp16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;

        // Convert to NSData
        NSData* m_nsdata;
        NSData* p_nsdata;

        if (use_fp16) {
            // Convert to FP16
            std::vector<__fp16> m_fp16(m_data.size());
            std::vector<__fp16> p_fp16(p_data.size());
            for (size_t i = 0; i < m_data.size(); i++) m_fp16[i] = (__fp16)m_data[i];
            for (size_t i = 0; i < p_data.size(); i++) p_fp16[i] = (__fp16)p_data[i];
            m_nsdata = [NSData dataWithBytes:m_fp16.data() length:m_fp16.size() * sizeof(__fp16)];
            p_nsdata = [NSData dataWithBytes:p_fp16.data() length:p_fp16.size() * sizeof(__fp16)];
        } else {
            m_nsdata = [NSData dataWithBytes:m_data.data() length:m_data.size() * sizeof(float)];
            p_nsdata = [NSData dataWithBytes:p_data.data() length:p_data.size() * sizeof(float)];
        }

        // Create weight constants
        MPSGraphTensor* whiten_m = [graph_ constantWithData:m_nsdata
                                                     shape:@[@1, @(enc_dim_)]
                                                  dataType:dtype];
        MPSGraphTensor* whiten_p = [graph_ constantWithData:p_nsdata
                                                     shape:@[@(enc_dim_), @(enc_dim_)]
                                                  dataType:dtype];

        // Build graph
        // Input: [N, enc_dim] FP16 from encoder
        input_ = [graph_ placeholderWithShape:@[@(num_patches_), @(enc_dim_)]
                                     dataType:dtype
                                         name:@"enc_features"];

        // Whitening: (x - mean) @ projection
        MPSGraphTensor* centered = [graph_ subtractionWithPrimaryTensor:input_
                                                        secondaryTensor:whiten_m name:nil];
        MPSGraphTensor* whitened = [graph_ matrixMultiplicationWithPrimaryTensor:centered
                                                                 secondaryTensor:whiten_p name:nil];

        // Cast to FP32 for output (retrieval needs FP32)
        if (use_fp16) {
            output_features_ = [graph_ castTensor:whitened toType:MPSDataTypeFloat32 name:nil];
        } else {
            output_features_ = whitened;
        }

        // Attention: L2 norm per patch, normalized to sum=1
        // Stays in native precision for computation
        MPSGraphTensor* sq = [graph_ squareWithTensor:whitened name:nil];
        MPSGraphTensor* l2_sq = [graph_ reductionSumWithTensor:sq axis:-1 name:nil];  // [N]

        // Normalize by total to get attention weights
        MPSGraphTensor* total = [graph_ reductionSumWithTensor:l2_sq axes:@[@0] name:nil];
        total = [graph_ reshapeTensor:total withShape:@[@1] name:nil];
        MPSGraphTensor* eps = [graph_ constantWithScalar:1e-8 shape:@[@1] dataType:dtype];
        total = [graph_ additionWithPrimaryTensor:total secondaryTensor:eps name:nil];
        MPSGraphTensor* attn = [graph_ divisionWithPrimaryTensor:l2_sq secondaryTensor:total name:nil];

        // Cast attention to FP32
        if (use_fp16) {
            output_attention_ = [graph_ castTensor:attn toType:MPSDataTypeFloat32 name:nil];
        } else {
            output_attention_ = attn;
        }

        is_loaded_ = true;
    }
}

void WhiteningGraph::compile() {
    // Pre-compilation not implemented - uses JIT compilation on first run
    // MPSGraph automatically caches compiled graphs internally
}

WhiteningOutput WhiteningGraph::run(MPSGraphTensorData* enc_out) {
    WhiteningOutput result = {nil, nil};

    if (@available(macOS 15.0, *)) {
        if (!is_loaded_) {
            throw std::runtime_error("Whitening weights not loaded");
        }

        NSDictionary* feeds = @{input_: enc_out};
        NSDictionary* results = [graph_ runWithMTLCommandQueue:ctx_->queue()
                                                         feeds:feeds
                                                 targetTensors:@[output_features_, output_attention_]
                                             targetOperations:nil];
        result.features = results[output_features_];
        result.attention = results[output_attention_];
    }
    return result;
}

void WhiteningGraph::run_async(MPSGraphTensorData* enc_out,
                               void (^completion)(WhiteningOutput output)) {
    if (@available(macOS 15.0, *)) {
        if (!is_loaded_) {
            throw std::runtime_error("Whitening weights not loaded");
        }

        NSDictionary* feeds = @{input_: enc_out};

        MPSGraphExecutionDescriptor* exec_desc = [MPSGraphExecutionDescriptor new];
        exec_desc.completionHandler = ^(NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results,
                                        NSError* error) {
            if (completion && !error) {
                WhiteningOutput output;
                output.features = results[output_features_];
                output.attention = results[output_attention_];
                completion(output);
            }
        };

        [graph_ runAsyncWithMTLCommandQueue:ctx_->queue()
                                      feeds:feeds
                              targetTensors:@[output_features_, output_attention_]
                           targetOperations:nil
                        executionDescriptor:exec_desc];
    }
}

WhiteningOutputBuffers WhiteningGraph::allocate_output_buffers() {
    WhiteningOutputBuffers buffers;

    if (@available(macOS 15.0, *)) {
        const size_t features_elements = num_patches_ * enc_dim_;
        const size_t attention_elements = num_patches_;

        // Allocate shared MTLBuffers (zero-copy between GPU and CPU)
        buffers.buf_features = [ctx_->device() newBufferWithLength:features_elements * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        buffers.buf_attention = [ctx_->device() newBufferWithLength:attention_elements * sizeof(float)
                                                            options:MTLResourceStorageModeShared];

        // Wrap in MPSGraphTensorData for resultsDictionary
        buffers.td_features = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffers.buf_features
                                                                      shape:@[@(num_patches_), @(enc_dim_)]
                                                                   dataType:MPSDataTypeFloat32];
        buffers.td_attention = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffers.buf_attention
                                                                       shape:@[@(num_patches_)]
                                                                    dataType:MPSDataTypeFloat32];

        // Direct pointers for zero-copy access
        buffers.ptr_features = static_cast<float*>([buffers.buf_features contents]);
        buffers.ptr_attention = static_cast<float*>([buffers.buf_attention contents]);
    }

    return buffers;
}

void WhiteningGraph::run_into(MPSGraphTensorData* enc_out, WhiteningOutputBuffers& buffers) {
    if (@available(macOS 15.0, *)) {
        if (!is_loaded_) {
            throw std::runtime_error("Whitening weights not loaded");
        }

        NSDictionary* feeds = @{input_: enc_out};
        NSDictionary* resultsDict = @{
            output_features_: buffers.td_features,
            output_attention_: buffers.td_attention
        };

        [graph_ runWithMTLCommandQueue:ctx_->queue()
                                 feeds:feeds
                      targetOperations:nil
                     resultsDictionary:resultsDict];
    }
}

void WhiteningGraph::run_async_into(MPSGraphTensorData* enc_out, WhiteningOutputBuffers& buffers,
                                    void (^completion)(void)) {
    if (@available(macOS 15.0, *)) {
        if (!is_loaded_) {
            throw std::runtime_error("Whitening weights not loaded");
        }

        NSDictionary* feeds = @{input_: enc_out};
        NSDictionary* resultsDict = @{
            output_features_: buffers.td_features,
            output_attention_: buffers.td_attention
        };

        MPSGraphExecutionDescriptor* exec_desc = [MPSGraphExecutionDescriptor new];
        exec_desc.completionHandler = ^(NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results,
                                        NSError* error) {
            if (completion) {
                completion();
            }
        };

        [graph_ runAsyncWithMTLCommandQueue:ctx_->queue()
                                      feeds:feeds
                           targetOperations:nil
                          resultsDictionary:resultsDict
                        executionDescriptor:exec_desc];
    }
}

}  // namespace mpsgraph
}  // namespace mast3r
