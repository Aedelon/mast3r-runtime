// MASt3R Runtime - Encoder Graph Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#import "encoder_graph.hpp"
#import "graph_builder.hpp"
#include <stdexcept>

namespace mast3r {
namespace mpsgraph {

EncoderGraph::EncoderGraph(std::shared_ptr<MPSGraphContext> ctx, const RuntimeConfig& config)
    : ctx_(std::move(ctx)), config_(config) {
    if (@available(macOS 15.0, *)) {
        graph_ = ctx_->create_graph();
        cfg_ = ModelConfig::from_variant(config.variant);

        // Calculate dimensions from config resolution (4:3 aspect ratio)
        img_h_ = config.resolution;
        img_w_ = static_cast<int>(config.resolution * 4.0f / 3.0f);
        patch_h_ = img_h_ / cfg_.patch_size;
        patch_w_ = img_w_ / cfg_.patch_size;
        num_patches_ = patch_h_ * patch_w_;
    } else {
        throw std::runtime_error("EncoderGraph requires macOS 15.0+");
    }
}

void EncoderGraph::load(safetensors::MultiSafetensorsFile& files) {
    if (@available(macOS 15.0, *)) {
        bool use_fp16 = (config_.precision == Precision::FP16);
        GraphBuilder gb(graph_, ctx_->device(), &files, use_fp16);

        // Load encoder weights
        std::string pe = cfg_.patch_embed_key();
        auto patch_w = gb.load(pe + "weight", @[@(cfg_.enc_dim), @3, @(cfg_.patch_size), @(cfg_.patch_size)]);
        auto patch_b = gb.load(pe + "bias", @[@(cfg_.enc_dim)]);

        // Encoder layers
        std::vector<EncoderLayer> enc(cfg_.enc_depth);
        for (int i = 0; i < cfg_.enc_depth; i++) {
            std::string k = cfg_.enc_block_key(i);
            auto& L = enc[i];
            L.n1w = gb.load(k + "norm1.weight", @[@(cfg_.enc_dim)]);
            L.n1b = gb.load(k + "norm1.bias", @[@(cfg_.enc_dim)]);
            L.n2w = gb.load(k + "norm2.weight", @[@(cfg_.enc_dim)]);
            L.n2b = gb.load(k + "norm2.bias", @[@(cfg_.enc_dim)]);
            L.qkvw = gb.load(k + "attn.qkv.weight", @[@(cfg_.enc_dim * 3), @(cfg_.enc_dim)]);
            L.qkvb = gb.load(k + "attn.qkv.bias", @[@(cfg_.enc_dim * 3)]);
            L.pw = gb.load(k + "attn.proj.weight", @[@(cfg_.enc_dim), @(cfg_.enc_dim)]);
            L.pb = gb.load(k + "attn.proj.bias", @[@(cfg_.enc_dim)]);
            L.f1w = gb.load(k + "mlp.fc1.weight", @[@(cfg_.enc_mlp), @(cfg_.enc_dim)]);
            L.f1b = gb.load(k + "mlp.fc1.bias", @[@(cfg_.enc_mlp)]);
            L.f2w = gb.load(k + "mlp.fc2.weight", @[@(cfg_.enc_dim), @(cfg_.enc_mlp)]);
            L.f2b = gb.load(k + "mlp.fc2.bias", @[@(cfg_.enc_dim)]);
        }

        // Encoder norm
        std::string en = cfg_.enc_norm_key();
        auto enc_nw = gb.load(en + "weight", @[@(cfg_.enc_dim)]);
        auto enc_nb = gb.load(en + "bias", @[@(cfg_.enc_dim)]);

        // Build graph
        input_ = [graph_ placeholderWithShape:@[@1, @(img_h_), @(img_w_), @3]
                                     dataType:MPSDataTypeUInt8 name:@"image_uint8"];

        // GPU preprocessing
        MPSGraphTensor* normalized = gb.imagenet_normalize(input_);

        // Patch embedding
        MPSGraphTensor* patches = gb.conv2d(normalized, patch_w, patch_b, cfg_.patch_size, 0);
        patches = [graph_ reshapeTensor:patches withShape:@[@(num_patches_), @(cfg_.enc_dim)] name:nil];

        // Encoder blocks
        MPSGraphTensor* x = patches;
        for (int i = 0; i < cfg_.enc_depth; i++) {
            auto& L = enc[i];
            MPSGraphTensor* res = x;
            x = gb.layer_norm(x, L.n1w, L.n1b);
            x = gb.self_attention(x, L.qkvw, L.qkvb, L.pw, L.pb, cfg_.enc_heads, cfg_.enc_head_dim);
            x = gb.add(x, res);
            res = x;
            x = gb.layer_norm(x, L.n2w, L.n2b);
            x = gb.linear(x, L.f1w, L.f1b);
            x = gb.gelu(x);
            x = gb.linear(x, L.f2w, L.f2b);
            x = gb.add(x, res);
        }

        // Final norm
        x = gb.layer_norm(x, enc_nw, enc_nb);

        // Output stays in FP16 for downstream graphs
        output_ = x;  // [N, enc_dim]
    }
}

void EncoderGraph::compile() {
    // Pre-compilation not implemented - uses JIT compilation on first run
    // MPSGraph automatically caches compiled graphs internally
}

MPSGraphTensorData* EncoderGraph::run(MPSGraphTensorData* input) {
    if (@available(macOS 15.0, *)) {
        NSDictionary* feeds = @{input_: input};
        NSDictionary* results = [graph_ runWithMTLCommandQueue:ctx_->queue()
                                                         feeds:feeds
                                                 targetTensors:@[output_]
                                             targetOperations:nil];
        return results[output_];
    }
    return nil;
}

void EncoderGraph::run_async(MPSGraphTensorData* input,
                             void (^completion)(MPSGraphTensorData* output)) {
    if (@available(macOS 15.0, *)) {
        NSDictionary* feeds = @{input_: input};

        MPSGraphExecutionDescriptor* exec_desc = [MPSGraphExecutionDescriptor new];
        exec_desc.completionHandler = ^(NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results,
                                        NSError* error) {
            if (completion && !error) {
                completion(results[output_]);
            }
        };

        [graph_ runAsyncWithMTLCommandQueue:ctx_->queue()
                                      feeds:feeds
                              targetTensors:@[output_]
                           targetOperations:nil
                        executionDescriptor:exec_desc];
    }
}

EncoderOutputBuffer EncoderGraph::allocate_output_buffer() {
    EncoderOutputBuffer buffer;

    if (@available(macOS 15.0, *)) {
        bool use_fp16 = (config_.precision == Precision::FP16);
        MPSDataType dtype = use_fp16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        size_t elem_size = use_fp16 ? sizeof(__fp16) : sizeof(float);
        size_t elements = num_patches_ * cfg_.enc_dim;

        // Allocate shared MTLBuffer (zero-copy between GPU and CPU)
        buffer.buf = [ctx_->device() newBufferWithLength:elements * elem_size
                                                 options:MTLResourceStorageModeShared];

        // Wrap in MPSGraphTensorData for resultsDictionary
        buffer.td = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer.buf
                                                            shape:@[@(num_patches_), @(cfg_.enc_dim)]
                                                         dataType:dtype];

        // Direct pointer for zero-copy access
        buffer.ptr = [buffer.buf contents];
    }

    return buffer;
}

void EncoderGraph::run_into(MPSGraphTensorData* input, EncoderOutputBuffer& buffer) {
    if (@available(macOS 15.0, *)) {
        NSDictionary* feeds = @{input_: input};
        NSDictionary* resultsDict = @{output_: buffer.td};

        [graph_ runWithMTLCommandQueue:ctx_->queue()
                                 feeds:feeds
                      targetOperations:nil
                     resultsDictionary:resultsDict];
    }
}

void EncoderGraph::run_async_into(MPSGraphTensorData* input, EncoderOutputBuffer& buffer,
                                  void (^completion)(void)) {
    if (@available(macOS 15.0, *)) {
        NSDictionary* feeds = @{input_: input};
        NSDictionary* resultsDict = @{output_: buffer.td};

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
