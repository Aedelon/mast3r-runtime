// MASt3R Runtime - Decoder+DPT Graph Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#import "decoder_graph.hpp"
#import "graph_builder.hpp"
#include <stdexcept>

namespace mast3r {
namespace mpsgraph {

DecoderGraph::DecoderGraph(std::shared_ptr<MPSGraphContext> ctx, const RuntimeConfig& config)
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
        throw std::runtime_error("DecoderGraph requires macOS 15.0+");
    }
}

void DecoderGraph::load(safetensors::MultiSafetensorsFile& files) {
    if (@available(macOS 15.0, *)) {
        bool use_fp16 = (config_.precision == Precision::FP16);
        GraphBuilder gb(graph_, ctx_->device(), &files, use_fp16);

        // Decoder embed (enc_dim → dec_dim)
        std::string de = cfg_.decoder_embed_key();
        auto e2d_w = gb.load(de + "weight", @[@(cfg_.dec_dim), @(cfg_.enc_dim)]);
        auto e2d_b = gb.load(de + "bias", @[@(cfg_.dec_dim)]);

        // Decoder layers
        std::vector<DecoderLayer> dec(cfg_.dec_depth);
        for (int i = 0; i < cfg_.dec_depth; i++) {
            std::string p = cfg_.dec_block_key(i);
            auto& L = dec[i];
            L.n1w = gb.load(p + "norm1.weight", @[@(cfg_.dec_dim)]);
            L.n1b = gb.load(p + "norm1.bias", @[@(cfg_.dec_dim)]);
            L.qkvw = gb.load(p + "attn.qkv.weight", @[@(3*cfg_.dec_dim), @(cfg_.dec_dim)]);
            L.qkvb = gb.load(p + "attn.qkv.bias", @[@(3*cfg_.dec_dim)]);
            L.pw = gb.load(p + "attn.proj.weight", @[@(cfg_.dec_dim), @(cfg_.dec_dim)]);
            L.pb = gb.load(p + "attn.proj.bias", @[@(cfg_.dec_dim)]);
            L.nyw = gb.load(p + "norm_y.weight", @[@(cfg_.dec_dim)]);
            L.nyb = gb.load(p + "norm_y.bias", @[@(cfg_.dec_dim)]);
            L.n3w = gb.load(p + "norm3.weight", @[@(cfg_.dec_dim)]);
            L.n3b = gb.load(p + "norm3.bias", @[@(cfg_.dec_dim)]);
            L.cqw = gb.load(p + "cross_attn.projq.weight", @[@(cfg_.dec_dim), @(cfg_.dec_dim)]);
            L.cqb = gb.load(p + "cross_attn.projq.bias", @[@(cfg_.dec_dim)]);
            L.ckw = gb.load(p + "cross_attn.projk.weight", @[@(cfg_.dec_dim), @(cfg_.dec_dim)]);
            L.ckb = gb.load(p + "cross_attn.projk.bias", @[@(cfg_.dec_dim)]);
            L.cvw = gb.load(p + "cross_attn.projv.weight", @[@(cfg_.dec_dim), @(cfg_.dec_dim)]);
            L.cvb = gb.load(p + "cross_attn.projv.bias", @[@(cfg_.dec_dim)]);
            L.cpw = gb.load(p + "cross_attn.proj.weight", @[@(cfg_.dec_dim), @(cfg_.dec_dim)]);
            L.cpb = gb.load(p + "cross_attn.proj.bias", @[@(cfg_.dec_dim)]);
            L.n2w = gb.load(p + "norm2.weight", @[@(cfg_.dec_dim)]);
            L.n2b = gb.load(p + "norm2.bias", @[@(cfg_.dec_dim)]);
            L.f1w = gb.load(p + "mlp.fc1.weight", @[@(cfg_.dec_mlp), @(cfg_.dec_dim)]);
            L.f1b = gb.load(p + "mlp.fc1.bias", @[@(cfg_.dec_mlp)]);
            L.f2w = gb.load(p + "mlp.fc2.weight", @[@(cfg_.dec_dim), @(cfg_.dec_mlp)]);
            L.f2b = gb.load(p + "mlp.fc2.bias", @[@(cfg_.dec_dim)]);
        }

        // DPT weights
        std::string head_prefix = cfg_.head_key();
        std::string dp = head_prefix + "dpt.";

        int ap0_in = cfg_.enc_dim;
        int ap_in = cfg_.dec_dim;

        auto ap0_1w = gb.load(dp+"act_postprocess.0.0.weight", @[@96, @(ap0_in), @1, @1]);
        auto ap0_1b = gb.load(dp+"act_postprocess.0.0.bias", @[@96]);
        auto ap0_2w = gb.load(dp+"act_postprocess.0.1.weight", @[@96, @96, @4, @4]);
        auto ap0_2b = gb.load(dp+"act_postprocess.0.1.bias", @[@96]);
        auto ap1_1w = gb.load(dp+"act_postprocess.1.0.weight", @[@192, @(ap_in), @1, @1]);
        auto ap1_1b = gb.load(dp+"act_postprocess.1.0.bias", @[@192]);
        auto ap1_2w = gb.load(dp+"act_postprocess.1.1.weight", @[@192, @192, @2, @2]);
        auto ap1_2b = gb.load(dp+"act_postprocess.1.1.bias", @[@192]);
        auto ap2_1w = gb.load(dp+"act_postprocess.2.0.weight", @[@384, @(ap_in), @1, @1]);
        auto ap2_1b = gb.load(dp+"act_postprocess.2.0.bias", @[@384]);
        auto ap3_1w = gb.load(dp+"act_postprocess.3.0.weight", @[@768, @(ap_in), @1, @1]);
        auto ap3_1b = gb.load(dp+"act_postprocess.3.0.bias", @[@768]);
        auto ap3_2w = gb.load(dp+"act_postprocess.3.1.weight", @[@768, @768, @3, @3]);
        auto ap3_2b = gb.load(dp+"act_postprocess.3.1.bias", @[@768]);

        auto lr0w = gb.load(dp+"scratch.layer_rn.0.weight", @[@256, @96, @3, @3]);
        auto lr1w = gb.load(dp+"scratch.layer_rn.1.weight", @[@256, @192, @3, @3]);
        auto lr2w = gb.load(dp+"scratch.layer_rn.2.weight", @[@256, @384, @3, @3]);
        auto lr3w = gb.load(dp+"scratch.layer_rn.3.weight", @[@256, @768, @3, @3]);

        auto hd0w = gb.load(dp+"head.0.weight", @[@128, @256, @3, @3]);
        auto hd0b = gb.load(dp+"head.0.bias", @[@128]);
        auto hd2w = gb.load(dp+"head.2.weight", @[@128, @128, @3, @3]);
        auto hd2b = gb.load(dp+"head.2.bias", @[@128]);
        auto hd4w = gb.load(dp+"head.4.weight", @[@4, @128, @1, @1]);
        auto hd4b = gb.load(dp+"head.4.bias", @[@4]);

        // Local features weights
        std::string lf = head_prefix + "head_local_features.";
        int lf_in = cfg_.enc_dim + cfg_.dec_dim;
        int lf_out = (cfg_.desc_dim + 1) * cfg_.patch_size * cfg_.patch_size;

        auto lf1w = gb.load(lf+"fc1.weight", @[@(cfg_.lf_hidden), @(lf_in)]);
        auto lf1b = gb.load(lf+"fc1.bias", @[@(cfg_.lf_hidden)]);
        auto lf2w = gb.load(lf+"fc2.weight", @[@(lf_out), @(cfg_.lf_hidden)]);
        auto lf2b = gb.load(lf+"fc2.bias", @[@(lf_out)]);

        // Build graph
        // Input: encoder output [N, enc_dim] FP16
        input_ = [graph_ placeholderWithShape:@[@(num_patches_), @(cfg_.enc_dim)]
                                     dataType:use_fp16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32
                                         name:@"enc_out"];

        // Store enc_out for DPT hook0 and local features
        MPSGraphTensor* enc_out = input_;

        // Decoder embed
        MPSGraphTensor* dec_in = gb.linear(enc_out, e2d_w, e2d_b);
        MPSGraphTensor* enc_proj = dec_in;  // For cross-attention
        MPSGraphTensor* x = dec_in;

        // DPT hook indices
        int hook1_idx = cfg_.dec_depth / 2 - 1;
        int hook2_idx = cfg_.dec_depth * 3 / 4 - 1;
        int hook3_idx = cfg_.dec_depth - 1;

        MPSGraphTensor* hooks[4];
        hooks[0] = enc_out;

        // Decoder blocks
        for (int i = 0; i < cfg_.dec_depth; i++) {
            auto& L = dec[i];
            MPSGraphTensor* res = x;
            x = gb.layer_norm(x, L.n1w, L.n1b);
            x = gb.self_attention(x, L.qkvw, L.qkvb, L.pw, L.pb, cfg_.dec_heads, cfg_.dec_head_dim);
            x = gb.add(x, res);
            res = x;
            MPSGraphTensor* xn = gb.layer_norm(x, L.n3w, L.n3b);
            MPSGraphTensor* yn = gb.layer_norm(enc_proj, L.nyw, L.nyb);
            x = gb.add(gb.cross_attention(xn, yn, L.cqw, L.cqb, L.ckw, L.ckb, L.cvw, L.cvb, L.cpw, L.cpb, cfg_.dec_heads, cfg_.dec_head_dim), res);
            res = x;
            x = gb.layer_norm(x, L.n2w, L.n2b);
            x = gb.linear(x, L.f1w, L.f1b);
            x = gb.gelu(x);
            x = gb.linear(x, L.f2w, L.f2b);
            x = gb.add(x, res);

            if (i == hook1_idx) hooks[1] = x;
            if (i == hook2_idx) hooks[2] = x;
            if (i == hook3_idx) hooks[3] = x;
        }
        MPSGraphTensor* dec_out = x;

        // DPT
        auto to_spatial = [&](MPSGraphTensor* t, int dim) {
            return [graph_ reshapeTensor:t withShape:@[@1, @(patch_h_), @(patch_w_), @(dim)] name:nil];
        };

        MPSGraphTensor* h0 = to_spatial(hooks[0], cfg_.enc_dim);
        MPSGraphTensor* h1 = to_spatial(hooks[1], cfg_.dec_dim);
        MPSGraphTensor* h2 = to_spatial(hooks[2], cfg_.dec_dim);
        MPSGraphTensor* h3 = to_spatial(hooks[3], cfg_.dec_dim);

        MPSGraphTensor* f0 = gb.conv_transpose2d(gb.conv2d(h0, ap0_1w, ap0_1b), ap0_2w, ap0_2b, 4);
        MPSGraphTensor* f1 = gb.conv_transpose2d(gb.conv2d(h1, ap1_1w, ap1_1b), ap1_2w, ap1_2b, 2);
        MPSGraphTensor* f2 = gb.conv2d(h2, ap2_1w, ap2_1b);
        MPSGraphTensor* f3 = gb.conv2d(gb.conv2d(h3, ap3_1w, ap3_1b), ap3_2w, ap3_2b, 2, 1);

        f0 = gb.conv2d(f0, lr0w, nil, 1, 1);
        f1 = gb.conv2d(f1, lr1w, nil, 1, 1);
        f2 = gb.conv2d(f2, lr2w, nil, 1, 1);
        f3 = gb.conv2d(f3, lr3w, nil, 1, 1);

        f1 = gb.upsample(f1, 2);
        f2 = gb.upsample(f2, 4);
        f3 = gb.upsample(f3, 8);
        MPSGraphTensor* fused = gb.add(gb.add(gb.add(f0, f1), f2), f3);

        MPSGraphTensor* head = gb.upsample(fused, 2);
        head = gb.relu(gb.conv2d(head, hd0w, hd0b, 1, 1));
        head = gb.upsample(head, 2);
        head = gb.relu(gb.conv2d(head, hd2w, hd2b, 1, 1));
        MPSGraphTensor* pts_conf = gb.conv2d(head, hd4w, hd4b);

        // Resize if needed (DUNE with patch_size=14)
        int dpt_h = patch_h_ * 16;
        int dpt_w = patch_w_ * 16;
        if (dpt_h != img_h_ || dpt_w != img_w_) {
            pts_conf = [graph_ resizeTensor:pts_conf size:@[@(img_h_), @(img_w_)]
                                      mode:MPSGraphResizeBilinear centerResult:YES alignCorners:NO
                                    layout:MPSGraphTensorNamedDataLayoutNHWC name:nil];
        }

        // Cast to FP32 for output
        if (use_fp16) {
            pts_conf = [graph_ castTensor:pts_conf toType:MPSDataTypeFloat32 name:nil];
        }

        // Split pts3d (3 channels) and conf (1 channel)
        NSArray<MPSGraphTensor*>* pts_conf_split = [graph_ splitTensor:pts_conf
                                                            splitSizes:@[@3, @1]
                                                                  axis:-1 name:nil];
        output_pts3d_ = pts_conf_split[0];
        output_conf_ = [graph_ squeezeTensor:pts_conf_split[1] axis:-1 name:nil];

        // Local features
        MPSGraphTensor* concat = [graph_ concatTensors:@[enc_out, dec_out] dimension:-1 name:nil];
        MPSGraphTensor* lf_result = gb.gelu(gb.linear(concat, lf1w, lf1b));
        lf_result = gb.linear(lf_result, lf2w, lf2b);
        lf_result = [graph_ reshapeTensor:lf_result withShape:@[@1, @(patch_h_), @(patch_w_), @(lf_out)] name:nil];
        lf_result = gb.pixel_shuffle(lf_result, cfg_.patch_size);
        NSArray<MPSGraphTensor*>* desc_split = [graph_ splitTensor:lf_result splitSizes:@[@(cfg_.desc_dim), @1] axis:-1 name:nil];
        MPSGraphTensor* desc_out = gb.l2_norm(desc_split[0]);

        if (use_fp16) {
            desc_out = [graph_ castTensor:desc_out toType:MPSDataTypeFloat32 name:nil];
        }
        output_desc_ = desc_out;
    }
}

void DecoderGraph::compile() {
    // Pre-compilation not implemented - uses JIT compilation on first run
    // MPSGraph automatically caches compiled graphs internally
}

DecoderOutput DecoderGraph::run(MPSGraphTensorData* enc_out) {
    DecoderOutput result = {nil, nil, nil};

    if (@available(macOS 15.0, *)) {
        NSDictionary* feeds = @{input_: enc_out};
        NSDictionary* results = [graph_ runWithMTLCommandQueue:ctx_->queue()
                                                         feeds:feeds
                                                 targetTensors:@[output_pts3d_, output_conf_, output_desc_]
                                             targetOperations:nil];
        result.pts3d = results[output_pts3d_];
        result.conf = results[output_conf_];
        result.desc = results[output_desc_];
    }
    return result;
}

void DecoderGraph::run_async(MPSGraphTensorData* enc_out,
                             void (^completion)(DecoderOutput output)) {
    if (@available(macOS 15.0, *)) {
        NSDictionary* feeds = @{input_: enc_out};

        MPSGraphExecutionDescriptor* exec_desc = [MPSGraphExecutionDescriptor new];
        exec_desc.completionHandler = ^(NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results,
                                        NSError* error) {
            if (completion && !error) {
                DecoderOutput output;
                output.pts3d = results[output_pts3d_];
                output.conf = results[output_conf_];
                output.desc = results[output_desc_];
                completion(output);
            }
        };

        [graph_ runAsyncWithMTLCommandQueue:ctx_->queue()
                                      feeds:feeds
                              targetTensors:@[output_pts3d_, output_conf_, output_desc_]
                           targetOperations:nil
                        executionDescriptor:exec_desc];
    }
}

DecoderOutputBuffers DecoderGraph::allocate_output_buffers() {
    DecoderOutputBuffers buffers;

    if (@available(macOS 15.0, *)) {
        const size_t pts3d_elements = img_h_ * img_w_ * 3;
        const size_t conf_elements = img_h_ * img_w_;
        const size_t desc_elements = img_h_ * img_w_ * cfg_.desc_dim;

        // Allocate shared MTLBuffers (zero-copy between GPU and CPU)
        buffers.buf_pts3d = [ctx_->device() newBufferWithLength:pts3d_elements * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        buffers.buf_conf = [ctx_->device() newBufferWithLength:conf_elements * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        buffers.buf_desc = [ctx_->device() newBufferWithLength:desc_elements * sizeof(float)
                                                       options:MTLResourceStorageModeShared];

        // Wrap in MPSGraphTensorData for resultsDictionary
        buffers.td_pts3d = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffers.buf_pts3d
                                                                   shape:@[@1, @(img_h_), @(img_w_), @3]
                                                                dataType:MPSDataTypeFloat32];
        buffers.td_conf = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffers.buf_conf
                                                                  shape:@[@1, @(img_h_), @(img_w_)]
                                                               dataType:MPSDataTypeFloat32];
        buffers.td_desc = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffers.buf_desc
                                                                  shape:@[@1, @(img_h_), @(img_w_), @(cfg_.desc_dim)]
                                                               dataType:MPSDataTypeFloat32];

        // Direct pointers for zero-copy CPU access
        buffers.ptr_pts3d = static_cast<float*>([buffers.buf_pts3d contents]);
        buffers.ptr_conf = static_cast<float*>([buffers.buf_conf contents]);
        buffers.ptr_desc = static_cast<float*>([buffers.buf_desc contents]);
    }

    return buffers;
}

void DecoderGraph::run_into(MPSGraphTensorData* enc_out, DecoderOutputBuffers& buffers) {
    if (@available(macOS 15.0, *)) {
        NSDictionary* feeds = @{input_: enc_out};
        NSDictionary* resultsDict = @{
            output_pts3d_: buffers.td_pts3d,
            output_conf_: buffers.td_conf,
            output_desc_: buffers.td_desc
        };

        [graph_ runWithMTLCommandQueue:ctx_->queue()
                                 feeds:feeds
                      targetOperations:nil
                     resultsDictionary:resultsDict];
    }
}

void DecoderGraph::run_async_into(MPSGraphTensorData* enc_out, DecoderOutputBuffers& buffers,
                                  void (^completion)(void)) {
    if (@available(macOS 15.0, *)) {
        NSDictionary* feeds = @{input_: enc_out};
        NSDictionary* resultsDict = @{
            output_pts3d_: buffers.td_pts3d,
            output_conf_: buffers.td_conf,
            output_desc_: buffers.td_desc
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
