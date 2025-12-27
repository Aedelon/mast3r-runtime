// MASt3R Runtime - MPSGraph Engine Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
// Uses MPSGraph with native SDPA (macOS 15+) for ~21x speedup.
// Supports both MASt3R (ViT-Large) and DUNE (ViT-Small/Base).

#import "mpsgraph_engine.hpp"
#import "mpsgraph_context.hpp"
#include "../common/safetensors.hpp"
#include <cmath>
#include <chrono>

namespace mast3r {
namespace mpsgraph {

// ============================================================================
// Model Configuration
// ============================================================================

struct ModelConfig {
    // Architecture
    int patch_size;
    bool is_dune;  // DUNE vs MASt3R architecture

    // Encoder
    int enc_dim;
    int enc_heads;
    int enc_head_dim;
    int enc_mlp;
    int enc_depth;

    // Decoder
    int dec_dim;
    int dec_heads;
    int dec_head_dim;
    int dec_mlp;
    int dec_depth;

    // Output
    int desc_dim;
    int lf_hidden;  // local features hidden dim

    // Weight prefixes (different for DUNE vs MASt3R)
    // MASt3R: patch_embed.proj, enc_blocks.X, dec_blocks.X, downstream_head1
    // DUNE:   encoder.patch_embed.proj, encoder.blocks.0.X, mast3r.dec_blocks.X, mast3r.downstream_head1

    std::string enc_block_key(int i) const {
        if (is_dune) {
            // DUNE: encoder.blocks.0.X (note the extra .0.)
            return "encoder.blocks.0." + std::to_string(i) + ".";
        } else {
            return "enc_blocks." + std::to_string(i) + ".";
        }
    }

    std::string dec_block_key(int i) const {
        if (is_dune) {
            return "mast3r.dec_blocks." + std::to_string(i) + ".";
        } else {
            return "dec_blocks." + std::to_string(i) + ".";
        }
    }

    std::string patch_embed_key() const {
        return is_dune ? "encoder.patch_embed.proj." : "patch_embed.proj.";
    }

    std::string enc_norm_key() const {
        return is_dune ? "encoder.norm." : "enc_norm.";
    }

    std::string decoder_embed_key() const {
        return is_dune ? "mast3r.decoder_embed." : "decoder_embed.";
    }

    std::string head_key() const {
        return is_dune ? "mast3r.downstream_head1." : "downstream_head1.";
    }

    static ModelConfig mast3r_vit_large() {
        return {
            .patch_size = 16,
            .is_dune = false,
            .enc_dim = 1024, .enc_heads = 16, .enc_head_dim = 64, .enc_mlp = 4096, .enc_depth = 24,
            .dec_dim = 768, .dec_heads = 12, .dec_head_dim = 64, .dec_mlp = 3072, .dec_depth = 12,
            .desc_dim = 24,
            .lf_hidden = 7168
        };
    }

    static ModelConfig dune_vit_small() {
        return {
            .patch_size = 14,
            .is_dune = true,
            .enc_dim = 384, .enc_heads = 6, .enc_head_dim = 64, .enc_mlp = 1536, .enc_depth = 12,
            .dec_dim = 768, .dec_heads = 12, .dec_head_dim = 64, .dec_mlp = 3072, .dec_depth = 12,
            .desc_dim = 24,
            .lf_hidden = 4096
        };
    }

    static ModelConfig dune_vit_base() {
        return {
            .patch_size = 14,
            .is_dune = true,
            .enc_dim = 768, .enc_heads = 12, .enc_head_dim = 64, .enc_mlp = 3072, .enc_depth = 12,
            .dec_dim = 768, .dec_heads = 12, .dec_head_dim = 64, .dec_mlp = 3072, .dec_depth = 12,
            .desc_dim = 24,
            .lf_hidden = 4096
        };
    }

    static ModelConfig from_variant(ModelVariant variant) {
        switch (variant) {
            case ModelVariant::MAST3R_VIT_LARGE:
                return mast3r_vit_large();
            case ModelVariant::DUNE_VIT_SMALL_336:
            case ModelVariant::DUNE_VIT_SMALL_448:
                return dune_vit_small();
            case ModelVariant::DUNE_VIT_BASE_336:
            case ModelVariant::DUNE_VIT_BASE_448:
                return dune_vit_base();
            default:
                return dune_vit_small();
        }
    }
};

constexpr float LN_EPS = 1e-6f;

// ============================================================================
// Graph Builder Helper
// ============================================================================

class API_AVAILABLE(macos(15.0)) GraphBuilder {
public:
    MPSGraph* graph_;
    id<MTLDevice> device_;
    safetensors::MultiSafetensorsFile* files_;

    GraphBuilder(MPSGraph* graph, id<MTLDevice> device, safetensors::MultiSafetensorsFile* files)
        : graph_(graph), device_(device), files_(files) {}

    MPSGraphTensor* load(const std::string& name, NSArray<NSNumber*>* shape) {
        auto data = files_->load_tensor_f32(name);
        NSData* nsdata = [NSData dataWithBytes:data.data() length:data.size() * sizeof(float)];
        return [graph_ constantWithData:nsdata shape:shape dataType:MPSDataTypeFloat32];
    }

    bool has_weight(const std::string& name) {
        return files_->has_tensor(name);
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
// MPSGraphEngine Implementation
// ============================================================================

MPSGraphEngine::MPSGraphEngine(const RuntimeConfig& config)
    : MPSGraphEngine(config, MPSGraphContext::shared()) {}

MPSGraphEngine::MPSGraphEngine(const RuntimeConfig& config, std::shared_ptr<MPSGraphContext> ctx)
    : config_(config), ctx_(std::move(ctx)) {
    if (@available(macOS 15.0, *)) {
        graph_ = ctx_->create_graph();
        spec_ = get_model_spec(config.variant);
    } else {
        throw std::runtime_error("MPSGraph SDPA requires macOS 15.0+");
    }
}

MPSGraphEngine::~MPSGraphEngine() = default;

bool MPSGraphEngine::is_available() {
    return MPSGraphContext::is_available();
}

std::string MPSGraphEngine::name() const {
    if (@available(macOS 15.0, *)) {
        return std::string("MPSGraph SDPA (") + ctx_->device_name() + ")";
    }
    return "MPSGraph (unavailable)";
}

void MPSGraphEngine::build_graph() {
    // This is called from load() after weights are loaded
}

void MPSGraphEngine::load(const std::string& model_path) {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            // Load safetensors files (supports split DUNE files)
            safetensors::MultiSafetensorsFile files;

            // Check if path is a directory (DUNE) or single file (MASt3R)
            std::ifstream test_file(model_path);
            if (test_file.good()) {
                // Single file - extract directory and add all files
                std::string dir = model_path.substr(0, model_path.rfind('/'));
                files.add_directory(dir);
            } else {
                // Directory path
                files.add_directory(model_path);
            }

            ModelConfig cfg = ModelConfig::from_variant(config_.variant);

            // Get image dimensions from config
            const int IMG_H = config_.resolution;
            const int IMG_W = static_cast<int>(config_.resolution * 4.0f / 3.0f);  // 4:3 aspect
            const int PATCH_H = IMG_H / cfg.patch_size;
            const int PATCH_W = IMG_W / cfg.patch_size;
            const int NUM_PATCHES = PATCH_H * PATCH_W;

            GraphBuilder gb(graph_, ctx_->device(), &files);

            // ================================================================
            // Load weights using ModelConfig methods
            // ================================================================

            // Patch embedding
            std::string pe = cfg.patch_embed_key();
            auto patch_w = gb.load(pe+"weight", @[@(cfg.enc_dim), @3, @(cfg.patch_size), @(cfg.patch_size)]);
            auto patch_b = gb.load(pe+"bias", @[@(cfg.enc_dim)]);

            // Encoder layers
            struct EncL { MPSGraphTensor *n1w,*n1b,*qkvw,*qkvb,*pw,*pb,*n2w,*n2b,*f1w,*f1b,*f2w,*f2b; };
            std::vector<EncL> enc(cfg.enc_depth);

            std::string en = cfg.enc_norm_key();
            auto enc_nw = gb.load(en+"weight", @[@(cfg.enc_dim)]);
            auto enc_nb = gb.load(en+"bias", @[@(cfg.enc_dim)]);

            for (int i = 0; i < cfg.enc_depth; i++) {
                std::string p = cfg.enc_block_key(i);
                enc[i] = {
                    gb.load(p+"norm1.weight", @[@(cfg.enc_dim)]),
                    gb.load(p+"norm1.bias", @[@(cfg.enc_dim)]),
                    gb.load(p+"attn.qkv.weight", @[@(3*cfg.enc_dim), @(cfg.enc_dim)]),
                    gb.load(p+"attn.qkv.bias", @[@(3*cfg.enc_dim)]),
                    gb.load(p+"attn.proj.weight", @[@(cfg.enc_dim), @(cfg.enc_dim)]),
                    gb.load(p+"attn.proj.bias", @[@(cfg.enc_dim)]),
                    gb.load(p+"norm2.weight", @[@(cfg.enc_dim)]),
                    gb.load(p+"norm2.bias", @[@(cfg.enc_dim)]),
                    gb.load(p+"mlp.fc1.weight", @[@(cfg.enc_mlp), @(cfg.enc_dim)]),
                    gb.load(p+"mlp.fc1.bias", @[@(cfg.enc_mlp)]),
                    gb.load(p+"mlp.fc2.weight", @[@(cfg.enc_dim), @(cfg.enc_mlp)]),
                    gb.load(p+"mlp.fc2.bias", @[@(cfg.enc_dim)])
                };
            }

            // Decoder layers
            struct DecL {
                MPSGraphTensor *n1w,*n1b,*qkvw,*qkvb,*pw,*pb;
                MPSGraphTensor *nyw,*nyb,*n3w,*n3b,*cqw,*cqb,*ckw,*ckb,*cvw,*cvb,*cpw,*cpb;
                MPSGraphTensor *n2w,*n2b,*f1w,*f1b,*f2w,*f2b;
            };
            std::vector<DecL> dec(cfg.dec_depth);

            std::string de = cfg.decoder_embed_key();
            auto e2d_w = gb.load(de+"weight", @[@(cfg.dec_dim), @(cfg.enc_dim)]);
            auto e2d_b = gb.load(de+"bias", @[@(cfg.dec_dim)]);

            for (int i = 0; i < cfg.dec_depth; i++) {
                std::string p = cfg.dec_block_key(i);
                dec[i] = {
                    gb.load(p+"norm1.weight", @[@(cfg.dec_dim)]),
                    gb.load(p+"norm1.bias", @[@(cfg.dec_dim)]),
                    gb.load(p+"attn.qkv.weight", @[@(3*cfg.dec_dim), @(cfg.dec_dim)]),
                    gb.load(p+"attn.qkv.bias", @[@(3*cfg.dec_dim)]),
                    gb.load(p+"attn.proj.weight", @[@(cfg.dec_dim), @(cfg.dec_dim)]),
                    gb.load(p+"attn.proj.bias", @[@(cfg.dec_dim)]),
                    gb.load(p+"norm_y.weight", @[@(cfg.dec_dim)]),
                    gb.load(p+"norm_y.bias", @[@(cfg.dec_dim)]),
                    gb.load(p+"norm3.weight", @[@(cfg.dec_dim)]),
                    gb.load(p+"norm3.bias", @[@(cfg.dec_dim)]),
                    gb.load(p+"cross_attn.projq.weight", @[@(cfg.dec_dim), @(cfg.dec_dim)]),
                    gb.load(p+"cross_attn.projq.bias", @[@(cfg.dec_dim)]),
                    gb.load(p+"cross_attn.projk.weight", @[@(cfg.dec_dim), @(cfg.dec_dim)]),
                    gb.load(p+"cross_attn.projk.bias", @[@(cfg.dec_dim)]),
                    gb.load(p+"cross_attn.projv.weight", @[@(cfg.dec_dim), @(cfg.dec_dim)]),
                    gb.load(p+"cross_attn.projv.bias", @[@(cfg.dec_dim)]),
                    gb.load(p+"cross_attn.proj.weight", @[@(cfg.dec_dim), @(cfg.dec_dim)]),
                    gb.load(p+"cross_attn.proj.bias", @[@(cfg.dec_dim)]),
                    gb.load(p+"norm2.weight", @[@(cfg.dec_dim)]),
                    gb.load(p+"norm2.bias", @[@(cfg.dec_dim)]),
                    gb.load(p+"mlp.fc1.weight", @[@(cfg.dec_mlp), @(cfg.dec_dim)]),
                    gb.load(p+"mlp.fc1.bias", @[@(cfg.dec_mlp)]),
                    gb.load(p+"mlp.fc2.weight", @[@(cfg.dec_dim), @(cfg.dec_mlp)]),
                    gb.load(p+"mlp.fc2.bias", @[@(cfg.dec_dim)])
                };
            }

            // ================================================================
            // DPT and Head weights
            // ================================================================

            std::string head_prefix = cfg.head_key();
            std::string lf = head_prefix + "head_local_features.";
            int lf_in = cfg.enc_dim + cfg.dec_dim;
            int lf_out = (cfg.desc_dim + 1) * cfg.patch_size * cfg.patch_size;

            auto lf1w = gb.load(lf+"fc1.weight", @[@(cfg.lf_hidden), @(lf_in)]);
            auto lf1b = gb.load(lf+"fc1.bias", @[@(cfg.lf_hidden)]);
            auto lf2w = gb.load(lf+"fc2.weight", @[@(lf_out), @(cfg.lf_hidden)]);
            auto lf2b = gb.load(lf+"fc2.bias", @[@(lf_out)]);

            // DPT weights
            std::string dp = head_prefix + "dpt.";

            // For now, load basic DPT layers
            // MASt3R uses enc_dim=1024 for hook0, dec_dim=768 for hooks 1-3
            // DUNE uses enc_dim for hook0, dec_dim for hooks 1-3

            int ap0_in = cfg.enc_dim;
            int ap_in = cfg.dec_dim;

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

            // ================================================================
            // Build graph
            // ================================================================

            input_placeholder_ = [graph_ placeholderWithShape:@[@1, @(IMG_H), @(IMG_W), @3]
                                                     dataType:MPSDataTypeFloat32 name:@"image"];

            // Patch embedding
            MPSGraphTensor* patches = gb.conv2d(input_placeholder_, patch_w, patch_b, cfg.patch_size, 0);
            patches = [graph_ reshapeTensor:patches withShape:@[@(NUM_PATCHES), @(cfg.enc_dim)] name:nil];

            // Encoder
            MPSGraphTensor* x = patches;
            for (int i = 0; i < cfg.enc_depth; i++) {
                auto& L = enc[i];
                MPSGraphTensor* res = x;
                x = gb.layer_norm(x, L.n1w, L.n1b);
                x = gb.self_attention(x, L.qkvw, L.qkvb, L.pw, L.pb, cfg.enc_heads, cfg.enc_head_dim);
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

            // DPT hook indices based on decoder depth
            int hook1_idx = cfg.dec_depth / 2 - 1;      // middle
            int hook2_idx = cfg.dec_depth * 3 / 4 - 1;  // 3/4
            int hook3_idx = cfg.dec_depth - 1;          // last

            MPSGraphTensor* hooks[4];
            hooks[0] = enc_out;

            for (int i = 0; i < cfg.dec_depth; i++) {
                auto& L = dec[i];
                MPSGraphTensor* res = x;
                x = gb.layer_norm(x, L.n1w, L.n1b);
                x = gb.self_attention(x, L.qkvw, L.qkvb, L.pw, L.pb, cfg.dec_heads, cfg.dec_head_dim);
                x = gb.add(x, res);
                res = x;
                MPSGraphTensor* xn = gb.layer_norm(x, L.n3w, L.n3b);
                MPSGraphTensor* yn = gb.layer_norm(enc_proj, L.nyw, L.nyb);
                x = gb.add(gb.cross_attention(xn, yn, L.cqw, L.cqb, L.ckw, L.ckb, L.cvw, L.cvb, L.cpw, L.cpb, cfg.dec_heads, cfg.dec_head_dim), res);
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
                return [graph_ reshapeTensor:t withShape:@[@1, @(PATCH_H), @(PATCH_W), @(dim)] name:nil];
            };
            MPSGraphTensor* h0 = to_spatial(hooks[0], cfg.enc_dim);
            MPSGraphTensor* h1 = to_spatial(hooks[1], cfg.dec_dim);
            MPSGraphTensor* h2 = to_spatial(hooks[2], cfg.dec_dim);
            MPSGraphTensor* h3 = to_spatial(hooks[3], cfg.dec_dim);

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

            // Resize to target resolution if needed (for DUNE with patch_size != 16)
            // DPT produces PATCH_H*16 x PATCH_W*16, but we need IMG_H x IMG_W
            int dpt_h = PATCH_H * 16;
            int dpt_w = PATCH_W * 16;
            if (dpt_h != IMG_H || dpt_w != IMG_W) {
                pts_conf = [graph_ resizeTensor:pts_conf size:@[@(IMG_H), @(IMG_W)]
                                          mode:MPSGraphResizeBilinear centerResult:YES alignCorners:NO
                                        layout:MPSGraphTensorNamedDataLayoutNHWC name:nil];
            }
            output_pts3d_conf_ = pts_conf;

            // Local features
            MPSGraphTensor* concat = [graph_ concatTensors:@[enc_out, dec_out] dimension:-1 name:nil];
            MPSGraphTensor* lf_result = gb.gelu(gb.linear(concat, lf1w, lf1b));
            lf_result = gb.linear(lf_result, lf2w, lf2b);
            lf_result = [graph_ reshapeTensor:lf_result withShape:@[@1, @(PATCH_H), @(PATCH_W), @(lf_out)] name:nil];
            lf_result = gb.pixel_shuffle(lf_result, cfg.patch_size);
            NSArray<MPSGraphTensor*>* desc_split = [graph_ splitTensor:lf_result splitSizes:@[@(cfg.desc_dim), @1] axis:-1 name:nil];
            output_descriptors_ = gb.l2_norm(desc_split[0]);

            is_loaded_ = true;
        }
    }
}

void MPSGraphEngine::warmup(int num_iterations) {
    if (!is_loaded_) return;

    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            const int IMG_H = config_.resolution;
            const int IMG_W = static_cast<int>(config_.resolution * 4.0f / 3.0f);

            std::vector<float> dummy(IMG_H * IMG_W * 3, 0.0f);
            MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                                                shape:@[@1, @(IMG_H), @(IMG_W), @3]];
            MPSNDArray* arr = [[MPSNDArray alloc] initWithDevice:ctx_->device() descriptor:desc];
            [arr writeBytes:(void*)dummy.data() strideBytes:nil];
            MPSGraphTensorData* td = [[MPSGraphTensorData alloc] initWithMPSNDArray:arr];
            NSDictionary* feeds = @{input_placeholder_: td};

            for (int i = 0; i < num_iterations; i++) {
                [graph_ runWithMTLCommandQueue:ctx_->queue() feeds:feeds
                                  targetTensors:@[output_pts3d_conf_, output_descriptors_] targetOperations:nil];
            }
        }
    }
}

void MPSGraphEngine::preprocess(const ImageView& img, float* output) {
    // ImageNet normalization: (x / 255 - mean) / std
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int idx = (y * img.width + x) * 3;
            for (int c = 0; c < 3; c++) {
                float val = static_cast<float>(img.data[idx + c]) / 255.0f;
                output[idx + c] = (val - mean[c]) / std[c];
            }
        }
    }
}

InferenceResult MPSGraphEngine::infer(const ImageView& img1, const ImageView& img2) {
    InferenceResult result;

    if (!is_loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            auto total_start = std::chrono::high_resolution_clock::now();

            const int IMG_H = config_.resolution;
            const int IMG_W = static_cast<int>(config_.resolution * 4.0f / 3.0f);
            const size_t img_size = IMG_H * IMG_W * 3;

            // Preprocess
            auto preprocess_start = std::chrono::high_resolution_clock::now();
            std::vector<float> img1_normalized(img_size);
            std::vector<float> img2_normalized(img_size);
            preprocess(img1, img1_normalized.data());
            preprocess(img2, img2_normalized.data());
            auto preprocess_end = std::chrono::high_resolution_clock::now();

            // Create tensor data
            MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                                                shape:@[@1, @(IMG_H), @(IMG_W), @3]];

            // Run inference on both images
            auto inference_start = std::chrono::high_resolution_clock::now();

            // Image 1
            MPSNDArray* arr1 = [[MPSNDArray alloc] initWithDevice:ctx_->device() descriptor:desc];
            [arr1 writeBytes:(void*)img1_normalized.data() strideBytes:nil];
            MPSGraphTensorData* td1 = [[MPSGraphTensorData alloc] initWithMPSNDArray:arr1];
            NSDictionary* results1 = [graph_ runWithMTLCommandQueue:ctx_->queue() feeds:@{input_placeholder_: td1}
                                                       targetTensors:@[output_pts3d_conf_, output_descriptors_] targetOperations:nil];

            // Image 2
            MPSNDArray* arr2 = [[MPSNDArray alloc] initWithDevice:ctx_->device() descriptor:desc];
            [arr2 writeBytes:(void*)img2_normalized.data() strideBytes:nil];
            MPSGraphTensorData* td2 = [[MPSGraphTensorData alloc] initWithMPSNDArray:arr2];
            NSDictionary* results2 = [graph_ runWithMTLCommandQueue:ctx_->queue() feeds:@{input_placeholder_: td2}
                                                       targetTensors:@[output_pts3d_conf_, output_descriptors_] targetOperations:nil];

            auto inference_end = std::chrono::high_resolution_clock::now();

            // Copy results
            result.height = IMG_H;
            result.width = IMG_W;
            result.desc_dim = spec_.desc_dim;

            size_t pts_size = IMG_H * IMG_W * 4;
            size_t desc_size = IMG_H * IMG_W * spec_.desc_dim;

            result.pts3d_1 = new float[IMG_H * IMG_W * 3];
            result.pts3d_2 = new float[IMG_H * IMG_W * 3];
            result.conf_1 = new float[IMG_H * IMG_W];
            result.conf_2 = new float[IMG_H * IMG_W];
            result.desc_1 = new float[desc_size];
            result.desc_2 = new float[desc_size];

            // Extract from pts3d_conf (4 channels: xyz + conf)
            std::vector<float> pts_conf1(pts_size), pts_conf2(pts_size);
            [[results1[output_pts3d_conf_] mpsndarray] readBytes:pts_conf1.data() strideBytes:nil];
            [[results2[output_pts3d_conf_] mpsndarray] readBytes:pts_conf2.data() strideBytes:nil];

            // Split pts3d and conf
            for (int i = 0; i < IMG_H * IMG_W; i++) {
                result.pts3d_1[i * 3 + 0] = pts_conf1[i * 4 + 0];
                result.pts3d_1[i * 3 + 1] = pts_conf1[i * 4 + 1];
                result.pts3d_1[i * 3 + 2] = pts_conf1[i * 4 + 2];
                result.conf_1[i] = pts_conf1[i * 4 + 3];

                result.pts3d_2[i * 3 + 0] = pts_conf2[i * 4 + 0];
                result.pts3d_2[i * 3 + 1] = pts_conf2[i * 4 + 1];
                result.pts3d_2[i * 3 + 2] = pts_conf2[i * 4 + 2];
                result.conf_2[i] = pts_conf2[i * 4 + 3];
            }

            // Descriptors
            [[results1[output_descriptors_] mpsndarray] readBytes:result.desc_1 strideBytes:nil];
            [[results2[output_descriptors_] mpsndarray] readBytes:result.desc_2 strideBytes:nil];

            auto total_end = std::chrono::high_resolution_clock::now();

            result.preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
            result.inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
            result.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        }
    }

    return result;
}

MatchResult MPSGraphEngine::match(
    const float* desc_1, const float* desc_2,
    int height, int width, int desc_dim,
    const MatchingConfig& config
) {
    MatchResult result;

    auto start = std::chrono::high_resolution_clock::now();

    const int N = height * width;
    std::vector<int> best_match_12(N), best_match_21(N);
    std::vector<float> best_sim_12(N, -1e9f), best_sim_21(N, -1e9f);

    // Find best matches
    for (int i = 0; i < N; i++) {
        const float* d1 = desc_1 + i * desc_dim;
        for (int j = 0; j < N; j++) {
            const float* d2 = desc_2 + j * desc_dim;
            float sim = 0.0f;
            for (int k = 0; k < desc_dim; k++) {
                sim += d1[k] * d2[k];
            }
            if (sim > best_sim_12[i]) {
                best_sim_12[i] = sim;
                best_match_12[i] = j;
            }
            if (sim > best_sim_21[j]) {
                best_sim_21[j] = sim;
                best_match_21[j] = i;
            }
        }
    }

    // Reciprocal matching
    for (int i = 0; i < N; i++) {
        int j = best_match_12[i];
        if (config.reciprocal && best_match_21[j] != i) continue;
        if (best_sim_12[i] < config.confidence_threshold) continue;

        result.idx_1.push_back(i);
        result.idx_2.push_back(j);
        result.confidence.push_back(best_sim_12[i]);

        int y1 = i / width, x1 = i % width;
        int y2 = j / width, x2 = j % width;
        result.pts2d_1.push_back(static_cast<float>(x1));
        result.pts2d_1.push_back(static_cast<float>(y1));
        result.pts2d_2.push_back(static_cast<float>(x2));
        result.pts2d_2.push_back(static_cast<float>(y2));
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.match_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

}  // namespace mpsgraph
}  // namespace mast3r
