// MASt3R Runtime - Metal Engine Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#import "metal_engine.hpp"
#import "metal_context.hpp"
#import "vit_buffers.hpp"
#import "common/model_loader.hpp"

#import <chrono>

// IEEE 754 single-precision to half-precision conversion
static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;

    if (exp <= 0) {
        // Underflow to zero
        return static_cast<uint16_t>(sign);
    } else if (exp >= 31) {
        // Overflow to infinity
        return static_cast<uint16_t>(sign | 0x7C00);
    }

    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

namespace mast3r {
namespace metal {

// ============================================================================
// Command Buffer Batching
// ============================================================================

/**
 * CommandBatch manages multiple compute passes in a single command buffer.
 * Reduces synchronization overhead by encoding all operations before commit.
 */
class CommandBatch {
public:
    explicit CommandBatch(MetalContext& ctx) : ctx_(ctx) {
        @autoreleasepool {
            buffer_ = [ctx.command_queue() commandBuffer];
            buffer_.label = @"MASt3R Batch";
        }
    }

    ~CommandBatch() {
        if (encoder_) {
            [encoder_ endEncoding];
        }
    }

    // Start a new compute pass (or continue existing)
    id<MTLComputeCommandEncoder> encoder() {
        if (!encoder_) {
            encoder_ = [buffer_ computeCommandEncoder];
        }
        return encoder_;
    }

    // End current compute pass (for memory barriers)
    void endPass() {
        if (encoder_) {
            [encoder_ endEncoding];
            encoder_ = nil;
        }
    }

    // Dispatch a kernel with automatic threadgroup sizing
    void dispatch(
        id<MTLComputePipelineState> pipeline,
        MTLSize gridSize,
        const char* label = nullptr
    ) {
        auto enc = encoder();
        [enc setComputePipelineState:pipeline];

        if (label) {
            [enc pushDebugGroup:[NSString stringWithUTF8String:label]];
        }

        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        if (label) {
            [enc popDebugGroup];
        }
    }

    // Dispatch with explicit threadgroup size
    void dispatchWithThreadgroups(
        id<MTLComputePipelineState> pipeline,
        MTLSize threadgroupsPerGrid,
        MTLSize threadsPerThreadgroup
    ) {
        auto enc = encoder();
        [enc setComputePipelineState:pipeline];
        [enc dispatchThreadgroups:threadgroupsPerGrid
            threadsPerThreadgroup:threadsPerThreadgroup];
    }

    // Memory barrier between passes
    void memoryBarrier() {
        if (encoder_) {
            [encoder_ memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    }

    // Commit and wait (blocking)
    void commitAndWait() {
        endPass();
        [buffer_ commit];
        [buffer_ waitUntilCompleted];
    }

    // Commit without waiting (async)
    void commit() {
        endPass();
        [buffer_ commit];
    }

    // Wait for completion (after async commit)
    void waitUntilCompleted() {
        [buffer_ waitUntilCompleted];
    }

    // Get GPU execution time (if profiling enabled)
    double gpuTimeMs() {
        if (@available(macOS 10.15, iOS 10.3, *)) {
            CFTimeInterval gpuStart = buffer_.GPUStartTime;
            CFTimeInterval gpuEnd = buffer_.GPUEndTime;
            return (gpuEnd - gpuStart) * 1000.0;
        }
        return 0.0;
    }

private:
    MetalContext& ctx_;
    id<MTLCommandBuffer> buffer_ = nil;
    id<MTLComputeCommandEncoder> encoder_ = nil;
};

// ============================================================================
// Metal Engine Implementation
// ============================================================================

MetalEngine::MetalEngine(const RuntimeConfig& config)
    : config_(config), spec_(get_model_spec(config.variant)) {
    if (!MetalContext::instance().is_available()) {
        throw std::runtime_error("Metal is not available on this system");
    }
    allocate_buffers();
}

MetalEngine::~MetalEngine() {
    @autoreleasepool {
        // I/O buffers
        buffer_img1_ = nil;
        buffer_img2_ = nil;
        buffer_preprocessed1_ = nil;
        buffer_preprocessed2_ = nil;
        buffer_pts3d_1_ = nil;
        buffer_pts3d_2_ = nil;
        buffer_desc_1_ = nil;
        buffer_desc_2_ = nil;
        buffer_conf_1_ = nil;
        buffer_conf_2_ = nil;

        // Intermediate buffers
        buffer_patches1_ = nil;
        buffer_patches2_ = nil;
        buffer_encoder1_ = nil;
        buffer_encoder2_ = nil;
        buffer_decoder1_ = nil;
        buffer_decoder2_ = nil;

        // Attention buffers
        buffer_qkv_ = nil;
        buffer_q_ = nil;
        buffer_k_ = nil;
        buffer_v_ = nil;
        buffer_attn_out_ = nil;
        buffer_attn_ = nil;
        buffer_mlp_ = nil;
        buffer_residual_ = nil;

        // Cross-attention buffers
        buffer_cross_q_ = nil;
        buffer_cross_k_ = nil;
        buffer_cross_v_ = nil;

        // RoPE positions
        buffer_rope_positions_ = nil;

        // Patch-level output buffers
        buffer_pts3d_patch1_ = nil;
        buffer_pts3d_patch2_ = nil;
        buffer_desc_patch1_ = nil;
        buffer_desc_patch2_ = nil;
        buffer_conf_patch1_ = nil;
        buffer_conf_patch2_ = nil;

        // DPT intermediate buffers
        for (int i = 0; i < DPT_NUM_HOOKS; i++) {
            buffer_dpt_hooks_[i] = nil;
            buffer_dpt_postprocess_[i] = nil;
            buffer_dpt_layer_rn_[i] = nil;
            buffer_dpt_refinenet_[i] = nil;
        }
        buffer_dpt_scratch1_ = nil;
        buffer_dpt_scratch2_ = nil;
        buffer_dpt_head_temp_ = nil;
        buffer_local_hidden_ = nil;
        buffer_fc1_out_ = nil;
        buffer_local_shuffle_ = nil;
    }

    // ViTBuffers destructor handles GPU weight cleanup
    vit_buffers_.reset();
}

void MetalEngine::allocate_buffers() {
    @autoreleasepool {
        auto& ctx = MetalContext::instance();
        const int res = config_.resolution;
        const int desc_dim = spec_.desc_dim;
        const int num_patches = spec_.num_patches(res);
        const int embed_dim = spec_.embed_dim;
        const int decoder_dim = spec_.decoder_dim;

        // Input image buffers (RGBA for Metal)
        buffer_img1_ = ctx.create_buffer(res * res * 4);
        buffer_img2_ = ctx.create_buffer(res * res * 4);

        // Preprocessed (float32, CHW) - use F32 for intermediate precision
        buffer_preprocessed1_ = ctx.create_buffer(3 * res * res * sizeof(float));
        buffer_preprocessed2_ = ctx.create_buffer(3 * res * res * sizeof(float));

        // Patch embeddings [num_patches, embed_dim] - FP32 for numerical stability
        buffer_patches1_ = ctx.create_buffer(num_patches * embed_dim * sizeof(float));
        buffer_patches2_ = ctx.create_buffer(num_patches * embed_dim * sizeof(float));

        // Encoder outputs [num_patches, embed_dim] - FP32 for numerical stability
        buffer_encoder1_ = ctx.create_buffer(num_patches * embed_dim * sizeof(float));
        buffer_encoder2_ = ctx.create_buffer(num_patches * embed_dim * sizeof(float));

        // Decoder outputs [num_patches, decoder_dim] - FP32 for numerical stability
        buffer_decoder1_ = ctx.create_buffer(num_patches * decoder_dim * sizeof(float));
        buffer_decoder2_ = ctx.create_buffer(num_patches * decoder_dim * sizeof(float));

        // Scratch buffers for attention (sized for largest operation)
        const int num_heads = spec_.num_heads;
        const int head_dim = embed_dim / num_heads;
        const int decoder_heads = spec_.decoder_heads;
        const int decoder_head_dim = decoder_dim / decoder_heads;

        // QKV projection output: [num_patches, 3 * embed_dim] - FP32 for stability
        size_t qkv_size = num_patches * 3 * embed_dim * sizeof(float);
        buffer_qkv_ = ctx.create_buffer(qkv_size);

        // Q, K, V after split: [num_patches, num_heads, head_dim] - FP32 for stability
        size_t qkv_split_size = num_patches * embed_dim * sizeof(float);
        buffer_q_ = ctx.create_buffer(qkv_split_size);
        buffer_k_ = ctx.create_buffer(qkv_split_size);
        buffer_v_ = ctx.create_buffer(qkv_split_size);

        // Attention output: [num_patches, embed_dim] - FP32 for stability
        buffer_attn_out_ = ctx.create_buffer(num_patches * embed_dim * sizeof(float));

        // Residual buffer for skip connections - FP32 for stability
        buffer_residual_ = ctx.create_buffer(num_patches * embed_dim * sizeof(float));

        // Attention scores: [max_heads, num_patches, num_patches] - FP32 for stability
        // Use max of encoder and decoder heads to support both
        const int max_heads = std::max(num_heads, decoder_heads);
        size_t attn_size = max_heads * num_patches * num_patches * sizeof(float);
        buffer_attn_ = ctx.create_buffer(attn_size);

        // MLP intermediate: [num_patches, 4 * embed_dim] - FP32 for stability
        size_t mlp_size = num_patches * 4 * embed_dim * sizeof(float);
        buffer_mlp_ = ctx.create_buffer(mlp_size);

        // Cross-attention buffers for decoder - FP32 for stability
        size_t cross_qkv_size = num_patches * decoder_dim * sizeof(float);
        buffer_cross_q_ = ctx.create_buffer(cross_qkv_size);
        buffer_cross_k_ = ctx.create_buffer(cross_qkv_size);
        buffer_cross_v_ = ctx.create_buffer(cross_qkv_size);

        // RoPE 2D positions: [num_patches, 2] - keep float for sin/cos
        buffer_rope_positions_ = ctx.create_buffer(num_patches * 2 * sizeof(float));

        // DPT outputs at 2x target resolution: (res/4)*2 = res/2
        // For 512 input: DPT outputs at 256x256
        const int dpt_res = res / 2;
        buffer_pts3d_patch1_ = ctx.create_buffer(dpt_res * dpt_res * 3 * sizeof(float));
        buffer_pts3d_patch2_ = ctx.create_buffer(dpt_res * dpt_res * 3 * sizeof(float));

        // Local features MLP output: [num_patches, (desc_dim + 1) * patch_size^2]
        // MASt3R: (24 + 1) * 16^2 = 25 * 256 = 6400
        // The +1 is for desc_conf channel, patch_size^2 for pixel shuffle
        const int patch_size = spec_.patch_size;  // 16 for MASt3R
        const int mlp_out_dim = (desc_dim + 1) * patch_size * patch_size;  // 6400
        buffer_desc_patch1_ = ctx.create_buffer(num_patches * mlp_out_dim * sizeof(float));
        buffer_desc_patch2_ = ctx.create_buffer(num_patches * mlp_out_dim * sizeof(float));

        buffer_conf_patch1_ = ctx.create_buffer(dpt_res * dpt_res * sizeof(float));
        buffer_conf_patch2_ = ctx.create_buffer(dpt_res * dpt_res * sizeof(float));

        // Final outputs (pixel level)
        // pts3d and conf are float (final output precision)
        // desc is half internally, converted to float at the end
        buffer_pts3d_1_ = ctx.create_buffer(res * res * 3 * sizeof(float));
        buffer_pts3d_2_ = ctx.create_buffer(res * res * 3 * sizeof(float));
        buffer_desc_1_ = ctx.create_buffer(res * res * desc_dim * sizeof(float));  // Final float output
        buffer_desc_2_ = ctx.create_buffer(res * res * desc_dim * sizeof(float));  // Final float output
        buffer_conf_1_ = ctx.create_buffer(res * res * sizeof(float));
        buffer_conf_2_ = ctx.create_buffer(res * res * sizeof(float));
        // Half precision intermediate for desc (before final conversion)
        buffer_desc_half_1_ = ctx.create_buffer(res * res * desc_dim * sizeof(uint16_t));
        buffer_desc_half_2_ = ctx.create_buffer(res * res * desc_dim * sizeof(uint16_t));

        // =========================================================================
        // DPT (Dense Prediction Transformer) intermediate buffers
        // =========================================================================
        // Spatial dimensions for DPT
        const int P = spec_.patch_size;
        const int pH = res / P;  // patches height
        const int pW = res / P;  // patches width

        // DPT layer dimensions (after act_postprocess)
        // layer_dims = [96, 192, 384, 768] for standard DPT
        const int layer_dims[4] = {96, 192, 384, 768};
        const int feature_dim = 256;  // DPT feature dimension

        // Hook outputs from decoder [4 buffers, each num_patches * decoder_dim]
        for (int i = 0; i < DPT_NUM_HOOKS; i++) {
            buffer_dpt_hooks_[i] = ctx.create_buffer(num_patches * decoder_dim * sizeof(float));
        }

        // After act_postprocess (scale-adjusted spatial dimensions)
        // Hook 0: 4x upsample -> [pH*4, pW*4, layer_dims[0]]
        // Hook 1: 2x upsample -> [pH*2, pW*2, layer_dims[1]]
        // Hook 2: same scale  -> [pH, pW, layer_dims[2]]
        // Hook 3: 0.5x       -> [pH/2, pW/2, layer_dims[3]]
        // After layer_rn, all get projected to feature_dim=256 and upsampled to [pH*4, pW*4]
        const int dpt_spatial_h = pH * 4;
        const int dpt_spatial_w = pW * 4;

        for (int i = 0; i < DPT_NUM_HOOKS; i++) {
            // act_postprocess output sizes vary by hook
            int h_scale = (i == 0) ? 4 : (i == 1) ? 2 : (i == 2) ? 1 : 1;
            int w_scale = h_scale;
            int spatial_h = pH * h_scale;
            int spatial_w = pW * w_scale;
            if (i == 3) {
                spatial_h = (pH + 1) / 2;  // Downsample by 2 for hook 3
                spatial_w = (pW + 1) / 2;
            }
            buffer_dpt_postprocess_[i] = ctx.create_buffer(
                spatial_h * spatial_w * layer_dims[i] * sizeof(float));

            // After layer_rn: [spatial_h, spatial_w, feature_dim]
            buffer_dpt_layer_rn_[i] = ctx.create_buffer(
                spatial_h * spatial_w * feature_dim * sizeof(float));

            // After refinenet (all at same resolution after upsampling)
            buffer_dpt_refinenet_[i] = ctx.create_buffer(
                dpt_spatial_h * dpt_spatial_w * feature_dim * sizeof(float));
        }

        // DPT scratch buffers for convolutions
        buffer_dpt_scratch1_ = ctx.create_buffer(
            dpt_spatial_h * dpt_spatial_w * feature_dim * sizeof(float));
        buffer_dpt_scratch2_ = ctx.create_buffer(
            dpt_spatial_h * dpt_spatial_w * feature_dim * sizeof(float));

        // DPT head temp buffer [H*2, W*2, 128] after first upsample
        const int last_dim = feature_dim / 2;  // 128
        buffer_dpt_head_temp_ = ctx.create_buffer(
            dpt_spatial_h * 2 * dpt_spatial_w * 2 * last_dim * sizeof(float));

        // Local features MLP buffers (FP32 for mixed precision)
        // in_dim = enc_dim + dec_dim = 1792, hidden_dim = 4 * in_dim = 7168
        const int local_in_dim = embed_dim + decoder_dim;  // 1792
        const int local_hidden_dim = 4 * local_in_dim;     // 7168
        buffer_local_hidden_ = ctx.create_buffer(
            num_patches * local_in_dim * sizeof(float));   // concat output
        buffer_fc1_out_ = ctx.create_buffer(
            num_patches * local_hidden_dim * sizeof(float));  // fc1+GELU output

        // Local features after pixel shuffle
        // MASt3R: out_dim = 6400 = 25 * 256 = (local_feat_dim + 1) * patch_size^2
        // DUNE:   out_dim = 4900 = 25 * 196 = (local_feat_dim + 1) * patch_size^2
        // pixel_shuffle: [pH, pW, out_dim] -> [H, W, local_feat_dim + 1]
        // local_feat_dim = 24 for both, plus 1 for desc_conf = 25 channels output
        buffer_local_shuffle_ = ctx.create_buffer(
            res * res * (spec_.desc_dim + 1) * sizeof(float));  // [H, W, 25]

        NSLog(@"[mast3r] Allocated buffers: %d patches, embed=%d, decoder=%d",
              num_patches, embed_dim, decoder_dim);
        NSLog(@"[mast3r] DPT buffers: spatial=%dx%d, feature_dim=%d, hooks=4",
              dpt_spatial_h, dpt_spatial_w, feature_dim);
    }
}

void MetalEngine::load(const std::string& /* model_path */) {
    @autoreleasepool {
        // Load weights from disk
        if (spec_.is_mast3r()) {
            weights_ = load_mast3r_model(config_.precision);
        } else {
            weights_ = load_dune_model(config_.variant, config_.precision);
        }

        // Create ViT buffers and upload weights to GPU
        auto& ctx = MetalContext::instance();
        vit_buffers_ = std::make_unique<ViTBuffers>(ctx.device(), spec_);
        vit_buffers_->upload(*weights_);

        // Optionally clear CPU weights to save memory
        // weights_.reset();

        is_loaded_ = true;
        NSLog(@"[mast3r] Model loaded: %s", variant_name(config_.variant));
    }
}

std::string MetalEngine::name() const {
    return "Metal (" + MetalContext::instance().device_name() + ")";
}

void MetalEngine::warmup(int num_iterations) {
    std::vector<uint8_t> dummy(config_.resolution * config_.resolution * 3, 128);
    ImageView view{dummy.data(), config_.resolution, config_.resolution, 3};

    for (int i = 0; i < num_iterations; ++i) {
        infer(view, view);
    }
}

InferenceResult MetalEngine::infer(const ImageView& img1, const ImageView& img2) {
    using Clock = std::chrono::high_resolution_clock;

    InferenceResult result;
    result.height = config_.resolution;
    result.width = config_.resolution;
    result.desc_dim = spec_.desc_dim;

    @autoreleasepool {
        auto& ctx = MetalContext::instance();

        // Create batched command buffer for all operations
        CommandBatch batch(ctx);

        auto t0 = Clock::now();

        // === PREPROCESSING PASS ===
        // Inlined preprocessing for both images
        {
            auto pipeline = ctx.get_pipeline("preprocess_fused");
            if (!pipeline) {
                pipeline = ctx.get_pipeline("preprocess_image");
            }

            if (pipeline) {
                const int res = config_.resolution;

                // Copy input images to GPU buffers
                std::memcpy(buffer_img1_.contents, img1.data, img1.size_bytes());
                std::memcpy(buffer_img2_.contents, img2.data, img2.size_bytes());

                // Prepare parameters
                struct PreprocessParams {
                    int src_width;
                    int src_height;
                    int dst_width;
                    int dst_height;
                    float scale_x;
                    float scale_y;
                    int crop_x;
                    int crop_y;
                };

                PreprocessParams params1 = {
                    img1.width, img1.height,
                    res, res,
                    float(img1.width) / float(res),
                    float(img1.height) / float(res),
                    0, 0
                };

                PreprocessParams params2 = {
                    img2.width, img2.height,
                    res, res,
                    float(img2.width) / float(res),
                    float(img2.height) / float(res),
                    0, 0
                };

                auto params_buffer1 = ctx.create_buffer(&params1, sizeof(params1));
                auto params_buffer2 = ctx.create_buffer(&params2, sizeof(params2));

                // Encode both image preprocessing in same command encoder
                auto enc = batch.encoder();
                [enc setComputePipelineState:pipeline];

                // Image 1
                [enc pushDebugGroup:@"Preprocess Image 1"];
                [enc setBuffer:buffer_img1_ offset:0 atIndex:0];
                [enc setBuffer:buffer_preprocessed1_ offset:0 atIndex:1];
                [enc setBuffer:params_buffer1 offset:0 atIndex:2];

                MTLSize gridSize = MTLSizeMake(res, res, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

                [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [enc popDebugGroup];

                // Image 2
                [enc pushDebugGroup:@"Preprocess Image 2"];
                [enc setBuffer:buffer_img2_ offset:0 atIndex:0];
                [enc setBuffer:buffer_preprocessed2_ offset:0 atIndex:1];
                [enc setBuffer:params_buffer2 offset:0 atIndex:2];

                [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [enc popDebugGroup];
            }
        }

        // Memory barrier between preprocessing and inference
        batch.memoryBarrier();

        // Commit preprocessing
        batch.commitAndWait();

        auto t1 = Clock::now();
        result.preprocess_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // === INFERENCE PASS ===
        if (vit_buffers_ && vit_buffers_->is_uploaded()) {
            // Run encoder on both images
            run_encoder(buffer_preprocessed1_, buffer_encoder1_, buffer_qkv_, buffer_mlp_);
            run_encoder(buffer_preprocessed2_, buffer_encoder2_, buffer_qkv_, buffer_mlp_);

            // Debug: check encoder output (FP32 data buffers)
            {
                float* enc_float = (float*)buffer_encoder1_.contents;
                int nan_f = 0; float min_f = INFINITY, max_f = -INFINITY;
                for (int i = 0; i < 1000; i++) {
                    float v = enc_float[i];
                    if (std::isnan(v)) nan_f++;
                    else { min_f = std::min(min_f, v); max_f = std::max(max_f, v); }
                }
                NSLog(@"[mast3r] Encoder output: nan=%d, range=[%.3f, %.3f]", nan_f, min_f, max_f);
            }

            // Check if DPT weights are available for full pipeline
            const auto& head = vit_buffers_->dpt_head();
            const bool use_full_dpt = (head.conv1_weight != nil);

            if (use_full_dpt) {
                // ===== FULL DPT PIPELINE =====
                // Copy hook buffer pointers to local array (ARC workaround)
                id<MTLBuffer> __unsafe_unretained hooks[DPT_NUM_HOOKS];
                for (int i = 0; i < DPT_NUM_HOOKS; i++) {
                    hooks[i] = buffer_dpt_hooks_[i];
                }

                // Run decoder with hook capture for DPT
                // Image 1 uses dec_blocks (use_decoder2 = false)
                run_decoder(buffer_encoder1_, buffer_encoder2_, buffer_decoder1_,
                            buffer_qkv_, buffer_mlp_, hooks, false);

                // Run full DPT pipeline for pts3d + conf
                run_dpt(buffer_encoder1_, hooks, buffer_pts3d_1_, buffer_conf_1_);

                // Run local features MLP for descriptors
                run_local_features_mlp(buffer_encoder1_, buffer_decoder1_, buffer_desc_1_);

                // Repeat for second image
                // Image 2 uses dec_blocks2 (use_decoder2 = true) for MASt3R
                run_decoder(buffer_encoder2_, buffer_encoder1_, buffer_decoder2_,
                            buffer_qkv_, buffer_mlp_, hooks, spec_.is_mast3r());
                run_dpt(buffer_encoder2_, hooks, buffer_pts3d_2_, buffer_conf_2_);
                run_local_features_mlp(buffer_encoder2_, buffer_decoder2_, buffer_desc_2_);

                NSLog(@"[mast3r] Full DPT pipeline complete");
            } else {
                // ===== FALLBACK PIPELINE (simplified heads) =====
                // Run decoder without hook capture
                // Image 1 uses dec_blocks, Image 2 uses dec_blocks2 for MASt3R
                run_decoder(buffer_encoder1_, buffer_encoder2_, buffer_decoder1_,
                            buffer_qkv_, buffer_mlp_, nullptr, false);
                run_decoder(buffer_encoder2_, buffer_encoder1_, buffer_decoder2_,
                            buffer_qkv_, buffer_mlp_, nullptr, spec_.is_mast3r());

                // Apply simplified DPT heads to decoder outputs (patch-level)
                run_dpt_heads(buffer_decoder1_, buffer_pts3d_patch1_, buffer_desc_patch1_, buffer_conf_patch1_);
                run_dpt_heads(buffer_decoder2_, buffer_pts3d_patch2_, buffer_desc_patch2_, buffer_conf_patch2_);

                // Upsample from patch-level to pixel-level
                run_upsample(buffer_pts3d_patch1_, buffer_desc_patch1_, buffer_conf_patch1_,
                             buffer_pts3d_1_, buffer_desc_1_, buffer_conf_1_);
                run_upsample(buffer_pts3d_patch2_, buffer_desc_patch2_, buffer_conf_patch2_,
                             buffer_pts3d_2_, buffer_desc_2_, buffer_conf_2_);

                NSLog(@"[mast3r] Fallback DPT pipeline complete");
            }
        } else {
            // Placeholder outputs when model not loaded
            const int res = config_.resolution;
            const int desc_dim = spec_.desc_dim;
            std::memset(buffer_pts3d_1_.contents, 0, res * res * 3 * sizeof(float));
            std::memset(buffer_pts3d_2_.contents, 0, res * res * 3 * sizeof(float));
            std::memset(buffer_desc_1_.contents, 0, res * res * desc_dim * sizeof(float));
            std::memset(buffer_desc_2_.contents, 0, res * res * desc_dim * sizeof(float));

            float* conf1 = static_cast<float*>(buffer_conf_1_.contents);
            float* conf2 = static_cast<float*>(buffer_conf_2_.contents);
            std::fill(conf1, conf1 + res * res, 1.0f);
            std::fill(conf2, conf2 + res * res, 1.0f);
        }

        auto t2 = Clock::now();
        result.inference_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();

        // Set pointers (Metal shared memory - zero copy!)
        result.pts3d_1 = static_cast<float*>(buffer_pts3d_1_.contents);
        result.pts3d_2 = static_cast<float*>(buffer_pts3d_2_.contents);
        result.desc_1 = static_cast<float*>(buffer_desc_1_.contents);
        result.desc_2 = static_cast<float*>(buffer_desc_2_.contents);
        result.conf_1 = static_cast<float*>(buffer_conf_1_.contents);
        result.conf_2 = static_cast<float*>(buffer_conf_2_.contents);

        result.total_ms = result.preprocess_ms + result.inference_ms;
    }

    return result;
}

void MetalEngine::preprocess_gpu(const ImageView& img, void* output_buffer) {
    // Legacy single-image preprocessing (kept for compatibility)
    @autoreleasepool {
        auto& ctx = MetalContext::instance();
        CommandBatch batch(ctx);

        // Encode preprocessing
        auto pipeline = ctx.get_pipeline("preprocess_image");
        if (!pipeline) return;

        // TODO: Setup and dispatch
        batch.commitAndWait();
    }
}

// ============================================================================
// ViT Encoder Forward Pass
// ============================================================================

void MetalEngine::run_encoder(
    id<MTLBuffer> input,      // Preprocessed image [3, H, W]
    id<MTLBuffer> output,     // Encoder output [num_patches, embed_dim]
    id<MTLBuffer> scratch_qkv,
    id<MTLBuffer> scratch_mlp
) {
    @autoreleasepool {
        auto& ctx = MetalContext::instance();
        const int res = config_.resolution;
        const int P = spec_.patch_size;
        const int num_patches = spec_.num_patches(res);
        const int D = spec_.embed_dim;
        const int num_heads = spec_.num_heads;
        const int head_dim = D / num_heads;

        // === PATCH EMBEDDING ===
        auto patch_pipeline = ctx.get_pipeline("patch_embed");
        if (patch_pipeline && vit_buffers_->patch_embed_weight()) {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:patch_pipeline];
            [enc setBuffer:input offset:0 atIndex:0];
            [enc setBuffer:output offset:0 atIndex:1];  // Output to patches buffer
            [enc setBuffer:vit_buffers_->patch_embed_weight() offset:0 atIndex:2];
            [enc setBuffer:vit_buffers_->patch_embed_bias() offset:0 atIndex:3];

            // dims: H, W, patch_size, embed_dim
            int dims[4] = {res, res, P, D};
            [enc setBytes:dims length:sizeof(dims) atIndex:4];

            // Grid: [num_patches, embed_dim]
            MTLSize grid = MTLSizeMake(num_patches, D, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            // Debug: check after patch_embed (FP32 data buffers)
            float* patch_out = (float*)output.contents;
            int nan_count = 0;
            float min_val = INFINITY, max_val = -INFINITY;
            for (int i = 0; i < 1000; i++) {
                float v = patch_out[i];
                if (std::isnan(v)) nan_count++;
                else { min_val = std::min(min_val, v); max_val = std::max(max_val, v); }
            }
            NSLog(@"[mast3r] After patch_embed: nan=%d, range=[%.3f, %.3f]", nan_count, min_val, max_val);
        }

        // === ADD POSITIONAL EMBEDDING (DUNE only) ===
        if (spec_.is_dune() && vit_buffers_->pos_embed()) {
            auto pos_pipeline = ctx.get_pipeline("add_pos_embed");
            if (pos_pipeline) {
                id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

                [enc setComputePipelineState:pos_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:vit_buffers_->pos_embed() offset:0 atIndex:1];

                int dims[2] = {num_patches, D};
                [enc setBytes:dims length:sizeof(dims) atIndex:2];

                MTLSize grid = MTLSizeMake(num_patches * D, 1, 1);
                MTLSize tg = MTLSizeMake(256, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];

                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }
        }

        // === TRANSFORMER BLOCKS ===
        auto ln_pipeline = ctx.get_pipeline("layer_norm");
        auto qkv_pipeline = ctx.get_pipeline("linear");
        auto attn_pipeline = ctx.get_pipeline("attention_tiled");
        auto proj_pipeline = ctx.get_pipeline("linear");
        auto mlp_fc1_pipeline = ctx.get_pipeline("mlp_fc1");
        auto mlp_fc2_pipeline = ctx.get_pipeline("mlp_fc2");
        auto residual_pipeline = ctx.get_pipeline("residual_add");

        for (int layer = 0; layer < spec_.depth; layer++) {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            cmd.label = [NSString stringWithFormat:@"Encoder Layer %d", layer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            // --- Pre-norm 1 ---
            if (ln_pipeline && vit_buffers_->encoder_norm1_weight(layer)) {
                [enc setComputePipelineState:ln_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:vit_buffers_->encoder_norm1_weight(layer) offset:0 atIndex:1];
                [enc setBuffer:vit_buffers_->encoder_norm1_bias(layer) offset:0 atIndex:2];

                int dims[2] = {num_patches, D};
                float eps = 1e-6f;
                [enc setBytes:dims length:sizeof(dims) atIndex:3];
                [enc setBytes:&eps length:sizeof(eps) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Debug layer 1 and 2 after norm1 - check ALL elements (FP32)
                if (layer == 1 || layer == 2) {
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    float* out = (float*)output.contents;
                    int total = num_patches * D;  // 1024 * 1024 = 1M
                    int nan = 0;
                    float min_v = INFINITY, max_v = -INFINITY;
                    for (int i = 0; i < total; i++) {
                        float v = out[i];
                        if (std::isnan(v) || std::isinf(v)) nan++;
                        else { min_v = std::min(min_v, v); max_v = std::max(max_v, v); }
                    }
                    NSLog(@"[mast3r] L%d after norm1: nan=%d/%d, range=[%.1f, %.1f]", layer, nan, total, min_v, max_v);
                    cmd = [ctx.command_queue() commandBuffer];
                    enc = [cmd computeCommandEncoder];
                }
            }

            // --- Self-Attention ---
            // Save residual for skip connection (GPU copy to avoid CPU sync)
            auto copy_pipeline = ctx.get_pipeline("buffer_copy");
            if (copy_pipeline) {
                [enc setComputePipelineState:copy_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_residual_ offset:0 atIndex:1];
                int copy_size = num_patches * D;
                [enc setBytes:&copy_size length:sizeof(copy_size) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(copy_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // QKV projection: output -> qkv [N, 3*D]
            if (qkv_pipeline && vit_buffers_->encoder_qkv_weight(layer)) {
                [enc setComputePipelineState:qkv_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_qkv_ offset:0 atIndex:1];
                [enc setBuffer:vit_buffers_->encoder_qkv_weight(layer) offset:0 atIndex:2];
                [enc setBuffer:vit_buffers_->encoder_qkv_bias(layer) offset:0 atIndex:3];

                int dims[3] = {num_patches, D, 3 * D};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, 3 * D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Debug layer 1 and 2 after qkv_proj (FP32)
                if (layer == 1 || layer == 2) {
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    float* qkv = (float*)buffer_qkv_.contents;
                    int total = num_patches * 3 * D;  // 1024 * 3072 = 3M
                    int nan = 0;
                    for (int i = 0; i < total; i++) if (std::isnan(qkv[i])) nan++;
                    NSLog(@"[mast3r] L%d after qkv_proj: nan=%d/%d", layer, nan, total);
                    cmd = [ctx.command_queue() commandBuffer];
                    enc = [cmd computeCommandEncoder];
                }
            }

            // Split QKV into Q, K, V: [N, 3*D] -> 3x [N, num_heads, head_dim]
            auto split_pipeline = ctx.get_pipeline("split_qkv");
            if (split_pipeline) {
                [enc setComputePipelineState:split_pipeline];
                [enc setBuffer:buffer_qkv_ offset:0 atIndex:0];
                [enc setBuffer:buffer_q_ offset:0 atIndex:1];
                [enc setBuffer:buffer_k_ offset:0 atIndex:2];
                [enc setBuffer:buffer_v_ offset:0 atIndex:3];

                int dims[3] = {num_patches, num_heads, head_dim};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Debug layer 1 and 2 after split_qkv (before RoPE) - FP32
                if (layer == 1 || layer == 2) {
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    float* q = (float*)buffer_q_.contents;
                    float* k = (float*)buffer_k_.contents;
                    int total = num_patches * D;  // 1024 * 1024 = 1M
                    int nan_q = 0, nan_k = 0;
                    for (int i = 0; i < total; i++) {
                        if (std::isnan(q[i])) nan_q++;
                        if (std::isnan(k[i])) nan_k++;
                    }
                    NSLog(@"[mast3r] L%d after split_qkv (pre-RoPE): Q nan=%d/%d, K nan=%d/%d", layer, nan_q, total, nan_k, total);
                    cmd = [ctx.command_queue() commandBuffer];
                    enc = [cmd computeCommandEncoder];
                }
            }

            // Apply RoPE 2D for MASt3R (layer 0 generates positions)
            if (spec_.is_mast3r() && layer == 0) {
                auto rope_pos_pipeline = ctx.get_pipeline("generate_rope_positions");
                if (rope_pos_pipeline) {
                    [enc setComputePipelineState:rope_pos_pipeline];
                    [enc setBuffer:buffer_rope_positions_ offset:0 atIndex:0];

                    int grid_size[2] = {res / P, res / P};
                    [enc setBytes:grid_size length:sizeof(grid_size) atIndex:1];

                    [enc dispatchThreads:MTLSizeMake(num_patches, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }

            if (spec_.is_mast3r()) {
                auto rope_pipeline = ctx.get_pipeline("apply_rope_2d");
                if (rope_pipeline) {
                    // Apply RoPE to Q
                    [enc setComputePipelineState:rope_pipeline];
                    [enc setBuffer:buffer_q_ offset:0 atIndex:0];
                    [enc setBuffer:buffer_rope_positions_ offset:0 atIndex:1];

                    int dims[3] = {num_patches, num_heads, head_dim};
                    float freq_base = 100.0f;
                    [enc setBytes:dims length:sizeof(dims) atIndex:2];
                    [enc setBytes:&freq_base length:sizeof(freq_base) atIndex:3];

                    [enc dispatchThreads:MTLSizeMake(num_patches, num_heads, head_dim / 4)
                        threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];

                    // Apply RoPE to K
                    [enc setBuffer:buffer_k_ offset:0 atIndex:0];
                    [enc dispatchThreads:MTLSizeMake(num_patches, num_heads, head_dim / 4)
                        threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Debug layer 1 and 2 after RoPE - check ALL elements (FP32)
                    if (layer == 1 || layer == 2) {
                        [enc endEncoding];
                        [cmd commit];
                        [cmd waitUntilCompleted];
                        float* q = (float*)buffer_q_.contents;
                        float* k = (float*)buffer_k_.contents;
                        int total = num_patches * D;  // 1024 * 1024 = 1M
                        int nan_q = 0, nan_k = 0;
                        for (int i = 0; i < total; i++) {
                            if (std::isnan(q[i])) nan_q++;
                            if (std::isnan(k[i])) nan_k++;
                        }
                        NSLog(@"[mast3r] L%d RoPE: Q nan=%d/%d, K nan=%d/%d", layer, nan_q, total, nan_k, total);
                        cmd = [ctx.command_queue() commandBuffer];
                        enc = [cmd computeCommandEncoder];
                    }
                }
            }

            // Multi-head Self-Attention using 3-pass parallel approach
            // Pass 1: Compute attention scores Q @ K^T
            auto scores_pipeline = ctx.get_pipeline("attention_scores");
            if (scores_pipeline) {
                [enc setComputePipelineState:scores_pipeline];
                [enc setBuffer:buffer_q_ offset:0 atIndex:0];
                [enc setBuffer:buffer_k_ offset:0 atIndex:1];
                [enc setBuffer:buffer_attn_ offset:0 atIndex:2];

                int dims3[3] = {num_patches, num_heads, head_dim};
                [enc setBytes:dims3 length:sizeof(dims3) atIndex:3];

                [enc dispatchThreads:MTLSizeMake(num_patches, num_patches, num_heads)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Debug layer 1 and 2 after attention_scores (FP32)
                if (layer == 1 || layer == 2) {
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    float* attn = (float*)buffer_attn_.contents;
                    int total = num_heads * num_patches * num_patches;  // 16 * 1024 * 1024 = 16M
                    int nan = 0;
                    float min_v = INFINITY, max_v = -INFINITY;
                    for (int i = 0; i < total; i++) {
                        float v = attn[i];
                        if (std::isnan(v) || std::isinf(v)) nan++;
                        else { min_v = std::min(min_v, v); max_v = std::max(max_v, v); }
                    }
                    NSLog(@"[mast3r] L%d attn_scores: nan=%d/%d, range=[%.1f, %.1f]", layer, nan, total, min_v, max_v);
                    cmd = [ctx.command_queue() commandBuffer];
                    enc = [cmd computeCommandEncoder];
                }
            }

            // Pass 2: Softmax over each row
            auto softmax_pipeline = ctx.get_pipeline("attention_softmax");
            if (softmax_pipeline) {
                [enc setComputePipelineState:softmax_pipeline];
                [enc setBuffer:buffer_attn_ offset:0 atIndex:0];

                int dims2[2] = {num_patches, num_heads};
                [enc setBytes:dims2 length:sizeof(dims2) atIndex:1];

                // Grid: [N, num_heads] - one thread per row
                [enc dispatchThreads:MTLSizeMake(num_patches, num_heads, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Debug layer 1 after softmax (FP32)
                if (layer == 1) {
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    float* attn = (float*)buffer_attn_.contents;
                    int total = num_heads * num_patches * num_patches;
                    int nan = 0;
                    for (int i = 0; i < total; i++) if (std::isnan(attn[i]) || std::isinf(attn[i])) nan++;
                    NSLog(@"[mast3r] L1 after softmax: nan=%d/%d", nan, total);
                    cmd = [ctx.command_queue() commandBuffer];
                    enc = [cmd computeCommandEncoder];
                }
            }

            // Pass 3: Compute output = scores @ V
            auto output_pipeline = ctx.get_pipeline("attention_output");
            if (output_pipeline) {
                [enc setComputePipelineState:output_pipeline];
                [enc setBuffer:buffer_attn_ offset:0 atIndex:0];
                [enc setBuffer:buffer_v_ offset:0 atIndex:1];
                [enc setBuffer:buffer_attn_out_ offset:0 atIndex:2];

                int dims3[3] = {num_patches, num_heads, head_dim};
                [enc setBytes:dims3 length:sizeof(dims3) atIndex:3];

                // Grid: [N, head_dim, num_heads] - one thread per output element
                [enc dispatchThreads:MTLSizeMake(num_patches, head_dim, num_heads)
                    threadsPerThreadgroup:MTLSizeMake(16, 8, 4)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Debug layer 1 after attention_output (FP32)
                if (layer == 1) {
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    float* attn_out = (float*)buffer_attn_out_.contents;
                    int total = num_patches * D;
                    int nan = 0;
                    for (int i = 0; i < total; i++) if (std::isnan(attn_out[i]) || std::isinf(attn_out[i])) nan++;
                    NSLog(@"[mast3r] L1 after attn_output: nan=%d/%d", nan, total);
                    cmd = [ctx.command_queue() commandBuffer];
                    enc = [cmd computeCommandEncoder];
                }
            }

            // Output projection: attn_out -> output
            auto proj_pipeline = ctx.get_pipeline("attention_output_proj");
            if (proj_pipeline && vit_buffers_->encoder_proj_weight(layer)) {
                [enc setComputePipelineState:proj_pipeline];
                [enc setBuffer:buffer_attn_out_ offset:0 atIndex:0];
                [enc setBuffer:output offset:0 atIndex:1];
                [enc setBuffer:vit_buffers_->encoder_proj_weight(layer) offset:0 atIndex:2];
                [enc setBuffer:vit_buffers_->encoder_proj_bias(layer) offset:0 atIndex:3];

                int dims[2] = {num_patches, D};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Residual connection for attention
            auto residual_pipeline = ctx.get_pipeline("residual_add");
            if (residual_pipeline) {
                [enc setComputePipelineState:residual_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_residual_ offset:0 atIndex:1];

                int size = num_patches * D;
                [enc setBytes:&size length:sizeof(size) atIndex:2];

                [enc dispatchThreads:MTLSizeMake(num_patches * D, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Debug layer 1 after attn residual (FP32)
                if (layer == 1) {
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    float* out = (float*)output.contents;
                    int total = num_patches * D;
                    int nan = 0;
                    float min_v = INFINITY, max_v = -INFINITY;
                    for (int i = 0; i < total; i++) {
                        float v = out[i];
                        if (std::isnan(v) || std::isinf(v)) nan++;
                        else { min_v = std::min(min_v, v); max_v = std::max(max_v, v); }
                    }
                    NSLog(@"[mast3r] L1 after attn+residual: nan=%d/%d, range=[%.1f, %.1f]", nan, total, min_v, max_v);
                    cmd = [ctx.command_queue() commandBuffer];
                    enc = [cmd computeCommandEncoder];
                }
            }

            // --- Pre-norm 2 + MLP ---
            // Save residual for MLP skip connection (GPU copy)
            if (copy_pipeline) {
                [enc setComputePipelineState:copy_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_residual_ offset:0 atIndex:1];
                int copy_size = num_patches * D;
                [enc setBytes:&copy_size length:sizeof(copy_size) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(copy_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            if (ln_pipeline && vit_buffers_->encoder_norm2_weight(layer)) {
                [enc setComputePipelineState:ln_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:vit_buffers_->encoder_norm2_weight(layer) offset:0 atIndex:1];
                [enc setBuffer:vit_buffers_->encoder_norm2_bias(layer) offset:0 atIndex:2];

                int dims[2] = {num_patches, D};
                float eps = 1e-6f;
                [enc setBytes:dims length:sizeof(dims) atIndex:3];
                [enc setBytes:&eps length:sizeof(eps) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // MLP: FC1 with GELU
            if (mlp_fc1_pipeline && vit_buffers_->encoder_mlp_fc1_weight(layer)) {
                [enc setComputePipelineState:mlp_fc1_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_mlp_ offset:0 atIndex:1];
                [enc setBuffer:vit_buffers_->encoder_mlp_fc1_weight(layer) offset:0 atIndex:2];
                [enc setBuffer:vit_buffers_->encoder_mlp_fc1_bias(layer) offset:0 atIndex:3];

                int dims[2] = {num_patches, D};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, 4 * D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Debug layer 1 after mlp_fc1 (FP32)
                if (layer == 1) {
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    float* mlp = (float*)buffer_mlp_.contents;
                    int total = num_patches * 4 * D;
                    int nan = 0;
                    for (int i = 0; i < total; i++) if (std::isnan(mlp[i]) || std::isinf(mlp[i])) nan++;
                    NSLog(@"[mast3r] L1 after mlp_fc1: nan=%d/%d", nan, total);
                    cmd = [ctx.command_queue() commandBuffer];
                    enc = [cmd computeCommandEncoder];
                }
            }

            // MLP: FC2
            if (mlp_fc2_pipeline && vit_buffers_->encoder_mlp_fc2_weight(layer)) {
                [enc setComputePipelineState:mlp_fc2_pipeline];
                [enc setBuffer:buffer_mlp_ offset:0 atIndex:0];
                [enc setBuffer:output offset:0 atIndex:1];
                [enc setBuffer:vit_buffers_->encoder_mlp_fc2_weight(layer) offset:0 atIndex:2];
                [enc setBuffer:vit_buffers_->encoder_mlp_fc2_bias(layer) offset:0 atIndex:3];

                int dims[2] = {num_patches, D};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Debug layer 1 after mlp_fc2 (FP32)
                if (layer == 1) {
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    float* out = (float*)output.contents;
                    int total = num_patches * D;
                    int nan = 0;
                    float min_v = INFINITY, max_v = -INFINITY;
                    for (int i = 0; i < total; i++) {
                        float v = out[i];
                        if (std::isnan(v) || std::isinf(v)) nan++;
                        else { min_v = std::min(min_v, v); max_v = std::max(max_v, v); }
                    }
                    NSLog(@"[mast3r] L1 after mlp_fc2 (pre-residual): nan=%d/%d, range=[%.1f, %.1f]", nan, total, min_v, max_v);
                    cmd = [ctx.command_queue() commandBuffer];
                    enc = [cmd computeCommandEncoder];
                }
            }

            // Residual connection for MLP
            if (residual_pipeline) {
                [enc setComputePipelineState:residual_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_residual_ offset:0 atIndex:1];

                int size = num_patches * D;
                [enc setBytes:&size length:sizeof(size) atIndex:2];

                [enc dispatchThreads:MTLSizeMake(num_patches * D, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            // Debug layer 0 internal stages (FP32)
            if (layer == 0) {
                // Check attention output
                float* attn = (float*)buffer_attn_out_.contents;
                float attn_min = INFINITY, attn_max = -INFINITY;
                int attn_nan = 0;
                for (int i = 0; i < num_patches * D; i++) {
                    float v = attn[i];
                    if (std::isnan(v) || std::isinf(v)) attn_nan++;
                    else { attn_min = std::min(attn_min, v); attn_max = std::max(attn_max, v); }
                }
                NSLog(@"[mast3r] L0 attn_out: nan=%d, range=[%.1f, %.1f]", attn_nan, attn_min, attn_max);

                // Check MLP buffer (fc1 output after GELU)
                float* mlp = (float*)buffer_mlp_.contents;
                float mlp_min = INFINITY, mlp_max = -INFINITY;
                int mlp_nan = 0;
                for (int i = 0; i < num_patches * 4 * D; i++) {
                    float v = mlp[i];
                    if (std::isnan(v) || std::isinf(v)) mlp_nan++;
                    else { mlp_min = std::min(mlp_min, v); mlp_max = std::max(mlp_max, v); }
                }
                NSLog(@"[mast3r] L0 mlp_fc1: nan=%d, range=[%.1f, %.1f]", mlp_nan, mlp_min, mlp_max);

                // Check residual buffer (saved before norm2)
                float* res = (float*)buffer_residual_.contents;
                float res_min = INFINITY, res_max = -INFINITY;
                int res_nan = 0;
                for (int i = 0; i < num_patches * D; i++) {
                    float v = res[i];
                    if (std::isnan(v) || std::isinf(v)) res_nan++;
                    else { res_min = std::min(res_min, v); res_max = std::max(res_max, v); }
                }
                NSLog(@"[mast3r] L0 residual (before mlp add): nan=%d, range=[%.1f, %.1f]", res_nan, res_min, res_max);
            }

            // Debug: check ALL values for first few layers (FP32)
            if (layer < 4) {
                float* layer_out = (float*)output.contents;
                int total = num_patches * D;  // 1024 * 1024 = 1M
                int nan_count = 0;
                float min_val = INFINITY, max_val = -INFINITY;
                for (int i = 0; i < total; i++) {
                    float v = layer_out[i];
                    if (std::isnan(v) || std::isinf(v)) nan_count++;
                    else { min_val = std::min(min_val, v); max_val = std::max(max_val, v); }
                }
                NSLog(@"[mast3r] Layer %d: nan/inf=%d/%d, range=[%.1f, %.1f]", layer, nan_count, total, min_val, max_val);
            }
        }

        NSLog(@"[mast3r] Encoder forward pass complete (%d layers)", spec_.depth);
    }
}

// ============================================================================
// DPT (Dense Prediction Transformer) - Full Architecture
// ============================================================================
//
// DPT Architecture:
// 1. Hooks at decoder layers [0, 6, 9, 12] capture intermediate features
// 2. act_postprocess: Channel projection + scale adjustment per hook
//    - [0]: Conv1x1 (embed->96) + ConvTranspose4x4 (4x upsample)
//    - [1]: Conv1x1 (768->192) + ConvTranspose2x2 (2x upsample)
//    - [2]: Conv1x1 (768->384) only (same scale)
//    - [3]: Conv1x1 (768->768) + Conv3x3 stride2 (0.5x downsample)
// 3. scratch.layer_rn: Conv3x3 projection to feature_dim (256)
// 4. refinenets: FeatureFusionBlocks with ResidualConvUnits
// 5. DPT head: Conv3x3(256->128) -> Upsample2x -> Conv3x3(128->128) -> ReLU -> Conv1x1(128->4)
// 6. Local features MLP: concat(enc,dec) -> fc1 -> GELU -> fc2 -> pixel_shuffle

void MetalEngine::run_dpt(
    id<MTLBuffer> encoder_output,                    // [num_patches, embed_dim]
    id<MTLBuffer> __unsafe_unretained * hook_outputs,  // [4] from decoder
    id<MTLBuffer> pts3d_out,                         // [H, W, 3]
    id<MTLBuffer> conf_out                           // [H, W]
) {
    @autoreleasepool {
        auto& ctx = MetalContext::instance();
        const int res = config_.resolution;
        const int P = spec_.patch_size;
        const int pH = res / P;  // patches height
        const int pW = res / P;  // patches width
        const int num_patches = pH * pW;
        const int feature_dim = 256;  // DPT feature dimension

        // Layer dimensions after act_postprocess
        const int layer_dims[4] = {96, 192, 384, 768};

        // Get pipelines
        auto reshape_pipeline = ctx.get_pipeline("reshape_patches_to_grid");
        auto conv1x1_pipeline = ctx.get_pipeline("conv2d_1x1");
        auto conv3x3_pipeline = ctx.get_pipeline("conv2d_3x3");
        auto conv3x3_relu_pipeline = ctx.get_pipeline("conv2d_3x3_relu");
        auto convT2x2_pipeline = ctx.get_pipeline("conv_transpose_2x2_stride2");
        auto convT4x4_pipeline = ctx.get_pipeline("conv_transpose_4x4_stride4");
        auto rcu_pass1_pipeline = ctx.get_pipeline("residual_conv_unit_pass1");
        auto rcu_pass2_pipeline = ctx.get_pipeline("residual_conv_unit_pass2");
        auto add_pipeline = ctx.get_pipeline("add_tensors");
        auto upsample_pipeline = ctx.get_pipeline("bilinear_upsample");

        if (!vit_buffers_ || !vit_buffers_->is_uploaded()) {
            NSLog(@"[mast3r] DPT: Weights not loaded, skipping");
            return;
        }

        // =====================================================================
        // Stage 1: act_postprocess - project and scale each hook
        // =====================================================================
        for (int i = 0; i < DPT_NUM_HOOKS; i++) {
            if (!hook_outputs[i]) continue;

            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            cmd.label = [NSString stringWithFormat:@"DPT act_postprocess[%d]", i];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            const auto& act = vit_buffers_->dpt_act_postprocess(i);

            // First reshape from [N, C] to [pH, pW, C]
            if (reshape_pipeline) {
                [enc setComputePipelineState:reshape_pipeline];
                [enc setBuffer:hook_outputs[i] offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:1];

                int hook_dim = (i == 0) ? spec_.embed_dim : spec_.decoder_dim;
                int dims[3] = {pH, pW, hook_dim};
                [enc setBytes:dims length:sizeof(dims) atIndex:2];

                [enc dispatchThreads:MTLSizeMake(num_patches, hook_dim, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Conv1x1: project channels
            if (conv1x1_pipeline && act.conv1_weight) {
                [enc setComputePipelineState:conv1x1_pipeline];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:1];
                [enc setBuffer:act.conv1_weight offset:0 atIndex:2];
                [enc setBuffer:act.conv1_bias offset:0 atIndex:3];

                int hook_dim = (i == 0) ? spec_.embed_dim : spec_.decoder_dim;
                int dims[4] = {pH, pW, hook_dim, layer_dims[i]};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(pH, pW, layer_dims[i])
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Scale adjustment (conv2 if present)
            if (act.conv2_weight) {
                if (i == 0 && convT4x4_pipeline) {
                    // 4x upsample via ConvTranspose4x4
                    [enc setComputePipelineState:convT4x4_pipeline];
                    [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:0];
                    [enc setBuffer:buffer_dpt_postprocess_[i] offset:0 atIndex:1];
                    [enc setBuffer:act.conv2_weight offset:0 atIndex:2];
                    [enc setBuffer:act.conv2_bias offset:0 atIndex:3];

                    int dims[3] = {pH, pW, layer_dims[i]};
                    [enc setBytes:dims length:sizeof(dims) atIndex:4];

                    [enc dispatchThreads:MTLSizeMake(pH * 4, pW * 4, layer_dims[i])
                        threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                } else if (i == 1 && convT2x2_pipeline) {
                    // 2x upsample via ConvTranspose2x2
                    [enc setComputePipelineState:convT2x2_pipeline];
                    [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:0];
                    [enc setBuffer:buffer_dpt_postprocess_[i] offset:0 atIndex:1];
                    [enc setBuffer:act.conv2_weight offset:0 atIndex:2];
                    [enc setBuffer:act.conv2_bias offset:0 atIndex:3];

                    int dims[3] = {pH, pW, layer_dims[i]};
                    [enc setBytes:dims length:sizeof(dims) atIndex:4];

                    [enc dispatchThreads:MTLSizeMake(pH * 2, pW * 2, layer_dims[i])
                        threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                } else if (i == 3 && conv3x3_pipeline) {
                    // 0.5x downsample via Conv3x3 stride2
                    // For simplicity, we use regular conv and then subsample
                    // TODO: Implement strided conv3x3
                    [enc setComputePipelineState:conv3x3_pipeline];
                    [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:0];
                    [enc setBuffer:buffer_dpt_postprocess_[i] offset:0 atIndex:1];
                    [enc setBuffer:act.conv2_weight offset:0 atIndex:2];
                    [enc setBuffer:act.conv2_bias offset:0 atIndex:3];

                    int oH = (pH + 1) / 2;
                    int oW = (pW + 1) / 2;
                    int dims[4] = {oH, oW, layer_dims[i], layer_dims[i]};
                    [enc setBytes:dims length:sizeof(dims) atIndex:4];

                    [enc dispatchThreads:MTLSizeMake(oH, oW, layer_dims[i])
                        threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                }
            } else {
                // i == 2: same scale, just copy from scratch2 to postprocess
                auto copy_pipeline = ctx.get_pipeline("buffer_copy");
                if (copy_pipeline) {
                    [enc setComputePipelineState:copy_pipeline];
                    [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:0];
                    [enc setBuffer:buffer_dpt_postprocess_[i] offset:0 atIndex:1];
                    int size = pH * pW * layer_dims[i];
                    [enc setBytes:&size length:sizeof(size) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(size, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // =====================================================================
        // Stage 2: layer_rn - Conv3x3 projection to feature_dim (256)
        // =====================================================================
        for (int i = 0; i < DPT_NUM_HOOKS; i++) {
            id<MTLBuffer> layer_rn_w = vit_buffers_->dpt_layer_rn_weight(i);
            if (!layer_rn_w) continue;

            // Determine spatial size for this hook
            int sH, sW;
            if (i == 0) { sH = pH * 4; sW = pW * 4; }
            else if (i == 1) { sH = pH * 2; sW = pW * 2; }
            else if (i == 2) { sH = pH; sW = pW; }
            else { sH = (pH + 1) / 2; sW = (pW + 1) / 2; }

            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            cmd.label = [NSString stringWithFormat:@"DPT layer_rn[%d]", i];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            if (conv3x3_pipeline) {
                [enc setComputePipelineState:conv3x3_pipeline];
                [enc setBuffer:buffer_dpt_postprocess_[i] offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_layer_rn_[i] offset:0 atIndex:1];
                [enc setBuffer:layer_rn_w offset:0 atIndex:2];

                // layer_rn has no bias, create zero bias
                static id<MTLBuffer> zero_bias = nil;
                if (!zero_bias) {
                    std::vector<float> zeros(feature_dim, 0.0f);
                    zero_bias = [ctx.device() newBufferWithBytes:zeros.data()
                                                          length:zeros.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                }
                [enc setBuffer:zero_bias offset:0 atIndex:3];

                int dims[4] = {sH, sW, layer_dims[i], feature_dim};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(sH, sW, feature_dim)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
            }

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // =====================================================================
        // Stage 3: RefineNets - bottom-up feature fusion
        // =====================================================================
        // Process from bottom (hook 3) to top (hook 0)
        // Each refinenet takes current features + upsampled features from below
        const int target_H = pH * 4;  // Final spatial resolution
        const int target_W = pW * 4;

        for (int i = 3; i >= 0; i--) {
            const auto& rn = vit_buffers_->dpt_refinenet(i);
            if (!rn.rcu1_conv1_weight) continue;

            // Determine spatial size for this level
            int sH, sW;
            if (i == 0) { sH = pH * 4; sW = pW * 4; }
            else if (i == 1) { sH = pH * 2; sW = pW * 2; }
            else if (i == 2) { sH = pH; sW = pW; }
            else { sH = (pH + 1) / 2; sW = (pW + 1) / 2; }

            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            cmd.label = [NSString stringWithFormat:@"DPT refinenet[%d]", i];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            // If not the bottom level, upsample and add previous refinenet output
            if (i < 3 && upsample_pipeline && add_pipeline) {
                // Upsample previous level to current resolution
                [enc setComputePipelineState:upsample_pipeline];
                [enc setBuffer:buffer_dpt_refinenet_[i + 1] offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:1];

                // Previous level dimensions
                int prev_sH, prev_sW;
                if (i + 1 == 1) { prev_sH = pH * 2; prev_sW = pW * 2; }
                else if (i + 1 == 2) { prev_sH = pH; prev_sW = pW; }
                else { prev_sH = (pH + 1) / 2; prev_sW = (pW + 1) / 2; }

                int in_dims[4] = {prev_sH, prev_sW, feature_dim, 0};
                int out_dims[2] = {sH, sW};
                [enc setBytes:in_dims length:sizeof(in_dims) atIndex:2];
                [enc setBytes:out_dims length:sizeof(out_dims) atIndex:3];

                [enc dispatchThreads:MTLSizeMake(sH, sW, feature_dim)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Add to layer_rn output
                [enc setComputePipelineState:add_pipeline];
                [enc setBuffer:buffer_dpt_layer_rn_[i] offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:1];
                [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:2];

                int size = sH * sW * feature_dim;
                [enc setBytes:&size length:sizeof(size) atIndex:3];

                [enc dispatchThreads:MTLSizeMake(size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                // Bottom level: just copy layer_rn to scratch2
                auto copy_pipeline = ctx.get_pipeline("buffer_copy");
                if (copy_pipeline) {
                    [enc setComputePipelineState:copy_pipeline];
                    [enc setBuffer:buffer_dpt_layer_rn_[i] offset:0 atIndex:0];
                    [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:1];
                    int size = sH * sW * feature_dim;
                    [enc setBytes:&size length:sizeof(size) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(size, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }

            // ResidualConvUnit 1: Pass 1
            if (rcu_pass1_pipeline) {
                [enc setComputePipelineState:rcu_pass1_pipeline];
                [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:1];
                [enc setBuffer:rn.rcu1_conv1_weight offset:0 atIndex:2];
                [enc setBuffer:rn.rcu1_conv1_bias offset:0 atIndex:3];

                int dims[3] = {sH, sW, feature_dim};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(sH, sW, feature_dim)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // ResidualConvUnit 1: Pass 2
            if (rcu_pass2_pipeline) {
                [enc setComputePipelineState:rcu_pass2_pipeline];
                [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:1];
                [enc setBuffer:buffer_dpt_refinenet_[i] offset:0 atIndex:2];
                [enc setBuffer:rn.rcu1_conv2_weight offset:0 atIndex:3];
                [enc setBuffer:rn.rcu1_conv2_bias offset:0 atIndex:4];

                int dims[3] = {sH, sW, feature_dim};
                [enc setBytes:dims length:sizeof(dims) atIndex:5];

                [enc dispatchThreads:MTLSizeMake(sH, sW, feature_dim)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // ResidualConvUnit 2: Pass 1
            if (rcu_pass1_pipeline) {
                [enc setComputePipelineState:rcu_pass1_pipeline];
                [enc setBuffer:buffer_dpt_refinenet_[i] offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:1];
                [enc setBuffer:rn.rcu2_conv1_weight offset:0 atIndex:2];
                [enc setBuffer:rn.rcu2_conv1_bias offset:0 atIndex:3];

                int dims[3] = {sH, sW, feature_dim};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(sH, sW, feature_dim)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // ResidualConvUnit 2: Pass 2 (output to refinenet buffer for next iteration)
            if (rcu_pass2_pipeline) {
                [enc setComputePipelineState:rcu_pass2_pipeline];
                [enc setBuffer:buffer_dpt_refinenet_[i] offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:1];
                [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:2];
                [enc setBuffer:rn.rcu2_conv2_weight offset:0 atIndex:3];
                [enc setBuffer:rn.rcu2_conv2_bias offset:0 atIndex:4];

                int dims[3] = {sH, sW, feature_dim};
                [enc setBytes:dims length:sizeof(dims) atIndex:5];

                [enc dispatchThreads:MTLSizeMake(sH, sW, feature_dim)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Output convolution
            if (conv1x1_pipeline && rn.out_conv_weight) {
                [enc setComputePipelineState:conv1x1_pipeline];
                [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_refinenet_[i] offset:0 atIndex:1];
                [enc setBuffer:rn.out_conv_weight offset:0 atIndex:2];
                [enc setBuffer:rn.out_conv_bias offset:0 atIndex:3];

                int dims[4] = {sH, sW, feature_dim, feature_dim};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(sH, sW, feature_dim)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
            }

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // =====================================================================
        // Stage 4: DPT Head - Final regression
        // Conv3x3(256->128) -> Upsample2x -> Conv3x3(128->128) -> ReLU -> Conv1x1(128->4)
        // =====================================================================
        const auto& head = vit_buffers_->dpt_head();

        if (head.conv1_weight) {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            cmd.label = @"DPT Head";
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            // Conv3x3: 256 -> 128
            if (conv3x3_pipeline) {
                [enc setComputePipelineState:conv3x3_pipeline];
                [enc setBuffer:buffer_dpt_refinenet_[0] offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:1];
                [enc setBuffer:head.conv1_weight offset:0 atIndex:2];
                [enc setBuffer:head.conv1_bias offset:0 atIndex:3];

                int dims[4] = {target_H, target_W, feature_dim, 128};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(target_H, target_W, 128)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Upsample 2x using bilinear
            if (upsample_pipeline) {
                [enc setComputePipelineState:upsample_pipeline];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_head_temp_ offset:0 atIndex:1];

                int in_dims[4] = {target_H, target_W, 128, 0};
                int out_dims[2] = {target_H * 2, target_W * 2};
                [enc setBytes:in_dims length:sizeof(in_dims) atIndex:2];
                [enc setBytes:out_dims length:sizeof(out_dims) atIndex:3];

                [enc dispatchThreads:MTLSizeMake(target_H * 2, target_W * 2, 128)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Conv3x3 with ReLU: 128 -> 128
            if (conv3x3_relu_pipeline) {
                [enc setComputePipelineState:conv3x3_relu_pipeline];
                [enc setBuffer:buffer_dpt_head_temp_ offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:1];
                [enc setBuffer:head.conv2_weight offset:0 atIndex:2];
                [enc setBuffer:head.conv2_bias offset:0 atIndex:3];

                int dims[4] = {target_H * 2, target_W * 2, 128, 128};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(target_H * 2, target_W * 2, 128)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Conv1x1: 128 -> 4 (pts3d[3] + conf[1])
            if (conv1x1_pipeline) {
                [enc setComputePipelineState:conv1x1_pipeline];
                [enc setBuffer:buffer_dpt_scratch1_ offset:0 atIndex:0];
                [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:1];
                [enc setBuffer:head.conv3_weight offset:0 atIndex:2];
                [enc setBuffer:head.conv3_bias offset:0 atIndex:3];

                int dims[4] = {target_H * 2, target_W * 2, 128, 4};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(target_H * 2, target_W * 2, 4)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Split output [H*2, W*2, 4] into intermediate buffers
            // Apply tanh * scale to pts3d and sigmoid to conf
            auto split_pipeline = ctx.get_pipeline("dpt_split_pts3d_conf");
            int dpt_H = target_H * 2;  // 256 for 512 input
            int dpt_W = target_W * 2;

            if (split_pipeline) {
                [enc setComputePipelineState:split_pipeline];
                [enc setBuffer:buffer_dpt_scratch2_ offset:0 atIndex:0];  // Input [H, W, 4]
                [enc setBuffer:buffer_pts3d_patch1_ offset:0 atIndex:1];  // Temp pts3d [H, W, 3]
                [enc setBuffer:buffer_conf_patch1_ offset:0 atIndex:2];   // Temp conf [H, W]

                int out_dims[2] = {dpt_H, dpt_W};
                float pts3d_scale = 10.0f;
                [enc setBytes:out_dims length:sizeof(out_dims) atIndex:3];
                [enc setBytes:&pts3d_scale length:sizeof(pts3d_scale) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(dpt_H, dpt_W, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                NSLog(@"[mast3r] Shader function not found: dpt_split_pts3d_conf");
            }

            // Bilinear upsample pts3d from [dpt_H, dpt_W, 3] to [res, res, 3]
            auto upsample_pipeline = ctx.get_pipeline("bilinear_upsample");
            if (upsample_pipeline) {
                // Upsample pts3d
                [enc setComputePipelineState:upsample_pipeline];
                [enc setBuffer:buffer_pts3d_patch1_ offset:0 atIndex:0];
                [enc setBuffer:pts3d_out offset:0 atIndex:1];

                int in_dims[4] = {dpt_H, dpt_W, 3, 0};
                int out_dims[2] = {res, res};
                [enc setBytes:in_dims length:sizeof(in_dims) atIndex:2];
                [enc setBytes:out_dims length:sizeof(out_dims) atIndex:3];

                [enc dispatchThreads:MTLSizeMake(res, res, 3)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Upsample conf (single channel)
                [enc setBuffer:buffer_conf_patch1_ offset:0 atIndex:0];
                [enc setBuffer:conf_out offset:0 atIndex:1];

                int conf_in_dims[4] = {dpt_H, dpt_W, 1, 0};
                [enc setBytes:conf_in_dims length:sizeof(conf_in_dims) atIndex:2];
                [enc setBytes:out_dims length:sizeof(out_dims) atIndex:3];

                [enc dispatchThreads:MTLSizeMake(res, res, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            }

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        NSLog(@"[mast3r] DPT pipeline complete: %dx%d spatial resolution", target_H * 2, target_W * 2);
    }
}

void MetalEngine::run_local_features_mlp(
    id<MTLBuffer> encoder_output,      // [num_patches, embed_dim]
    id<MTLBuffer> decoder_output,      // [num_patches, decoder_dim]
    id<MTLBuffer> desc_out             // [H, W, desc_dim]
) {
    @autoreleasepool {
        auto& ctx = MetalContext::instance();
        const int res = config_.resolution;
        const int P = spec_.patch_size;
        const int num_patches = spec_.num_patches(res);
        const int embed_dim = spec_.embed_dim;
        const int decoder_dim = spec_.decoder_dim;

        if (!vit_buffers_ || !vit_buffers_->is_uploaded()) {
            NSLog(@"[mast3r] Local features MLP: Weights not loaded, skipping");
            return;
        }

        const auto& mlp = vit_buffers_->local_features_mlp();
        if (!mlp.fc1_weight) {
            NSLog(@"[mast3r] Local features MLP: No weights, skipping");
            return;
        }

        // Dimensions computed from model spec
        // MASt3R: fc1=[7168, 1792], fc2=[6400, 7168], out=(24+1)*16^2=6400
        // DUNE:   fc1=[4608, 1152], fc2=[4900, 4608], out=(24+1)*14^2=4900
        const int in_dim = embed_dim + decoder_dim;
        const int hidden_dim = 4 * in_dim;
        const int local_feat_dim = spec_.desc_dim;   // 24 for both DUNE and MASt3R
        const int out_dim = (local_feat_dim + 1) * P * P;  // (24+1) * patch_size^2

        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        cmd.label = @"Local Features MLP";
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // ========== Step 1: Concatenate encoder and decoder outputs ==========
        // [N, embed_dim] + [N, decoder_dim] -> [N, in_dim]
        auto concat_pipeline = ctx.get_pipeline("concat_features");
        if (concat_pipeline) {
            [enc setComputePipelineState:concat_pipeline];
            [enc setBuffer:encoder_output offset:0 atIndex:0];
            [enc setBuffer:decoder_output offset:0 atIndex:1];
            [enc setBuffer:buffer_local_hidden_ offset:0 atIndex:2];

            int dims[3] = {num_patches, embed_dim, decoder_dim};
            [enc setBytes:dims length:sizeof(dims) atIndex:3];

            [enc dispatchThreads:MTLSizeMake(num_patches, in_dim, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // ========== Step 2: FC1 linear projection ==========
        // [N, 1792] @ [7168, 1792]^T + bias -> [N, 7168]
        auto linear_pipeline = ctx.get_pipeline("linear");
        if (linear_pipeline) {
            [enc setComputePipelineState:linear_pipeline];
            [enc setBuffer:buffer_local_hidden_ offset:0 atIndex:0];  // input [N, 1792]
            [enc setBuffer:buffer_fc1_out_ offset:0 atIndex:1];       // output [N, 7168]
            [enc setBuffer:mlp.fc1_weight offset:0 atIndex:2];        // [7168, 1792]
            [enc setBuffer:mlp.fc1_bias offset:0 atIndex:3];          // [7168]

            int dims[3] = {num_patches, in_dim, hidden_dim};  // N, D_in, D_out
            [enc setBytes:dims length:sizeof(dims) atIndex:4];

            [enc dispatchThreads:MTLSizeMake(num_patches, hidden_dim, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // ========== Step 3: GELU activation in-place ==========
        auto gelu_pipeline = ctx.get_pipeline("gelu");
        if (gelu_pipeline) {
            [enc setComputePipelineState:gelu_pipeline];
            [enc setBuffer:buffer_fc1_out_ offset:0 atIndex:0];

            int size = num_patches * hidden_dim;
            [enc setBytes:&size length:sizeof(size) atIndex:1];

            [enc dispatchThreads:MTLSizeMake(size, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // ========== Step 4: FC2 linear projection ==========
        // [N, 7168] @ [6400, 7168]^T + bias -> [N, 6400]
        if (linear_pipeline) {
            [enc setComputePipelineState:linear_pipeline];
            [enc setBuffer:buffer_fc1_out_ offset:0 atIndex:0];       // input [N, 7168]
            [enc setBuffer:buffer_desc_patch1_ offset:0 atIndex:1];   // output [N, 6400]
            [enc setBuffer:mlp.fc2_weight offset:0 atIndex:2];        // [6400, 7168]
            [enc setBuffer:mlp.fc2_bias offset:0 atIndex:3];          // [6400]

            int dims[3] = {num_patches, hidden_dim, out_dim};  // N, D_in, D_out
            [enc setBytes:dims length:sizeof(dims) atIndex:4];

            [enc dispatchThreads:MTLSizeMake(num_patches, out_dim, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // Debug: commit and check MLP output
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        {
            float* mlp_out = (float*)buffer_desc_patch1_.contents;
            int nan_count = 0;
            float min_val = INFINITY, max_val = -INFINITY;
            for (int i = 0; i < num_patches * out_dim && i < 100000; i++) {
                if (std::isnan(mlp_out[i])) nan_count++;
                else {
                    min_val = std::min(min_val, mlp_out[i]);
                    max_val = std::max(max_val, mlp_out[i]);
                }
            }
            NSLog(@"[mast3r] MLP fc2 output: nan=%d, range=[%.3f, %.3f]", nan_count, min_val, max_val);
        }

        // Start new command buffer for pixel shuffle and normalization
        cmd = [ctx.command_queue() commandBuffer];
        enc = [cmd computeCommandEncoder];

        // ========== Step 5: Pixel shuffle ==========
        // [pH, pW, 6400] -> [H, W, 25] with factor=16 (patch_size)
        // Input: buffer_desc_patch1_ as [pH*pW, 6400] = [1024, 6400]
        // Need to treat as [pH, pW, 6400] then pixel_shuffle to [H, W, 25]
        auto shuffle_pipeline = ctx.get_pipeline("pixel_shuffle");
        int pH = res / P;  // 32
        int pW = res / P;  // 32
        const int shuffle_factor = P;  // 16 (patch_size, as in official impl)
        const int out_channels = out_dim / (shuffle_factor * shuffle_factor);  // 6400 / 256 = 25

        if (shuffle_pipeline) {
            [enc setComputePipelineState:shuffle_pipeline];
            [enc setBuffer:buffer_desc_patch1_ offset:0 atIndex:0];   // [pH, pW, 6400]
            [enc setBuffer:buffer_local_shuffle_ offset:0 atIndex:1]; // [H, W, 25]

            int dims[4] = {pH, pW, out_channels, shuffle_factor};  // H, W, C, r
            [enc setBytes:dims length:sizeof(dims) atIndex:2];

            [enc dispatchThreads:MTLSizeMake(res, res, out_channels)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // ========== Step 6: L2 normalize first 24 channels ==========
        // buffer_local_shuffle_ has [H, W, 25], we normalize first 24 channels
        // For now, just copy to output. TODO: proper normalization
        // The current desc_dim in ModelSpec is 256 but actual is 24
        // We'll output 24 channels normalized

        // For now, copy and normalize directly to desc_out
        // Use normalize_descriptors_dpt kernel
        auto norm_pipeline = ctx.get_pipeline("normalize_descriptors_dpt");
        if (norm_pipeline) {
            // First copy 24 channels to desc_out, then normalize
            // TODO: Need a proper kernel for this
            // For now, skip normalization and just copy
        }

        // Simple workaround: copy first 24 channels to output
        // This is a temporary solution - proper implementation needed
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Manual copy with L2 normalization (CPU fallback for now)
        {
            float* shuffle_out = (float*)buffer_local_shuffle_.contents;
            float* desc_final = (float*)desc_out.contents;

            for (int y = 0; y < res; y++) {
                for (int x = 0; x < res; x++) {
                    int idx = (y * res + x) * out_channels;  // 25 channels
                    int out_idx = (y * res + x) * local_feat_dim;  // 24 channels

                    // Compute L2 norm of first 24 channels
                    float norm_sq = 0.0f;
                    for (int c = 0; c < local_feat_dim; c++) {
                        float v = shuffle_out[idx + c];
                        norm_sq += v * v;
                    }
                    float inv_norm = 1.0f / sqrtf(norm_sq + 1e-12f);

                    // Normalize and copy
                    for (int c = 0; c < local_feat_dim; c++) {
                        desc_final[out_idx + c] = shuffle_out[idx + c] * inv_norm;
                    }
                }
            }

            // Check output
            int nan_count = 0;
            float min_val = INFINITY, max_val = -INFINITY;
            for (int i = 0; i < res * res * local_feat_dim; i++) {
                if (std::isnan(desc_final[i])) nan_count++;
                else {
                    min_val = std::min(min_val, desc_final[i]);
                    max_val = std::max(max_val, desc_final[i]);
                }
            }
            NSLog(@"[mast3r] Desc output (L2 norm): nan=%d, range=[%.3f, %.3f]", nan_count, min_val, max_val);
        }

        NSLog(@"[mast3r] Local features MLP complete (desc_dim=%d)", local_feat_dim);
    }
}

void MetalEngine::run_dpt_heads(
    id<MTLBuffer> decoder_output,
    id<MTLBuffer> pts3d_out,
    id<MTLBuffer> desc_out,
    id<MTLBuffer> conf_out
) {
    @autoreleasepool {
        auto& ctx = MetalContext::instance();
        const int res = config_.resolution;
        const int num_patches = spec_.num_patches(res);
        const int decoder_dim = spec_.decoder_dim;
        const int desc_dim = spec_.desc_dim;

        // =====================================================================
        // FALLBACK: Use simple linear projections from decoder output
        // The full DPT architecture requires multi-scale refinenets which
        // we haven't fully implemented yet. This fallback uses direct linear
        // projections with Xavier-initialized weights for testing.
        // =====================================================================

        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        cmd.label = @"DPT Heads (fallback)";
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Get the existing linear projection kernels
        auto pts3d_pipeline = ctx.get_pipeline("pts3d_head");
        auto desc_pipeline = ctx.get_pipeline("desc_head");
        auto conf_pipeline = ctx.get_pipeline("conf_head");
        auto norm_pipeline = ctx.get_pipeline("normalize_descriptors_dpt");

        // Create fallback projection weights if needed (lazy initialization)
        // These are Xavier-initialized random weights for testing
        static id<MTLBuffer> fallback_pts3d_w = nil;
        static id<MTLBuffer> fallback_pts3d_b = nil;
        static id<MTLBuffer> fallback_desc_w = nil;
        static id<MTLBuffer> fallback_desc_b = nil;
        static id<MTLBuffer> fallback_conf_w = nil;
        static id<MTLBuffer> fallback_conf_b = nil;

        if (!fallback_pts3d_w) {
            // Xavier initialization: std = sqrt(2 / (in + out))
            float pts3d_std = sqrtf(2.0f / (float)(decoder_dim + 3));
            float desc_std = sqrtf(2.0f / (float)(decoder_dim + desc_dim));
            float conf_std = sqrtf(2.0f / (float)(decoder_dim + 1));

            // Allocate FP16 weights (uint16_t for half precision)
            std::vector<uint16_t> pts3d_w(3 * decoder_dim);
            std::vector<uint16_t> pts3d_b(3);
            std::vector<uint16_t> desc_w(desc_dim * decoder_dim);
            std::vector<uint16_t> desc_b(desc_dim);
            std::vector<uint16_t> conf_w(decoder_dim);
            std::vector<uint16_t> conf_b(1);

            // Initialize biases to zero
            for (size_t i = 0; i < pts3d_b.size(); i++) pts3d_b[i] = f32_to_f16(0.0f);
            for (size_t i = 0; i < desc_b.size(); i++) desc_b[i] = f32_to_f16(0.0f);
            for (size_t i = 0; i < conf_b.size(); i++) conf_b[i] = f32_to_f16(0.0f);

            srand(42);  // Deterministic for reproducibility
            for (int i = 0; i < 3 * decoder_dim; i++) {
                float val = pts3d_std * (2.0f * (float)rand() / RAND_MAX - 1.0f);
                pts3d_w[i] = f32_to_f16(val);
            }
            for (int i = 0; i < desc_dim * decoder_dim; i++) {
                float val = desc_std * (2.0f * (float)rand() / RAND_MAX - 1.0f);
                desc_w[i] = f32_to_f16(val);
            }
            for (int i = 0; i < decoder_dim; i++) {
                float val = conf_std * (2.0f * (float)rand() / RAND_MAX - 1.0f);
                conf_w[i] = f32_to_f16(val);
            }

            // Create Metal buffers (FP16 = 2 bytes per element)
            auto device = ctx.device();
            fallback_pts3d_w = [device newBufferWithBytes:pts3d_w.data()
                                                   length:pts3d_w.size() * sizeof(uint16_t)
                                                  options:MTLResourceStorageModeShared];
            fallback_pts3d_b = [device newBufferWithBytes:pts3d_b.data()
                                                   length:pts3d_b.size() * sizeof(uint16_t)
                                                  options:MTLResourceStorageModeShared];
            fallback_desc_w = [device newBufferWithBytes:desc_w.data()
                                                  length:desc_w.size() * sizeof(uint16_t)
                                                 options:MTLResourceStorageModeShared];
            fallback_desc_b = [device newBufferWithBytes:desc_b.data()
                                                  length:desc_b.size() * sizeof(uint16_t)
                                                 options:MTLResourceStorageModeShared];
            fallback_conf_w = [device newBufferWithBytes:conf_w.data()
                                                  length:conf_w.size() * sizeof(uint16_t)
                                                 options:MTLResourceStorageModeShared];
            fallback_conf_b = [device newBufferWithBytes:conf_b.data()
                                                  length:conf_b.size() * sizeof(uint16_t)
                                                 options:MTLResourceStorageModeShared];

            NSLog(@"[mast3r] Created fallback projection weights (Xavier init, FP16)");
        }

        // --- Pts3D Head ---
        if (pts3d_pipeline && fallback_pts3d_w) {
            [enc setComputePipelineState:pts3d_pipeline];
            [enc setBuffer:decoder_output offset:0 atIndex:0];
            [enc setBuffer:pts3d_out offset:0 atIndex:1];
            [enc setBuffer:fallback_pts3d_w offset:0 atIndex:2];
            [enc setBuffer:fallback_pts3d_b offset:0 atIndex:3];

            int dims[2] = {num_patches, decoder_dim};
            float scale = 10.0f;
            [enc setBytes:dims length:sizeof(dims) atIndex:4];
            [enc setBytes:&scale length:sizeof(scale) atIndex:5];

            [enc dispatchThreads:MTLSizeMake(num_patches, 3, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // --- Descriptor Head ---
        if (desc_pipeline && fallback_desc_w) {
            [enc setComputePipelineState:desc_pipeline];
            [enc setBuffer:decoder_output offset:0 atIndex:0];
            [enc setBuffer:desc_out offset:0 atIndex:1];
            [enc setBuffer:fallback_desc_w offset:0 atIndex:2];
            [enc setBuffer:fallback_desc_b offset:0 atIndex:3];

            int dims[3] = {num_patches, decoder_dim, desc_dim};
            [enc setBytes:dims length:sizeof(dims) atIndex:4];

            [enc dispatchThreads:MTLSizeMake(num_patches, desc_dim, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // --- Normalize Descriptors ---
        if (norm_pipeline) {
            [enc setComputePipelineState:norm_pipeline];
            [enc setBuffer:desc_out offset:0 atIndex:0];

            int dims[2] = {num_patches, desc_dim};
            [enc setBytes:dims length:sizeof(dims) atIndex:1];

            [enc dispatchThreads:MTLSizeMake(num_patches, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // --- Confidence Head ---
        if (conf_pipeline && fallback_conf_w) {
            [enc setComputePipelineState:conf_pipeline];
            [enc setBuffer:decoder_output offset:0 atIndex:0];
            [enc setBuffer:conf_out offset:0 atIndex:1];
            [enc setBuffer:fallback_conf_w offset:0 atIndex:2];
            [enc setBuffer:fallback_conf_b offset:0 atIndex:3];

            int dims[2] = {num_patches, decoder_dim};
            [enc setBytes:dims length:sizeof(dims) atIndex:4];

            [enc dispatchThreads:MTLSizeMake(num_patches, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        NSLog(@"[mast3r] DPT heads complete (fallback)");
    }
}

// ============================================================================
// Upsampling from Patch to Pixel Level
// ============================================================================

void MetalEngine::run_upsample(
    id<MTLBuffer> patch_pts3d,
    id<MTLBuffer> patch_desc,
    id<MTLBuffer> patch_conf,
    id<MTLBuffer> pixel_pts3d,
    id<MTLBuffer> pixel_desc,
    id<MTLBuffer> pixel_conf
) {
    @autoreleasepool {
        auto& ctx = MetalContext::instance();
        const int res = config_.resolution;
        const int P = spec_.patch_size;
        const int patches_h = res / P;
        const int patches_w = res / P;
        const int desc_dim = spec_.desc_dim;

        auto upsample_pipeline = ctx.get_pipeline("bilinear_upsample");
        if (!upsample_pipeline) {
            NSLog(@"[mast3r] Upsample pipeline not found, skipping upsampling");
            return;
        }

        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        cmd.label = @"Upsampling";
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:upsample_pipeline];

        // === Upsample Pts3D [patches_h, patches_w, 3] -> [H, W, 3] ===
        [enc pushDebugGroup:@"Upsample Pts3D"];
        [enc setBuffer:patch_pts3d offset:0 atIndex:0];
        [enc setBuffer:pixel_pts3d offset:0 atIndex:1];

        int in_dims[4] = {patches_h, patches_w, 3, 0};
        int out_dims[2] = {res, res};
        [enc setBytes:in_dims length:sizeof(in_dims) atIndex:2];
        [enc setBytes:out_dims length:sizeof(out_dims) atIndex:3];

        [enc dispatchThreads:MTLSizeMake(res, res, 3)
            threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [enc popDebugGroup];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // === Upsample Descriptors [patches_h, patches_w, D] -> [H, W, D] ===
        [enc pushDebugGroup:@"Upsample Desc"];
        [enc setBuffer:patch_desc offset:0 atIndex:0];
        [enc setBuffer:pixel_desc offset:0 atIndex:1];

        in_dims[2] = desc_dim;
        [enc setBytes:in_dims length:sizeof(in_dims) atIndex:2];
        [enc setBytes:out_dims length:sizeof(out_dims) atIndex:3];

        // Dispatch in batches to avoid huge grid
        for (int c_start = 0; c_start < desc_dim; c_start += 64) {
            int c_end = std::min(c_start + 64, desc_dim);
            [enc dispatchThreads:MTLSizeMake(res, res, c_end - c_start)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        }
        [enc popDebugGroup];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // === Upsample Confidence [patches_h, patches_w, 1] -> [H, W] ===
        [enc pushDebugGroup:@"Upsample Conf"];
        [enc setBuffer:patch_conf offset:0 atIndex:0];
        [enc setBuffer:pixel_conf offset:0 atIndex:1];

        in_dims[2] = 1;
        [enc setBytes:in_dims length:sizeof(in_dims) atIndex:2];
        [enc setBytes:out_dims length:sizeof(out_dims) atIndex:3];

        [enc dispatchThreads:MTLSizeMake(res, res, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc popDebugGroup];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        NSLog(@"[mast3r] Upsampling complete");
    }
}

// ============================================================================
// ViT Decoder Forward Pass
// ============================================================================

void MetalEngine::run_decoder(
    id<MTLBuffer> decoder_tokens,   // Initial decoder tokens (encoder output)
    id<MTLBuffer> encoder_output,   // Other encoder output for cross-attention
    id<MTLBuffer> output,           // Decoder output
    id<MTLBuffer> scratch_qkv,
    id<MTLBuffer> scratch_mlp,
    id<MTLBuffer> __unsafe_unretained * hook_outputs,  // Optional: [4] buffers for DPT hooks
    bool use_decoder2  // If true, use dec_blocks2 weights (for MASt3R image 2)
) {
    @autoreleasepool {
        auto& ctx = MetalContext::instance();
        const int res = config_.resolution;
        const int num_patches = spec_.num_patches(res);
        const int D = spec_.decoder_dim;
        const int num_heads = spec_.decoder_heads;
        const int head_dim = D / num_heads;

        // Weight accessors - select between decoder and decoder2
        auto get_norm1_w = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_norm1_weight(l) : vit_buffers_->decoder_norm1_weight(l); };
        auto get_norm1_b = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_norm1_bias(l) : vit_buffers_->decoder_norm1_bias(l); };
        auto get_norm2_w = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_norm2_weight(l) : vit_buffers_->decoder_norm2_weight(l); };
        auto get_norm2_b = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_norm2_bias(l) : vit_buffers_->decoder_norm2_bias(l); };
        auto get_qkv_w = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_qkv_weight(l) : vit_buffers_->decoder_qkv_weight(l); };
        auto get_qkv_b = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_qkv_bias(l) : vit_buffers_->decoder_qkv_bias(l); };
        auto get_proj_w = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_proj_weight(l) : vit_buffers_->decoder_proj_weight(l); };
        auto get_proj_b = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_proj_bias(l) : vit_buffers_->decoder_proj_bias(l); };
        auto get_cross_norm_w = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_cross_norm_weight(l) : vit_buffers_->decoder_cross_norm_weight(l); };
        auto get_cross_norm_b = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_cross_norm_bias(l) : vit_buffers_->decoder_cross_norm_bias(l); };
        auto get_cross_q_w = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_cross_q_weight(l) : vit_buffers_->decoder_cross_q_weight(l); };
        auto get_cross_q_b = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_cross_q_bias(l) : vit_buffers_->decoder_cross_q_bias(l); };
        auto get_cross_kv_w = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_cross_kv_weight(l) : vit_buffers_->decoder_cross_kv_weight(l); };
        auto get_cross_kv_b = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_cross_kv_bias(l) : vit_buffers_->decoder_cross_kv_bias(l); };
        auto get_cross_proj_w = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_cross_proj_weight(l) : vit_buffers_->decoder_cross_proj_weight(l); };
        auto get_cross_proj_b = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_cross_proj_bias(l) : vit_buffers_->decoder_cross_proj_bias(l); };
        auto get_mlp_fc1_w = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_mlp_fc1_weight(l) : vit_buffers_->decoder_mlp_fc1_weight(l); };
        auto get_mlp_fc1_b = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_mlp_fc1_bias(l) : vit_buffers_->decoder_mlp_fc1_bias(l); };
        auto get_mlp_fc2_w = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_mlp_fc2_weight(l) : vit_buffers_->decoder_mlp_fc2_weight(l); };
        auto get_mlp_fc2_b = [&](int l) { return use_decoder2 ? vit_buffers_->decoder2_mlp_fc2_bias(l) : vit_buffers_->decoder_mlp_fc2_bias(l); };

        // DPT hook layers: [0, 6, 9, 12] for decoder_depth=12
        // These correspond to hook indices 0, 1, 2, 3
        const int dpt_hooks[4] = {0, 6, 9, 12};

        auto ln_pipeline = ctx.get_pipeline("layer_norm");
        auto qkv_pipeline = ctx.get_pipeline("linear");
        auto split_pipeline = ctx.get_pipeline("split_qkv");
        auto mha_pipeline = ctx.get_pipeline("multihead_self_attention");
        auto cross_attn_pipeline = ctx.get_pipeline("cross_attention");
        auto proj_pipeline = ctx.get_pipeline("attention_output_proj");
        auto residual_pipeline = ctx.get_pipeline("residual_add");
        auto mlp_fc1_pipeline = ctx.get_pipeline("mlp_fc1");
        auto mlp_fc2_pipeline = ctx.get_pipeline("mlp_fc2");
        auto copy_pipeline = ctx.get_pipeline("buffer_copy");

        // Copy encoder output to decoder output as initial tokens (GPU copy)
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (copy_pipeline) {
                [enc setComputePipelineState:copy_pipeline];
                [enc setBuffer:decoder_tokens offset:0 atIndex:0];
                [enc setBuffer:output offset:0 atIndex:1];
                int copy_size = num_patches * D;
                [enc setBytes:&copy_size length:sizeof(copy_size) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(copy_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // Capture hook 0 (before any layer processing)
        if (hook_outputs && hook_outputs[0] && copy_pipeline) {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:copy_pipeline];
            [enc setBuffer:output offset:0 atIndex:0];
            [enc setBuffer:hook_outputs[0] offset:0 atIndex:1];
            int copy_size = num_patches * D;
            [enc setBytes:&copy_size length:sizeof(copy_size) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(copy_size, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        for (int layer = 0; layer < spec_.decoder_depth; layer++) {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            cmd.label = [NSString stringWithFormat:@"Decoder Layer %d", layer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            // === SELF-ATTENTION ===
            // Save residual (GPU copy)
            if (copy_pipeline) {
                [enc setComputePipelineState:copy_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_residual_ offset:0 atIndex:1];
                int copy_size = num_patches * D;
                [enc setBytes:&copy_size length:sizeof(copy_size) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(copy_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Pre-norm
            if (ln_pipeline && get_norm1_w(layer)) {
                [enc setComputePipelineState:ln_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:get_norm1_w(layer) offset:0 atIndex:1];
                [enc setBuffer:get_norm1_b(layer) offset:0 atIndex:2];

                int dims[2] = {num_patches, D};
                float eps = 1e-6f;
                [enc setBytes:dims length:sizeof(dims) atIndex:3];
                [enc setBytes:&eps length:sizeof(eps) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Self-attention QKV projection
            if (qkv_pipeline && get_qkv_w(layer)) {
                [enc setComputePipelineState:qkv_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_qkv_ offset:0 atIndex:1];
                [enc setBuffer:get_qkv_w(layer) offset:0 atIndex:2];
                [enc setBuffer:get_qkv_b(layer) offset:0 atIndex:3];

                int dims[3] = {num_patches, D, 3 * D};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, 3 * D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Split QKV
            if (split_pipeline) {
                [enc setComputePipelineState:split_pipeline];
                [enc setBuffer:buffer_qkv_ offset:0 atIndex:0];
                [enc setBuffer:buffer_q_ offset:0 atIndex:1];
                [enc setBuffer:buffer_k_ offset:0 atIndex:2];
                [enc setBuffer:buffer_v_ offset:0 atIndex:3];

                int dims[3] = {num_patches, num_heads, head_dim};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Apply RoPE 2D for MASt3R decoder self-attention
            if (spec_.is_mast3r()) {
                auto rope_pipeline = ctx.get_pipeline("apply_rope_2d");
                if (rope_pipeline) {
                    // Apply RoPE to Q
                    // Grid: [num_patches, num_heads, head_dim/4]
                    // Decoder: head_dim=64 -> 16 rotation pairs per half
                    [enc setComputePipelineState:rope_pipeline];
                    [enc setBuffer:buffer_q_ offset:0 atIndex:0];
                    [enc setBuffer:buffer_rope_positions_ offset:0 atIndex:1];

                    int rope_dims[3] = {num_patches, num_heads, head_dim};
                    float freq_base = 100.0f;
                    [enc setBytes:rope_dims length:sizeof(rope_dims) atIndex:2];
                    [enc setBytes:&freq_base length:sizeof(freq_base) atIndex:3];

                    [enc dispatchThreads:MTLSizeMake(num_patches, num_heads, head_dim / 4)
                        threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];

                    // Apply RoPE to K
                    [enc setBuffer:buffer_k_ offset:0 atIndex:0];
                    [enc dispatchThreads:MTLSizeMake(num_patches, num_heads, head_dim / 4)
                        threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }

            // Self-attention using 3-pass parallel approach
            auto scores_pipeline = ctx.get_pipeline("attention_scores");
            auto softmax_pipeline = ctx.get_pipeline("attention_softmax");
            auto attn_output_pipeline = ctx.get_pipeline("attention_output");

            // Pass 1: attention scores
            if (scores_pipeline) {
                [enc setComputePipelineState:scores_pipeline];
                [enc setBuffer:buffer_q_ offset:0 atIndex:0];
                [enc setBuffer:buffer_k_ offset:0 atIndex:1];
                [enc setBuffer:buffer_attn_ offset:0 atIndex:2];

                int dims3[3] = {num_patches, num_heads, head_dim};
                [enc setBytes:dims3 length:sizeof(dims3) atIndex:3];

                [enc dispatchThreads:MTLSizeMake(num_patches, num_patches, num_heads)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Pass 2: softmax
            if (softmax_pipeline) {
                [enc setComputePipelineState:softmax_pipeline];
                [enc setBuffer:buffer_attn_ offset:0 atIndex:0];

                int dims2[2] = {num_patches, num_heads};
                [enc setBytes:dims2 length:sizeof(dims2) atIndex:1];

                [enc dispatchThreads:MTLSizeMake(num_patches, num_heads, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Pass 3: output
            if (attn_output_pipeline) {
                [enc setComputePipelineState:attn_output_pipeline];
                [enc setBuffer:buffer_attn_ offset:0 atIndex:0];
                [enc setBuffer:buffer_v_ offset:0 atIndex:1];
                [enc setBuffer:buffer_attn_out_ offset:0 atIndex:2];

                int dims3[3] = {num_patches, num_heads, head_dim};
                [enc setBytes:dims3 length:sizeof(dims3) atIndex:3];

                [enc dispatchThreads:MTLSizeMake(num_patches, head_dim, num_heads)
                    threadsPerThreadgroup:MTLSizeMake(16, 8, 4)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Self-attention output projection
            if (proj_pipeline && get_proj_w(layer)) {
                [enc setComputePipelineState:proj_pipeline];
                [enc setBuffer:buffer_attn_out_ offset:0 atIndex:0];
                [enc setBuffer:output offset:0 atIndex:1];
                [enc setBuffer:get_proj_w(layer) offset:0 atIndex:2];
                [enc setBuffer:get_proj_b(layer) offset:0 atIndex:3];

                int dims[2] = {num_patches, D};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Residual for self-attention
            if (residual_pipeline) {
                [enc setComputePipelineState:residual_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_residual_ offset:0 atIndex:1];

                int size = num_patches * D;
                [enc setBytes:&size length:sizeof(size) atIndex:2];

                [enc dispatchThreads:MTLSizeMake(num_patches * D, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // === CROSS-ATTENTION ===
            // Save residual (GPU copy)
            if (copy_pipeline) {
                [enc setComputePipelineState:copy_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_residual_ offset:0 atIndex:1];
                int copy_size = num_patches * D;
                [enc setBytes:&copy_size length:sizeof(copy_size) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(copy_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Cross-attention pre-norm
            if (ln_pipeline && get_cross_norm_w(layer)) {
                [enc setComputePipelineState:ln_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:get_cross_norm_w(layer) offset:0 atIndex:1];
                [enc setBuffer:get_cross_norm_b(layer) offset:0 atIndex:2];

                int dims[2] = {num_patches, D};
                float eps = 1e-6f;
                [enc setBytes:dims length:sizeof(dims) atIndex:3];
                [enc setBytes:&eps length:sizeof(eps) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Cross-attention Q projection (from decoder)
            if (qkv_pipeline && get_cross_q_w(layer)) {
                [enc setComputePipelineState:qkv_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_cross_q_ offset:0 atIndex:1];
                [enc setBuffer:get_cross_q_w(layer) offset:0 atIndex:2];
                [enc setBuffer:get_cross_q_b(layer) offset:0 atIndex:3];

                int dims[3] = {num_patches, D, D};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Apply RoPE 2D for MASt3R cross-attention Q (decoder positions)
            if (spec_.is_mast3r()) {
                auto rope_pipeline = ctx.get_pipeline("apply_rope_2d");
                if (rope_pipeline) {
                    [enc setComputePipelineState:rope_pipeline];
                    [enc setBuffer:buffer_cross_q_ offset:0 atIndex:0];
                    [enc setBuffer:buffer_rope_positions_ offset:0 atIndex:1];

                    int rope_dims[3] = {num_patches, num_heads, head_dim};
                    float freq_base = 100.0f;
                    [enc setBytes:rope_dims length:sizeof(rope_dims) atIndex:2];
                    [enc setBytes:&freq_base length:sizeof(freq_base) atIndex:3];

                    [enc dispatchThreads:MTLSizeMake(num_patches, num_heads, head_dim / 4)
                        threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }

            // Cross-attention K/V projection (from encoder) - combined projection
            // Output is [N, 2*D] which we'll split into K and V
            if (qkv_pipeline && get_cross_kv_w(layer)) {
                // Project encoder output to combined K/V buffer
                // We reuse buffer_qkv_ to store the [N, 2*D] output
                [enc setComputePipelineState:qkv_pipeline];
                [enc setBuffer:encoder_output offset:0 atIndex:0];
                [enc setBuffer:buffer_qkv_ offset:0 atIndex:1];
                [enc setBuffer:get_cross_kv_w(layer) offset:0 atIndex:2];
                [enc setBuffer:get_cross_kv_b(layer) offset:0 atIndex:3];

                int dims[3] = {num_patches, D, 2 * D};  // Output is 2*D (K and V)
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, 2 * D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Apply RoPE 2D for MASt3R cross-attention K (encoder positions)
            // K is in the first half of buffer_qkv_ [0:N*D]
            if (spec_.is_mast3r()) {
                auto rope_pipeline = ctx.get_pipeline("apply_rope_2d");
                if (rope_pipeline) {
                    [enc setComputePipelineState:rope_pipeline];
                    [enc setBuffer:buffer_qkv_ offset:0 atIndex:0];  // K at offset 0
                    [enc setBuffer:buffer_rope_positions_ offset:0 atIndex:1];

                    int rope_dims[3] = {num_patches, num_heads, head_dim};
                    float freq_base = 100.0f;
                    [enc setBytes:rope_dims length:sizeof(rope_dims) atIndex:2];
                    [enc setBytes:&freq_base length:sizeof(freq_base) atIndex:3];

                    [enc dispatchThreads:MTLSizeMake(num_patches, num_heads, head_dim / 4)
                        threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }

            // Cross-attention computation (use optimized version)
            // Note: buffer_qkv_ contains [K, V] from the cross_kv projection
            // K = buffer_qkv_[0:D], V = buffer_qkv_[D:2D]
            auto cross_opt_pipeline = ctx.get_pipeline("cross_attention_optimized");
            if (!cross_opt_pipeline) {
                cross_opt_pipeline = cross_attn_pipeline;  // Fallback
            }
            if (cross_opt_pipeline) {
                [enc setComputePipelineState:cross_opt_pipeline];
                [enc setBuffer:buffer_cross_q_ offset:0 atIndex:0];
                // Use buffer_qkv_ for K (first half) and V (second half)
                [enc setBuffer:buffer_qkv_ offset:0 atIndex:1];  // K starts at offset 0
                [enc setBuffer:buffer_qkv_ offset:num_patches * D * sizeof(float) atIndex:2];  // V starts at offset D
                [enc setBuffer:buffer_attn_out_ offset:0 atIndex:3];

                int dims[4] = {num_patches, num_patches, num_heads, head_dim};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, num_heads, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Cross-attention output projection
            if (proj_pipeline && get_cross_proj_w(layer)) {
                [enc setComputePipelineState:proj_pipeline];
                [enc setBuffer:buffer_attn_out_ offset:0 atIndex:0];
                [enc setBuffer:output offset:0 atIndex:1];
                [enc setBuffer:get_cross_proj_w(layer) offset:0 atIndex:2];
                [enc setBuffer:get_cross_proj_b(layer) offset:0 atIndex:3];

                int dims[2] = {num_patches, D};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Residual for cross-attention
            if (residual_pipeline) {
                [enc setComputePipelineState:residual_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_residual_ offset:0 atIndex:1];

                int size = num_patches * D;
                [enc setBytes:&size length:sizeof(size) atIndex:2];

                [enc dispatchThreads:MTLSizeMake(num_patches * D, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // === MLP ===
            // Save residual (GPU copy)
            if (copy_pipeline) {
                [enc setComputePipelineState:copy_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_residual_ offset:0 atIndex:1];
                int copy_size = num_patches * D;
                [enc setBytes:&copy_size length:sizeof(copy_size) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(copy_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // MLP pre-norm
            if (ln_pipeline && get_norm2_w(layer)) {
                [enc setComputePipelineState:ln_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:get_norm2_w(layer) offset:0 atIndex:1];
                [enc setBuffer:get_norm2_b(layer) offset:0 atIndex:2];

                int dims[2] = {num_patches, D};
                float eps = 1e-6f;
                [enc setBytes:dims length:sizeof(dims) atIndex:3];
                [enc setBytes:&eps length:sizeof(eps) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // MLP FC1
            if (mlp_fc1_pipeline && get_mlp_fc1_w(layer)) {
                [enc setComputePipelineState:mlp_fc1_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_mlp_ offset:0 atIndex:1];
                [enc setBuffer:get_mlp_fc1_w(layer) offset:0 atIndex:2];
                [enc setBuffer:get_mlp_fc1_b(layer) offset:0 atIndex:3];

                int dims[2] = {num_patches, D};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, 4 * D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // MLP FC2
            if (mlp_fc2_pipeline && get_mlp_fc2_w(layer)) {
                [enc setComputePipelineState:mlp_fc2_pipeline];
                [enc setBuffer:buffer_mlp_ offset:0 atIndex:0];
                [enc setBuffer:output offset:0 atIndex:1];
                [enc setBuffer:get_mlp_fc2_w(layer) offset:0 atIndex:2];
                [enc setBuffer:get_mlp_fc2_b(layer) offset:0 atIndex:3];

                int dims[2] = {num_patches, D};
                [enc setBytes:dims length:sizeof(dims) atIndex:4];

                [enc dispatchThreads:MTLSizeMake(num_patches, D, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Residual for MLP
            if (residual_pipeline) {
                [enc setComputePipelineState:residual_pipeline];
                [enc setBuffer:output offset:0 atIndex:0];
                [enc setBuffer:buffer_residual_ offset:0 atIndex:1];

                int size = num_patches * D;
                [enc setBytes:&size length:sizeof(size) atIndex:2];

                [enc dispatchThreads:MTLSizeMake(num_patches * D, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // Capture hook outputs for DPT (after layer completion)
            // Hooks at layers [0, 6, 9, 12] -> indices 0, 1, 2, 3
            // Note: hook 0 is captured before the loop, hooks 1-3 after layers 6, 9, 12
            if (hook_outputs && copy_pipeline) {
                int hook_idx = -1;
                if (layer == 6) hook_idx = 1;
                else if (layer == 9) hook_idx = 2;
                else if (layer == 12 || layer == spec_.decoder_depth - 1) hook_idx = 3;

                if (hook_idx > 0 && hook_outputs[hook_idx]) {
                    [enc setComputePipelineState:copy_pipeline];
                    [enc setBuffer:output offset:0 atIndex:0];
                    [enc setBuffer:hook_outputs[hook_idx] offset:0 atIndex:1];
                    int copy_size = num_patches * D;
                    [enc setBytes:&copy_size length:sizeof(copy_size) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(copy_size, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        NSLog(@"[mast3r] Decoder forward pass complete (%d layers)", spec_.decoder_depth);
    }
}

MatchResult MetalEngine::match(
    const float* desc_1, const float* desc_2,
    int height, int width, int desc_dim,
    const MatchingConfig& config
) {
    @autoreleasepool {
        auto& ctx = MetalContext::instance();
        const int N = height * width;
        const int K = config.top_k;

        // Allocate buffers
        auto desc1_buf = ctx.create_buffer(desc_1, N * desc_dim * sizeof(float));
        auto desc2_buf = ctx.create_buffer(desc_2, N * desc_dim * sizeof(float));
        auto topk_indices = ctx.create_buffer(N * K * sizeof(int));
        auto topk_scores = ctx.create_buffer(N * K * sizeof(float));
        auto match_count = ctx.create_buffer(sizeof(int));
        auto matches_1 = ctx.create_buffer(K * sizeof(int));
        auto matches_2 = ctx.create_buffer(K * sizeof(int));
        auto match_scores = ctx.create_buffer(K * sizeof(float));

        // Zero match count
        std::memset(match_count.contents, 0, sizeof(int));

        CommandBatch batch(ctx);

        // === NORMALIZE DESCRIPTORS ===
        auto norm_pipeline = ctx.get_pipeline("normalize_descriptors");
        if (norm_pipeline) {
            auto enc = batch.encoder();
            [enc setComputePipelineState:norm_pipeline];

            int dims[2] = {N, desc_dim};
            auto dims_buf = ctx.create_buffer(dims, sizeof(dims));

            [enc setBuffer:desc1_buf offset:0 atIndex:0];
            [enc setBuffer:dims_buf offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:desc2_buf offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        batch.memoryBarrier();

        // === TOP-K SELECTION ===
        auto topk_pipeline = ctx.get_pipeline("row_topk");
        if (topk_pipeline) {
            auto enc = batch.encoder();
            [enc setComputePipelineState:topk_pipeline];

            // Forward: desc_1 vs desc_2
            // (Would need similarity matrix, but we use direct top-k here)
            // For real impl, compute sim matrix or use approximate methods
        }

        // === RECIPROCAL MATCHING ===
        auto match_pipeline = ctx.get_pipeline("check_reciprocity");
        if (match_pipeline) {
            // TODO: Encode reciprocal matching
        }

        batch.commitAndWait();

        // Read results
        MatchResult result;
        int num_matches = *static_cast<int*>(match_count.contents);
        num_matches = std::min(num_matches, K);

        if (num_matches > 0) {
            int* m1 = static_cast<int*>(matches_1.contents);
            int* m2 = static_cast<int*>(matches_2.contents);
            float* ms = static_cast<float*>(match_scores.contents);

            result.idx_1.assign(m1, m1 + num_matches);
            result.idx_2.assign(m2, m2 + num_matches);
            result.confidence.assign(ms, ms + num_matches);

            // Convert indices to 2D coordinates
            result.pts2d_1.resize(num_matches * 2);
            result.pts2d_2.resize(num_matches * 2);

            for (int i = 0; i < num_matches; ++i) {
                result.pts2d_1[i * 2 + 0] = float(m1[i] % width);
                result.pts2d_1[i * 2 + 1] = float(m1[i] / width);
                result.pts2d_2[i * 2 + 0] = float(m2[i] % width);
                result.pts2d_2[i * 2 + 1] = float(m2[i] / width);
            }
        }

        return result;
    }
}

}  // namespace metal
}  // namespace mast3r
