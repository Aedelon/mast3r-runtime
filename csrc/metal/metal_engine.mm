// MASt3R Runtime - Metal Engine Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#import "metal_engine.hpp"
#import "metal_context.hpp"

#import <chrono>

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

MetalEngine::MetalEngine(const RuntimeConfig& config) : config_(config) {
    if (!MetalContext::instance().is_available()) {
        throw std::runtime_error("Metal is not available on this system");
    }
    allocate_buffers();
}

MetalEngine::~MetalEngine() {
    @autoreleasepool {
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
    }
}

void MetalEngine::allocate_buffers() {
    @autoreleasepool {
        auto& ctx = MetalContext::instance();
        const int res = config_.resolution;
        const int desc_dim = 256;

        // Input image buffers (RGBA for Metal)
        buffer_img1_ = ctx.create_buffer(res * res * 4);
        buffer_img2_ = ctx.create_buffer(res * res * 4);

        // Preprocessed (float16, CHW)
        buffer_preprocessed1_ = ctx.create_buffer(3 * res * res * sizeof(uint16_t));
        buffer_preprocessed2_ = ctx.create_buffer(3 * res * res * sizeof(uint16_t));

        // Outputs
        buffer_pts3d_1_ = ctx.create_buffer(res * res * 3 * sizeof(float));
        buffer_pts3d_2_ = ctx.create_buffer(res * res * 3 * sizeof(float));
        buffer_desc_1_ = ctx.create_buffer(res * res * desc_dim * sizeof(float));
        buffer_desc_2_ = ctx.create_buffer(res * res * desc_dim * sizeof(float));
        buffer_conf_1_ = ctx.create_buffer(res * res * sizeof(float));
        buffer_conf_2_ = ctx.create_buffer(res * res * sizeof(float));
    }
}

void MetalEngine::load(const std::string& model_path) {
    // TODO: Load model weights and build compute graph
    is_loaded_ = true;
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
    result.desc_dim = 256;

    @autoreleasepool {
        auto& ctx = MetalContext::instance();

        // Create batched command buffer for all operations
        CommandBatch batch(ctx);

        auto t0 = Clock::now();

        // === PREPROCESSING PASS ===
        preprocess_batched(batch, img1, img2);

        // Memory barrier between preprocessing and inference
        batch.memoryBarrier();

        auto t1 = Clock::now();
        result.preprocess_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // === INFERENCE PASS ===
        // TODO: Add ViT encoder + decoder passes here
        // For now, placeholder zeros

        // === MATCHING PASS ===
        // TODO: Add top-k matching pass

        // Commit all operations at once
        batch.commitAndWait();

        auto t2 = Clock::now();
        result.inference_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();

        // Placeholder: fill with zeros
        const int res = config_.resolution;
        std::memset(buffer_pts3d_1_.contents, 0, res * res * 3 * sizeof(float));
        std::memset(buffer_pts3d_2_.contents, 0, res * res * 3 * sizeof(float));
        std::memset(buffer_desc_1_.contents, 0, res * res * 256 * sizeof(float));
        std::memset(buffer_desc_2_.contents, 0, res * res * 256 * sizeof(float));

        float* conf1 = static_cast<float*>(buffer_conf_1_.contents);
        float* conf2 = static_cast<float*>(buffer_conf_2_.contents);
        std::fill(conf1, conf1 + res * res, 1.0f);
        std::fill(conf2, conf2 + res * res, 1.0f);

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

void MetalEngine::preprocess_batched(CommandBatch& batch, const ImageView& img1, const ImageView& img2) {
    @autoreleasepool {
        auto& ctx = MetalContext::instance();

        // Get fused preprocessing pipeline
        auto pipeline = ctx.get_pipeline("preprocess_fused");
        if (!pipeline) {
            // Fallback: try regular pipeline
            pipeline = ctx.get_pipeline("preprocess_image");
        }

        if (!pipeline) {
            // CPU fallback if no shader available
            return;
        }

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
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);  // Match PREPROCESS_TILE_SIZE

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
