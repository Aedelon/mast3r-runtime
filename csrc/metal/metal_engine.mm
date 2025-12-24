// MASt3R Runtime - Metal Engine Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#import "metal_engine.hpp"
#import "metal_context.hpp"

#import <chrono>

namespace mast3r {
namespace metal {

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

        // Preprocessed (float, CHW)
        buffer_preprocessed1_ = ctx.create_buffer(3 * res * res * sizeof(float));
        buffer_preprocessed2_ = ctx.create_buffer(3 * res * res * sizeof(float));

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
    // TODO: Load model weights and build MPSGraph
    // For now, just mark as loaded
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

        // Preprocessing on GPU
        auto t0 = Clock::now();
        preprocess_gpu(img1, (__bridge void*)buffer_preprocessed1_);
        preprocess_gpu(img2, (__bridge void*)buffer_preprocessed2_);
        auto t1 = Clock::now();
        result.preprocess_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // TODO: Run inference with MPSGraph
        auto t2 = Clock::now();

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

        auto t3 = Clock::now();
        result.inference_ms = std::chrono::duration<float, std::milli>(t3 - t2).count();

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
    @autoreleasepool {
        auto& ctx = MetalContext::instance();

        // Get preprocessing pipeline
        auto pipeline = ctx.get_pipeline("preprocess_image");
        if (!pipeline) {
            // Fallback to CPU preprocessing
            // TODO: Implement CPU fallback
            return;
        }

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // TODO: Set buffers and dispatch
        // [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        // [encoder setBuffer:outputBuffer offset:0 atIndex:1];

        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

MatchResult MetalEngine::match(
    const float* desc_1, const float* desc_2,
    int height, int width, int desc_dim,
    const MatchingConfig& config
) {
    // TODO: GPU matching with Metal
    // For now, return empty result
    return MatchResult{};
}

}  // namespace metal
}  // namespace mast3r