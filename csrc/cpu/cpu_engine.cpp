// MASt3R Runtime - CPU Engine Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include "cpu_engine.hpp"

#include <chrono>
#include <stdexcept>

#include "common/model_loader.hpp"
#include "preprocessing.hpp"

namespace mast3r {
namespace cpu {

CPUEngine::CPUEngine(const RuntimeConfig& config)
    : config_(config),
      spec_(get_model_spec(config.variant)),
      preprocessor_(std::make_unique<Preprocessor>(config.resolution)) {
    allocate_buffers();
}

CPUEngine::~CPUEngine() = default;

CPUEngine::CPUEngine(CPUEngine&&) noexcept = default;
CPUEngine& CPUEngine::operator=(CPUEngine&&) noexcept = default;

void CPUEngine::allocate_buffers() {
    const int res = config_.resolution;
    const int desc_dim = spec_.desc_dim;

    // Image buffers [1, 3, H, W]
    buffer_img1_.resize(3 * res * res);
    buffer_img2_.resize(3 * res * res);

    // Output buffers
    buffer_pts3d_1_.resize(res * res * 3);
    buffer_pts3d_2_.resize(res * res * 3);
    buffer_desc_1_.resize(res * res * desc_dim);
    buffer_desc_2_.resize(res * res * desc_dim);
    buffer_conf_1_.resize(res * res);
    buffer_conf_2_.resize(res * res);
}

void CPUEngine::load(const std::string& /* model_path */) {
    // Load based on architecture type
    if (spec_.is_mast3r()) {
        weights_ = load_mast3r_model(config_.precision);
    } else {
        weights_ = load_dune_model(config_.variant, config_.precision);
    }
    is_loaded_ = true;
}

void CPUEngine::warmup(int num_iterations) {
    if (!is_loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    // Create dummy image
    std::vector<uint8_t> dummy(config_.resolution * config_.resolution * 3, 128);
    ImageView dummy_view{dummy.data(), config_.resolution, config_.resolution, 3};

    for (int i = 0; i < num_iterations; ++i) {
        infer(dummy_view, dummy_view);
    }
}

InferenceResult CPUEngine::infer(const ImageView& img1, const ImageView& img2) {
    using Clock = std::chrono::high_resolution_clock;

    InferenceResult result;
    result.height = config_.resolution;
    result.width = config_.resolution;
    result.desc_dim = spec_.desc_dim;

    // Preprocessing
    auto t0 = Clock::now();
    preprocess(img1, buffer_img1_.data());
    preprocess(img2, buffer_img2_.data());
    auto t1 = Clock::now();
    result.preprocess_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // TODO: Implement actual inference
    // For now, fill with zeros (placeholder)
    auto t2 = Clock::now();

    std::fill(buffer_pts3d_1_.begin(), buffer_pts3d_1_.end(), 0.0f);
    std::fill(buffer_pts3d_2_.begin(), buffer_pts3d_2_.end(), 0.0f);
    std::fill(buffer_desc_1_.begin(), buffer_desc_1_.end(), 0.0f);
    std::fill(buffer_desc_2_.begin(), buffer_desc_2_.end(), 0.0f);
    std::fill(buffer_conf_1_.begin(), buffer_conf_1_.end(), 1.0f);
    std::fill(buffer_conf_2_.begin(), buffer_conf_2_.end(), 1.0f);

    auto t3 = Clock::now();
    result.inference_ms = std::chrono::duration<float, std::milli>(t3 - t2).count();

    // Set pointers
    result.pts3d_1 = buffer_pts3d_1_.data();
    result.pts3d_2 = buffer_pts3d_2_.data();
    result.desc_1 = buffer_desc_1_.data();
    result.desc_2 = buffer_desc_2_.data();
    result.conf_1 = buffer_conf_1_.data();
    result.conf_2 = buffer_conf_2_.data();

    result.total_ms = result.preprocess_ms + result.inference_ms;

    return result;
}

void CPUEngine::preprocess(const ImageView& img, float* output) {
    preprocessor_->process(img, output);
}

MatchResult CPUEngine::match(
    const float* desc_1, const float* desc_2,
    int height, int width, int desc_dim,
    const MatchingConfig& config
) {
    // TODO: Implement CPU matching
    // For now, return empty result
    return MatchResult{};
}

}  // namespace cpu
}  // namespace mast3r