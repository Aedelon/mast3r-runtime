// MASt3R Runtime - CPU Engine
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#pragma once

#include <memory>
#include <vector>

#include "common/types.hpp"

namespace mast3r {
namespace cpu {

// Forward declarations
class Preprocessor;
class AttentionLayer;
class Matcher;

/**
 * CPU-based inference engine.
 *
 * Uses OpenMP for parallelization and BLAS for matrix operations.
 * Reference implementation - correct but not the fastest.
 */
class CPUEngine {
public:
    explicit CPUEngine(const RuntimeConfig& config);
    ~CPUEngine();

    // Non-copyable
    CPUEngine(const CPUEngine&) = delete;
    CPUEngine& operator=(const CPUEngine&) = delete;

    // Moveable
    CPUEngine(CPUEngine&&) noexcept;
    CPUEngine& operator=(CPUEngine&&) noexcept;

    // Load model weights
    void load(const std::string& model_path);

    // Check if ready
    bool is_ready() const { return is_loaded_; }

    // Get engine name
    std::string name() const { return "CPU"; }

    // Warmup
    void warmup(int num_iterations = 3);

    // Run inference on stereo pair
    InferenceResult infer(const ImageView& img1, const ImageView& img2);

    // Match descriptors
    MatchResult match(
        const float* desc_1, const float* desc_2,
        int height, int width, int desc_dim,
        const MatchingConfig& config
    );

private:
    RuntimeConfig config_;
    bool is_loaded_ = false;

    // Model weights
    std::unique_ptr<ModelWeights> weights_;

    // Processing components
    std::unique_ptr<Preprocessor> preprocessor_;

    // Internal buffers
    std::vector<float> buffer_img1_;
    std::vector<float> buffer_img2_;
    std::vector<float> buffer_pts3d_1_;
    std::vector<float> buffer_pts3d_2_;
    std::vector<float> buffer_desc_1_;
    std::vector<float> buffer_desc_2_;
    std::vector<float> buffer_conf_1_;
    std::vector<float> buffer_conf_2_;

    // Internal methods
    void allocate_buffers();
    void preprocess(const ImageView& img, float* output);
    void run_encoder(const float* input, float* output);
    void run_decoder(const float* enc1, const float* enc2,
                     float* pts3d, float* desc, float* conf);
};

}  // namespace cpu
}  // namespace mast3r