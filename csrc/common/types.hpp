// MASt3R Runtime - Common types
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mast3r {

// Precision types
enum class Precision { FP32, FP16, INT8, INT4 };

// Model variants
enum class ModelVariant {
    DUNE_VIT_SMALL_14,
    DUNE_VIT_BASE_14,
    MAST3R_VIT_LARGE,
    DUST3R_224_LINEAR
};

// Runtime configuration
struct RuntimeConfig {
    ModelVariant variant = ModelVariant::DUNE_VIT_SMALL_14;
    int resolution = 336;
    Precision precision = Precision::FP16;
    int num_threads = 4;
    bool use_gpu = true;
};

// Matching configuration
struct MatchingConfig {
    int top_k = 512;
    bool reciprocal = true;
    float confidence_threshold = 0.5f;
};

// Inference result (views into memory, no ownership)
struct InferenceResult {
    // 3D points [H, W, 3]
    float* pts3d_1 = nullptr;
    float* pts3d_2 = nullptr;

    // Descriptors [H, W, D]
    float* desc_1 = nullptr;
    float* desc_2 = nullptr;

    // Confidence [H, W]
    float* conf_1 = nullptr;
    float* conf_2 = nullptr;

    // Dimensions
    int height = 0;
    int width = 0;
    int desc_dim = 0;

    // Timing
    float preprocess_ms = 0.0f;
    float inference_ms = 0.0f;
    float total_ms = 0.0f;
};

// Match result
struct MatchResult {
    std::vector<int64_t> idx_1;
    std::vector<int64_t> idx_2;
    std::vector<float> pts2d_1;  // [N, 2] flattened
    std::vector<float> pts2d_2;  // [N, 2] flattened
    std::vector<float> pts3d_1;  // [N, 3] flattened
    std::vector<float> pts3d_2;  // [N, 3] flattened
    std::vector<float> confidence;

    float match_ms = 0.0f;

    size_t num_matches() const { return idx_1.size(); }
};

// Model weights container
struct ModelWeights {
    std::unordered_map<std::string, std::vector<float>> tensors;
    std::unordered_map<std::string, std::vector<int64_t>> shapes;

    bool has(const std::string& name) const {
        return tensors.find(name) != tensors.end();
    }

    const float* data(const std::string& name) const {
        return tensors.at(name).data();
    }

    std::vector<int64_t> shape(const std::string& name) const {
        return shapes.at(name);
    }
};

// Image data (non-owning view)
struct ImageView {
    const uint8_t* data = nullptr;
    int height = 0;
    int width = 0;
    int channels = 3;

    size_t size_bytes() const {
        return static_cast<size_t>(height) * width * channels;
    }
};

}  // namespace mast3r