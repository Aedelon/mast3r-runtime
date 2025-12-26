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
    DUNE_VIT_SMALL_336,
    DUNE_VIT_SMALL_448,
    DUNE_VIT_BASE_336,
    DUNE_VIT_BASE_448,
    MAST3R_VIT_LARGE
};

// Architecture type
enum class ArchType {
    DUNE,    // DINOv2 encoder + decoder (patch_size=14)
    MAST3R   // CroCoNet encoder + decoder (patch_size=16, RoPE 2D)
};

// Model specification
struct ModelSpec {
    ArchType arch;
    int patch_size;
    int embed_dim;
    int num_heads;
    int depth;
    int decoder_dim;
    int decoder_heads;
    int decoder_depth;
    int native_resolution;
    int desc_dim;

    bool is_dune() const { return arch == ArchType::DUNE; }
    bool is_mast3r() const { return arch == ArchType::MAST3R; }
    int num_patches(int res) const { return (res / patch_size) * (res / patch_size); }
};

// Get model spec for variant
inline ModelSpec get_model_spec(ModelVariant variant) {
    switch (variant) {
        case ModelVariant::DUNE_VIT_SMALL_336:
            // desc_dim=24: MLP output = (24+1) * 14^2 = 4900
            return {ArchType::DUNE, 14, 384, 6, 12, 768, 12, 12, 336, 24};
        case ModelVariant::DUNE_VIT_SMALL_448:
            return {ArchType::DUNE, 14, 384, 6, 12, 768, 12, 12, 448, 24};
        case ModelVariant::DUNE_VIT_BASE_336:
            return {ArchType::DUNE, 14, 768, 12, 12, 768, 12, 12, 336, 24};
        case ModelVariant::DUNE_VIT_BASE_448:
            return {ArchType::DUNE, 14, 768, 12, 12, 768, 12, 12, 448, 24};
        case ModelVariant::MAST3R_VIT_LARGE:
            // desc_dim=24 is the actual local_feat_dim from official MASt3R (not 256)
            // MLP output: 6400 = (24+1) * 16^2, pixel_shuffle gives 25 channels, first 24 are descriptors
            return {ArchType::MAST3R, 16, 1024, 16, 24, 768, 12, 12, 512, 24};
        default:
            return {ArchType::DUNE, 14, 384, 6, 12, 768, 12, 12, 336, 256};
    }
}

// Get variant name
inline const char* variant_name(ModelVariant v) {
    switch (v) {
        case ModelVariant::DUNE_VIT_SMALL_336: return "dune_vit_small_336";
        case ModelVariant::DUNE_VIT_SMALL_448: return "dune_vit_small_448";
        case ModelVariant::DUNE_VIT_BASE_336: return "dune_vit_base_336";
        case ModelVariant::DUNE_VIT_BASE_448: return "dune_vit_base_448";
        case ModelVariant::MAST3R_VIT_LARGE: return "mast3r_vit_large";
        default: return "unknown";
    }
}

// Parse variant from string
inline ModelVariant parse_variant(const std::string& s) {
    if (s == "dune_vit_small_336") return ModelVariant::DUNE_VIT_SMALL_336;
    if (s == "dune_vit_small_448") return ModelVariant::DUNE_VIT_SMALL_448;
    if (s == "dune_vit_base_336") return ModelVariant::DUNE_VIT_BASE_336;
    if (s == "dune_vit_base_448") return ModelVariant::DUNE_VIT_BASE_448;
    if (s == "mast3r_vit_large") return ModelVariant::MAST3R_VIT_LARGE;
    return ModelVariant::DUNE_VIT_SMALL_336;
}

// Runtime configuration
struct RuntimeConfig {
    ModelVariant variant = ModelVariant::DUNE_VIT_SMALL_336;
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

// Tensor data type
enum class DType { F32, F16, BF16, I64, I32, I16, I8, U8 };

// Single tensor storage
struct Tensor {
    std::vector<uint8_t> data;      // Raw bytes
    std::vector<int64_t> shape;
    DType dtype = DType::F32;

    size_t num_elements() const {
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }

    size_t element_size() const {
        switch (dtype) {
            case DType::F32: case DType::I32: return 4;
            case DType::F16: case DType::BF16: case DType::I16: return 2;
            case DType::I64: return 8;
            default: return 1;
        }
    }

    // Get raw pointer (for GPU upload)
    const void* raw() const { return data.data(); }
    void* raw() { return data.data(); }

    // Get typed pointers (caller must verify dtype first)
    const float* as_f32() const {
        return reinterpret_cast<const float*>(data.data());
    }

    const uint16_t* as_f16() const {
        return reinterpret_cast<const uint16_t*>(data.data());
    }

    const uint16_t* as_bf16() const {
        return reinterpret_cast<const uint16_t*>(data.data());
    }

    // Check dtype
    bool is_f32() const { return dtype == DType::F32; }
    bool is_f16() const { return dtype == DType::F16; }
    bool is_bf16() const { return dtype == DType::BF16; }

    // Convert BF16 to F16 in-place (for Metal which doesn't support BF16)
    void bf16_to_f16_inplace() {
        if (dtype != DType::BF16) return;

        auto* ptr = reinterpret_cast<uint16_t*>(data.data());
        size_t n = num_elements();

        for (size_t i = 0; i < n; i++) {
            // BF16: seeeeeee efffffff (8-bit exp, 7-bit mantissa)
            // F16:  seeeeeff ffffffff (5-bit exp, 10-bit mantissa)
            uint16_t bf = ptr[i];
            uint32_t sign = (bf >> 15) & 1;
            int32_t exp = ((bf >> 7) & 0xFF) - 127;  // BF16 exp bias
            uint32_t mant = bf & 0x7F;

            // Clamp exponent for F16 range
            if (exp > 15) {
                exp = 15;
                mant = 0x3FF;  // Inf
            } else if (exp < -14) {
                exp = -15;
                mant = 0;  // Zero/denorm
            }

            uint16_t f16 = static_cast<uint16_t>(
                (sign << 15) |
                ((exp + 15) << 10) |
                (mant << 3)
            );
            ptr[i] = f16;
        }

        dtype = DType::F16;
    }

    // Convert F32 to F16 in-place (for Metal optimization)
    void f32_to_f16_inplace() {
        if (dtype != DType::F32) return;

        size_t n = num_elements();
        std::vector<uint8_t> new_data(n * 2);  // F16 is 2 bytes

        const float* src = reinterpret_cast<const float*>(data.data());
        uint16_t* dst = reinterpret_cast<uint16_t*>(new_data.data());

        for (size_t i = 0; i < n; i++) {
            float f = src[i];
            uint32_t bits = *reinterpret_cast<uint32_t*>(&f);

            uint32_t sign = (bits >> 31) & 1;
            int32_t exp = ((bits >> 23) & 0xFF) - 127;  // F32 exp bias
            uint32_t mant = bits & 0x7FFFFF;  // 23-bit mantissa

            // Handle special cases
            if (exp == 128) {
                // Inf or NaN
                dst[i] = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));
            } else if (exp > 15) {
                // Overflow → Inf
                dst[i] = static_cast<uint16_t>((sign << 15) | 0x7C00);
            } else if (exp < -14) {
                // Underflow → zero or denorm
                if (exp >= -24) {
                    // Denormalized
                    mant = (mant | 0x800000) >> (-exp - 14 + 1);
                    dst[i] = static_cast<uint16_t>((sign << 15) | (mant >> 13));
                } else {
                    // Zero
                    dst[i] = static_cast<uint16_t>(sign << 15);
                }
            } else {
                // Normal case
                dst[i] = static_cast<uint16_t>(
                    (sign << 15) |
                    ((exp + 15) << 10) |
                    (mant >> 13)
                );
            }
        }

        data = std::move(new_data);
        dtype = DType::F16;
    }
};

// Model weights container
struct ModelWeights {
    std::unordered_map<std::string, Tensor> tensors;

    bool has(const std::string& name) const {
        return tensors.find(name) != tensors.end();
    }

    const Tensor& get(const std::string& name) const {
        return tensors.at(name);
    }

    Tensor& get(const std::string& name) {
        return tensors.at(name);
    }

    std::vector<int64_t> shape(const std::string& name) const {
        return tensors.at(name).shape;
    }

    DType dtype(const std::string& name) const {
        return tensors.at(name).dtype;
    }

    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        for (const auto& [k, _] : tensors) result.push_back(k);
        return result;
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