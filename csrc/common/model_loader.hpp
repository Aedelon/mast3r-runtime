// MASt3R Runtime - Model weight loader
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "types.hpp"

namespace mast3r {

// Supported weight formats
enum class WeightFormat {
    SAFETENSORS,  // HuggingFace safetensors
    GGUF,         // GGML format (quantized)
    NPZ,          // NumPy format
    RAW           // Raw binary
};

// Model loader interface
class ModelLoader {
public:
    virtual ~ModelLoader() = default;

    // Load weights from file
    virtual std::unique_ptr<ModelWeights> load(
        const std::filesystem::path& path
    ) = 0;

    // Check if format is supported
    virtual bool supports(WeightFormat format) const = 0;

    // Factory
    static std::unique_ptr<ModelLoader> create(WeightFormat format);
};

// Safetensors loader
class SafetensorsLoader : public ModelLoader {
public:
    std::unique_ptr<ModelWeights> load(
        const std::filesystem::path& path
    ) override;

    bool supports(WeightFormat format) const override {
        return format == WeightFormat::SAFETENSORS;
    }
};

// Get default model path for variant
std::filesystem::path get_default_model_path(
    ModelVariant variant,
    Precision precision = Precision::FP16
);

// Get cache directory
std::filesystem::path get_cache_dir();

// Load DUNE model (encoder + decoder)
std::unique_ptr<ModelWeights> load_dune_model(
    ModelVariant variant,
    Precision precision = Precision::FP16
);

// Load MASt3R unified model (main weights only)
std::unique_ptr<ModelWeights> load_mast3r_model(
    Precision precision = Precision::FP16
);

// Complete MASt3R model with all components
struct MASt3RModel {
    std::unique_ptr<ModelWeights> main_weights;       // unified.safetensors
    std::unique_ptr<ModelWeights> retrieval_weights;  // retrieval.safetensors
    std::filesystem::path codebook_path;              // codebook.pkl (loaded by Python)

    bool has_retrieval = false;
    bool has_codebook = false;
};

// Load complete MASt3R model with retrieval head
MASt3RModel load_mast3r_complete(
    Precision precision = Precision::FP16
);

}  // namespace mast3r