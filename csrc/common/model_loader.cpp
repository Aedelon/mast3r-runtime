// MASt3R Runtime - Model weight loader implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include "model_loader.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace mast3r {

std::unique_ptr<ModelLoader> ModelLoader::create(WeightFormat format) {
    switch (format) {
        case WeightFormat::SAFETENSORS:
            return std::make_unique<SafetensorsLoader>();
        default:
            throw std::runtime_error("Unsupported weight format");
    }
}

std::unique_ptr<ModelWeights> SafetensorsLoader::load(
    const std::filesystem::path& path
) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Model file not found: " + path.string());
    }

    auto weights = std::make_unique<ModelWeights>();

    // Open file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open: " + path.string());
    }

    // Read header size (first 8 bytes, little-endian uint64)
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    // Read header JSON
    std::string header_json(header_size, '\0');
    file.read(header_json.data(), header_size);

    // TODO: Parse header JSON to get tensor metadata
    // For now, this is a stub - actual implementation needs JSON parsing

    // The header contains:
    // {
    //   "tensor_name": {
    //     "dtype": "F32" | "F16" | "BF16",
    //     "shape": [dim1, dim2, ...],
    //     "data_offsets": [start, end]
    //   },
    //   ...
    // }

    return weights;
}

std::filesystem::path get_cache_dir() {
    // Check environment variable first
    if (const char* env = std::getenv("MAST3R_CACHE_DIR")) {
        return std::filesystem::path(env);
    }

    // Default: ~/.cache/mast3r_runtime
    std::filesystem::path home;

#ifdef _WIN32
    if (const char* userprofile = std::getenv("USERPROFILE")) {
        home = userprofile;
    }
#else
    if (const char* home_env = std::getenv("HOME")) {
        home = home_env;
    }
#endif

    return home / ".cache" / "mast3r_runtime";
}

std::filesystem::path get_default_model_path(
    ModelVariant variant,
    Precision precision
) {
    std::string variant_str;
    switch (variant) {
        case ModelVariant::DUNE_VIT_SMALL_14:
            variant_str = "dune_vit_small_14";
            break;
        case ModelVariant::DUNE_VIT_BASE_14:
            variant_str = "dune_vit_base_14";
            break;
        case ModelVariant::MAST3R_VIT_LARGE:
            variant_str = "mast3r_vit_large";
            break;
        case ModelVariant::DUST3R_224_LINEAR:
            variant_str = "dust3r_224_linear";
            break;
    }

    std::string precision_str;
    switch (precision) {
        case Precision::FP32:
            precision_str = "fp32";
            break;
        case Precision::FP16:
            precision_str = "fp16";
            break;
        case Precision::INT8:
            precision_str = "int8";
            break;
        case Precision::INT4:
            precision_str = "int4";
            break;
    }

    std::string filename = variant_str + "_" + precision_str + ".safetensors";
    return get_cache_dir() / "models" / filename;
}

}  // namespace mast3r