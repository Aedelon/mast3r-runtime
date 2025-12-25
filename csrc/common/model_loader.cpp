// MASt3R Runtime - Model weight loader implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include "model_loader.hpp"
#include "safetensors.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace mast3r {

namespace {

// Convert safetensors dtype to our dtype
DType convert_dtype(safetensors::DType st_dtype) {
    switch (st_dtype) {
        case safetensors::DType::F32: return DType::F32;
        case safetensors::DType::F16: return DType::F16;
        case safetensors::DType::BF16: return DType::BF16;
        case safetensors::DType::I64: return DType::I64;
        case safetensors::DType::I32: return DType::I32;
        case safetensors::DType::I16: return DType::I16;
        case safetensors::DType::I8: return DType::I8;
        case safetensors::DType::U8: return DType::U8;
        default: return DType::F32;
    }
}

}  // namespace

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

    // Open safetensors file
    safetensors::SafetensorsFile st_file(path.string());

    // Load all tensors (keep original dtype, no conversion)
    for (const auto& name : st_file.tensor_names()) {
        const auto& info = st_file.tensor_info(name);

        Tensor tensor;
        tensor.shape = info.shape;
        tensor.dtype = convert_dtype(info.dtype);
        tensor.data = st_file.load_tensor_raw(name);

        weights->tensors[name] = std::move(tensor);
    }

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
    Precision /* precision */
) {
    std::string variant_str;
    switch (variant) {
        case ModelVariant::DUNE_VIT_SMALL_336:
            variant_str = "dune_vit_small_336";
            break;
        case ModelVariant::DUNE_VIT_SMALL_448:
            variant_str = "dune_vit_small_448";
            break;
        case ModelVariant::DUNE_VIT_BASE_336:
            variant_str = "dune_vit_base_336";
            break;
        case ModelVariant::DUNE_VIT_BASE_448:
            variant_str = "dune_vit_base_448";
            break;
        case ModelVariant::MAST3R_VIT_LARGE:
            variant_str = "mast3r_vit_large";
            break;
    }

    // Return path to safetensors directory
    return get_cache_dir() / "safetensors" / variant_str;
}

// Load encoder + decoder for DUNE models
std::unique_ptr<ModelWeights> load_dune_model(
    ModelVariant variant,
    Precision precision
) {
    auto base_path = get_default_model_path(variant, precision);

    auto encoder_path = base_path / "encoder.safetensors";
    auto decoder_path = base_path / "decoder.safetensors";

    if (!std::filesystem::exists(encoder_path)) {
        throw std::runtime_error("Encoder not found: " + encoder_path.string());
    }
    if (!std::filesystem::exists(decoder_path)) {
        throw std::runtime_error("Decoder not found: " + decoder_path.string());
    }

    auto loader = ModelLoader::create(WeightFormat::SAFETENSORS);

    // Load encoder
    auto encoder_weights = loader->load(encoder_path);

    // Load decoder and merge
    auto decoder_weights = loader->load(decoder_path);

    // Merge into single weights object with prefixes
    auto weights = std::make_unique<ModelWeights>();

    for (auto& [name, tensor] : encoder_weights->tensors) {
        weights->tensors["encoder." + name] = std::move(tensor);
    }

    for (auto& [name, tensor] : decoder_weights->tensors) {
        weights->tensors["decoder." + name] = std::move(tensor);
    }

    return weights;
}

// Load unified MASt3R model (main model only)
std::unique_ptr<ModelWeights> load_mast3r_model(
    Precision precision
) {
    auto base_path = get_default_model_path(ModelVariant::MAST3R_VIT_LARGE, precision);

    auto unified_path = base_path / "unified.safetensors";

    if (!std::filesystem::exists(unified_path)) {
        throw std::runtime_error("Model not found: " + unified_path.string());
    }

    auto loader = ModelLoader::create(WeightFormat::SAFETENSORS);
    return loader->load(unified_path);
}

// Load complete MASt3R model with retrieval head
MASt3RModel load_mast3r_complete(Precision precision) {
    auto base_path = get_default_model_path(ModelVariant::MAST3R_VIT_LARGE, precision);

    MASt3RModel model;

    // Load unified model
    auto unified_path = base_path / "unified.safetensors";
    if (!std::filesystem::exists(unified_path)) {
        throw std::runtime_error("Model not found: " + unified_path.string());
    }

    auto loader = ModelLoader::create(WeightFormat::SAFETENSORS);
    model.main_weights = loader->load(unified_path);

    // Load retrieval head (optional)
    auto retrieval_path = base_path / "retrieval.safetensors";
    if (std::filesystem::exists(retrieval_path)) {
        model.retrieval_weights = loader->load(retrieval_path);
        model.has_retrieval = true;
    }

    // Codebook path (loaded separately, it's a pickle)
    auto codebook_path = base_path / "codebook.pkl";
    if (std::filesystem::exists(codebook_path)) {
        model.codebook_path = codebook_path;
        model.has_codebook = true;
    }

    return model;
}

}  // namespace mast3r
