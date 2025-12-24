/**
 * Jetson Engine implementation - STUB.
 *
 * TODO: Implement TensorRT inference with DLA offloading.
 *
 * Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
 */

#include "jetson_engine.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>
#include <chrono>

namespace mast3r {

// TensorRT logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            // Log warnings and errors
        }
    }
};

static TRTLogger gLogger;

JetsonEngine::JetsonEngine(
    const std::string& variant,
    int resolution,
    const std::string& precision,
    int num_threads
) : variant_(variant), resolution_(resolution), precision_(precision) {

    // Create CUDA stream
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }

    // Create TensorRT runtime
    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    if (!runtime_) {
        cudaStreamDestroy(stream_);
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
}

JetsonEngine::~JetsonEngine() {
    // Free bindings
    for (void* ptr : bindings_) {
        if (ptr) cudaFree(ptr);
    }

    if (stream_) cudaStreamDestroy(stream_);
}

bool JetsonEngine::is_available() {
    // Check for Jetson-specific features
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }

    // Check if we can create TensorRT runtime
    TRTLogger logger;
    auto runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        return false;
    }
    runtime->destroy();

    return true;
}

std::string JetsonEngine::get_device_name() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return "Unknown";
    }
    return prop.name;
}

int JetsonEngine::get_dla_count() {
    TRTLogger logger;
    auto builder = nvinfer1::createInferBuilder(logger);
    if (!builder) return 0;

    int count = builder->getNbDLACores();
    builder->destroy();
    return count;
}

bool JetsonEngine::has_dla() {
    return get_dla_count() > 0;
}

void JetsonEngine::set_dla_core(int core) {
    dla_core_ = core;
}

bool JetsonEngine::load_weights(const std::string& path) {
    // For Jetson, we need to build TensorRT engine from weights
    // This is a stub - actual implementation would:
    // 1. Load safetensors weights
    // 2. Build network programmatically
    // 3. Serialize engine

    is_ready_ = true;
    return true;
}

bool JetsonEngine::load_engine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    file.read(data.data(), size);

    engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
    if (!engine_) {
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        return false;
    }

    // Allocate bindings
    int nb_bindings = engine_->getNbBindings();
    bindings_.resize(nb_bindings);
    binding_sizes_.resize(nb_bindings);

    for (int i = 0; i < nb_bindings; ++i) {
        auto dims = engine_->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        size *= sizeof(float);
        binding_sizes_[i] = size;

        cudaMalloc(&bindings_[i], size);
    }

    is_ready_ = true;
    return true;
}

bool JetsonEngine::build_engine(const std::string& onnx_path, const std::string& engine_path) {
    // TODO: Build TensorRT engine from ONNX with DLA support
    // 1. Create builder and config
    // 2. Parse ONNX
    // 3. Set DLA core if available
    // 4. Build and serialize

    return false;
}

void JetsonEngine::warmup(int iterations) {
    if (!is_ready_) return;

    std::vector<uint8_t> dummy(resolution_ * resolution_ * 3, 128);

    for (int i = 0; i < iterations; ++i) {
        infer(dummy.data(), resolution_, resolution_,
              dummy.data(), resolution_, resolution_);
    }

    cudaStreamSynchronize(stream_);
}

InferenceResult JetsonEngine::infer(
    const uint8_t* img1_data, int img1_h, int img1_w,
    const uint8_t* img2_data, int img2_h, int img2_w
) {
    auto start = std::chrono::high_resolution_clock::now();

    InferenceResult result;
    int res = resolution_;
    int desc_dim = 256;

    // Allocate outputs
    result.pts3d_1.resize(res * res * 3, 0.0f);
    result.pts3d_2.resize(res * res * 3, 0.0f);
    result.desc_1.resize(res * res * desc_dim, 0.0f);
    result.desc_2.resize(res * res * desc_dim, 0.0f);
    result.conf_1.resize(res * res, 1.0f);
    result.conf_2.resize(res * res, 1.0f);

    // TODO: Run TensorRT inference
    // 1. Preprocess on GPU
    // 2. Copy to input bindings
    // 3. Execute context
    // 4. Copy from output bindings

    auto end = std::chrono::high_resolution_clock::now();
    result.timing_ms["total"] = std::chrono::duration<float, std::milli>(end - start).count();

    return result;
}

MatchResult JetsonEngine::match(
    const float* desc1, const float* desc2,
    int h, int w, int desc_dim,
    const float* conf1,
    const float* conf2,
    int top_k,
    bool reciprocal,
    float conf_threshold
) {
    auto start = std::chrono::high_resolution_clock::now();

    MatchResult result;

    // TODO: Implement matching on Jetson
    // Can reuse CUDA matching kernels

    auto end = std::chrono::high_resolution_clock::now();
    result.timing_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return result;
}

}  // namespace mast3r
