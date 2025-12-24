/**
 * CUDA Engine implementation - STUB.
 *
 * TODO: Implement actual CUDA inference.
 *
 * Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
 */

#include "cuda_engine.hpp"
#include "model_loader.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <chrono>

namespace mast3r {

struct CUDAEngine::Impl {
    // Weight tensors on device
    std::vector<float*> d_weights;
    ModelWeights weights;
};

CUDAEngine::CUDAEngine(
    const std::string& variant,
    int resolution,
    const std::string& precision,
    int num_threads
) : variant_(variant), resolution_(resolution), precision_(precision) {

    // Check CUDA availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    // Create stream
    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }

    // Create cuBLAS handle
    cublasStatus_t status = cublasCreate(&cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(stream_);
        throw std::runtime_error("Failed to create cuBLAS handle");
    }

    cublasSetStream(cublas_handle_, stream_);

    impl_ = std::make_unique<Impl>();
}

CUDAEngine::~CUDAEngine() {
    // Free device memory
    if (d_img1_) cudaFree(d_img1_);
    if (d_img2_) cudaFree(d_img2_);
    if (d_weights_) cudaFree(d_weights_);

    for (auto ptr : impl_->d_weights) {
        if (ptr) cudaFree(ptr);
    }

    if (cublas_handle_) cublasDestroy(cublas_handle_);
    if (stream_) cudaStreamDestroy(stream_);
}

CUDAEngine::CUDAEngine(CUDAEngine&& other) noexcept
    : impl_(std::move(other.impl_)),
      variant_(std::move(other.variant_)),
      resolution_(other.resolution_),
      precision_(std::move(other.precision_)),
      is_ready_(other.is_ready_),
      stream_(other.stream_),
      cublas_handle_(other.cublas_handle_),
      d_img1_(other.d_img1_),
      d_img2_(other.d_img2_),
      d_weights_(other.d_weights_) {
    other.stream_ = nullptr;
    other.cublas_handle_ = nullptr;
    other.d_img1_ = nullptr;
    other.d_img2_ = nullptr;
    other.d_weights_ = nullptr;
    other.is_ready_ = false;
}

CUDAEngine& CUDAEngine::operator=(CUDAEngine&& other) noexcept {
    if (this != &other) {
        // Cleanup current
        if (d_img1_) cudaFree(d_img1_);
        if (d_img2_) cudaFree(d_img2_);
        if (d_weights_) cudaFree(d_weights_);
        if (cublas_handle_) cublasDestroy(cublas_handle_);
        if (stream_) cudaStreamDestroy(stream_);

        // Move
        impl_ = std::move(other.impl_);
        variant_ = std::move(other.variant_);
        resolution_ = other.resolution_;
        precision_ = std::move(other.precision_);
        is_ready_ = other.is_ready_;
        stream_ = other.stream_;
        cublas_handle_ = other.cublas_handle_;
        d_img1_ = other.d_img1_;
        d_img2_ = other.d_img2_;
        d_weights_ = other.d_weights_;

        other.stream_ = nullptr;
        other.cublas_handle_ = nullptr;
        other.d_img1_ = nullptr;
        other.d_img2_ = nullptr;
        other.d_weights_ = nullptr;
        other.is_ready_ = false;
    }
    return *this;
}

bool CUDAEngine::is_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

std::string CUDAEngine::get_device_name() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return "Unknown";
    }
    return prop.name;
}

std::pair<int, int> CUDAEngine::get_compute_capability() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return {0, 0};
    }
    return {prop.major, prop.minor};
}

bool CUDAEngine::load_weights(const std::string& path) {
    impl_->weights = load_safetensors(path);

    // TODO: Upload weights to GPU
    // For now, just mark as ready
    is_ready_ = true;
    return true;
}

void CUDAEngine::warmup(int iterations) {
    if (!is_ready_) return;

    // Allocate dummy input
    std::vector<uint8_t> dummy(resolution_ * resolution_ * 3, 128);

    for (int i = 0; i < iterations; ++i) {
        infer(dummy.data(), resolution_, resolution_,
              dummy.data(), resolution_, resolution_);
    }

    cudaStreamSynchronize(stream_);
}

InferenceResult CUDAEngine::infer(
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

    // TODO: Implement actual CUDA inference
    // 1. Upload images to GPU
    // 2. Run preprocessing kernel
    // 3. Run encoder
    // 4. Run decoder
    // 5. Download results

    auto end = std::chrono::high_resolution_clock::now();
    result.timing_ms["total"] = std::chrono::duration<float, std::milli>(end - start).count();

    return result;
}

MatchResult CUDAEngine::match(
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

    // TODO: Implement CUDA matching
    // 1. Upload descriptors to GPU
    // 2. Compute similarity matrix with cuBLAS
    // 3. Find top-k matches
    // 4. Apply reciprocal filtering
    // 5. Download results

    auto end = std::chrono::high_resolution_clock::now();
    result.timing_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return result;
}

}  // namespace mast3r
