/**
 * CUDA Engine - NVIDIA GPU inference backend.
 *
 * Uses cuBLAS for matrix operations and custom CUDA kernels.
 *
 * Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <string>
#include <vector>

#include "types.hpp"

namespace mast3r {

class CUDAEngine {
public:
    CUDAEngine(
        const std::string& variant,
        int resolution,
        const std::string& precision,
        int num_threads = 4
    );
    ~CUDAEngine();

    // Non-copyable
    CUDAEngine(const CUDAEngine&) = delete;
    CUDAEngine& operator=(const CUDAEngine&) = delete;

    // Move semantics
    CUDAEngine(CUDAEngine&&) noexcept;
    CUDAEngine& operator=(CUDAEngine&&) noexcept;

    // Engine interface
    bool load_weights(const std::string& path);
    void warmup(int iterations = 3);

    InferenceResult infer(
        const uint8_t* img1_data, int img1_h, int img1_w,
        const uint8_t* img2_data, int img2_h, int img2_w
    );

    MatchResult match(
        const float* desc1, const float* desc2,
        int h, int w, int desc_dim,
        const float* conf1 = nullptr,
        const float* conf2 = nullptr,
        int top_k = 512,
        bool reciprocal = true,
        float conf_threshold = 0.5f
    );

    // Queries
    static bool is_available();
    static std::string get_device_name();
    static std::pair<int, int> get_compute_capability();

    bool is_ready() const { return is_ready_; }
    std::string name() const { return "CUDA (cuBLAS)"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::string variant_;
    int resolution_;
    std::string precision_;
    ModelSpec spec_;
    bool is_ready_ = false;

    // CUDA handles
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;

    // Device memory buffers
    float* d_img1_ = nullptr;
    float* d_img2_ = nullptr;
    float* d_weights_ = nullptr;
};

}  // namespace mast3r