/**
 * Jetson Engine - TensorRT + DLA inference backend.
 *
 * Optimized for NVIDIA Jetson Orin with DLA offloading.
 *
 * Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "types.hpp"

namespace mast3r {

class JetsonEngine {
public:
    JetsonEngine(
        const std::string& variant,
        int resolution,
        const std::string& precision,
        int num_threads = 4
    );
    ~JetsonEngine();

    // Non-copyable
    JetsonEngine(const JetsonEngine&) = delete;
    JetsonEngine& operator=(const JetsonEngine&) = delete;

    // Engine interface
    bool load_weights(const std::string& path);
    bool load_engine(const std::string& engine_path);
    bool build_engine(const std::string& onnx_path, const std::string& engine_path);
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
    static int get_dla_count();
    static bool has_dla();

    bool is_ready() const { return is_ready_; }
    std::string name() const { return "Jetson (TensorRT + DLA)"; }

    // DLA configuration
    void set_dla_core(int core);
    int get_dla_core() const { return dla_core_; }

private:
    struct TRTDeleter {
        template <typename T>
        void operator()(T* ptr) const {
            if (ptr) ptr->destroy();
        }
    };

    template <typename T>
    using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

    std::string variant_;
    int resolution_;
    std::string precision_;
    ModelSpec spec_;
    bool is_ready_ = false;
    int dla_core_ = -1;  // -1 = GPU, 0/1 = DLA core

    // TensorRT objects
    TRTUniquePtr<nvinfer1::IRuntime> runtime_;
    TRTUniquePtr<nvinfer1::ICudaEngine> engine_;
    TRTUniquePtr<nvinfer1::IExecutionContext> context_;

    // CUDA resources
    cudaStream_t stream_ = nullptr;
    std::vector<void*> bindings_;
    std::vector<size_t> binding_sizes_;
};

}  // namespace mast3r
