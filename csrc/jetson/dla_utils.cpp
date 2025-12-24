/**
 * DLA (Deep Learning Accelerator) utilities - STUB.
 *
 * Jetson Orin has 2 DLA cores that can run inference
 * in parallel with the GPU for maximum throughput.
 *
 * Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
 */

#include <NvInfer.h>
#include <string>
#include <vector>

namespace mast3r {
namespace dla {

/**
 * Check which layers can run on DLA.
 *
 * Not all TensorRT layers are DLA-compatible.
 * This function analyzes the network and reports compatibility.
 */
struct LayerCompatibility {
    std::string name;
    std::string type;
    bool dla_compatible;
    std::string reason;
};

std::vector<LayerCompatibility> analyze_dla_compatibility(
    nvinfer1::INetworkDefinition* network
) {
    std::vector<LayerCompatibility> results;

    // TODO: Iterate layers and check DLA compatibility
    // DLA-compatible layers (Orin):
    // - Convolution (with restrictions)
    // - Pooling
    // - Activation (ReLU, Sigmoid, Tanh)
    // - ElementWise
    // - Scale
    // - Softmax
    //
    // NOT compatible:
    // - MatMul (use Convolution instead)
    // - LayerNorm (decompose into primitives)
    // - GELU (approximate with supported activations)

    return results;
}

/**
 * Configure builder for DLA execution.
 */
void configure_dla(
    nvinfer1::IBuilderConfig* config,
    int dla_core,
    bool allow_gpu_fallback
) {
    if (dla_core < 0) return;

    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config->setDLACore(dla_core);

    if (allow_gpu_fallback) {
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }

    // Enable FP16 for DLA (required)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
}

/**
 * Profile DLA vs GPU execution.
 */
struct ProfilingResult {
    float gpu_ms;
    float dla_ms;
    float speedup;
    int layers_on_dla;
    int layers_on_gpu;
};

ProfilingResult profile_dla_execution(
    nvinfer1::ICudaEngine* engine,
    nvinfer1::IExecutionContext* context,
    int warmup_iterations,
    int profile_iterations
) {
    ProfilingResult result = {};

    // TODO: Profile execution
    // 1. Run warmup
    // 2. Time GPU-only execution
    // 3. Time DLA execution
    // 4. Report layer distribution

    return result;
}

}  // namespace dla
}  // namespace mast3r
