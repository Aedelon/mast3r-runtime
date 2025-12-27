// MASt3R Runtime - GPU Tensor Handle
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
// Lazy-copy pattern: data stays on GPU until explicit .numpy() call.

#pragma once

#include <vector>
#include <memory>
#include <cstdint>

#ifdef __OBJC__
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif

namespace mast3r {
namespace mpsgraph {

// Forward declaration for pimpl
class GPUTensorImpl;

/// GPU Tensor Handle - lazy copy pattern
/// Data remains on GPU until numpy() is called.
/// This eliminates ~420ms of copy overhead when data isn't needed on CPU.
class GPUTensor {
public:
    GPUTensor();
    ~GPUTensor();

    // Move-only (GPU resources shouldn't be copied)
    GPUTensor(GPUTensor&& other) noexcept;
    GPUTensor& operator=(GPUTensor&& other) noexcept;
    GPUTensor(const GPUTensor&) = delete;
    GPUTensor& operator=(const GPUTensor&) = delete;

    /// Shape of the tensor
    const std::vector<int64_t>& shape() const;

    /// Number of elements
    size_t numel() const;

    /// Size in bytes
    size_t nbytes() const;

    /// Number of dimensions
    size_t ndim() const;

    /// Copy to CPU (this is the expensive operation)
    /// Returns a newly allocated float array. Caller owns the memory.
    float* to_cpu() const;

    /// Copy to pre-allocated buffer (for NumPy buffer protocol)
    void copy_to(float* dst) const;

    /// Check if tensor is valid
    bool is_valid() const;

#ifdef __OBJC__
    /// Create from MPSGraphTensorData (internal use)
    static GPUTensor from_tensor_data(MPSGraphTensorData* data, std::vector<int64_t> shape);
#endif

private:
    std::unique_ptr<GPUTensorImpl> impl_;
};

/// Result containing GPU tensor handles instead of CPU arrays
struct GPUInferenceResult {
    GPUTensor pts3d_1;
    GPUTensor pts3d_2;
    GPUTensor conf_1;
    GPUTensor conf_2;
    GPUTensor desc_1;
    GPUTensor desc_2;

    int height = 0;
    int width = 0;
    int desc_dim = 0;

    // Timing (still available)
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double total_ms = 0.0;
};

}  // namespace mpsgraph
}  // namespace mast3r
