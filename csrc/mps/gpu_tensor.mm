// MASt3R Runtime - GPU Tensor Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#import "gpu_tensor.hpp"
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace mast3r {
namespace mpsgraph {

// ============================================================================
// Implementation class (pimpl pattern for Objective-C++ isolation)
// ============================================================================

class GPUTensorImpl {
public:
    // ARC manages the lifetime automatically - no manual retain/release
    MPSGraphTensorData* tensor_data_ = nil;
    std::vector<int64_t> shape_;
    size_t numel_ = 0;

    GPUTensorImpl() = default;

    GPUTensorImpl(MPSGraphTensorData* data, std::vector<int64_t> shape)
        : tensor_data_(data), shape_(std::move(shape)) {  // ARC retains automatically
        numel_ = 1;
        for (auto dim : shape_) {
            numel_ *= dim;
        }
    }

    ~GPUTensorImpl() {
        // ARC releases automatically when tensor_data_ goes out of scope
        tensor_data_ = nil;
    }

    // Non-copyable
    GPUTensorImpl(const GPUTensorImpl&) = delete;
    GPUTensorImpl& operator=(const GPUTensorImpl&) = delete;

    // Movable
    GPUTensorImpl(GPUTensorImpl&& other) noexcept
        : tensor_data_(other.tensor_data_),
          shape_(std::move(other.shape_)),
          numel_(other.numel_) {
        other.tensor_data_ = nil;
        other.numel_ = 0;
    }

    GPUTensorImpl& operator=(GPUTensorImpl&& other) noexcept {
        if (this != &other) {
            // ARC handles the old tensor_data_ release
            tensor_data_ = other.tensor_data_;
            shape_ = std::move(other.shape_);
            numel_ = other.numel_;
            other.tensor_data_ = nil;
            other.numel_ = 0;
        }
        return *this;
    }

    float* to_cpu() const {
        if (!tensor_data_) return nullptr;

        float* result = new float[numel_];
        [[tensor_data_ mpsndarray] readBytes:result strideBytes:nil];
        return result;
    }

    void copy_to(float* dst) const {
        if (!tensor_data_ || !dst) return;
        [[tensor_data_ mpsndarray] readBytes:dst strideBytes:nil];
    }

    bool is_valid() const {
        return tensor_data_ != nil;
    }
};

// ============================================================================
// GPUTensor public interface
// ============================================================================

GPUTensor::GPUTensor() : impl_(std::make_unique<GPUTensorImpl>()) {}

GPUTensor::~GPUTensor() = default;

GPUTensor::GPUTensor(GPUTensor&& other) noexcept = default;
GPUTensor& GPUTensor::operator=(GPUTensor&& other) noexcept = default;

const std::vector<int64_t>& GPUTensor::shape() const {
    return impl_->shape_;
}

size_t GPUTensor::numel() const {
    return impl_->numel_;
}

size_t GPUTensor::nbytes() const {
    return impl_->numel_ * sizeof(float);
}

size_t GPUTensor::ndim() const {
    return impl_->shape_.size();
}

float* GPUTensor::to_cpu() const {
    return impl_->to_cpu();
}

void GPUTensor::copy_to(float* dst) const {
    impl_->copy_to(dst);
}

bool GPUTensor::is_valid() const {
    return impl_ && impl_->is_valid();
}

GPUTensor GPUTensor::from_tensor_data(MPSGraphTensorData* data, std::vector<int64_t> shape) {
    GPUTensor tensor;
    tensor.impl_ = std::make_unique<GPUTensorImpl>(data, std::move(shape));
    return tensor;
}

}  // namespace mpsgraph
}  // namespace mast3r
