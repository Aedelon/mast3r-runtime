// MASt3R Runtime - MPSGraph Context Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#import "mpsgraph_context.hpp"

namespace mast3r {
namespace mpsgraph {

// ============================================================================
// Static members
// ============================================================================

std::shared_ptr<MPSGraphContext> MPSGraphContext::shared_instance_;
std::mutex MPSGraphContext::shared_mutex_;

// ============================================================================
// BufferPool Implementation
// ============================================================================

BufferPool::BufferPool(id<MTLDevice> device, size_t max_buffers)
    : device_(device), max_buffers_(max_buffers) {}

BufferPool::~BufferPool() {
    clear();
}

id<MTLBuffer> BufferPool::acquire(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find a buffer that fits
    for (auto it = available_.begin(); it != available_.end(); ++it) {
        if ([*it length] >= size) {
            id<MTLBuffer> buffer = *it;
            available_.erase(it);
            return buffer;
        }
    }

    // Create new buffer
    id<MTLBuffer> buffer = [device_ newBufferWithLength:size
                                                options:MTLResourceStorageModeShared];
    total_bytes_ += size;
    return buffer;
}

void BufferPool::release(id<MTLBuffer> buffer) {
    if (buffer == nil) return;

    std::lock_guard<std::mutex> lock(mutex_);

    if (available_.size() < max_buffers_) {
        available_.push_back(buffer);
    } else {
        // Pool full, let buffer be deallocated
        total_bytes_ -= [buffer length];
    }
}

void BufferPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    available_.clear();
    total_bytes_ = 0;
}

// ============================================================================
// MPSGraphContext Implementation
// ============================================================================

MPSGraphContext::MPSGraphContext() {
    if (@available(macOS 15.0, *)) {
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) {
            throw std::runtime_error("No Metal device available");
        }

        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) {
            throw std::runtime_error("Failed to create Metal command queue");
        }

        buffer_pool_ = std::make_unique<BufferPool>(device_);
    } else {
        throw std::runtime_error("MPSGraph SDPA requires macOS 15.0+");
    }
}

MPSGraphContext::~MPSGraphContext() {
    // Synchronize before cleanup
    synchronize();
}

std::shared_ptr<MPSGraphContext> MPSGraphContext::shared() {
    std::lock_guard<std::mutex> lock(shared_mutex_);

    if (!shared_instance_) {
        shared_instance_ = std::shared_ptr<MPSGraphContext>(new MPSGraphContext());
    }

    return shared_instance_;
}

std::shared_ptr<MPSGraphContext> MPSGraphContext::create() {
    return std::shared_ptr<MPSGraphContext>(new MPSGraphContext());
}

bool MPSGraphContext::is_available() {
    if (@available(macOS 15.0, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
    return false;
}

std::string MPSGraphContext::device_name() const {
    if (@available(macOS 15.0, *)) {
        return std::string([[device_ name] UTF8String]);
    }
    return "unknown";
}

size_t MPSGraphContext::recommended_working_set_size() const {
    if (@available(macOS 15.0, *)) {
        return [device_ recommendedMaxWorkingSetSize];
    }
    return 0;
}

size_t MPSGraphContext::max_buffer_length() const {
    if (@available(macOS 15.0, *)) {
        return [device_ maxBufferLength];
    }
    return 0;
}

bool MPSGraphContext::supports_family(MTLGPUFamily family) const {
    if (@available(macOS 15.0, *)) {
        return [device_ supportsFamily:family];
    }
    return false;
}

MPSGraph* MPSGraphContext::create_graph() {
    if (@available(macOS 15.0, *)) {
        return [[MPSGraph alloc] init];
    }
    return nil;
}

void MPSGraphContext::synchronize() {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
}

// ============================================================================
// PooledBuffer Implementation
// ============================================================================

PooledBuffer::PooledBuffer(std::shared_ptr<MPSGraphContext> ctx, size_t size)
    : ctx_(std::move(ctx)) {
    if (@available(macOS 15.0, *)) {
        buffer_ = ctx_->buffer_pool().acquire(size);
    }
}

PooledBuffer::~PooledBuffer() {
    if (@available(macOS 15.0, *)) {
        if (buffer_ && ctx_) {
            ctx_->buffer_pool().release(buffer_);
        }
    }
}

PooledBuffer::PooledBuffer(PooledBuffer&& other) noexcept
    : ctx_(std::move(other.ctx_)), buffer_(other.buffer_) {
    other.buffer_ = nil;
}

PooledBuffer& PooledBuffer::operator=(PooledBuffer&& other) noexcept {
    if (this != &other) {
        if (@available(macOS 15.0, *)) {
            if (buffer_ && ctx_) {
                ctx_->buffer_pool().release(buffer_);
            }
        }
        ctx_ = std::move(other.ctx_);
        buffer_ = other.buffer_;
        other.buffer_ = nil;
    }
    return *this;
}

}  // namespace mpsgraph
}  // namespace mast3r
