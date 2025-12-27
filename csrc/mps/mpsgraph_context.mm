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
// BufferPool Implementation (Size-Class Pool for O(1) allocation)
// ============================================================================

BufferPool::BufferPool(id<MTLDevice> device, size_t max_per_class)
    : device_(device), max_per_class_(max_per_class) {}

BufferPool::~BufferPool() {
    clear();
}

size_t BufferPool::next_power_of_2(size_t n) {
    if (n == 0) return 1ULL << MIN_SIZE_CLASS;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    // Clamp to min/max size class
    if (n < (1ULL << MIN_SIZE_CLASS)) n = 1ULL << MIN_SIZE_CLASS;
    if (n > (1ULL << MAX_SIZE_CLASS)) n = 1ULL << MAX_SIZE_CLASS;
    return n;
}

size_t BufferPool::size_class_index(size_t size) {
    size_t power = next_power_of_2(size);
    // Count trailing zeros to get log2
    size_t log2 = 0;
    while ((power >> log2) > 1) log2++;
    // Clamp to valid range
    if (log2 < MIN_SIZE_CLASS) log2 = MIN_SIZE_CLASS;
    if (log2 > MAX_SIZE_CLASS) log2 = MAX_SIZE_CLASS;
    return log2 - MIN_SIZE_CLASS;
}

id<MTLBuffer> BufferPool::acquire(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t idx = size_class_index(size);
    auto& pool = pools_[idx];

    // O(1) pop from size class
    if (!pool.empty()) {
        id<MTLBuffer> buffer = pool.back();
        pool.pop_back();
        return buffer;
    }

    // Allocate new buffer at size class granularity
    size_t alloc_size = 1ULL << (idx + MIN_SIZE_CLASS);
    id<MTLBuffer> buffer = [device_ newBufferWithLength:alloc_size
                                                options:MTLResourceStorageModeShared];
    total_bytes_ += alloc_size;
    return buffer;
}

void BufferPool::release(id<MTLBuffer> buffer) {
    if (buffer == nil) return;

    std::lock_guard<std::mutex> lock(mutex_);

    size_t idx = size_class_index([buffer length]);
    auto& pool = pools_[idx];

    if (pool.size() < max_per_class_) {
        pool.push_back(buffer);
    } else {
        // Pool full, let buffer be deallocated
        total_bytes_ -= [buffer length];
    }
}

void BufferPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pool : pools_) {
        pool.clear();
    }
    total_bytes_ = 0;
}

void BufferPool::warmup(const std::vector<size_t>& expected_sizes) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t size : expected_sizes) {
        size_t idx = size_class_index(size);
        size_t alloc_size = 1ULL << (idx + MIN_SIZE_CLASS);

        // Pre-allocate 2 buffers per size class for double-buffering
        for (int i = 0; i < 2; i++) {
            id<MTLBuffer> buffer = [device_ newBufferWithLength:alloc_size
                                                        options:MTLResourceStorageModeShared];
            pools_[idx].push_back(buffer);
            total_bytes_ += alloc_size;
        }
    }
}

size_t BufferPool::pooled_count() const {
    size_t count = 0;
    for (const auto& pool : pools_) {
        count += pool.size();
    }
    return count;
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
