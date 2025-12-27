// MASt3R Runtime - MPSGraph Context
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
// Shared context for MPSGraph resources (device, queue, buffer pool).

#pragma once

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <array>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>

namespace mast3r {
namespace mpsgraph {

// ============================================================================
// Buffer Pool - Reusable GPU buffers
// ============================================================================

class API_AVAILABLE(macos(15.0)) BufferPool {
public:
    explicit BufferPool(id<MTLDevice> device, size_t max_per_class = 4);
    ~BufferPool();

    // Acquire a buffer of at least the given size (O(1) with size classes)
    id<MTLBuffer> acquire(size_t size);

    // Release a buffer back to the pool
    void release(id<MTLBuffer> buffer);

    // Clear all pooled buffers
    void clear();

    // Warmup: pre-allocate buffers for expected sizes
    void warmup(const std::vector<size_t>& expected_sizes);

    // Stats
    size_t pooled_count() const;
    size_t total_bytes() const { return total_bytes_; }

private:
    // Round up to next power of 2 for size class
    static size_t next_power_of_2(size_t n);

    // Get size class index (log2 of power of 2)
    static size_t size_class_index(size_t size);

    id<MTLDevice> device_;

    // Size-class pools: index = log2(size), each holds buffers of that size class
    // Classes: 4KB, 8KB, 16KB, 32KB, 64KB, 128KB, 256KB, 512KB, 1MB, 2MB, 4MB, 8MB, 16MB, 32MB, 64MB, 128MB, 256MB, 512MB
    static constexpr size_t MIN_SIZE_CLASS = 12;  // 4KB = 2^12
    static constexpr size_t MAX_SIZE_CLASS = 29;  // 512MB = 2^29
    static constexpr size_t NUM_SIZE_CLASSES = MAX_SIZE_CLASS - MIN_SIZE_CLASS + 1;

    std::array<std::vector<id<MTLBuffer>>, NUM_SIZE_CLASSES> pools_;
    std::mutex mutex_;
    size_t max_per_class_;
    size_t total_bytes_ = 0;
};

// ============================================================================
// MPSGraph Context - Singleton for shared resources
// ============================================================================

class API_AVAILABLE(macos(15.0)) MPSGraphContext {
public:
    // Get the shared context (creates on first call)
    static std::shared_ptr<MPSGraphContext> shared();

    // Create a new isolated context (for testing/multi-device)
    static std::shared_ptr<MPSGraphContext> create();

    // Check availability
    static bool is_available();

    ~MPSGraphContext();

    // Accessors
    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }
    BufferPool& buffer_pool() { return *buffer_pool_; }

    // Device info
    std::string device_name() const;
    size_t recommended_working_set_size() const;
    size_t max_buffer_length() const;
    bool supports_family(MTLGPUFamily family) const;

    // Create a new graph (graphs are NOT shared, but context is)
    MPSGraph* create_graph();

    // Synchronization
    void synchronize();

private:
    MPSGraphContext();

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    std::unique_ptr<BufferPool> buffer_pool_;

    static std::shared_ptr<MPSGraphContext> shared_instance_;
    static std::mutex shared_mutex_;
};

// ============================================================================
// RAII Buffer Handle
// ============================================================================

class API_AVAILABLE(macos(15.0)) PooledBuffer {
public:
    PooledBuffer(std::shared_ptr<MPSGraphContext> ctx, size_t size);
    ~PooledBuffer();

    // Non-copyable, movable
    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;
    PooledBuffer(PooledBuffer&& other) noexcept;
    PooledBuffer& operator=(PooledBuffer&& other) noexcept;

    id<MTLBuffer> buffer() const { return buffer_; }
    void* contents() const { return [buffer_ contents]; }
    size_t length() const { return [buffer_ length]; }

private:
    std::shared_ptr<MPSGraphContext> ctx_;
    id<MTLBuffer> buffer_ = nil;
};

}  // namespace mpsgraph
}  // namespace mast3r
