// MASt3R Runtime - Metal Context
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#else
// Forward declarations for C++ headers
typedef void* MTLDevicePtr;
typedef void* MTLCommandQueuePtr;
typedef void* MTLLibraryPtr;
typedef void* MTLBufferPtr;
#endif

namespace mast3r {
namespace metal {

/**
 * Metal context singleton.
 *
 * Manages Metal device, command queue, and shader library.
 * Thread-safe for command buffer creation.
 */
class MetalContext {
public:
    // Singleton access
    static MetalContext& instance();

    // Non-copyable
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    // Check if Metal is available
    bool is_available() const;

    // Get device name
    std::string device_name() const;

    // Check if unified memory (Apple Silicon)
    bool has_unified_memory() const;

#ifdef __OBJC__
    // Objective-C accessors
    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> command_queue() const { return command_queue_; }
    id<MTLLibrary> library() const { return library_; }

    // Get compute pipeline
    id<MTLComputePipelineState> get_pipeline(const std::string& name);

    // Create buffer
    id<MTLBuffer> create_buffer(size_t size, bool shared = true);

    // Create buffer from data
    id<MTLBuffer> create_buffer(const void* data, size_t size, bool shared = true);
#endif

private:
    MetalContext();
    ~MetalContext();

#ifdef __OBJC__
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> command_queue_ = nil;
    id<MTLLibrary> library_ = nil;

    // Pipeline cache
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
#endif

    bool is_available_ = false;
};

}  // namespace metal
}  // namespace mast3r