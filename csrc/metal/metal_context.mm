// MASt3R Runtime - Metal Context Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#import "metal_context.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace mast3r {
namespace metal {

MetalContext& MetalContext::instance() {
    static MetalContext instance;
    return instance;
}

MetalContext::MetalContext() {
    @autoreleasepool {
        // Get default Metal device
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            NSLog(@"[mast3r] Metal not available");
            is_available_ = false;
            return;
        }

        is_available_ = true;
        NSLog(@"[mast3r] Metal device: %@", device_.name);

        // Create command queue
        command_queue_ = [device_ newCommandQueue];
        if (!command_queue_) {
            NSLog(@"[mast3r] Failed to create command queue");
            is_available_ = false;
            return;
        }

        // Load shader library
        NSError* error = nil;
        NSString* libraryPath = @MAST3R_METALLIB_PATH;

        // Check if metallib exists at compile-time path
        if ([[NSFileManager defaultManager] fileExistsAtPath:libraryPath]) {
            NSURL* libraryURL = [NSURL fileURLWithPath:libraryPath];
            library_ = [device_ newLibraryWithURL:libraryURL error:&error];
        } else {
            // Try loading from bundle
            NSBundle* bundle = [NSBundle mainBundle];
            libraryPath = [bundle pathForResource:@"mast3r" ofType:@"metallib"];
            if (libraryPath) {
                NSURL* libraryURL = [NSURL fileURLWithPath:libraryPath];
                library_ = [device_ newLibraryWithURL:libraryURL error:&error];
            }
        }

        if (!library_) {
            NSLog(@"[mast3r] Failed to load Metal library: %@", error);
            // Continue without shaders - will use MPSGraph
        } else {
            NSLog(@"[mast3r] Metal library loaded");
        }
    }
}

MetalContext::~MetalContext() {
    @autoreleasepool {
        pipelines_.clear();
        library_ = nil;
        command_queue_ = nil;
        device_ = nil;
    }
}

bool MetalContext::is_available() const {
    return is_available_;
}

std::string MetalContext::device_name() const {
    if (!device_) return "Unknown";

    @autoreleasepool {
        return std::string([device_.name UTF8String]);
    }
}

bool MetalContext::has_unified_memory() const {
    if (!device_) return false;
    return device_.hasUnifiedMemory;
}

id<MTLComputePipelineState> MetalContext::get_pipeline(const std::string& name) {
    // Check cache
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) {
        return it->second;
    }

    if (!library_) {
        NSLog(@"[mast3r] No shader library loaded");
        return nil;
    }

    @autoreleasepool {
        NSString* funcName = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> function = [library_ newFunctionWithName:funcName];

        if (!function) {
            NSLog(@"[mast3r] Shader function not found: %@", funcName);
            return nil;
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device_
            newComputePipelineStateWithFunction:function
            error:&error];

        if (!pipeline) {
            NSLog(@"[mast3r] Failed to create pipeline: %@", error);
            return nil;
        }

        pipelines_[name] = pipeline;
        return pipeline;
    }
}

id<MTLBuffer> MetalContext::create_buffer(size_t size, bool shared) {
    MTLResourceOptions options = shared
        ? MTLResourceStorageModeShared
        : MTLResourceStorageModePrivate;

    return [device_ newBufferWithLength:size options:options];
}

id<MTLBuffer> MetalContext::create_buffer(const void* data, size_t size, bool shared) {
    MTLResourceOptions options = shared
        ? MTLResourceStorageModeShared
        : MTLResourceStorageModePrivate;

    return [device_ newBufferWithBytes:data length:size options:options];
}

}  // namespace metal
}  // namespace mast3r