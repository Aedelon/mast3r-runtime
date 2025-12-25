// MASt3R Runtime - Metal Preprocessing Shaders
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include <metal_stdlib>
using namespace metal;

// ImageNet normalization constants
constant float3 IMAGENET_MEAN = float3(0.485, 0.456, 0.406);
constant float3 IMAGENET_STD = float3(0.229, 0.224, 0.225);

// Preprocessing parameters
struct PreprocessParams {
    int src_width;
    int src_height;
    int dst_width;
    int dst_height;
    float scale_x;
    float scale_y;
    int crop_x;
    int crop_y;
};

/**
 * Bilinear resize + center crop + normalize.
 *
 * Input: RGBA uint8 [H, W, 4]
 * Output: float [3, dst_H, dst_W] (CHW format, normalized)
 */
kernel void preprocess_image(
    texture2d<float, access::sample> input [[texture(0)]],
    device float* output [[buffer(0)]],
    constant PreprocessParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(params.dst_width) || gid.y >= uint(params.dst_height)) {
        return;
    }

    // Calculate source coordinates with crop offset
    float src_x = (float(gid.x) + 0.5f + params.crop_x) * params.scale_x;
    float src_y = (float(gid.y) + 0.5f + params.crop_y) * params.scale_y;

    // Bilinear sampling (texture sampler handles interpolation)
    float2 coord = float2(src_x / float(params.src_width),
                          src_y / float(params.src_height));

    constexpr sampler linear_sampler(filter::linear);
    float4 pixel = input.sample(linear_sampler, coord);

    // Normalize with ImageNet stats
    float3 rgb = pixel.rgb;
    float3 normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD;

    // Write to output in CHW format
    int dst_size = params.dst_width * params.dst_height;
    int pixel_idx = gid.y * params.dst_width + gid.x;

    output[0 * dst_size + pixel_idx] = normalized.r;  // R channel
    output[1 * dst_size + pixel_idx] = normalized.g;  // G channel
    output[2 * dst_size + pixel_idx] = normalized.b;  // B channel
}

/**
 * Fast resize using threadgroup shared memory.
 *
 * Uses tile-based loading to reduce global memory bandwidth.
 * Each threadgroup processes a TILE_SIZE x TILE_SIZE output tile,
 * pre-loading required input pixels into shared memory.
 */
constant int PREPROCESS_TILE_SIZE = 16;
constant int PREPROCESS_TILE_MARGIN = 2;  // For bilinear interpolation
constant int PREPROCESS_SHARED_SIZE = PREPROCESS_TILE_SIZE + PREPROCESS_TILE_MARGIN;

kernel void resize_bilinear_tiled(
    device const uchar4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant PreprocessParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Shared memory tile for input pixels
    threadgroup float4 tile[PREPROCESS_SHARED_SIZE][PREPROCESS_SHARED_SIZE];

    // Calculate tile bounds in source image
    float tile_src_x0 = float(tgid.x * PREPROCESS_TILE_SIZE) * params.scale_x;
    float tile_src_y0 = float(tgid.y * PREPROCESS_TILE_SIZE) * params.scale_y;

    int src_x_start = max(0, int(floor(tile_src_x0)) - 1);
    int src_y_start = max(0, int(floor(tile_src_y0)) - 1);

    // Cooperatively load input tile into shared memory
    for (int dy = tid.y; dy < PREPROCESS_SHARED_SIZE; dy += PREPROCESS_TILE_SIZE) {
        for (int dx = tid.x; dx < PREPROCESS_SHARED_SIZE; dx += PREPROCESS_TILE_SIZE) {
            int src_x = clamp(src_x_start + dx, 0, params.src_width - 1);
            int src_y = clamp(src_y_start + dy, 0, params.src_height - 1);
            uchar4 pixel = input[src_y * params.src_width + src_x];
            tile[dy][dx] = float4(pixel) / 255.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now compute output using shared memory
    if (gid.x >= uint(params.dst_width) || gid.y >= uint(params.dst_height)) {
        return;
    }

    // Source coordinates
    float src_x = (float(gid.x) + 0.5f) * params.scale_x - 0.5f;
    float src_y = (float(gid.y) + 0.5f) * params.scale_y - 0.5f;

    // Local tile coordinates
    float local_x = src_x - float(src_x_start);
    float local_y = src_y - float(src_y_start);

    int lx0 = int(floor(local_x));
    int ly0 = int(floor(local_y));
    int lx1 = min(lx0 + 1, PREPROCESS_SHARED_SIZE - 1);
    int ly1 = min(ly0 + 1, PREPROCESS_SHARED_SIZE - 1);
    lx0 = max(lx0, 0);
    ly0 = max(ly0, 0);

    float fx = local_x - float(lx0);
    float fy = local_y - float(ly0);

    // Bilinear interpolation from shared memory (fast!)
    float4 p00 = tile[ly0][lx0];
    float4 p01 = tile[ly0][lx1];
    float4 p10 = tile[ly1][lx0];
    float4 p11 = tile[ly1][lx1];

    float4 result = mix(mix(p00, p01, fx), mix(p10, p11, fx), fy);

    output[gid.y * params.dst_width + gid.x] = result;
}

/**
 * Fused resize + normalize + HWC->CHW in single pass.
 * Reduces memory bandwidth by avoiding intermediate buffer.
 */
kernel void preprocess_fused(
    device const uchar4* input [[buffer(0)]],
    device float* output [[buffer(1)]],  // CHW format
    constant PreprocessParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup float4 tile[PREPROCESS_SHARED_SIZE][PREPROCESS_SHARED_SIZE];

    float tile_src_x0 = float(tgid.x * PREPROCESS_TILE_SIZE) * params.scale_x;
    float tile_src_y0 = float(tgid.y * PREPROCESS_TILE_SIZE) * params.scale_y;

    int src_x_start = max(0, int(floor(tile_src_x0)) - 1);
    int src_y_start = max(0, int(floor(tile_src_y0)) - 1);

    // Load tile
    for (int dy = tid.y; dy < PREPROCESS_SHARED_SIZE; dy += PREPROCESS_TILE_SIZE) {
        for (int dx = tid.x; dx < PREPROCESS_SHARED_SIZE; dx += PREPROCESS_TILE_SIZE) {
            int src_x = clamp(src_x_start + dx, 0, params.src_width - 1);
            int src_y = clamp(src_y_start + dy, 0, params.src_height - 1);
            uchar4 pixel = input[src_y * params.src_width + src_x];
            tile[dy][dx] = float4(pixel) / 255.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid.x >= uint(params.dst_width) || gid.y >= uint(params.dst_height)) {
        return;
    }

    // Bilinear from shared memory
    float src_x = (float(gid.x) + 0.5f) * params.scale_x - 0.5f;
    float src_y = (float(gid.y) + 0.5f) * params.scale_y - 0.5f;

    float local_x = src_x - float(src_x_start);
    float local_y = src_y - float(src_y_start);

    int lx0 = clamp(int(floor(local_x)), 0, PREPROCESS_SHARED_SIZE - 1);
    int ly0 = clamp(int(floor(local_y)), 0, PREPROCESS_SHARED_SIZE - 1);
    int lx1 = min(lx0 + 1, PREPROCESS_SHARED_SIZE - 1);
    int ly1 = min(ly0 + 1, PREPROCESS_SHARED_SIZE - 1);

    float fx = local_x - float(lx0);
    float fy = local_y - float(ly0);

    float4 result = mix(mix(tile[ly0][lx0], tile[ly0][lx1], fx),
                        mix(tile[ly1][lx0], tile[ly1][lx1], fx), fy);

    // Normalize with ImageNet stats and write CHW
    float3 normalized = (result.rgb - IMAGENET_MEAN) / IMAGENET_STD;

    int dst_size = params.dst_width * params.dst_height;
    int pixel_idx = gid.y * params.dst_width + gid.x;

    output[0 * dst_size + pixel_idx] = normalized.r;
    output[1 * dst_size + pixel_idx] = normalized.g;
    output[2 * dst_size + pixel_idx] = normalized.b;
}

/**
 * HWC to CHW conversion with normalization.
 */
kernel void hwc_to_chw_normalize(
    device const float4* input [[buffer(0)]],  // [H, W] float4 (RGBA)
    device float* output [[buffer(1)]],         // [3, H, W] float
    constant int2& size [[buffer(2)]],          // width, height
    uint2 gid [[thread_position_in_grid]]
) {
    int width = size.x;
    int height = size.y;

    if (gid.x >= uint(width) || gid.y >= uint(height)) {
        return;
    }

    int pixel_idx = gid.y * width + gid.x;
    int total_pixels = width * height;

    float4 pixel = input[pixel_idx];

    // Normalize and write CHW
    output[0 * total_pixels + pixel_idx] = (pixel.r - IMAGENET_MEAN.r) / IMAGENET_STD.r;
    output[1 * total_pixels + pixel_idx] = (pixel.g - IMAGENET_MEAN.g) / IMAGENET_STD.g;
    output[2 * total_pixels + pixel_idx] = (pixel.b - IMAGENET_MEAN.b) / IMAGENET_STD.b;
}