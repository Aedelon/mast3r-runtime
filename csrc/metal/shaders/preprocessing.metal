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
    texture2d<float, access::read> input [[texture(0)]],
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

    float4 pixel = input.sample(sampler(filter::linear), coord);

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
 */
kernel void resize_bilinear(
    device const uchar4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant PreprocessParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(params.dst_width) || gid.y >= uint(params.dst_height)) {
        return;
    }

    // Source coordinates
    float src_x = (float(gid.x) + 0.5f) * params.scale_x - 0.5f;
    float src_y = (float(gid.y) + 0.5f) * params.scale_y - 0.5f;

    // Integer and fractional parts
    int x0 = int(floor(src_x));
    int y0 = int(floor(src_y));
    int x1 = min(x0 + 1, params.src_width - 1);
    int y1 = min(y0 + 1, params.src_height - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float fx = src_x - float(x0);
    float fy = src_y - float(y0);

    // Read 4 neighbors
    uchar4 p00 = input[y0 * params.src_width + x0];
    uchar4 p01 = input[y0 * params.src_width + x1];
    uchar4 p10 = input[y1 * params.src_width + x0];
    uchar4 p11 = input[y1 * params.src_width + x1];

    // Bilinear interpolation
    float4 f00 = float4(p00) / 255.0f;
    float4 f01 = float4(p01) / 255.0f;
    float4 f10 = float4(p10) / 255.0f;
    float4 f11 = float4(p11) / 255.0f;

    float4 result = mix(mix(f00, f01, fx), mix(f10, f11, fx), fy);

    output[gid.y * params.dst_width + gid.x] = result;
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