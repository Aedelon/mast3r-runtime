/**
 * CUDA preprocessing kernels - STUB.
 *
 * Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace mast3r {
namespace cuda {

// ImageNet normalization constants
__constant__ float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
__constant__ float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

/**
 * Bilinear resize + normalize kernel.
 *
 * Input: HWC uint8 RGB
 * Output: CHW float32 normalized
 */
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int src_h, int src_w,
    int dst_h, int dst_w
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    // Compute source coordinates
    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    // Clamp to valid range
    src_x = fmaxf(0.0f, fminf(src_x, src_w - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_h - 1.0f));

    // Bilinear interpolation
    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float wx = src_x - x0;
    float wy = src_y - y0;

    for (int c = 0; c < 3; ++c) {
        float v00 = input[(y0 * src_w + x0) * 3 + c];
        float v01 = input[(y0 * src_w + x1) * 3 + c];
        float v10 = input[(y1 * src_w + x0) * 3 + c];
        float v11 = input[(y1 * src_w + x1) * 3 + c];

        float value = (1 - wy) * ((1 - wx) * v00 + wx * v01) +
                      wy * ((1 - wx) * v10 + wx * v11);

        // Normalize: [0,255] -> [0,1] -> ImageNet
        value = value / 255.0f;
        value = (value - IMAGENET_MEAN[c]) / IMAGENET_STD[c];

        // Store in CHW format
        output[c * dst_h * dst_w + y * dst_w + x] = value;
    }
}

void launch_preprocess(
    const uint8_t* d_input,
    float* d_output,
    int src_h, int src_w,
    int dst_h, int dst_w,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, src_h, src_w, dst_h, dst_w
    );
}

}  // namespace cuda
}  // namespace mast3r
