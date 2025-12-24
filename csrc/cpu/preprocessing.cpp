// MASt3R Runtime - CPU Preprocessing Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include "preprocessing.hpp"

#include <algorithm>
#include <cmath>

#ifdef MAST3R_HAS_OPENMP
#include <omp.h>
#endif

namespace mast3r {
namespace cpu {

Preprocessor::Preprocessor(int target_resolution)
    : resolution_(target_resolution) {
    resize_buffer_.resize(resolution_ * resolution_ * 3);
}

void Preprocessor::process(const ImageView& img, float* output) {
    // Step 1: Resize and center crop
    resize_and_crop(img, resize_buffer_.data());

    // Step 2: Normalize and convert HWC -> CHW
    normalize(resize_buffer_.data(), output);
}

void Preprocessor::resize_and_crop(const ImageView& img, uint8_t* output) {
    const int src_h = img.height;
    const int src_w = img.width;
    const int dst_size = resolution_;

    // Calculate scale to make smaller dimension = target
    const float scale = std::max(
        static_cast<float>(dst_size) / src_h,
        static_cast<float>(dst_size) / src_w
    );

    const int new_h = static_cast<int>(src_h * scale);
    const int new_w = static_cast<int>(src_w * scale);

    // Resize to intermediate size
    std::vector<uint8_t> resized(new_h * new_w * 3);
    bilinear_resize(img.data, src_h, src_w, resized.data(), new_h, new_w);

    // Center crop
    const int start_h = (new_h - dst_size) / 2;
    const int start_w = (new_w - dst_size) / 2;

#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < dst_size; ++y) {
        for (int x = 0; x < dst_size; ++x) {
            const int src_idx = ((start_h + y) * new_w + (start_w + x)) * 3;
            const int dst_idx = (y * dst_size + x) * 3;
            output[dst_idx + 0] = resized[src_idx + 0];
            output[dst_idx + 1] = resized[src_idx + 1];
            output[dst_idx + 2] = resized[src_idx + 2];
        }
    }
}

void Preprocessor::normalize(const uint8_t* input, float* output) {
    const int size = resolution_ * resolution_;

#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
        const int pixel_idx = i * 3;

        // Convert to [0, 1] and apply ImageNet normalization
        // Also convert HWC -> CHW
        for (int c = 0; c < 3; ++c) {
            const float val = static_cast<float>(input[pixel_idx + c]) / 255.0f;
            const float normalized = (val - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            output[c * size + i] = normalized;
        }
    }
}

void bilinear_resize(
    const uint8_t* src, int src_h, int src_w,
    uint8_t* dst, int dst_h, int dst_w,
    int channels
) {
    const float scale_y = static_cast<float>(src_h) / dst_h;
    const float scale_x = static_cast<float>(src_w) / dst_w;

#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int dst_y = 0; dst_y < dst_h; ++dst_y) {
        for (int dst_x = 0; dst_x < dst_w; ++dst_x) {
            // Source coordinates
            const float src_y = dst_y * scale_y;
            const float src_x = dst_x * scale_x;

            // Integer and fractional parts
            const int y0 = static_cast<int>(src_y);
            const int x0 = static_cast<int>(src_x);
            const int y1 = std::min(y0 + 1, src_h - 1);
            const int x1 = std::min(x0 + 1, src_w - 1);

            const float fy = src_y - y0;
            const float fx = src_x - x0;

            // Bilinear weights
            const float w00 = (1.0f - fx) * (1.0f - fy);
            const float w01 = fx * (1.0f - fy);
            const float w10 = (1.0f - fx) * fy;
            const float w11 = fx * fy;

            // Interpolate each channel
            for (int c = 0; c < channels; ++c) {
                const float v00 = src[(y0 * src_w + x0) * channels + c];
                const float v01 = src[(y0 * src_w + x1) * channels + c];
                const float v10 = src[(y1 * src_w + x0) * channels + c];
                const float v11 = src[(y1 * src_w + x1) * channels + c];

                const float val = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                dst[(dst_y * dst_w + dst_x) * channels + c] =
                    static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }
}

}  // namespace cpu
}  // namespace mast3r