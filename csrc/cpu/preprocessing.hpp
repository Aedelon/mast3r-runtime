// MASt3R Runtime - CPU Preprocessing
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#pragma once

#include "common/types.hpp"

namespace mast3r {
namespace cpu {

// ImageNet normalization constants
constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

/**
 * CPU image preprocessor.
 *
 * Handles:
 * - Resize with aspect ratio preservation
 * - Center crop
 * - Normalization with ImageNet stats
 * - HWC -> CHW conversion
 */
class Preprocessor {
public:
    explicit Preprocessor(int target_resolution);

    /**
     * Process image for inference.
     *
     * @param img Input image [H, W, 3] RGB uint8
     * @param output Output buffer [3, res, res] float32, must be pre-allocated
     */
    void process(const ImageView& img, float* output);

private:
    int resolution_;

    // Resize buffer
    std::vector<uint8_t> resize_buffer_;

    void resize_and_crop(const ImageView& img, uint8_t* output);
    void normalize(const uint8_t* input, float* output);
};

/**
 * Bilinear interpolation resize.
 */
void bilinear_resize(
    const uint8_t* src, int src_h, int src_w,
    uint8_t* dst, int dst_h, int dst_w,
    int channels = 3
);

}  // namespace cpu
}  // namespace mast3r