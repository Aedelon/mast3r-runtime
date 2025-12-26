// MASt3R Runtime - DPT Head Shaders
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DPT Head Projections
// ============================================================================

/**
 * Linear projection for head output.
 *
 * Projects decoder output to target dimension.
 * Input:  [num_patches, decoder_dim]
 * Output: [num_patches, out_dim]
 */
kernel void head_linear(
    device const float* input [[buffer(0)]],      // [N, D_in] - activations in FP32
    device float* output [[buffer(1)]],           // [N, D_out] - activations in FP32
    device const half* weight [[buffer(2)]],      // [D_out, D_in] - weights in FP16
    device const half* bias [[buffer(3)]],        // [D_out] - weights in FP16
    constant int3& dims [[buffer(4)]],            // N, D_in, D_out
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D_in = dims.y;
    const int D_out = dims.z;

    int n = gid.x;
    int d = gid.y;

    if (n >= N || d >= D_out) return;

    float sum = float(bias[d]);
    for (int i = 0; i < D_in; i++) {
        sum += input[n * D_in + i] * float(weight[d * D_in + i]);
    }

    output[n * D_out + d] = sum;
}

/**
 * 3D points head with optional activation.
 *
 * Projects to 3 channels and applies tanh scaling.
 * Output range: [-scale, scale] for normalized 3D coordinates.
 */
kernel void pts3d_head(
    device const float* input [[buffer(0)]],      // [N, D] - activations in FP32
    device float* output [[buffer(1)]],           // [N, 3] - output in FP32
    device const half* weight [[buffer(2)]],      // [3, D] - weights in FP16
    device const half* bias [[buffer(3)]],        // [3] - weights in FP16
    constant int2& dims [[buffer(4)]],            // N, D
    constant float& scale [[buffer(5)]],          // Output scale (e.g., 10.0)
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;

    int n = gid.x;
    int c = gid.y;  // 0, 1, or 2 for x, y, z

    if (n >= N || c >= 3) return;

    float sum = float(bias[c]);
    for (int i = 0; i < D; i++) {
        sum += input[n * D + i] * float(weight[c * D + i]);
    }

    // Apply tanh and scale for bounded 3D coordinates
    output[n * 3 + c] = tanh(sum) * scale;
}

/**
 * Descriptor head with L2 normalization.
 *
 * Projects to descriptor dimension and normalizes.
 */
kernel void desc_head(
    device const float* input [[buffer(0)]],      // [N, D_in] - activations in FP32
    device float* output [[buffer(1)]],           // [N, desc_dim] - output in FP32
    device const half* weight [[buffer(2)]],      // [desc_dim, D_in] - weights in FP16
    device const half* bias [[buffer(3)]],        // [desc_dim] - weights in FP16
    constant int3& dims [[buffer(4)]],            // N, D_in, desc_dim
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D_in = dims.y;
    const int desc_dim = dims.z;

    int n = gid.x;
    int d = gid.y;

    if (n >= N || d >= desc_dim) return;

    // Linear projection
    float sum = float(bias[d]);
    for (int i = 0; i < D_in; i++) {
        sum += input[n * D_in + i] * float(weight[d * D_in + i]);
    }

    output[n * desc_dim + d] = sum;
}

/**
 * L2 normalize descriptors (in-place) - DPT head version.
 */
kernel void normalize_descriptors_dpt(
    device float* desc [[buffer(0)]],             // [N, D] in/out - activations in FP32
    constant int2& dims [[buffer(1)]],            // N, D
    uint gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;

    if (gid >= uint(N)) return;

    device float* row = desc + gid * D;

    // Compute L2 norm
    float norm_sq = 0.0f;
    for (int i = 0; i < D; i++) {
        float val = row[i];
        norm_sq += val * val;
    }

    float inv_norm = rsqrt(norm_sq + 1e-12f);

    // Normalize in-place
    for (int i = 0; i < D; i++) {
        row[i] *= inv_norm;
    }
}

/**
 * Confidence head with sigmoid activation.
 *
 * Projects to 1 channel and applies sigmoid.
 * Output range: [0, 1].
 */
kernel void conf_head(
    device const float* input [[buffer(0)]],      // [N, D] - activations in FP32
    device float* output [[buffer(1)]],           // [N] - output in FP32
    device const half* weight [[buffer(2)]],      // [1, D] - weights in FP16
    device const half* bias [[buffer(3)]],        // [1] - weights in FP16
    constant int2& dims [[buffer(4)]],            // N, D
    uint gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;

    if (gid >= uint(N)) return;

    float sum = float(bias[0]);
    for (int i = 0; i < D; i++) {
        sum += input[gid * D + i] * float(weight[i]);
    }

    // Sigmoid activation
    output[gid] = 1.0f / (1.0f + exp(-sum));
}

// ============================================================================
// Upsampling (for full DPT)
// ============================================================================

/**
 * Bilinear upsample from patches to pixel grid.
 *
 * Input:  [patches_h, patches_w, C]
 * Output: [H, W, C]
 */
kernel void bilinear_upsample(
    device const float* input [[buffer(0)]],      // [pH, pW, C] - activations in FP32
    device float* output [[buffer(1)]],           // [H, W, C] - activations in FP32
    constant int4& in_dims [[buffer(2)]],         // pH, pW, C, _
    constant int2& out_dims [[buffer(3)]],        // H, W
    uint3 gid [[thread_position_in_grid]]
) {
    const int pH = in_dims.x;
    const int pW = in_dims.y;
    const int C = in_dims.z;
    const int H = out_dims.x;
    const int W = out_dims.y;

    int y = gid.x;
    int x = gid.y;
    int c = gid.z;

    if (y >= H || x >= W || c >= C) return;

    // Map output pixel to input patch coordinate
    float src_y = (float(y) + 0.5f) * float(pH) / float(H) - 0.5f;
    float src_x = (float(x) + 0.5f) * float(pW) / float(W) - 0.5f;

    // Clamp to valid range
    src_y = clamp(src_y, 0.0f, float(pH - 1));
    src_x = clamp(src_x, 0.0f, float(pW - 1));

    // Bilinear interpolation
    int y0 = int(src_y);
    int x0 = int(src_x);
    int y1 = min(y0 + 1, pH - 1);
    int x1 = min(x0 + 1, pW - 1);

    float fy = src_y - float(y0);
    float fx = src_x - float(x0);

    float v00 = input[(y0 * pW + x0) * C + c];
    float v01 = input[(y0 * pW + x1) * C + c];
    float v10 = input[(y1 * pW + x0) * C + c];
    float v11 = input[(y1 * pW + x1) * C + c];

    float v0 = v00 * (1.0f - fx) + v01 * fx;
    float v1 = v10 * (1.0f - fx) + v11 * fx;

    output[(y * W + x) * C + c] = v0 * (1.0f - fy) + v1 * fy;
}

/**
 * Reshape patches to 2D grid.
 *
 * Input:  [num_patches, C] with patches in row-major order
 * Output: [patches_h, patches_w, C]
 */
kernel void reshape_patches_to_grid(
    device const float* input [[buffer(0)]],      // [N, C] - activations in FP32
    device float* output [[buffer(1)]],           // [pH, pW, C] - activations in FP32
    constant int3& dims [[buffer(2)]],            // patches_h, patches_w, C
    uint2 gid [[thread_position_in_grid]]
) {
    const int pH = dims.x;
    const int pW = dims.y;
    const int C = dims.z;
    const int N = pH * pW;

    int patch_idx = gid.x;
    int c = gid.y;

    if (patch_idx >= N || c >= C) return;

    // Input is already in row-major patch order
    output[patch_idx * C + c] = input[patch_idx * C + c];
}

// ============================================================================
// Conv2D Kernels for Full DPT
// ============================================================================

/**
 * 1x1 Convolution (pointwise).
 *
 * Used for channel projection in act_postprocess.
 * Input:  [H, W, C_in]
 * Output: [H, W, C_out]
 * Weight: [C_out, C_in]
 */
kernel void conv2d_1x1(
    device const float* input [[buffer(0)]],      // [H, W, C_in] - activations in FP32
    device float* output [[buffer(1)]],           // [H, W, C_out] - activations in FP32
    device const half* weight [[buffer(2)]],      // [C_out, C_in] - weights in FP16
    device const half* bias [[buffer(3)]],        // [C_out] - weights in FP16
    constant int4& dims [[buffer(4)]],            // H, W, C_in, C_out
    uint3 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int C_in = dims.z;
    const int C_out = dims.w;

    int y = gid.x;
    int x = gid.y;
    int c_out = gid.z;

    if (y >= H || x >= W || c_out >= C_out) return;

    float sum = float(bias[c_out]);
    int pixel_idx = y * W + x;

    for (int c_in = 0; c_in < C_in; c_in++) {
        sum += input[pixel_idx * C_in + c_in] * float(weight[c_out * C_in + c_in]);
    }

    output[pixel_idx * C_out + c_out] = sum;
}

/**
 * 1x1 Convolution with ReLU activation.
 */
kernel void conv2d_1x1_relu(
    device const float* input [[buffer(0)]],      // [H, W, C_in] - activations in FP32
    device float* output [[buffer(1)]],           // [H, W, C_out] - activations in FP32
    device const half* weight [[buffer(2)]],      // [C_out, C_in] - weights in FP16
    device const half* bias [[buffer(3)]],        // [C_out] - weights in FP16
    constant int4& dims [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int C_in = dims.z;
    const int C_out = dims.w;

    int y = gid.x;
    int x = gid.y;
    int c_out = gid.z;

    if (y >= H || x >= W || c_out >= C_out) return;

    float sum = float(bias[c_out]);
    int pixel_idx = y * W + x;

    for (int c_in = 0; c_in < C_in; c_in++) {
        sum += input[pixel_idx * C_in + c_in] * float(weight[c_out * C_in + c_in]);
    }

    output[pixel_idx * C_out + c_out] = max(0.0f, sum);
}

/**
 * 3x3 Convolution with padding=1.
 *
 * Used for scratch.layer_rn and refinenet blocks.
 * Input:  [H, W, C_in]
 * Output: [H, W, C_out]
 * Weight: [C_out, C_in, 3, 3]
 */
kernel void conv2d_3x3(
    device const float* input [[buffer(0)]],     // [H, W, C_in] - activations in FP32
    device float* output [[buffer(1)]],          // [H, W, C_out] - activations in FP32
    device const half* weight [[buffer(2)]],     // [C_out, C_in, 3, 3] - weights in FP16
    device const half* bias [[buffer(3)]],       // [C_out] - weights in FP16
    constant int4& dims [[buffer(4)]],           // H, W, C_in, C_out
    uint3 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int C_in = dims.z;
    const int C_out = dims.w;

    int y = gid.x;
    int x = gid.y;
    int c_out = gid.z;

    if (y >= H || x >= W || c_out >= C_out) return;

    float sum = float(bias[c_out]);

    // 3x3 convolution with padding=1
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int iy = y + ky - 1;  // padding=1
                int ix = x + kx - 1;

                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    float val = input[(iy * W + ix) * C_in + c_in];
                    // Weight layout: [C_out, C_in, 3, 3]
                    int w_idx = ((c_out * C_in + c_in) * 3 + ky) * 3 + kx;
                    sum += val * float(weight[w_idx]);
                }
            }
        }
    }

    output[(y * W + x) * C_out + c_out] = sum;
}

/**
 * 3x3 Convolution with ReLU activation.
 */
kernel void conv2d_3x3_relu(
    device const float* input [[buffer(0)]],     // [H, W, C_in] - activations in FP32
    device float* output [[buffer(1)]],          // [H, W, C_out] - activations in FP32
    device const half* weight [[buffer(2)]],     // [C_out, C_in, 3, 3] - weights in FP16
    device const half* bias [[buffer(3)]],       // [C_out] - weights in FP16
    constant int4& dims [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int C_in = dims.z;
    const int C_out = dims.w;

    int y = gid.x;
    int x = gid.y;
    int c_out = gid.z;

    if (y >= H || x >= W || c_out >= C_out) return;

    float sum = float(bias[c_out]);

    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int iy = y + ky - 1;
                int ix = x + kx - 1;

                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    float val = input[(iy * W + ix) * C_in + c_in];
                    int w_idx = ((c_out * C_in + c_in) * 3 + ky) * 3 + kx;
                    sum += val * float(weight[w_idx]);
                }
            }
        }
    }

    output[(y * W + x) * C_out + c_out] = max(0.0f, sum);
}

/**
 * Transposed Convolution (deconvolution) for upsampling.
 *
 * Stride=2, kernel=2 for 2x upsampling.
 * Input:  [H, W, C]
 * Output: [H*2, W*2, C]
 * Weight: [C, C, 2, 2]
 */
kernel void conv_transpose_2x2_stride2(
    device const float* input [[buffer(0)]],      // [H, W, C] - activations in FP32
    device float* output [[buffer(1)]],           // [H*2, W*2, C] - activations in FP32
    device const half* weight [[buffer(2)]],      // [C, C, 2, 2] - weights in FP16
    device const half* bias [[buffer(3)]],        // [C] - weights in FP16
    constant int3& dims [[buffer(4)]],            // H, W, C
    uint3 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int C = dims.z;
    const int H_out = H * 2;
    const int W_out = W * 2;

    int y_out = gid.x;
    int x_out = gid.y;
    int c = gid.z;

    if (y_out >= H_out || x_out >= W_out || c >= C) return;

    float sum = float(bias[c]);

    // For transposed conv with stride=2, kernel=2:
    // Each output pixel is influenced by at most 1 input pixel
    // y_in = y_out / 2, x_in = x_out / 2
    // kernel position = (y_out % 2, x_out % 2)

    int y_in = y_out / 2;
    int x_in = x_out / 2;
    int ky = y_out % 2;
    int kx = x_out % 2;

    if (y_in < H && x_in < W) {
        for (int c_in = 0; c_in < C; c_in++) {
            float val = input[(y_in * W + x_in) * C + c_in];
            // Weight layout: [C_out, C_in, 2, 2]
            int w_idx = ((c * C + c_in) * 2 + ky) * 2 + kx;
            sum += val * float(weight[w_idx]);
        }
    }

    output[(y_out * W_out + x_out) * C + c] = sum;
}

/**
 * Transposed Convolution 4x4 with stride=4 for 4x upsampling.
 *
 * Input:  [H, W, C]
 * Output: [H*4, W*4, C]
 */
kernel void conv_transpose_4x4_stride4(
    device const float* input [[buffer(0)]],     // [H, W, C] - activations in FP32
    device float* output [[buffer(1)]],          // [H*4, W*4, C] - activations in FP32
    device const half* weight [[buffer(2)]],     // [C, C, 4, 4] - weights in FP16
    device const half* bias [[buffer(3)]],       // [C] - weights in FP16
    constant int3& dims [[buffer(4)]],           // H, W, C
    uint3 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int C = dims.z;
    const int H_out = H * 4;
    const int W_out = W * 4;

    int y_out = gid.x;
    int x_out = gid.y;
    int c = gid.z;

    if (y_out >= H_out || x_out >= W_out || c >= C) return;

    float sum = float(bias[c]);

    int y_in = y_out / 4;
    int x_in = x_out / 4;
    int ky = y_out % 4;
    int kx = x_out % 4;

    if (y_in < H && x_in < W) {
        for (int c_in = 0; c_in < C; c_in++) {
            float val = input[(y_in * W + x_in) * C + c_in];
            int w_idx = ((c * C + c_in) * 4 + ky) * 4 + kx;
            sum += val * float(weight[w_idx]);
        }
    }

    output[(y_out * W_out + x_out) * C + c] = sum;
}

/**
 * Element-wise tensor addition.
 *
 * Used for residual connections and feature fusion.
 */
kernel void add_tensors(
    device const float* a [[buffer(0)]],          // activations in FP32
    device const float* b [[buffer(1)]],          // activations in FP32
    device float* output [[buffer(2)]],           // activations in FP32
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(size)) return;
    output[gid] = a[gid] + b[gid];
}

/**
 * Element-wise tensor addition (in-place on b).
 */
kernel void add_tensors_inplace(
    device const float* a [[buffer(0)]],          // activations in FP32
    device float* b [[buffer(1)]],                // in/out - activations in FP32
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(size)) return;
    b[gid] += a[gid];
}

/**
 * ResidualConvUnit: two 3x3 convs with ReLU and residual.
 *
 * Used in RefineNet blocks.
 * x -> ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> + x
 *
 * This is a fused version for efficiency.
 * Input/Output: [H, W, C]
 */
kernel void residual_conv_unit_pass1(
    device const float* input [[buffer(0)]],      // [H, W, C] - activations in FP32
    device float* temp [[buffer(1)]],             // [H, W, C] intermediate - activations in FP32
    device const half* weight1 [[buffer(2)]],     // [C, C, 3, 3] - weights in FP16
    device const half* bias1 [[buffer(3)]],       // [C] - weights in FP16
    constant int3& dims [[buffer(4)]],            // H, W, C
    uint3 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int C = dims.z;

    int y = gid.x;
    int x = gid.y;
    int c_out = gid.z;

    if (y >= H || x >= W || c_out >= C) return;

    // First: ReLU on input
    // Then: Conv3x3
    float sum = float(bias1[c_out]);

    for (int c_in = 0; c_in < C; c_in++) {
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int iy = y + ky - 1;
                int ix = x + kx - 1;

                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    // Apply ReLU to input
                    float val = max(0.0f, input[(iy * W + ix) * C + c_in]);
                    int w_idx = ((c_out * C + c_in) * 3 + ky) * 3 + kx;
                    sum += val * float(weight1[w_idx]);
                }
            }
        }
    }

    temp[(y * W + x) * C + c_out] = sum;
}

kernel void residual_conv_unit_pass2(
    device const float* input [[buffer(0)]],      // [H, W, C] original input - activations in FP32
    device const float* temp [[buffer(1)]],       // [H, W, C] from pass1 - activations in FP32
    device float* output [[buffer(2)]],           // [H, W, C] - activations in FP32
    device const half* weight2 [[buffer(3)]],     // [C, C, 3, 3] - weights in FP16
    device const half* bias2 [[buffer(4)]],       // [C] - weights in FP16
    constant int3& dims [[buffer(5)]],            // H, W, C
    uint3 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int C = dims.z;

    int y = gid.x;
    int x = gid.y;
    int c_out = gid.z;

    if (y >= H || x >= W || c_out >= C) return;

    // ReLU on temp, then Conv3x3, then add residual
    float sum = float(bias2[c_out]);

    for (int c_in = 0; c_in < C; c_in++) {
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int iy = y + ky - 1;
                int ix = x + kx - 1;

                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    float val = max(0.0f, temp[(iy * W + ix) * C + c_in]);
                    int w_idx = ((c_out * C + c_in) * 3 + ky) * 3 + kx;
                    sum += val * float(weight2[w_idx]);
                }
            }
        }
    }

    // Add residual
    int pixel_idx = (y * W + x) * C + c_out;
    output[pixel_idx] = sum + input[pixel_idx];
}

/**
 * Pixel shuffle (depth to space) for upsampling.
 *
 * Rearranges [H, W, C*r*r] to [H*r, W*r, C].
 * Used in local features head.
 */
kernel void pixel_shuffle(
    device const float* input [[buffer(0)]],      // [H, W, C*r*r] - activations in FP32
    device float* output [[buffer(1)]],           // [H*r, W*r, C] - activations in FP32
    constant int4& dims [[buffer(2)]],            // H, W, C, r
    uint3 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int C = dims.z;
    const int r = dims.w;

    const int H_out = H * r;
    const int W_out = W * r;

    int y_out = gid.x;
    int x_out = gid.y;
    int c = gid.z;

    if (y_out >= H_out || x_out >= W_out || c >= C) return;

    // Map output to input
    int y_in = y_out / r;
    int x_in = x_out / r;
    int dy = y_out % r;
    int dx = x_out % r;

    // Input channel = c * r * r + dy * r + dx
    int c_in = c * r * r + dy * r + dx;

    output[(y_out * W_out + x_out) * C + c] = input[(y_in * W + x_in) * (C * r * r) + c_in];
}

/**
 * Reshape from [N, C] to [H, W, C] for patch tokens.
 *
 * Excludes CLS token (first token).
 */
kernel void reshape_tokens_to_grid(
    device const float* tokens [[buffer(0)]],     // [N+1, C] with CLS - activations in FP32
    device float* grid [[buffer(1)]],             // [H, W, C] - activations in FP32
    constant int3& dims [[buffer(2)]],            // H, W, C
    uint3 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int C = dims.z;

    int y = gid.x;
    int x = gid.y;
    int c = gid.z;

    if (y >= H || x >= W || c >= C) return;

    // Skip CLS token (index 0), patch tokens start at 1
    int token_idx = y * W + x + 1;  // +1 to skip CLS
    grid[(y * W + x) * C + c] = tokens[token_idx * C + c];
}

// ============================================================================
// Fused DPT Head (all outputs in one pass)
// ============================================================================

/**
 * Fused DPT head computation.
 *
 * Computes pts3d, desc, and conf from decoder output in a single pass.
 * More efficient than separate kernels for small outputs.
 */
kernel void dpt_head_fused(
    device const float* decoder_out [[buffer(0)]],    // [N, D] - activations in FP32
    device float* pts3d [[buffer(1)]],                // [N, 3] - output in FP32
    device float* desc [[buffer(2)]],                 // [N, desc_dim] - output in FP32
    device float* conf [[buffer(3)]],                 // [N] - output in FP32
    device const half* pts3d_weight [[buffer(4)]],    // [3, D] - weights in FP16
    device const half* pts3d_bias [[buffer(5)]],      // [3] - weights in FP16
    device const half* desc_weight [[buffer(6)]],     // [desc_dim, D] - weights in FP16
    device const half* desc_bias [[buffer(7)]],       // [desc_dim] - weights in FP16
    device const half* conf_weight [[buffer(8)]],     // [1, D] - weights in FP16
    device const half* conf_bias [[buffer(9)]],       // [1] - weights in FP16
    constant int3& dims [[buffer(10)]],               // N, D, desc_dim
    constant float& pts3d_scale [[buffer(11)]],       // pts3d output scale
    uint gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;
    const int desc_dim = dims.z;

    if (gid >= uint(N)) return;

    const int in_offset = gid * D;

    // === Pts3D ===
    for (int c = 0; c < 3; c++) {
        float sum = float(pts3d_bias[c]);
        for (int i = 0; i < D; i++) {
            sum += decoder_out[in_offset + i] * float(pts3d_weight[c * D + i]);
        }
        pts3d[gid * 3 + c] = tanh(sum) * pts3d_scale;
    }

    // === Descriptor ===
    float norm_sq = 0.0f;
    for (int c = 0; c < desc_dim; c++) {
        float sum = float(desc_bias[c]);
        for (int i = 0; i < D; i++) {
            sum += decoder_out[in_offset + i] * float(desc_weight[c * D + i]);
        }
        desc[gid * desc_dim + c] = sum;
        norm_sq += sum * sum;
    }

    // L2 normalize descriptor
    float inv_norm = rsqrt(norm_sq + 1e-12f);
    for (int c = 0; c < desc_dim; c++) {
        desc[gid * desc_dim + c] *= inv_norm;
    }

    // === Confidence ===
    float conf_sum = float(conf_bias[0]);
    for (int i = 0; i < D; i++) {
        conf_sum += decoder_out[in_offset + i] * float(conf_weight[i]);
    }
    conf[gid] = 1.0f / (1.0f + exp(-conf_sum));
}

// ============================================================================
// Simplified DPT Kernels (for initial implementation)
// ============================================================================

/**
 * GELU activation function (fast approximation).
 *
 * Uses the tanh approximation:
 * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 */
inline float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
    const float coef = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + coef * x3)));
}

/**
 * Simplified DPT head for pts3d and confidence.
 *
 * Takes decoder output and produces pts3d (3 channels) + conf (1 channel).
 * Uses the final 1x1 conv weights (head.4: [4, 128, 1, 1]).
 *
 * NOTE: This is a simplified version that directly projects from decoder_dim.
 * The full implementation would use multi-scale features through refinenets.
 *
 * Input:  [N, decoder_dim]
 * Output: pts3d [N, 3], conf [N]
 * Weight: [4, in_dim] where in_dim matches decoder_dim for simplified version
 */
kernel void dpt_pts3d_conf_simple(
    device const float* input [[buffer(0)]],      // [N, D] - activations in FP32
    device float* pts3d [[buffer(1)]],            // [N, 3] - output in FP32
    device float* conf [[buffer(2)]],             // [N] - output in FP32
    device const half* weight [[buffer(3)]],      // [4, D] - weights in FP16
    device const half* bias [[buffer(4)]],        // [4] - weights in FP16
    constant int2& dims [[buffer(5)]],            // N, D
    constant float& scale [[buffer(6)]],          // pts3d scale
    uint gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;

    if (gid >= uint(N)) return;

    const int in_offset = gid * D;

    // Project to 4 channels: [x, y, z, conf]
    float outputs[4];
    for (int c = 0; c < 4; c++) {
        float sum = float(bias[c]);
        for (int i = 0; i < D; i++) {
            sum += input[in_offset + i] * float(weight[c * D + i]);
        }
        outputs[c] = sum;
    }

    // pts3d: apply tanh and scale
    pts3d[gid * 3 + 0] = tanh(outputs[0]) * scale;
    pts3d[gid * 3 + 1] = tanh(outputs[1]) * scale;
    pts3d[gid * 3 + 2] = tanh(outputs[2]) * scale;

    // conf: apply sigmoid
    conf[gid] = 1.0f / (1.0f + exp(-outputs[3]));
}

/**
 * Split DPT head output [H, W, 4] into pts3d [H, W, 3] and conf [H, W].
 *
 * Applies tanh * scale to pts3d and sigmoid to conf.
 *
 * Input:  [H, W, 4] - raw DPT head output (channels: x, y, z, conf) in FP32
 * Output: pts3d [H, W, 3], conf [H, W] in FP32
 */
kernel void dpt_split_pts3d_conf(
    device const float* input [[buffer(0)]],      // [H, W, 4] - activations in FP32
    device float* pts3d [[buffer(1)]],            // [H, W, 3] - output in FP32
    device float* conf [[buffer(2)]],             // [H, W] - output in FP32
    constant int2& dims [[buffer(3)]],            // H, W
    constant float& scale [[buffer(4)]],          // pts3d scale (e.g., 10.0)
    uint2 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;

    if (gid.x >= uint(H) || gid.y >= uint(W)) return;

    int idx = gid.x * W + gid.y;

    // Read 4 channels from input
    float x = input[idx * 4 + 0];
    float y = input[idx * 4 + 1];
    float z = input[idx * 4 + 2];
    float c = input[idx * 4 + 3];

    // pts3d: apply tanh and scale
    pts3d[idx * 3 + 0] = tanh(x) * scale;
    pts3d[idx * 3 + 1] = tanh(y) * scale;
    pts3d[idx * 3 + 2] = tanh(z) * scale;

    // conf: apply sigmoid
    conf[idx] = 1.0f / (1.0f + exp(-c));
}

/**
 * Concatenate encoder and decoder features.
 *
 * Combines encoder output [N, embed_dim] and decoder output [N, decoder_dim]
 * into a single tensor [N, embed_dim + decoder_dim] for the local features MLP.
 *
 * Input1: [N, embed_dim] - encoder features
 * Input2: [N, decoder_dim] - decoder features
 * Output: [N, embed_dim + decoder_dim] - concatenated features
 */
kernel void concat_features(
    device const float* encoder_out [[buffer(0)]],    // [N, embed_dim] - activations in FP32
    device const float* decoder_out [[buffer(1)]],    // [N, decoder_dim] - activations in FP32
    device float* output [[buffer(2)]],               // [N, embed_dim + decoder_dim] - activations in FP32
    constant int3& dims [[buffer(3)]],                // N, embed_dim, decoder_dim
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int embed_dim = dims.y;
    const int decoder_dim = dims.z;
    const int total_dim = embed_dim + decoder_dim;

    int n = gid.x;
    int d = gid.y;

    if (n >= N || d >= total_dim) return;

    if (d < embed_dim) {
        // First part: encoder features
        output[n * total_dim + d] = encoder_out[n * embed_dim + d];
    } else {
        // Second part: decoder features
        output[n * total_dim + d] = decoder_out[n * decoder_dim + (d - embed_dim)];
    }
}

/**
 * Local features MLP for descriptors.
 *
 * MLP architecture: decoder_output -> fc1 -> GELU -> fc2 -> descriptors
 * Then L2 normalize the output.
 *
 * Input:  [N, in_dim]
 * Output: [N, out_dim]
 * fc1:    [hidden_dim, in_dim]
 * fc2:    [out_dim, hidden_dim]
 */
kernel void local_features_mlp(
    device const float* input [[buffer(0)]],      // [N, in_dim] - activations in FP32
    device float* output [[buffer(1)]],           // [N, out_dim] - output in FP32
    device const half* fc1_weight [[buffer(2)]],  // [hidden_dim, in_dim] - weights in FP16
    device const half* fc1_bias [[buffer(3)]],    // [hidden_dim] - weights in FP16
    device const half* fc2_weight [[buffer(4)]],  // [out_dim, hidden_dim] - weights in FP16
    device const half* fc2_bias [[buffer(5)]],    // [out_dim] - weights in FP16
    constant int4& dims [[buffer(6)]],            // N, in_dim, hidden_dim, out_dim
    uint gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int in_dim = dims.y;
    const int hidden_dim = dims.z;
    const int out_dim = dims.w;

    if (gid >= uint(N)) return;

    const int in_offset = gid * in_dim;
    device float* out_row = output + gid * out_dim;

    // Process in chunks to handle large hidden_dim (up to 8192)
    // Use block-based approach: compute hidden in blocks, accumulate fc2
    constexpr int BLOCK_SIZE = 256;

    // Initialize output with bias
    for (int o = 0; o < out_dim; o++) {
        out_row[o] = float(fc2_bias[o]);
    }

    // Process hidden layer in blocks
    for (int block_start = 0; block_start < hidden_dim; block_start += BLOCK_SIZE) {
        int block_end = min(block_start + BLOCK_SIZE, hidden_dim);
        int block_len = block_end - block_start;

        // Compute this block of hidden activations
        float hidden[BLOCK_SIZE];
        for (int h = 0; h < block_len; h++) {
            int h_global = block_start + h;
            float sum = float(fc1_bias[h_global]);
            for (int i = 0; i < in_dim; i++) {
                sum += input[in_offset + i] * float(fc1_weight[h_global * in_dim + i]);
            }
            hidden[h] = gelu(sum);
        }

        // Accumulate fc2 contributions from this block
        for (int o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            for (int h = 0; h < block_len; h++) {
                int h_global = block_start + h;
                sum += hidden[h] * float(fc2_weight[o * hidden_dim + h_global]);
            }
            out_row[o] += sum;
        }
    }

    // L2 normalize
    float norm_sq = 0.0f;
    for (int o = 0; o < out_dim; o++) {
        norm_sq += out_row[o] * out_row[o];
    }
    float inv_norm = rsqrt(norm_sq + 1e-12f);
    for (int o = 0; o < out_dim; o++) {
        out_row[o] *= inv_norm;
    }
}

/**
 * Convert half precision buffer to float.
 *
 * Used for final output conversion to Python.
 */
kernel void half_to_float(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(size)) return;
    output[gid] = float(input[gid]);
}
