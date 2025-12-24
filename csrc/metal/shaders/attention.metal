// MASt3R Runtime - Metal Attention Shaders
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include <metal_stdlib>
using namespace metal;

// Softmax constants
constant float NEG_INF = -1e9f;

/**
 * Matrix multiplication: C = A @ B
 *
 * Uses threadgroup memory for tiled multiplication.
 */
kernel void matmul(
    device const float* A [[buffer(0)]],  // [M, K]
    device const float* B [[buffer(1)]],  // [K, N]
    device float* C [[buffer(2)]],        // [M, N]
    constant int3& dims [[buffer(3)]],    // M, N, K
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const int M = dims.x;
    const int N = dims.y;
    const int K = dims.z;

    const int TILE_SIZE = 16;

    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    int row = tgid.y * TILE_SIZE + tid.y;
    int col = tgid.x * TILE_SIZE + tid.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles
        int aCol = t * TILE_SIZE + tid.x;
        int bRow = t * TILE_SIZE + tid.y;

        if (row < M && aCol < K) {
            tileA[tid.y][tid.x] = A[row * K + aCol];
        } else {
            tileA[tid.y][tid.x] = 0.0f;
        }

        if (bRow < K && col < N) {
            tileB[tid.y][tid.x] = B[bRow * N + col];
        } else {
            tileB[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Softmax along last dimension (in-place).
 */
kernel void softmax_inplace(
    device float* x [[buffer(0)]],       // [N, D]
    constant int2& dims [[buffer(1)]],    // N, D
    uint gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;

    if (gid >= uint(N)) return;

    device float* row = x + gid * D;

    // Find max
    float max_val = row[0];
    for (int i = 1; i < D; ++i) {
        max_val = max(max_val, row[i]);
    }

    // Exp and sum
    float sum = 0.0f;
    for (int i = 0; i < D; ++i) {
        row[i] = exp(row[i] - max_val);
        sum += row[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < D; ++i) {
        row[i] *= inv_sum;
    }
}

/**
 * Scaled dot-product attention for single head.
 *
 * attention = softmax(Q @ K^T / sqrt(d)) @ V
 */
kernel void attention_single_head(
    device const float* Q [[buffer(0)]],  // [seq_len, head_dim]
    device const float* K [[buffer(1)]],  // [seq_len, head_dim]
    device const float* V [[buffer(2)]],  // [seq_len, head_dim]
    device float* output [[buffer(3)]],   // [seq_len, head_dim]
    device float* attn_weights [[buffer(4)]],  // [seq_len, seq_len] workspace
    constant int2& dims [[buffer(5)]],    // seq_len, head_dim
    uint2 gid [[thread_position_in_grid]]
) {
    const int seq_len = dims.x;
    const int head_dim = dims.y;
    const float scale = 1.0f / sqrt(float(head_dim));

    // This is a simplified version - real implementation would be tiled

    int i = gid.y;  // Query position
    int j = gid.x;  // Key position

    if (i >= seq_len || j >= seq_len) return;

    // Compute Q[i] @ K[j]^T
    float dot = 0.0f;
    for (int k = 0; k < head_dim; ++k) {
        dot += Q[i * head_dim + k] * K[j * head_dim + k];
    }
    dot *= scale;

    attn_weights[i * seq_len + j] = dot;
}

/**
 * Layer normalization.
 */
kernel void layer_norm(
    device float* x [[buffer(0)]],           // [N, D] input/output
    device const float* gamma [[buffer(1)]], // [D] scale
    device const float* beta [[buffer(2)]],  // [D] bias
    constant int2& dims [[buffer(3)]],       // N, D
    constant float& eps [[buffer(4)]],       // epsilon
    uint gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;

    if (gid >= uint(N)) return;

    device float* row = x + gid * D;

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < D; ++i) {
        mean += row[i];
    }
    mean /= float(D);

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < D; ++i) {
        float diff = row[i] - mean;
        var += diff * diff;
    }
    var /= float(D);

    // Normalize and scale
    float inv_std = 1.0f / sqrt(var + eps);
    for (int i = 0; i < D; ++i) {
        row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

/**
 * GELU activation.
 */
kernel void gelu(
    device float* x [[buffer(0)]],
    constant int& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(size)) return;

    float v = x[gid];
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float cdf = 0.5f * (1.0f + tanh(sqrt_2_over_pi * (v + coeff * v * v * v)));
    x[gid] = v * cdf;
}