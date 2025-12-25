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

// ============================================================================
// FlashAttention-style Tiled Attention
// ============================================================================

constant int ATTN_TILE_SIZE = 32;      // Tile size for Q/K blocks
constant int ATTN_HEAD_DIM = 64;       // Common head dimension (64 or 128)

/**
 * Tiled attention computation (FlashAttention-style).
 *
 * Computes attention in tiles to maximize SRAM reuse:
 * - Load Q tile, iterate over K/V tiles
 * - Accumulate softmax and output incrementally
 * - Avoids materializing full attention matrix
 *
 * For each query tile:
 *   for each key tile:
 *     S = Q_tile @ K_tile^T / sqrt(d)
 *     Update running softmax and output
 */
kernel void attention_tiled(
    device const half* Q [[buffer(0)]],     // [seq_len, head_dim] F16
    device const half* K [[buffer(1)]],     // [seq_len, head_dim] F16
    device const half* V [[buffer(2)]],     // [seq_len, head_dim] F16
    device half* output [[buffer(3)]],       // [seq_len, head_dim] F16
    constant int2& dims [[buffer(4)]],       // seq_len, head_dim
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const int seq_len = dims.x;
    const int head_dim = dims.y;
    const float scale = rsqrt(float(head_dim));

    // Shared memory for tiles
    threadgroup float q_tile[ATTN_TILE_SIZE][ATTN_HEAD_DIM];
    threadgroup float k_tile[ATTN_TILE_SIZE][ATTN_HEAD_DIM];
    threadgroup float v_tile[ATTN_TILE_SIZE][ATTN_HEAD_DIM];
    threadgroup float s_tile[ATTN_TILE_SIZE][ATTN_TILE_SIZE];  // Attention scores

    const int q_start = tgid.y * ATTN_TILE_SIZE;
    const int local_q = tid.y;
    const int global_q = q_start + local_q;

    // Running statistics for online softmax
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float output_acc[ATTN_HEAD_DIM];

    for (int d = 0; d < head_dim; ++d) {
        output_acc[d] = 0.0f;
    }

    // Load Q tile once (reused across all K tiles)
    if (global_q < seq_len && tid.x < uint(head_dim)) {
        q_tile[local_q][tid.x] = float(Q[global_q * head_dim + tid.x]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over K/V tiles
    for (int k_start = 0; k_start < seq_len; k_start += ATTN_TILE_SIZE) {
        const int local_k = tid.y;
        const int global_k = k_start + local_k;

        // Load K and V tiles
        if (global_k < seq_len && tid.x < uint(head_dim)) {
            k_tile[local_k][tid.x] = float(K[global_k * head_dim + tid.x]);
            v_tile[local_k][tid.x] = float(V[global_k * head_dim + tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute S = Q @ K^T for this tile
        if (local_q < ATTN_TILE_SIZE && tid.x < ATTN_TILE_SIZE) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_tile[local_q][d] * k_tile[tid.x][d];
            }
            s_tile[local_q][tid.x] = dot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax update
        if (global_q < seq_len && tid.x == 0) {
            // Find new max in this tile
            float tile_max = -INFINITY;
            for (int j = 0; j < ATTN_TILE_SIZE && (k_start + j) < seq_len; ++j) {
                tile_max = max(tile_max, s_tile[local_q][j]);
            }

            // Update running max and rescale
            float old_max = row_max;
            row_max = max(row_max, tile_max);

            // Rescale previous accumulations
            float rescale = exp(old_max - row_max);
            row_sum *= rescale;
            for (int d = 0; d < head_dim; ++d) {
                output_acc[d] *= rescale;
            }

            // Add this tile's contribution
            for (int j = 0; j < ATTN_TILE_SIZE && (k_start + j) < seq_len; ++j) {
                float w = exp(s_tile[local_q][j] - row_max);
                row_sum += w;
                for (int d = 0; d < head_dim; ++d) {
                    output_acc[d] += w * v_tile[j][d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write output
    if (global_q < seq_len && tid.x == 0) {
        float inv_sum = 1.0f / row_sum;
        for (int d = 0; d < head_dim; ++d) {
            output[global_q * head_dim + d] = half(output_acc[d] * inv_sum);
        }
    }
}

/**
 * Multi-head attention with tiling.
 * Each threadgroup handles one query position across all heads.
 */
kernel void multihead_attention_tiled(
    device const half* Q [[buffer(0)]],      // [batch, num_heads, seq_len, head_dim]
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant int4& dims [[buffer(4)]],        // batch, num_heads, seq_len, head_dim
    uint3 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const int batch = dims.x;
    const int num_heads = dims.y;
    const int seq_len = dims.z;
    const int head_dim = dims.w;

    const int b = tgid.z;
    const int h = tgid.y;

    if (b >= batch || h >= num_heads) return;

    // Offset to this batch/head
    const int stride = seq_len * head_dim;
    const int offset = (b * num_heads + h) * stride;

    device const half* Q_head = Q + offset;
    device const half* K_head = K + offset;
    device const half* V_head = V + offset;
    device half* O_head = output + offset;

    // Use same tiled logic as single-head
    // (Implementation would duplicate attention_tiled logic with offset pointers)

    // Simplified: Just compute one query per thread for now
    const int q_idx = tgid.x * ATTN_TILE_SIZE + tid.y;
    if (q_idx >= seq_len) return;

    const float scale = rsqrt(float(head_dim));

    // Accumulate attention output
    float acc[ATTN_HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) acc[d] = 0.0f;

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Compute attention scores and apply to V
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += float(Q_head[q_idx * head_dim + d]) *
                   float(K_head[k_idx * head_dim + d]);
        }
        dot *= scale;

        // Online softmax
        float old_max = row_max;
        row_max = max(row_max, dot);
        float rescale = exp(old_max - row_max);
        row_sum = row_sum * rescale + exp(dot - row_max);

        for (int d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * rescale + exp(dot - row_max) * float(V_head[k_idx * head_dim + d]);
        }
    }

    // Write output
    float inv_sum = 1.0f / row_sum;
    for (int d = 0; d < head_dim; ++d) {
        O_head[q_idx * head_dim + d] = half(acc[d] * inv_sum);
    }
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