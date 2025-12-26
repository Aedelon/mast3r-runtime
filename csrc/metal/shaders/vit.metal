// MASt3R Runtime - ViT Metal Shaders
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Error Function (erf) - Winitzki Approximation
// ============================================================================
// Approximation with ~0.035% relative error
// Source: https://www.cs.uaf.edu/2010/spring/cs481/section/1/lecture/02_23_planet.html

inline float erf_approx(float x) {
    const float pi = 3.14159265359f;
    const float a = 8.0f * (pi - 3.0f) / (3.0f * pi * (4.0f - pi));  // ~0.140012

    float sign_x = (x >= 0.0f) ? 1.0f : -1.0f;
    float x2 = x * x;

    // erf(x) = sign(x) * sqrt(1 - exp(-x² * (4/π + a*x²) / (1 + a*x²)))
    float inner = -x2 * (4.0f / pi + a * x2) / (1.0f + a * x2);
    return sign_x * sqrt(1.0f - exp(inner));
}

// ============================================================================
// Patch Embedding
// ============================================================================

/**
 * Patch embedding via convolution.
 *
 * Converts image [3, H, W] to patches [num_patches, embed_dim].
 * Each patch is a flattened patch_size x patch_size x 3 region,
 * projected to embed_dim via learned linear weights.
 *
 * Input:  [3, H, W] float (CHW format, normalized)
 * Output: [num_patches, embed_dim] float
 * Weight: [embed_dim, 3 * patch_size * patch_size] half
 * Bias:   [embed_dim] half
 */
kernel void patch_embed(
    device const float* input [[buffer(0)]],      // [3, H, W] float (preprocessed)
    device float* output [[buffer(1)]],            // [num_patches, embed_dim] float
    device const half* weight [[buffer(2)]],      // [embed_dim, 3*P*P] half
    device const half* bias [[buffer(3)]],        // [embed_dim] half
    constant int4& dims [[buffer(4)]],            // H, W, patch_size, embed_dim
    uint2 gid [[thread_position_in_grid]]
) {
    const int H = dims.x;
    const int W = dims.y;
    const int P = dims.z;           // patch_size (14 or 16)
    const int D = dims.w;           // embed_dim

    const int patches_h = H / P;
    const int patches_w = W / P;
    const int num_patches = patches_h * patches_w;
    const int patch_dim = 3 * P * P;  // Input dimension per patch

    // gid.x = patch index, gid.y = embed dimension
    int patch_idx = gid.x;
    int embed_idx = gid.y;

    if (patch_idx >= num_patches || embed_idx >= D) return;

    // Patch grid position
    int py = patch_idx / patches_w;
    int px = patch_idx % patches_w;

    // Compute dot product: sum over input channels and patch pixels
    // Accumulate in float for numerical stability
    float sum = float(bias[embed_idx]);

    for (int c = 0; c < 3; c++) {
        for (int dy = 0; dy < P; dy++) {
            for (int dx = 0; dx < P; dx++) {
                int y = py * P + dy;
                int x = px * P + dx;

                // Input index: CHW format
                int input_idx = c * H * W + y * W + x;

                // Weight index: [embed_dim, 3*P*P]
                int weight_idx = embed_idx * patch_dim + c * P * P + dy * P + dx;

                sum += input[input_idx] * float(weight[weight_idx]);
            }
        }
    }

    output[patch_idx * D + embed_idx] = sum;
}

/**
 * Add positional embeddings to patch tokens.
 *
 * Supports:
 * - DUNE: Learned absolute position embeddings
 * - Optional: cls_token prepending
 */
kernel void add_pos_embed(
    device float* x [[buffer(0)]],                // [num_patches, embed_dim] in/out (FP32)
    device const half* pos_embed [[buffer(1)]],   // [num_patches, embed_dim] (FP16 weights)
    constant int2& dims [[buffer(2)]],            // num_patches, embed_dim
    uint gid [[thread_position_in_grid]]
) {
    const int num_patches = dims.x;
    const int embed_dim = dims.y;
    const int total = num_patches * embed_dim;

    if (gid >= uint(total)) return;

    x[gid] += float(pos_embed[gid]);
}

/**
 * Prepend CLS token to sequence.
 *
 * Input:  [num_patches, embed_dim] float
 * Output: [1 + num_patches, embed_dim] float
 */
kernel void prepend_cls_token(
    device const float* input [[buffer(0)]],      // [num_patches, embed_dim] (FP32)
    device float* output [[buffer(1)]],           // [1 + num_patches, embed_dim] (FP32)
    device const half* cls_token [[buffer(2)]],   // [embed_dim] (FP16 weights)
    constant int2& dims [[buffer(3)]],            // num_patches, embed_dim
    uint gid [[thread_position_in_grid]]
) {
    const int num_patches = dims.x;
    const int D = dims.y;
    const int total_out = (1 + num_patches) * D;

    if (gid >= uint(total_out)) return;

    int seq_idx = gid / D;
    int dim_idx = gid % D;

    if (seq_idx == 0) {
        // CLS token (convert from FP16 weight to FP32)
        output[gid] = float(cls_token[dim_idx]);
    } else {
        // Copy from input (already FP32)
        int input_idx = (seq_idx - 1) * D + dim_idx;
        output[gid] = input[input_idx];
    }
}

// ============================================================================
// MLP Block
// ============================================================================

/**
 * MLP: Linear -> GELU -> Linear
 *
 * First linear expands by 4x, second projects back.
 * Mixed precision: FP32 activations, FP16 weights for memory efficiency.
 */
kernel void mlp_fc1(
    device const float* input [[buffer(0)]],      // [N, D] - FP32 activations
    device float* output [[buffer(1)]],           // [N, 4*D] - FP32 output
    device const half* weight [[buffer(2)]],      // [4*D, D] - FP16 weights
    device const half* bias [[buffer(3)]],        // [4*D] - FP16 bias
    constant int2& dims [[buffer(4)]],            // N, D
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;
    const int D4 = 4 * D;

    int n = gid.x;  // sequence position
    int d = gid.y;  // hidden dimension (0 to 4*D-1)

    if (n >= N || d >= D4) return;

    // Accumulate in float for numerical stability
    float sum = float(bias[d]);
    for (int i = 0; i < D; i++) {
        sum += input[n * D + i] * float(weight[d * D + i]);
    }

    // GELU activation using erf approximation for numerical stability
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    const float inv_sqrt2 = 0.7071067811865476f;
    float result = sum * 0.5f * (1.0f + erf_approx(sum * inv_sqrt2));

    output[n * D4 + d] = result;
}

kernel void mlp_fc2(
    device const float* input [[buffer(0)]],      // [N, 4*D] - FP32 input
    device float* output [[buffer(1)]],           // [N, D] - FP32 output
    device const half* weight [[buffer(2)]],      // [D, 4*D] - FP16 weights
    device const half* bias [[buffer(3)]],        // [D] - FP16 bias
    constant int2& dims [[buffer(4)]],            // N, D
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;
    const int D4 = 4 * D;

    int n = gid.x;
    int d = gid.y;

    if (n >= N || d >= D) return;

    float sum = float(bias[d]);
    for (int i = 0; i < D4; i++) {
        sum += input[n * D4 + i] * float(weight[d * D4 + i]);
    }

    output[n * D + d] = sum;
}

// ============================================================================
// Residual Addition
// ============================================================================

kernel void residual_add(
    device float* x [[buffer(0)]],               // [N, D] in/out (FP32)
    device const float* residual [[buffer(1)]],  // [N, D] (FP32)
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(size)) return;
    x[gid] += residual[gid];
}

// ============================================================================
// Linear Projection (general purpose)
// ============================================================================

kernel void linear(
    device const float* input [[buffer(0)]],      // [N, D_in] (FP32 data)
    device float* output [[buffer(1)]],           // [N, D_out] (FP32 data)
    device const half* weight [[buffer(2)]],      // [D_out, D_in] (FP16 weights)
    device const half* bias [[buffer(3)]],        // [D_out] (FP16 weights)
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

// ============================================================================
// RoPE 2D (for MASt3R/CroCoNet)
// ============================================================================

/**
 * Generate 2D RoPE positions for a grid.
 *
 * Output: [num_patches, 2] with (y, x) coordinates
 */
kernel void generate_rope_positions(
    device float* positions [[buffer(0)]],        // [num_patches, 2]
    constant int2& grid_size [[buffer(1)]],       // patches_h, patches_w
    uint gid [[thread_position_in_grid]]
) {
    const int patches_h = grid_size.x;
    const int patches_w = grid_size.y;
    const int num_patches = patches_h * patches_w;

    if (gid >= uint(num_patches)) return;

    int py = gid / patches_w;
    int px = gid % patches_w;

    positions[gid * 2 + 0] = float(py);  // y coordinate
    positions[gid * 2 + 1] = float(px);  // x coordinate
}

/**
 * Apply RoPE 2D to Q and K tensors.
 *
 * RoPE 2D splits head_dim into two halves:
 * - First half (indices 0 to head_dim/2-1) uses Y position
 * - Second half (indices head_dim/2 to head_dim-1) uses X position
 *
 * Within each half, rotate_half pairs index d with index d + D/2:
 * - For d in [0, D/4): rotate pair (d, d + D/4) with positive/negative
 *
 * Python reference (pos_embed.py):
 *   y, x = tokens.chunk(2, dim=-1)  # Split into Y and X halves
 *   y = apply_rope1d(y, positions[:,:,0], cos, sin)  # Y positions
 *   x = apply_rope1d(x, positions[:,:,1], cos, sin)  # X positions
 *
 * apply_rope1d with rotate_half:
 *   rotate_half([a1..aD/2, b1..bD/2]) = [-b1..-bD/2, a1..aD/2]
 *   result = tokens * cos + rotate_half(tokens) * sin
 */
kernel void apply_rope_2d(
    device float* qk [[buffer(0)]],               // [N, num_heads, head_dim] in/out (FP32)
    device const float* positions [[buffer(1)]],  // [N, 2] y,x positions (FP32)
    constant int3& dims [[buffer(2)]],            // N, num_heads, head_dim
    constant float& freq_base [[buffer(3)]],      // typically 100.0
    uint3 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;

    // head_dim is split into 2 halves: Y half and X half
    // Each half has D = head_dim / 2 dimensions
    const int D = head_dim / 2;        // e.g., 32 for head_dim=64
    const int D_half = D / 2;          // e.g., 16 - number of rotation pairs per half

    int n = gid.x;      // sequence position
    int h = gid.y;      // head index
    int p = gid.z;      // rotation pair index (0 to D_half - 1)

    if (n >= N || h >= num_heads || p >= D_half) return;

    // Get positions
    float pos_y = positions[n * 2 + 0];
    float pos_x = positions[n * 2 + 1];

    // Compute frequency for this pair
    float inv_freq = 1.0f / pow(freq_base, float(2 * p) / float(D));

    // Base offset for this token and head
    int base = n * num_heads * head_dim + h * head_dim;

    // ========== Y HALF (indices 0 to D-1) ==========
    {
        float angle = pos_y * inv_freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        int idx1 = base + p;
        int idx2 = base + p + D_half;

        float v1 = qk[idx1];
        float v2 = qk[idx2];

        qk[idx1] = v1 * cos_val - v2 * sin_val;
        qk[idx2] = v2 * cos_val + v1 * sin_val;
    }

    // ========== X HALF (indices D to head_dim-1) ==========
    {
        float angle = pos_x * inv_freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        int idx1 = base + D + p;
        int idx2 = base + D + p + D_half;

        float v1 = qk[idx1];
        float v2 = qk[idx2];

        qk[idx1] = v1 * cos_val - v2 * sin_val;
        qk[idx2] = v2 * cos_val + v1 * sin_val;
    }
}

// ============================================================================
// QKV Split and Reshape for Multi-Head Attention
// ============================================================================

/**
 * Split QKV projection output into Q, K, V.
 *
 * Input:  [N, 3*embed_dim]
 * Output: Q, K, V each [N, num_heads, head_dim]
 */
kernel void split_qkv(
    device const float* qkv [[buffer(0)]],        // [N, 3*D]
    device float* Q [[buffer(1)]],                // [N, num_heads, head_dim]
    device float* K [[buffer(2)]],
    device float* V [[buffer(3)]],
    constant int3& dims [[buffer(4)]],            // N, num_heads, head_dim
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;
    const int D = num_heads * head_dim;

    int n = gid.x;
    int hd = gid.y;  // combined head*head_dim index

    if (n >= N || hd >= D) return;

    int h = hd / head_dim;
    int d = hd % head_dim;

    int out_idx = n * D + h * head_dim + d;
    int qkv_base = n * 3 * D;

    Q[out_idx] = qkv[qkv_base + hd];
    K[out_idx] = qkv[qkv_base + D + hd];
    V[out_idx] = qkv[qkv_base + 2 * D + hd];
}

// ============================================================================
// Simple Parallel Self-Attention (3-pass approach)
// ============================================================================

/**
 * Pass 1: Compute attention scores Q @ K^T
 * Each thread computes one element of the attention matrix.
 * Grid: [N, N, num_heads]
 */
kernel void attention_scores(
    device const float* Q [[buffer(0)]],          // [N, num_heads, head_dim] (FP32)
    device const float* K [[buffer(1)]],          // [N, num_heads, head_dim] (FP32)
    device float* scores [[buffer(2)]],           // [num_heads, N, N] (FP32)
    constant int3& dims [[buffer(3)]],            // N, num_heads, head_dim
    uint3 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;

    const int i = gid.x;  // Query index
    const int j = gid.y;  // Key index
    const int h = gid.z;  // Head index

    if (i >= N || j >= N || h >= num_heads) return;

    const float scale = rsqrt(float(head_dim));
    const int stride = num_heads * head_dim;

    // Compute dot product Q[i, h, :] @ K[j, h, :]
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += Q[i * stride + h * head_dim + d] *
               K[j * stride + h * head_dim + d];
    }

    scores[h * N * N + i * N + j] = dot * scale;
}

/**
 * Pass 2: Softmax over each row of attention scores
 * Each thread processes one row.
 * Grid: [N, num_heads, 1]
 */
kernel void attention_softmax(
    device float* scores [[buffer(0)]],           // [num_heads, N, N] (FP32)
    constant int2& dims [[buffer(1)]],            // N, num_heads
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;

    const int i = gid.x;  // Row index
    const int h = gid.y;  // Head index

    if (i >= N || h >= num_heads) return;

    const int row_start = h * N * N + i * N;

    // Find max for numerical stability
    float max_val = scores[row_start];
    for (int j = 1; j < N; j++) {
        max_val = max(max_val, scores[row_start + j]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        float exp_val = exp(scores[row_start + j] - max_val);
        scores[row_start + j] = exp_val;
        sum += exp_val;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < N; j++) {
        scores[row_start + j] = scores[row_start + j] * inv_sum;
    }
}

/**
 * Pass 3: Compute attention output = scores @ V
 * Each thread computes one element of the output.
 * Grid: [N, head_dim, num_heads]
 */
kernel void attention_output(
    device const float* scores [[buffer(0)]],     // [num_heads, N, N] (FP32)
    device const float* V [[buffer(1)]],          // [N, num_heads, head_dim] (FP32)
    device float* output [[buffer(2)]],           // [N, num_heads, head_dim] (FP32)
    constant int3& dims [[buffer(3)]],            // N, num_heads, head_dim
    uint3 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;

    const int i = gid.x;      // Output row (query index)
    const int d = gid.y;      // Output dimension
    const int h = gid.z;      // Head index

    if (i >= N || d >= head_dim || h >= num_heads) return;

    const int stride = num_heads * head_dim;
    const int score_row = h * N * N + i * N;

    // Weighted sum: sum_j scores[i,j] * V[j,h,d]
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += scores[score_row + j] * V[j * stride + h * head_dim + d];
    }

    output[i * stride + h * head_dim + d] = sum;
}

// ============================================================================
// Optimized Self-Attention with Tiling and SIMD (legacy)
// ============================================================================

// Tile sizes for attention computation
constant int TILE_Q = 32;       // Queries per tile
constant int TILE_K = 32;       // Keys per tile
constant int WARP_SIZE = 32;    // Metal SIMD width

/**
 * Optimized multi-head self-attention using tiling and shared memory.
 *
 * Each threadgroup handles TILE_Q queries.
 * Threads cooperate to:
 * 1. Load K/V tiles into shared memory
 * 2. Compute Q @ K^T dot products in parallel
 * 3. Use SIMD for fast softmax (max, sum reductions)
 * 4. Accumulate attention output
 *
 * Grid: [ceil(N/TILE_Q), num_heads]
 * Threadgroup: [TILE_K, TILE_Q] or [32, 32]
 */
kernel void multihead_attention_optimized(
    device const float* Q [[buffer(0)]],          // [N, num_heads, head_dim] (FP32)
    device const float* K [[buffer(1)]],          // [N, num_heads, head_dim] (FP32)
    device const float* V [[buffer(2)]],          // [N, num_heads, head_dim] (FP32)
    device float* output [[buffer(3)]],           // [N, num_heads, head_dim] (FP32)
    constant int3& dims [[buffer(4)]],            // N, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;
    const float scale = rsqrt(float(head_dim));

    const int h = tgid.y;  // Head index
    const int q_tile_start = tgid.x * TILE_Q;

    if (h >= num_heads) return;

    const int stride = num_heads * head_dim;

    // Shared memory for K and V tiles (use float for accumulation precision)
    threadgroup float k_tile[TILE_K][64 + 1];  // +1 to avoid bank conflicts
    threadgroup float v_tile[TILE_K][64 + 1];

    // Each thread handles one query position within the tile
    const int local_q = tid.y;
    const int global_q = q_tile_start + local_q;

    // Online softmax accumulators (per thread/query)
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Output accumulator - stored in registers
    float acc[64];  // Max head_dim
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Load Q for this thread once
    float q_local[64];
    if (global_q < N) {
        for (int d = 0; d < head_dim; d++) {
            q_local[d] = Q[global_q * stride + h * head_dim + d];
        }
    }

    // Iterate over K/V tiles
    for (int k_tile_start = 0; k_tile_start < N; k_tile_start += TILE_K) {
        // Cooperative loading of K and V tiles into shared memory
        const int local_k = tid.x;
        const int global_k = k_tile_start + local_k;

        // Each thread loads one row of K and V
        if (global_k < N && local_q < TILE_K) {
            for (int d = tid.y; d < head_dim; d += TILE_Q) {
                k_tile[local_k][d] = K[global_k * stride + h * head_dim + d];
                v_tile[local_k][d] = V[global_k * stride + h * head_dim + d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention scores for this tile
        if (global_q < N) {
            // Find tile max for numerical stability
            float tile_max = -INFINITY;

            for (int j = 0; j < TILE_K && (k_tile_start + j) < N; j++) {
                // Compute Q[q] @ K[k_tile_start + j]
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_local[d] * k_tile[j][d];
                }
                dot *= scale;
                tile_max = max(tile_max, dot);
            }

            // Online softmax update
            float old_max = row_max;
            row_max = max(row_max, tile_max);
            float rescale = exp(old_max - row_max);

            // Rescale previous accumulations
            row_sum *= rescale;
            for (int d = 0; d < head_dim; d++) {
                acc[d] *= rescale;
            }

            // Add this tile's contribution
            for (int j = 0; j < TILE_K && (k_tile_start + j) < N; j++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_local[d] * k_tile[j][d];
                }
                dot *= scale;

                float w = exp(dot - row_max);
                row_sum += w;

                for (int d = 0; d < head_dim; d++) {
                    acc[d] += w * v_tile[j][d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write output
    if (global_q < N) {
        float inv_sum = 1.0f / row_sum;
        for (int d = 0; d < head_dim; d++) {
            output[global_q * stride + h * head_dim + d] = acc[d] * inv_sum;
        }
    }
}

/**
 * Ultra-optimized attention using SIMD intrinsics.
 *
 * Each SIMD group (32 threads) handles one query.
 * Threads within the group compute partial dot products in parallel.
 * Uses simd_sum for fast reductions.
 */
kernel void multihead_attention_simd(
    device const float* Q [[buffer(0)]],          // [N, num_heads, head_dim] (FP32)
    device const float* K [[buffer(1)]],          // [N, num_heads, head_dim] (FP32)
    device const float* V [[buffer(2)]],          // [N, num_heads, head_dim] (FP32)
    device float* output [[buffer(3)]],           // [N, num_heads, head_dim] (FP32)
    constant int3& dims [[buffer(4)]],            // N, num_heads, head_dim
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;
    const float scale = rsqrt(float(head_dim));

    // Each thread handles one (query, head) pair
    // SIMD lanes cooperate on dot product computation
    const int q_idx = gid.x;
    const int h = gid.y;

    if (q_idx >= N || h >= num_heads) return;

    const int stride = num_heads * head_dim;

    // Load Q for this query (all lanes load same Q)
    float q_local[64];
    for (int d = 0; d < head_dim; d++) {
        q_local[d] = Q[q_idx * stride + h * head_dim + d];
    }

    // Online softmax accumulators
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc[64];
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Process keys in chunks - each SIMD lane handles different keys
    for (int k_base = 0; k_base < N; k_base += WARP_SIZE) {
        int k_idx = k_base + simd_lane;

        float dot = 0.0f;
        if (k_idx < N) {
            // Compute dot product Q @ K^T
            for (int d = 0; d < head_dim; d++) {
                dot += q_local[d] * K[k_idx * stride + h * head_dim + d];
            }
            dot *= scale;
        } else {
            dot = -INFINITY;
        }

        // SIMD reduction to find max across lanes
        float tile_max = simd_max(dot);

        // Online softmax update
        float old_max = row_max;
        row_max = max(row_max, tile_max);
        float rescale = exp(old_max - row_max);

        row_sum *= rescale;
        for (int d = 0; d < head_dim; d++) {
            acc[d] *= rescale;
        }

        // Compute exp(dot - row_max) and accumulate V contribution
        float w = (k_idx < N) ? exp(dot - row_max) : 0.0f;

        // Sum weights across SIMD lanes
        row_sum += simd_sum(w);

        // Each lane contributes its weighted V to shared accumulator
        if (k_idx < N) {
            for (int d = 0; d < head_dim; d++) {
                float v_val = V[k_idx * stride + h * head_dim + d];
                acc[d] += simd_sum(w * v_val);
            }
        }
    }

    // Only lane 0 writes output (all lanes have same result due to simd_sum)
    if (simd_lane == 0) {
        float inv_sum = 1.0f / row_sum;
        for (int d = 0; d < head_dim; d++) {
            output[q_idx * stride + h * head_dim + d] = acc[d] * inv_sum;
        }
    }
}

/**
 * Cross-attention optimized with tiling.
 */
kernel void cross_attention_optimized(
    device const float* Q [[buffer(0)]],          // [N_dec, num_heads, head_dim] (FP32)
    device const float* K [[buffer(1)]],          // [N_enc, num_heads, head_dim] (FP32)
    device const float* V [[buffer(2)]],          // [N_enc, num_heads, head_dim] (FP32)
    device float* output [[buffer(3)]],           // [N_dec, num_heads, head_dim] (FP32)
    constant int4& dims [[buffer(4)]],            // N_dec, N_enc, num_heads, head_dim
    uint2 gid [[thread_position_in_grid]]
) {
    const int N_dec = dims.x;
    const int N_enc = dims.y;
    const int num_heads = dims.z;
    const int head_dim = dims.w;
    const float scale = rsqrt(float(head_dim));

    int i = gid.x;   // Decoder query position
    int h = gid.y;   // Head index

    if (i >= N_dec || h >= num_heads) return;

    const int stride = num_heads * head_dim;

    // Load Q once
    float q_local[64];
    for (int d = 0; d < head_dim; d++) {
        q_local[d] = Q[i * stride + h * head_dim + d];
    }

    // Online softmax
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc[64];
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Two-pass for numerical stability on long sequences
    // Pass 1: Find max
    for (int j = 0; j < N_enc; j++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_local[d] * K[j * stride + h * head_dim + d];
        }
        row_max = max(row_max, dot * scale);
    }

    // Pass 2: Compute softmax and accumulate
    for (int j = 0; j < N_enc; j++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_local[d] * K[j * stride + h * head_dim + d];
        }
        dot *= scale;

        float w = exp(dot - row_max);
        row_sum += w;

        for (int d = 0; d < head_dim; d++) {
            acc[d] += w * V[j * stride + h * head_dim + d];
        }
    }

    // Normalize and write
    float inv_sum = 1.0f / row_sum;
    for (int d = 0; d < head_dim; d++) {
        output[i * stride + h * head_dim + d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// Original Self-Attention (kept for reference/fallback)
// ============================================================================

/**
 * Compute attention scores: Q @ K^T / sqrt(head_dim)
 *
 * Input Q:  [N, num_heads, head_dim] (FP32)
 * Input K:  [N, num_heads, head_dim] (FP32)
 * Output:   [num_heads, N, N] attention scores (pre-softmax) (FP32)
 */
kernel void compute_attention_scores(
    device const float* Q [[buffer(0)]],          // [N, num_heads, head_dim] (FP32)
    device const float* K [[buffer(1)]],          // [N, num_heads, head_dim] (FP32)
    device float* scores [[buffer(2)]],           // [num_heads, N, N] (FP32)
    constant int3& dims [[buffer(3)]],            // N, num_heads, head_dim
    uint3 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;
    const float scale = rsqrt(float(head_dim));

    int h = gid.z;   // head index
    int i = gid.y;   // query position
    int j = gid.x;   // key position

    if (h >= num_heads || i >= N || j >= N) return;

    // Compute Q[i] @ K[j]^T for head h
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float q_val = Q[i * num_heads * head_dim + h * head_dim + d];
        float k_val = K[j * num_heads * head_dim + h * head_dim + d];
        dot += q_val * k_val;
    }

    // Store scaled attention score
    scores[h * N * N + i * N + j] = dot * scale;
}

/**
 * Softmax on attention scores (per row, per head) - legacy version.
 * NOTE: Renamed to avoid conflict with attention_softmax in 3-pass approach.
 */
kernel void attention_softmax_legacy(
    device float* scores [[buffer(0)]],           // [num_heads, N, N] in/out (FP32)
    constant int2& dims [[buffer(1)]],            // num_heads, N
    uint2 gid [[thread_position_in_grid]]
) {
    const int num_heads = dims.x;
    const int N = dims.y;

    int h = gid.y;   // head index
    int i = gid.x;   // query position (row)

    if (h >= num_heads || i >= N) return;

    const int row_start = h * N * N + i * N;

    // Find max for numerical stability
    float max_val = scores[row_start];
    for (int j = 1; j < N; j++) {
        max_val = max(max_val, scores[row_start + j]);
    }

    // Exp and sum
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        float exp_val = exp(scores[row_start + j] - max_val);
        scores[row_start + j] = exp_val;
        sum += exp_val;
    }

    // Normalize
    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int j = 0; j < N; j++) {
        scores[row_start + j] = scores[row_start + j] * inv_sum;
    }
}

/**
 * Apply attention: output = attn_probs @ V
 *
 * Input attn:  [num_heads, N, N] - attention probabilities (FP32)
 * Input V:     [N, num_heads, head_dim] (FP32)
 * Output:      [N, num_heads, head_dim] (FP32)
 */
kernel void apply_attention(
    device const float* attn [[buffer(0)]],       // [num_heads, N, N] (FP32)
    device const float* V [[buffer(1)]],          // [N, num_heads, head_dim] (FP32)
    device float* output [[buffer(2)]],           // [N, num_heads, head_dim] (FP32)
    constant int3& dims [[buffer(3)]],            // N, num_heads, head_dim
    uint3 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;

    int i = gid.x;   // query position
    int h = gid.y;   // head index
    int d = gid.z;   // dimension

    if (i >= N || h >= num_heads || d >= head_dim) return;

    // output[i, h, d] = sum_j(attn[h, i, j] * V[j, h, d])
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        float attn_weight = attn[h * N * N + i * N + j];
        float v_val = V[j * num_heads * head_dim + h * head_dim + d];
        sum += attn_weight * v_val;
    }

    output[i * num_heads * head_dim + h * head_dim + d] = sum;
}

/**
 * Fused self-attention for a single head (optimized).
 *
 * Computes: softmax(Q @ K^T / sqrt(d)) @ V
 * Uses online softmax to avoid materializing full attention matrix.
 *
 * Each thread handles one query position.
 */
kernel void self_attention_fused(
    device const float* Q [[buffer(0)]],          // [N, head_dim] (FP32)
    device const float* K [[buffer(1)]],          // [N, head_dim] (FP32)
    device const float* V [[buffer(2)]],          // [N, head_dim] (FP32)
    device float* output [[buffer(3)]],           // [N, head_dim] (FP32)
    constant int2& dims [[buffer(4)]],            // N, head_dim
    uint gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int head_dim = dims.y;
    const float scale = rsqrt(float(head_dim));

    int i = gid;  // query position
    if (i >= N) return;

    // Online softmax variables
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Output accumulator
    float acc[128];  // Max head_dim = 128
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Iterate over all keys
    for (int j = 0; j < N; j++) {
        // Compute Q[i] @ K[j]^T
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[i * head_dim + d] * K[j * head_dim + d];
        }
        dot *= scale;

        // Online softmax update
        float old_max = row_max;
        row_max = max(row_max, dot);
        float rescale = exp(old_max - row_max);
        float exp_score = exp(dot - row_max);

        // Rescale previous accumulation
        row_sum = row_sum * rescale + exp_score;
        for (int d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * rescale + exp_score * V[j * head_dim + d];
        }
    }

    // Final normalization and write output
    float inv_sum = 1.0f / row_sum;
    for (int d = 0; d < head_dim; d++) {
        output[i * head_dim + d] = acc[d] * inv_sum;
    }
}

/**
 * Multi-head self-attention.
 *
 * Dispatched with grid [N, num_heads].
 */
kernel void multihead_self_attention(
    device const float* Q [[buffer(0)]],          // [N, num_heads, head_dim] (FP32)
    device const float* K [[buffer(1)]],          // [N, num_heads, head_dim] (FP32)
    device const float* V [[buffer(2)]],          // [N, num_heads, head_dim] (FP32)
    device float* output [[buffer(3)]],           // [N, num_heads, head_dim] (FP32)
    constant int3& dims [[buffer(4)]],            // N, num_heads, head_dim
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;
    const float scale = rsqrt(float(head_dim));

    int i = gid.x;   // query position
    int h = gid.y;   // head index

    if (i >= N || h >= num_heads) return;

    // Pointers for this head
    const int stride = num_heads * head_dim;

    // Online softmax variables
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Output accumulator (max head_dim = 128)
    float acc[128];
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Iterate over all keys
    for (int j = 0; j < N; j++) {
        // Compute Q[i,h] @ K[j,h]^T
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = Q[i * stride + h * head_dim + d];
            float k_val = K[j * stride + h * head_dim + d];
            dot += q_val * k_val;
        }
        dot *= scale;

        // Online softmax update
        float old_max = row_max;
        row_max = max(row_max, dot);
        float rescale = exp(old_max - row_max);
        float exp_score = exp(dot - row_max);

        // Rescale previous accumulation
        row_sum = row_sum * rescale + exp_score;
        for (int d = 0; d < head_dim; d++) {
            float v_val = V[j * stride + h * head_dim + d];
            acc[d] = acc[d] * rescale + exp_score * v_val;
        }
    }

    // Final normalization and write output
    float inv_sum = 1.0f / row_sum;
    for (int d = 0; d < head_dim; d++) {
        output[i * stride + h * head_dim + d] = acc[d] * inv_sum;
    }
}

/**
 * Cross-attention for decoder.
 *
 * Q from decoder tokens, K/V from encoder output.
 *
 * Q:      [N_dec, num_heads, head_dim] (FP32)
 * K, V:   [N_enc, num_heads, head_dim] (FP32)
 * Output: [N_dec, num_heads, head_dim] (FP32)
 */
kernel void cross_attention(
    device const float* Q [[buffer(0)]],          // [N_dec, num_heads, head_dim] (FP32)
    device const float* K [[buffer(1)]],          // [N_enc, num_heads, head_dim] (FP32)
    device const float* V [[buffer(2)]],          // [N_enc, num_heads, head_dim] (FP32)
    device float* output [[buffer(3)]],           // [N_dec, num_heads, head_dim] (FP32)
    constant int4& dims [[buffer(4)]],            // N_dec, N_enc, num_heads, head_dim
    uint2 gid [[thread_position_in_grid]]
) {
    const int N_dec = dims.x;
    const int N_enc = dims.y;
    const int num_heads = dims.z;
    const int head_dim = dims.w;
    const float scale = rsqrt(float(head_dim));

    int i = gid.x;   // decoder query position
    int h = gid.y;   // head index

    if (i >= N_dec || h >= num_heads) return;

    const int stride_dec = num_heads * head_dim;
    const int stride_enc = num_heads * head_dim;

    // Online softmax variables
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Output accumulator
    float acc[128];
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Iterate over all encoder keys
    for (int j = 0; j < N_enc; j++) {
        // Compute Q[i,h] @ K[j,h]^T
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = Q[i * stride_dec + h * head_dim + d];
            float k_val = K[j * stride_enc + h * head_dim + d];
            dot += q_val * k_val;
        }
        dot *= scale;

        // Online softmax update
        float old_max = row_max;
        row_max = max(row_max, dot);
        float rescale = exp(old_max - row_max);
        float exp_score = exp(dot - row_max);

        row_sum = row_sum * rescale + exp_score;
        for (int d = 0; d < head_dim; d++) {
            float v_val = V[j * stride_enc + h * head_dim + d];
            acc[d] = acc[d] * rescale + exp_score * v_val;
        }
    }

    // Final normalization
    float inv_sum = 1.0f / row_sum;
    for (int d = 0; d < head_dim; d++) {
        output[i * stride_dec + h * head_dim + d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// Output Projection after Attention
// ============================================================================

kernel void attention_output_proj(
    device const float* attn_out [[buffer(0)]],   // [N, num_heads, head_dim] (FP32)
    device float* output [[buffer(1)]],           // [N, embed_dim] (FP32)
    device const half* weight [[buffer(2)]],      // [embed_dim, embed_dim] (FP16 weights)
    device const half* bias [[buffer(3)]],        // [embed_dim] (FP16 weights)
    constant int2& dims [[buffer(4)]],            // N, embed_dim
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;

    int n = gid.x;
    int d = gid.y;

    if (n >= N || d >= D) return;

    float sum = float(bias[d]);
    for (int i = 0; i < D; i++) {
        sum += attn_out[n * D + i] * float(weight[d * D + i]);
    }

    output[n * D + d] = sum;
}

// ============================================================================
// Buffer Copy (GPU-only, avoids CPU<->GPU sync)
// ============================================================================

/**
 * Fast GPU buffer copy to avoid CPU synchronization.
 * Use instead of std::memcpy on Metal buffers.
 */
kernel void buffer_copy(
    device const float* src [[buffer(0)]],        // (FP32)
    device float* dst [[buffer(1)]],              // (FP32)
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(size)) {
        dst[gid] = src[gid];
    }
}

/**
 * Vectorized buffer copy (4x throughput).
 * Size = number of elements / 4.
 */
kernel void buffer_copy_vec4(
    device const float4* src [[buffer(0)]],       // (FP32 vectorized)
    device float4* dst [[buffer(1)]],             // (FP32 vectorized)
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(size)) {
        dst[gid] = src[gid];
    }
}
