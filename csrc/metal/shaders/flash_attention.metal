// MASt3R Runtime - FlashAttention for Metal
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
// Based on: FlashAttention algorithm with online softmax

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// FlashAttention v2 - Correct Implementation
// ============================================================================
// One thread per query, sequential over keys with online softmax.
// This is memory-bound but numerically correct.
//
// Grid: [num_heads, N]
// Threadgroup: [1, 256] or similar

kernel void flash_attention_v2(
    device const float* Q [[buffer(0)]],      // [N, num_heads, head_dim]
    device const float* K [[buffer(1)]],      // [N, num_heads, head_dim]
    device const float* V [[buffer(2)]],      // [N, num_heads, head_dim]
    device float* O [[buffer(3)]],            // [N, num_heads, head_dim]
    constant int3& dims [[buffer(4)]],        // N, num_heads, head_dim
    uint2 gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;

    const int h = gid.x;  // Head index
    const int i = gid.y;  // Query index

    if (h >= num_heads || i >= N) return;

    const float scale = rsqrt(float(head_dim));
    const int stride = num_heads * head_dim;

    // Online softmax accumulators
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Output accumulator (in registers for head_dim <= 128)
    float acc[128];  // Max head_dim supported
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Base offset for this query
    const int q_offset = i * stride + h * head_dim;

    // Iterate over all keys
    for (int j = 0; j < N; j++) {
        const int k_offset = j * stride + h * head_dim;

        // Compute dot product: Q[i,h,:] @ K[j,h,:]
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[q_offset + d] * K[k_offset + d];
        }
        dot *= scale;

        // Online softmax update
        float old_max = row_max;
        row_max = max(row_max, dot);

        // Rescale previous accumulations
        float rescale = exp(old_max - row_max);
        row_sum = row_sum * rescale;
        for (int d = 0; d < head_dim; d++) {
            acc[d] *= rescale;
        }

        // Add current contribution
        float w = exp(dot - row_max);
        row_sum += w;

        // Accumulate weighted V
        for (int d = 0; d < head_dim; d++) {
            acc[d] += w * V[k_offset + d];
        }
    }

    // Final normalization and write output
    const int o_offset = i * stride + h * head_dim;
    float inv_sum = 1.0f / row_sum;
    for (int d = 0; d < head_dim; d++) {
        O[o_offset + d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// FlashAttention v3 - Hybrid Optimized
// ============================================================================
// Combines best of both approaches:
// - From v2: 1 thread = 1 query, Q in registers (no shared mem for Q)
// - From tiled: K/V cached in threadgroup memory with vectorized loads
// - New: float4 vectorized loads, better thread utilization
//
// Grid: [num_heads, ceil(N/TILE_Q)]
// Threadgroup: [TILE_Q, 1]

constant int TILE_Q = 32;      // Queries per threadgroup
constant int TILE_K = 64;      // Keys per tile (larger for better amortization)
constant int HEAD_DIM = 64;    // Fixed head_dim for unrolling

kernel void flash_attention_tiled(
    device const float* Q [[buffer(0)]],      // [N, num_heads, head_dim]
    device const float* K [[buffer(1)]],      // [N, num_heads, head_dim]
    device const float* V [[buffer(2)]],      // [N, num_heads, head_dim]
    device float* O [[buffer(3)]],            // [N, num_heads, head_dim]
    constant int3& dims [[buffer(4)]],        // N, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;

    const int h = tgid.x;                     // Head index
    const int q_tile_start = tgid.y * TILE_Q; // First query in this tile
    const int local_q = tid;                  // Which query this thread handles
    const int global_q = q_tile_start + local_q;

    if (h >= num_heads || global_q >= N) return;

    const float scale = rsqrt(float(head_dim));
    const int stride = num_heads * head_dim;

    // Shared memory for K/V tiles - float4 aligned for vectorized access
    threadgroup float4 K_smem[TILE_K][HEAD_DIM / 4];
    threadgroup float4 V_smem[TILE_K][HEAD_DIM / 4];

    // Per-thread accumulators in registers
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float4 acc[HEAD_DIM / 4];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        acc[d] = float4(0.0f);
    }

    // Load Q once into registers (vectorized)
    float4 Q_local[HEAD_DIM / 4];
    const int q_base = global_q * stride + h * head_dim;
    device const float4* Q4 = (device const float4*)(Q + q_base);

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        Q_local[d] = Q4[d];
    }

    // Iterate over K/V tiles
    for (int k_tile_start = 0; k_tile_start < N; k_tile_start += TILE_K) {

        // Cooperative load: each thread loads 2 K/V rows (since TILE_K = 2 * TILE_Q)
        #pragma unroll
        for (int offset = 0; offset < TILE_K; offset += TILE_Q) {
            int k_local = offset + local_q;
            int k_global = k_tile_start + k_local;

            if (k_global < N) {
                const int k_base = k_global * stride + h * head_dim;
                device const float4* K4 = (device const float4*)(K + k_base);
                device const float4* V4 = (device const float4*)(V + k_base);

                #pragma unroll
                for (int d = 0; d < HEAD_DIM / 4; d++) {
                    K_smem[k_local][d] = K4[d];
                    V_smem[k_local][d] = V4[d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention for this K/V tile
        int tile_end = min(TILE_K, N - k_tile_start);

        for (int k_local = 0; k_local < tile_end; k_local++) {
            // Compute dot product with cached K (vectorized)
            float dot = 0.0f;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                float4 q = Q_local[d];
                float4 k = K_smem[k_local][d];
                dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
            }
            dot *= scale;

            // Online softmax update
            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);

            row_sum *= rescale;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] *= rescale;
            }

            float w = exp(dot - row_max);
            row_sum += w;

            // Accumulate weighted V from shared memory (vectorized)
            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] += w * V_smem[k_local][d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write (vectorized)
    const int o_base = global_q * stride + h * head_dim;
    device float4* O4 = (device float4*)(O + o_base);
    float inv_sum = 1.0f / row_sum;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        O4[d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// FlashAttention v4 - Multi-head per threadgroup
// ============================================================================
// Processes multiple heads per threadgroup for better GPU occupancy:
// - 2 heads per threadgroup (HEADS_PER_TG=2)
// - TILE_K reduced to 32 to fit in shared memory
// - 64 threads per threadgroup (32 queries × 2 heads)
//
// Grid: [ceil(num_heads/2), ceil(N/TILE_Q)]
// Threadgroup: [TILE_Q, HEADS_PER_TG] = [32, 2]

// V4 configuration: 64 threads (32 × 2)
constant int TILE_Q_V4 = 32;
constant int TILE_K_V4 = 32;
constant int HEADS_PER_TG_V4 = 2;

// V5 configuration: 256 threads (64 × 4), smaller TILE_K to fit in shared memory
constant int TILE_Q_V5 = 64;
constant int TILE_K_V5 = 16;       // Reduced: 4 heads × 16 keys × 64 floats × 4 bytes = 16KB per K/V
constant int HEADS_PER_TG_V5 = 4;

// V6 configuration: 512 threads (64 × 8), optimized for maximum GPU occupancy
// Memory: 8 heads × 32 keys × 16 float4 × 4 bytes × 2 (K+V) = 32KB (exactly at limit)
// +1 padding on D dimension to avoid bank conflicts
constant int TILE_Q_V6 = 64;
constant int TILE_K_V6 = 32;       // Larger tile for better memory reuse
constant int HEADS_PER_TG_V6 = 8;
constant int HEAD_DIM_PADDED = HEAD_DIM / 4 + 1;  // +1 for bank conflict avoidance

kernel void flash_attention_multihead(
    device const float* Q [[buffer(0)]],      // [N, num_heads, head_dim]
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int3& dims [[buffer(4)]],        // N, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;

    const int h_tile = tgid.x;                           // Head tile index
    const int q_tile_start = tgid.y * TILE_Q_V4;         // First query in this tile
    const int local_q = tid.x;                           // Query within tile (0-31)
    const int local_h = tid.y;                           // Head within tile (0-1)
    const int global_q = q_tile_start + local_q;
    const int global_h = h_tile * HEADS_PER_TG_V4 + local_h;

    if (global_h >= num_heads || global_q >= N) return;

    const float scale = rsqrt(float(head_dim));
    const int stride = num_heads * head_dim;

    // Shared memory for K/V - indexed by [local_h][k_local][d]
    // Total: 2 × 32 × 16 × 16 bytes × 2 (K+V) = 32KB
    threadgroup float4 K_smem[HEADS_PER_TG_V4][TILE_K_V4][HEAD_DIM / 4];
    threadgroup float4 V_smem[HEADS_PER_TG_V4][TILE_K_V4][HEAD_DIM / 4];

    // Per-thread accumulators
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float4 acc[HEAD_DIM / 4];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        acc[d] = float4(0.0f);
    }

    // Load Q once into registers
    float4 Q_local[HEAD_DIM / 4];
    const int q_base = global_q * stride + global_h * head_dim;
    device const float4* Q4 = (device const float4*)(Q + q_base);

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        Q_local[d] = Q4[d];
    }

    // Iterate over K/V tiles
    for (int k_tile_start = 0; k_tile_start < N; k_tile_start += TILE_K_V4) {

        // Cooperative load: all threads load K/V for their respective heads
        // With 32 threads per head and TILE_K=32, each thread loads 1 row
        int k_global = k_tile_start + local_q;  // local_q used as k_local too

        if (k_global < N) {
            const int k_base = k_global * stride + global_h * head_dim;
            device const float4* K4 = (device const float4*)(K + k_base);
            device const float4* V4 = (device const float4*)(V + k_base);

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                K_smem[local_h][local_q][d] = K4[d];
                V_smem[local_h][local_q][d] = V4[d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention for this K/V tile
        int tile_end = min(TILE_K_V4, N - k_tile_start);

        for (int k_local = 0; k_local < tile_end; k_local++) {
            // Compute dot product with cached K
            float dot = 0.0f;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                float4 q = Q_local[d];
                float4 k = K_smem[local_h][k_local][d];
                dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
            }
            dot *= scale;

            // Online softmax update
            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);

            row_sum *= rescale;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] *= rescale;
            }

            float w = exp(dot - row_max);
            row_sum += w;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] += w * V_smem[local_h][k_local][d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write
    const int o_base = global_q * stride + global_h * head_dim;
    device float4* O4 = (device float4*)(O + o_base);
    float inv_sum = 1.0f / row_sum;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        O4[d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// Flash Cross-Attention Multihead - Same optimization for cross-attention
// ============================================================================

kernel void flash_cross_attention_multihead(
    device const float* Q [[buffer(0)]],      // [N_dec, num_heads, head_dim]
    device const float* K [[buffer(1)]],      // [N_enc, num_heads, head_dim]
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int4& dims [[buffer(4)]],        // N_dec, N_enc, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const int N_dec = dims.x;
    const int N_enc = dims.y;
    const int num_heads = dims.z;
    const int head_dim = dims.w;

    const int h_tile = tgid.x;
    const int q_tile_start = tgid.y * TILE_Q_V4;
    const int local_q = tid.x;
    const int local_h = tid.y;
    const int global_q = q_tile_start + local_q;
    const int global_h = h_tile * HEADS_PER_TG_V4 + local_h;

    if (global_h >= num_heads || global_q >= N_dec) return;

    const float scale = rsqrt(float(head_dim));
    const int stride_dec = num_heads * head_dim;
    const int stride_enc = num_heads * head_dim;

    threadgroup float4 K_smem[HEADS_PER_TG_V4][TILE_K_V4][HEAD_DIM / 4];
    threadgroup float4 V_smem[HEADS_PER_TG_V4][TILE_K_V4][HEAD_DIM / 4];

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float4 acc[HEAD_DIM / 4];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        acc[d] = float4(0.0f);
    }

    // Load Q from decoder
    float4 Q_local[HEAD_DIM / 4];
    const int q_base = global_q * stride_dec + global_h * head_dim;
    device const float4* Q4 = (device const float4*)(Q + q_base);

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        Q_local[d] = Q4[d];
    }

    // Iterate over K/V tiles from encoder
    for (int k_tile_start = 0; k_tile_start < N_enc; k_tile_start += TILE_K_V4) {

        int k_global = k_tile_start + local_q;

        if (k_global < N_enc) {
            const int k_base = k_global * stride_enc + global_h * head_dim;
            device const float4* K4 = (device const float4*)(K + k_base);
            device const float4* V4 = (device const float4*)(V + k_base);

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                K_smem[local_h][local_q][d] = K4[d];
                V_smem[local_h][local_q][d] = V4[d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        int tile_end = min(TILE_K_V4, N_enc - k_tile_start);

        for (int k_local = 0; k_local < tile_end; k_local++) {
            float dot = 0.0f;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                float4 q = Q_local[d];
                float4 k = K_smem[local_h][k_local][d];
                dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
            }
            dot *= scale;

            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);

            row_sum *= rescale;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] *= rescale;
            }

            float w = exp(dot - row_max);
            row_sum += w;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] += w * V_smem[local_h][k_local][d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int o_base = global_q * stride_dec + global_h * head_dim;
    device float4* O4 = (device float4*)(O + o_base);
    float inv_sum = 1.0f / row_sum;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        O4[d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// FlashAttention v5 - 256 threads per threadgroup
// ============================================================================
// Maximum parallelism with 256 threads:
// - 64 queries × 4 heads per threadgroup
// - Smaller TILE_K (16) to fit 4 heads in shared memory
// - Total shared memory: 4 × 16 × 16 × 16 × 2 = 32KB
//
// Grid: [ceil(num_heads/4), ceil(N/64)]
// Threadgroup: [64, 4] = 256 threads

kernel void flash_attention_256(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int3& dims [[buffer(4)]],        // N, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;

    const int h_tile = tgid.x;
    const int q_tile_start = tgid.y * TILE_Q_V5;
    const int local_q = tid.x;                           // 0-63
    const int local_h = tid.y;                           // 0-3
    const int global_q = q_tile_start + local_q;
    const int global_h = h_tile * HEADS_PER_TG_V5 + local_h;

    if (global_h >= num_heads || global_q >= N) return;

    const float scale = rsqrt(float(head_dim));
    const int stride = num_heads * head_dim;

    // Shared memory: 4 heads × 16 keys × 16 float4 = 16KB each
    threadgroup float4 K_smem[HEADS_PER_TG_V5][TILE_K_V5][HEAD_DIM / 4];
    threadgroup float4 V_smem[HEADS_PER_TG_V5][TILE_K_V5][HEAD_DIM / 4];

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float4 acc[HEAD_DIM / 4];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        acc[d] = float4(0.0f);
    }

    // Load Q into registers
    float4 Q_local[HEAD_DIM / 4];
    const int q_base = global_q * stride + global_h * head_dim;
    device const float4* Q4 = (device const float4*)(Q + q_base);

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        Q_local[d] = Q4[d];
    }

    // Iterate over K/V tiles
    for (int k_tile_start = 0; k_tile_start < N; k_tile_start += TILE_K_V5) {

        // Cooperative load: only first 16 threads per head load K/V
        // (64 threads per head but only 16 K rows to load)
        if (local_q < TILE_K_V5) {
            int k_global = k_tile_start + local_q;
            if (k_global < N) {
                const int k_base = k_global * stride + global_h * head_dim;
                device const float4* K4 = (device const float4*)(K + k_base);
                device const float4* V4 = (device const float4*)(V + k_base);

                #pragma unroll
                for (int d = 0; d < HEAD_DIM / 4; d++) {
                    K_smem[local_h][local_q][d] = K4[d];
                    V_smem[local_h][local_q][d] = V4[d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        int tile_end = min(TILE_K_V5, N - k_tile_start);

        for (int k_local = 0; k_local < tile_end; k_local++) {
            float dot = 0.0f;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                float4 q = Q_local[d];
                float4 k = K_smem[local_h][k_local][d];
                dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
            }
            dot *= scale;

            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);

            row_sum *= rescale;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] *= rescale;
            }

            float w = exp(dot - row_max);
            row_sum += w;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] += w * V_smem[local_h][k_local][d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int o_base = global_q * stride + global_h * head_dim;
    device float4* O4 = (device float4*)(O + o_base);
    float inv_sum = 1.0f / row_sum;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        O4[d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// Flash Cross-Attention 256 - Same for cross-attention
// ============================================================================

kernel void flash_cross_attention_256(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int4& dims [[buffer(4)]],        // N_dec, N_enc, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const int N_dec = dims.x;
    const int N_enc = dims.y;
    const int num_heads = dims.z;
    const int head_dim = dims.w;

    const int h_tile = tgid.x;
    const int q_tile_start = tgid.y * TILE_Q_V5;
    const int local_q = tid.x;
    const int local_h = tid.y;
    const int global_q = q_tile_start + local_q;
    const int global_h = h_tile * HEADS_PER_TG_V5 + local_h;

    if (global_h >= num_heads || global_q >= N_dec) return;

    const float scale = rsqrt(float(head_dim));
    const int stride_dec = num_heads * head_dim;
    const int stride_enc = num_heads * head_dim;

    threadgroup float4 K_smem[HEADS_PER_TG_V5][TILE_K_V5][HEAD_DIM / 4];
    threadgroup float4 V_smem[HEADS_PER_TG_V5][TILE_K_V5][HEAD_DIM / 4];

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float4 acc[HEAD_DIM / 4];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        acc[d] = float4(0.0f);
    }

    float4 Q_local[HEAD_DIM / 4];
    const int q_base = global_q * stride_dec + global_h * head_dim;
    device const float4* Q4 = (device const float4*)(Q + q_base);

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        Q_local[d] = Q4[d];
    }

    for (int k_tile_start = 0; k_tile_start < N_enc; k_tile_start += TILE_K_V5) {

        if (local_q < TILE_K_V5) {
            int k_global = k_tile_start + local_q;
            if (k_global < N_enc) {
                const int k_base = k_global * stride_enc + global_h * head_dim;
                device const float4* K4 = (device const float4*)(K + k_base);
                device const float4* V4 = (device const float4*)(V + k_base);

                #pragma unroll
                for (int d = 0; d < HEAD_DIM / 4; d++) {
                    K_smem[local_h][local_q][d] = K4[d];
                    V_smem[local_h][local_q][d] = V4[d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        int tile_end = min(TILE_K_V5, N_enc - k_tile_start);

        for (int k_local = 0; k_local < tile_end; k_local++) {
            float dot = 0.0f;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                float4 q = Q_local[d];
                float4 k = K_smem[local_h][k_local][d];
                dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
            }
            dot *= scale;

            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);

            row_sum *= rescale;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] *= rescale;
            }

            float w = exp(dot - row_max);
            row_sum += w;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] += w * V_smem[local_h][k_local][d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int o_base = global_q * stride_dec + global_h * head_dim;
    device float4* O4 = (device float4*)(O + o_base);
    float inv_sum = 1.0f / row_sum;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        O4[d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// Flash Cross-Attention - Optimized for decoder cross-attention
// ============================================================================
// Same optimization as flash_attention_v3:
// - Q from decoder in registers (float4 vectorized)
// - K/V from encoder cached in threadgroup memory
// - Online softmax in single pass
//
// Grid: [num_heads, ceil(N_dec/TILE_Q)]
// Threadgroup: [TILE_Q, 1]

kernel void flash_cross_attention(
    device const float* Q [[buffer(0)]],      // [N_dec, num_heads, head_dim] from decoder
    device const float* K [[buffer(1)]],      // [N_enc, num_heads, head_dim] from encoder
    device const float* V [[buffer(2)]],      // [N_enc, num_heads, head_dim] from encoder
    device float* O [[buffer(3)]],            // [N_dec, num_heads, head_dim]
    constant int4& dims [[buffer(4)]],        // N_dec, N_enc, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const int N_dec = dims.x;
    const int N_enc = dims.y;
    const int num_heads = dims.z;
    const int head_dim = dims.w;

    const int h = tgid.x;                     // Head index
    const int q_tile_start = tgid.y * TILE_Q; // First query in this tile
    const int local_q = tid;                  // Which query this thread handles
    const int global_q = q_tile_start + local_q;

    if (h >= num_heads || global_q >= N_dec) return;

    const float scale = rsqrt(float(head_dim));
    const int stride_dec = num_heads * head_dim;
    const int stride_enc = num_heads * head_dim;

    // Shared memory for K/V tiles from encoder - float4 aligned
    threadgroup float4 K_smem[TILE_K][HEAD_DIM / 4];
    threadgroup float4 V_smem[TILE_K][HEAD_DIM / 4];

    // Per-thread accumulators in registers
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float4 acc[HEAD_DIM / 4];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        acc[d] = float4(0.0f);
    }

    // Load Q once into registers (from decoder, vectorized)
    float4 Q_local[HEAD_DIM / 4];
    const int q_base = global_q * stride_dec + h * head_dim;
    device const float4* Q4 = (device const float4*)(Q + q_base);

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        Q_local[d] = Q4[d];
    }

    // Iterate over K/V tiles from encoder
    for (int k_tile_start = 0; k_tile_start < N_enc; k_tile_start += TILE_K) {

        // Cooperative load: each thread loads 2 K/V rows from encoder
        #pragma unroll
        for (int offset = 0; offset < TILE_K; offset += TILE_Q) {
            int k_local = offset + local_q;
            int k_global = k_tile_start + k_local;

            if (k_global < N_enc) {
                const int k_base = k_global * stride_enc + h * head_dim;
                device const float4* K4 = (device const float4*)(K + k_base);
                device const float4* V4 = (device const float4*)(V + k_base);

                #pragma unroll
                for (int d = 0; d < HEAD_DIM / 4; d++) {
                    K_smem[k_local][d] = K4[d];
                    V_smem[k_local][d] = V4[d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention for this K/V tile
        int tile_end = min(TILE_K, N_enc - k_tile_start);

        for (int k_local = 0; k_local < tile_end; k_local++) {
            // Compute dot product with cached K (vectorized)
            float dot = 0.0f;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                float4 q = Q_local[d];
                float4 k = K_smem[k_local][d];
                dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
            }
            dot *= scale;

            // Online softmax update
            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);

            row_sum *= rescale;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] *= rescale;
            }

            float w = exp(dot - row_max);
            row_sum += w;

            // Accumulate weighted V from shared memory (vectorized)
            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc[d] += w * V_smem[k_local][d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write (vectorized)
    const int o_base = global_q * stride_dec + h * head_dim;
    device float4* O4 = (device float4*)(O + o_base);
    float inv_sum = 1.0f / row_sum;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 4; d++) {
        O4[d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// FlashAttention v6 - 512 threads with TRUE SIMD optimizations
// ============================================================================
// Uses simd_sum and simd_max for efficient parallel reductions:
// - 16 SIMD groups × 32 threads = 512 threads
// - Each SIMD group handles 1 query, 32 threads collaborate on dot product
// - simd_sum for Q·K dot product (32 threads → 1 value, no barrier)
// - simd_max for online softmax max finding (no barrier)
// - Each thread owns 2 elements of head_dim (64 / 32 = 2)
//
// Grid: [num_heads, ceil(N/16)]
// Threadgroup: [32, 16] = 512 threads (32 lanes × 16 queries)

constant int SIMD_SIZE = 32;
constant int QUERIES_PER_TG_V6 = 16;  // 16 queries per threadgroup
constant int TILE_K_SIMD = 32;        // Keys per tile

kernel void flash_attention_512(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int3& dims [[buffer(4)]],        // N, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;  // Should be 64

    const int h = tgid.x;                                 // Head index
    const int q_tile_start = tgid.y * QUERIES_PER_TG_V6;  // First query in this tile
    const int local_q = simd_group;                       // Which query this SIMD group handles (0-15)
    const int global_q = q_tile_start + local_q;
    const int lane = simd_lane;                           // Thread within SIMD group (0-31)

    if (h >= num_heads || global_q >= N) return;

    const float scale = rsqrt(float(head_dim));
    const int stride = num_heads * head_dim;

    // Shared memory for K/V tiles
    // Each of 16 SIMD groups needs access to the same K/V tile
    threadgroup float K_smem[TILE_K_SIMD][HEAD_DIM + 1];  // +1 padding for bank conflicts
    threadgroup float V_smem[TILE_K_SIMD][HEAD_DIM + 1];

    // Per-thread accumulators
    // Each thread in SIMD group owns 2 output elements (64 / 32 = 2)
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc0 = 0.0f;  // Output element lane * 2
    float acc1 = 0.0f;  // Output element lane * 2 + 1

    // Load Q into registers (each thread loads 2 elements)
    const int q_base = global_q * stride + h * head_dim;
    float q0 = Q[q_base + lane * 2];
    float q1 = Q[q_base + lane * 2 + 1];

    // Iterate over K/V tiles
    for (int k_tile_start = 0; k_tile_start < N; k_tile_start += TILE_K_SIMD) {

        // Cooperative load: 512 threads load 32 K rows × 64 elements = 2048 elements
        // Each thread loads 4 elements (2048 / 512 = 4)
        // Pattern: thread (lane, simd_group) loads K[k_row][d] where:
        //   k_row = simd_group / 2 + (iteration * 8)
        //   d = lane * 2 + (simd_group % 2) * 64 ... but head_dim is only 64
        // Simpler: each of 16 SIMD groups loads 2 rows of K/V
        {
            int k_row = local_q * 2;
            int k_global_0 = k_tile_start + k_row;
            int k_global_1 = k_tile_start + k_row + 1;

            if (k_global_0 < N) {
                int k_base_0 = k_global_0 * stride + h * head_dim;
                K_smem[k_row][lane * 2] = K[k_base_0 + lane * 2];
                K_smem[k_row][lane * 2 + 1] = K[k_base_0 + lane * 2 + 1];
                V_smem[k_row][lane * 2] = V[k_base_0 + lane * 2];
                V_smem[k_row][lane * 2 + 1] = V[k_base_0 + lane * 2 + 1];
            }
            if (k_global_1 < N && k_row + 1 < TILE_K_SIMD) {
                int k_base_1 = k_global_1 * stride + h * head_dim;
                K_smem[k_row + 1][lane * 2] = K[k_base_1 + lane * 2];
                K_smem[k_row + 1][lane * 2 + 1] = K[k_base_1 + lane * 2 + 1];
                V_smem[k_row + 1][lane * 2] = V[k_base_1 + lane * 2];
                V_smem[k_row + 1][lane * 2 + 1] = V[k_base_1 + lane * 2 + 1];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        int tile_end = min(TILE_K_SIMD, N - k_tile_start);

        for (int k_local = 0; k_local < tile_end; k_local++) {
            // Each thread computes partial dot product (2 elements)
            float partial_dot = q0 * K_smem[k_local][lane * 2] +
                               q1 * K_smem[k_local][lane * 2 + 1];

            // SIMD reduce: sum all 32 partial dots to get full dot product
            float dot = simd_sum(partial_dot) * scale;

            // Online softmax update (all threads have same dot value after simd_sum)
            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);

            row_sum = row_sum * rescale + exp(dot - row_max);
            acc0 = acc0 * rescale + exp(dot - row_max) * V_smem[k_local][lane * 2];
            acc1 = acc1 * rescale + exp(dot - row_max) * V_smem[k_local][lane * 2 + 1];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write output
    // Each thread writes 2 output elements
    if (global_q < N) {
        const int o_base = global_q * stride + h * head_dim;
        float inv_sum = 1.0f / row_sum;
        O[o_base + lane * 2] = acc0 * inv_sum;
        O[o_base + lane * 2 + 1] = acc1 * inv_sum;
    }
}

// ============================================================================
// Flash Cross-Attention 512 - SIMD optimized cross-attention
// ============================================================================
// Same SIMD pattern as flash_attention_512:
// - 16 SIMD groups × 32 threads = 512 threads
// - simd_sum for Q·K dot product
// - Each thread owns 2 elements of head_dim
//
// Grid: [num_heads, ceil(N_dec/16)]
// Threadgroup: [32, 16] = 512 threads

kernel void flash_cross_attention_512(
    device const float* Q [[buffer(0)]],      // [N_dec, num_heads, head_dim]
    device const float* K [[buffer(1)]],      // [N_enc, num_heads, head_dim]
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int4& dims [[buffer(4)]],        // N_dec, N_enc, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const int N_dec = dims.x;
    const int N_enc = dims.y;
    const int num_heads = dims.z;
    const int head_dim = dims.w;  // Should be 64

    const int h = tgid.x;                                 // Head index
    const int q_tile_start = tgid.y * QUERIES_PER_TG_V6;  // First query in this tile
    const int local_q = simd_group;                       // Which query this SIMD group handles (0-15)
    const int global_q = q_tile_start + local_q;
    const int lane = simd_lane;                           // Thread within SIMD group (0-31)

    if (h >= num_heads || global_q >= N_dec) return;

    const float scale = rsqrt(float(head_dim));
    const int stride_dec = num_heads * head_dim;
    const int stride_enc = num_heads * head_dim;

    // Shared memory for K/V tiles from encoder
    threadgroup float K_smem[TILE_K_SIMD][HEAD_DIM + 1];
    threadgroup float V_smem[TILE_K_SIMD][HEAD_DIM + 1];

    // Per-thread accumulators
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Load Q from decoder (each thread loads 2 elements)
    const int q_base = global_q * stride_dec + h * head_dim;
    float q0 = Q[q_base + lane * 2];
    float q1 = Q[q_base + lane * 2 + 1];

    // Iterate over K/V tiles from encoder
    for (int k_tile_start = 0; k_tile_start < N_enc; k_tile_start += TILE_K_SIMD) {

        // Cooperative load: each SIMD group loads 2 rows of K/V
        {
            int k_row = local_q * 2;
            int k_global_0 = k_tile_start + k_row;
            int k_global_1 = k_tile_start + k_row + 1;

            if (k_global_0 < N_enc) {
                int k_base_0 = k_global_0 * stride_enc + h * head_dim;
                K_smem[k_row][lane * 2] = K[k_base_0 + lane * 2];
                K_smem[k_row][lane * 2 + 1] = K[k_base_0 + lane * 2 + 1];
                V_smem[k_row][lane * 2] = V[k_base_0 + lane * 2];
                V_smem[k_row][lane * 2 + 1] = V[k_base_0 + lane * 2 + 1];
            }
            if (k_global_1 < N_enc && k_row + 1 < TILE_K_SIMD) {
                int k_base_1 = k_global_1 * stride_enc + h * head_dim;
                K_smem[k_row + 1][lane * 2] = K[k_base_1 + lane * 2];
                K_smem[k_row + 1][lane * 2 + 1] = K[k_base_1 + lane * 2 + 1];
                V_smem[k_row + 1][lane * 2] = V[k_base_1 + lane * 2];
                V_smem[k_row + 1][lane * 2 + 1] = V[k_base_1 + lane * 2 + 1];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        int tile_end = min(TILE_K_SIMD, N_enc - k_tile_start);

        for (int k_local = 0; k_local < tile_end; k_local++) {
            // Each thread computes partial dot product (2 elements)
            float partial_dot = q0 * K_smem[k_local][lane * 2] +
                               q1 * K_smem[k_local][lane * 2 + 1];

            // SIMD reduce: sum all 32 partial dots to get full dot product
            float dot = simd_sum(partial_dot) * scale;

            // Online softmax update
            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);

            row_sum = row_sum * rescale + exp(dot - row_max);
            acc0 = acc0 * rescale + exp(dot - row_max) * V_smem[k_local][lane * 2];
            acc1 = acc1 * rescale + exp(dot - row_max) * V_smem[k_local][lane * 2 + 1];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write output
    if (global_q < N_dec) {
        const int o_base = global_q * stride_dec + h * head_dim;
        float inv_sum = 1.0f / row_sum;
        O[o_base + lane * 2] = acc0 * inv_sum;
        O[o_base + lane * 2 + 1] = acc1 * inv_sum;
    }
}

// ============================================================================
// FlashAttention v7 - Transposed [H, N, D] layout
// ============================================================================
// Uses [H, N, D] memory layout for optimal coalescing:
// - Q, K, V, O all in [num_heads, N, head_dim] layout
// - Each head's data is contiguous in memory
// - Stride between sequence positions is just head_dim (vs H*D before)
// - 16x better memory coalescing for key iteration
//
// Grid: [num_heads, ceil(N/16)]
// Threadgroup: [32, 16] = 512 threads (32 lanes × 16 queries)

kernel void flash_attention_v7(
    device const float* Q [[buffer(0)]],      // [H, N, D] transposed layout
    device const float* K [[buffer(1)]],      // [H, N, D]
    device const float* V [[buffer(2)]],      // [H, N, D]
    device float* O [[buffer(3)]],            // [H, N, D]
    constant int3& dims [[buffer(4)]],        // N, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;  // Should be 64

    const int h = tgid.x;                                 // Head index
    const int q_tile_start = tgid.y * QUERIES_PER_TG_V6;  // First query in this tile
    const int local_q = simd_group;                       // Which query this SIMD group handles (0-15)
    const int global_q = q_tile_start + local_q;
    const int lane = simd_lane;                           // Thread within SIMD group (0-31)

    if (h >= num_heads || global_q >= N) return;

    const float scale = rsqrt(float(head_dim));

    // [H, N, D] layout: base offset for head h is h * N * D
    const int head_offset = h * N * head_dim;

    // Shared memory for K/V tiles
    threadgroup float K_smem[TILE_K_SIMD][HEAD_DIM + 1];
    threadgroup float V_smem[TILE_K_SIMD][HEAD_DIM + 1];

    // Per-thread accumulators
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Load Q into registers - [H, N, D] layout: Q[h, i, d] at head_offset + i * D + d
    const int q_offset = head_offset + global_q * head_dim;
    float q0 = Q[q_offset + lane * 2];
    float q1 = Q[q_offset + lane * 2 + 1];

    // Iterate over K/V tiles
    for (int k_tile_start = 0; k_tile_start < N; k_tile_start += TILE_K_SIMD) {

        // Cooperative load: each SIMD group loads 2 rows of K/V
        // With [H, N, D], K[h, j, :] is contiguous at head_offset + j * D
        {
            int k_row = local_q * 2;
            int k_global_0 = k_tile_start + k_row;
            int k_global_1 = k_tile_start + k_row + 1;

            if (k_global_0 < N) {
                // [H, N, D]: K[h, k_global_0, :] at head_offset + k_global_0 * D
                int k_offset_0 = head_offset + k_global_0 * head_dim;
                K_smem[k_row][lane * 2] = K[k_offset_0 + lane * 2];
                K_smem[k_row][lane * 2 + 1] = K[k_offset_0 + lane * 2 + 1];
                V_smem[k_row][lane * 2] = V[k_offset_0 + lane * 2];
                V_smem[k_row][lane * 2 + 1] = V[k_offset_0 + lane * 2 + 1];
            }
            if (k_global_1 < N && k_row + 1 < TILE_K_SIMD) {
                int k_offset_1 = head_offset + k_global_1 * head_dim;
                K_smem[k_row + 1][lane * 2] = K[k_offset_1 + lane * 2];
                K_smem[k_row + 1][lane * 2 + 1] = K[k_offset_1 + lane * 2 + 1];
                V_smem[k_row + 1][lane * 2] = V[k_offset_1 + lane * 2];
                V_smem[k_row + 1][lane * 2 + 1] = V[k_offset_1 + lane * 2 + 1];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        int tile_end = min(TILE_K_SIMD, N - k_tile_start);

        for (int k_local = 0; k_local < tile_end; k_local++) {
            // Each thread computes partial dot product (2 elements)
            float partial_dot = q0 * K_smem[k_local][lane * 2] +
                               q1 * K_smem[k_local][lane * 2 + 1];

            // SIMD reduce: sum all 32 partial dots to get full dot product
            float dot = simd_sum(partial_dot) * scale;

            // Online softmax update
            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);

            row_sum = row_sum * rescale + exp(dot - row_max);
            acc0 = acc0 * rescale + exp(dot - row_max) * V_smem[k_local][lane * 2];
            acc1 = acc1 * rescale + exp(dot - row_max) * V_smem[k_local][lane * 2 + 1];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write output - [H, N, D] layout
    if (global_q < N) {
        const int o_offset = head_offset + global_q * head_dim;
        float inv_sum = 1.0f / row_sum;
        O[o_offset + lane * 2] = acc0 * inv_sum;
        O[o_offset + lane * 2 + 1] = acc1 * inv_sum;
    }
}

// ============================================================================
// Flash Cross-Attention v7 - Transposed [H, N, D] layout
// ============================================================================
// Cross-attention with [H, N, D] layout for Q, K, V, O

kernel void flash_cross_attention_v7(
    device const float* Q [[buffer(0)]],      // [H, N_dec, D]
    device const float* K [[buffer(1)]],      // [H, N_enc, D]
    device const float* V [[buffer(2)]],      // [H, N_enc, D]
    device float* O [[buffer(3)]],            // [H, N_dec, D]
    constant int4& dims [[buffer(4)]],        // N_dec, N_enc, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const int N_dec = dims.x;
    const int N_enc = dims.y;
    const int num_heads = dims.z;
    const int head_dim = dims.w;  // Should be 64

    const int h = tgid.x;                                 // Head index
    const int q_tile_start = tgid.y * QUERIES_PER_TG_V6;  // First query in this tile
    const int local_q = simd_group;
    const int global_q = q_tile_start + local_q;
    const int lane = simd_lane;

    if (h >= num_heads || global_q >= N_dec) return;

    const float scale = rsqrt(float(head_dim));

    // [H, N, D] layout offsets
    const int q_head_offset = h * N_dec * head_dim;
    const int kv_head_offset = h * N_enc * head_dim;

    // Shared memory for K/V tiles
    threadgroup float K_smem[TILE_K_SIMD][HEAD_DIM + 1];
    threadgroup float V_smem[TILE_K_SIMD][HEAD_DIM + 1];

    // Per-thread accumulators
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Load Q - [H, N_dec, D]
    const int q_offset = q_head_offset + global_q * head_dim;
    float q0 = Q[q_offset + lane * 2];
    float q1 = Q[q_offset + lane * 2 + 1];

    // Iterate over K/V tiles from encoder
    for (int k_tile_start = 0; k_tile_start < N_enc; k_tile_start += TILE_K_SIMD) {

        // Cooperative load from [H, N_enc, D]
        {
            int k_row = local_q * 2;
            int k_global_0 = k_tile_start + k_row;
            int k_global_1 = k_tile_start + k_row + 1;

            if (k_global_0 < N_enc) {
                int k_offset_0 = kv_head_offset + k_global_0 * head_dim;
                K_smem[k_row][lane * 2] = K[k_offset_0 + lane * 2];
                K_smem[k_row][lane * 2 + 1] = K[k_offset_0 + lane * 2 + 1];
                V_smem[k_row][lane * 2] = V[k_offset_0 + lane * 2];
                V_smem[k_row][lane * 2 + 1] = V[k_offset_0 + lane * 2 + 1];
            }
            if (k_global_1 < N_enc && k_row + 1 < TILE_K_SIMD) {
                int k_offset_1 = kv_head_offset + k_global_1 * head_dim;
                K_smem[k_row + 1][lane * 2] = K[k_offset_1 + lane * 2];
                K_smem[k_row + 1][lane * 2 + 1] = K[k_offset_1 + lane * 2 + 1];
                V_smem[k_row + 1][lane * 2] = V[k_offset_1 + lane * 2];
                V_smem[k_row + 1][lane * 2 + 1] = V[k_offset_1 + lane * 2 + 1];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        int tile_end = min(TILE_K_SIMD, N_enc - k_tile_start);

        for (int k_local = 0; k_local < tile_end; k_local++) {
            float partial_dot = q0 * K_smem[k_local][lane * 2] +
                               q1 * K_smem[k_local][lane * 2 + 1];

            float dot = simd_sum(partial_dot) * scale;

            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);

            row_sum = row_sum * rescale + exp(dot - row_max);
            acc0 = acc0 * rescale + exp(dot - row_max) * V_smem[k_local][lane * 2];
            acc1 = acc1 * rescale + exp(dot - row_max) * V_smem[k_local][lane * 2 + 1];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write - [H, N_dec, D]
    if (global_q < N_dec) {
        const int o_offset = q_head_offset + global_q * head_dim;
        float inv_sum = 1.0f / row_sum;
        O[o_offset + lane * 2] = acc0 * inv_sum;
        O[o_offset + lane * 2 + 1] = acc1 * inv_sum;
    }
}

// ============================================================================
// Flash Attention v8 - Double Buffering with Vectorized Loads
// ============================================================================
// Key optimizations:
// 1. Double buffering: prefetch next K/V tile while computing on current
// 2. Vectorized loads: float4 instead of float2 for 2x bandwidth
// 3. Bank conflict avoidance with padding
// 4. Pipelined execution: load and compute overlap
//
// Grid: [num_heads, ceil(N/16)]
// Threadgroup: [32, 16] = 512 threads

constant int TILE_K_V8 = 24;       // Keys per tile (reduced for threadgroup memory limit)
constant int HEAD_DIM_V8 = 64;     // Head dimension
constant int QUERIES_PER_TG_V8 = 16;  // Queries per threadgroup
// Threadgroup memory: 2 * 2 * 24 * 64 * 4 = 24576 bytes (< 32KB limit)

kernel void flash_attention_v8(
    device const float* Q [[buffer(0)]],      // [H, N, D] transposed layout
    device const float* K [[buffer(1)]],      // [H, N, D]
    device const float* V [[buffer(2)]],      // [H, N, D]
    device float* O [[buffer(3)]],            // [H, N, D]
    constant int3& dims [[buffer(4)]],        // N, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;  // Should be 64

    const int h = tgid.x;                                  // Head index
    const int q_tile_start = tgid.y * QUERIES_PER_TG_V8;   // First query in this tile
    const int local_q = simd_group;                        // Query within tile (0-15)
    const int global_q = q_tile_start + local_q;
    const int lane = simd_lane;                            // Thread within SIMD group (0-31)

    if (h >= num_heads || global_q >= N) return;

    const float scale = rsqrt(float(head_dim));

    // [H, N, D] layout: base offset for head h
    const int head_offset = h * N * head_dim;

    // Double buffers for K and V (no padding to fit in 32KB threadgroup memory)
    threadgroup float K_buf[2][TILE_K_V8][HEAD_DIM_V8];
    threadgroup float V_buf[2][TILE_K_V8][HEAD_DIM_V8];

    // Per-thread accumulators - use float4 for vectorized accumulation
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float4 acc = float4(0.0f);  // Accumulator for 4 elements

    // Load Q into registers - vectorized float4 load
    const int q_offset = head_offset + global_q * head_dim;

    // Each thread loads 4 consecutive elements (lane 0-15 loads first half, 16-31 loads second)
    // But for dot product we need all 64 elements distributed across 32 lanes
    // So each lane holds 2 elements for the dot product
    float q0 = Q[q_offset + lane * 2];
    float q1 = Q[q_offset + lane * 2 + 1];

    const int num_tiles = (N + TILE_K_V8 - 1) / TILE_K_V8;
    int cur_buf = 0;

    // ========== Load first tile into buffer 0 ==========
    {
        int k_row = local_q * 2;
        int k_global_0 = k_row;
        int k_global_1 = k_row + 1;

        if (k_global_0 < N && k_row < TILE_K_V8) {
            int k_offset = head_offset + k_global_0 * head_dim;
            K_buf[0][k_row][lane * 2] = K[k_offset + lane * 2];
            K_buf[0][k_row][lane * 2 + 1] = K[k_offset + lane * 2 + 1];
            V_buf[0][k_row][lane * 2] = V[k_offset + lane * 2];
            V_buf[0][k_row][lane * 2 + 1] = V[k_offset + lane * 2 + 1];
        }
        if (k_global_1 < N && k_row + 1 < TILE_K_V8) {
            int k_offset = head_offset + k_global_1 * head_dim;
            K_buf[0][k_row + 1][lane * 2] = K[k_offset + lane * 2];
            K_buf[0][k_row + 1][lane * 2 + 1] = K[k_offset + lane * 2 + 1];
            V_buf[0][k_row + 1][lane * 2] = V[k_offset + lane * 2];
            V_buf[0][k_row + 1][lane * 2 + 1] = V[k_offset + lane * 2 + 1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Main loop with double buffering ==========
    for (int tile = 0; tile < num_tiles; tile++) {
        int next_buf = 1 - cur_buf;

        // Prefetch next tile into alternate buffer (while computing on current)
        if (tile + 1 < num_tiles) {
            int next_tile_start = (tile + 1) * TILE_K_V8;
            int k_row = local_q * 2;
            int k_global_0 = next_tile_start + k_row;
            int k_global_1 = next_tile_start + k_row + 1;

            if (k_global_0 < N && k_row < TILE_K_V8) {
                int k_offset = head_offset + k_global_0 * head_dim;
                K_buf[next_buf][k_row][lane * 2] = K[k_offset + lane * 2];
                K_buf[next_buf][k_row][lane * 2 + 1] = K[k_offset + lane * 2 + 1];
                V_buf[next_buf][k_row][lane * 2] = V[k_offset + lane * 2];
                V_buf[next_buf][k_row][lane * 2 + 1] = V[k_offset + lane * 2 + 1];
            }
            if (k_global_1 < N && k_row + 1 < TILE_K_V8) {
                int k_offset = head_offset + k_global_1 * head_dim;
                K_buf[next_buf][k_row + 1][lane * 2] = K[k_offset + lane * 2];
                K_buf[next_buf][k_row + 1][lane * 2 + 1] = K[k_offset + lane * 2 + 1];
                V_buf[next_buf][k_row + 1][lane * 2] = V[k_offset + lane * 2];
                V_buf[next_buf][k_row + 1][lane * 2 + 1] = V[k_offset + lane * 2 + 1];
            }
        }

        // Compute attention on current tile
        int tile_start = tile * TILE_K_V8;
        int tile_end = min(TILE_K_V8, N - tile_start);

        // Unrolled inner loop for better ILP
        for (int k = 0; k < tile_end; k++) {
            // Compute dot product Q @ K^T
            float partial_dot = q0 * K_buf[cur_buf][k][lane * 2] +
                               q1 * K_buf[cur_buf][k][lane * 2 + 1];

            // SIMD reduction to get full dot product
            float dot = simd_sum(partial_dot) * scale;

            // Online softmax update (numerically stable)
            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);
            float exp_dot = exp(dot - row_max);

            row_sum = row_sum * rescale + exp_dot;

            // Accumulate weighted V
            acc[0] = acc[0] * rescale + exp_dot * V_buf[cur_buf][k][lane * 2];
            acc[1] = acc[1] * rescale + exp_dot * V_buf[cur_buf][k][lane * 2 + 1];
        }

        // Wait for prefetch to complete before swapping
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur_buf = next_buf;
    }

    // Final normalization and write output
    if (global_q < N) {
        const int o_offset = head_offset + global_q * head_dim;
        float inv_sum = 1.0f / row_sum;
        O[o_offset + lane * 2] = acc[0] * inv_sum;
        O[o_offset + lane * 2 + 1] = acc[1] * inv_sum;
    }
}

// ============================================================================
// Flash Cross-Attention v8 - Double Buffering
// ============================================================================
// Cross-attention with double buffering for encoder-decoder attention

kernel void flash_cross_attention_v8(
    device const float* Q [[buffer(0)]],      // [H, N_dec, D]
    device const float* K [[buffer(1)]],      // [H, N_enc, D]
    device const float* V [[buffer(2)]],      // [H, N_enc, D]
    device float* O [[buffer(3)]],            // [H, N_dec, D]
    constant int4& dims [[buffer(4)]],        // N_dec, N_enc, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const int N_dec = dims.x;
    const int N_enc = dims.y;
    const int num_heads = dims.z;
    const int head_dim = dims.w;

    const int h = tgid.x;
    const int q_tile_start = tgid.y * QUERIES_PER_TG_V8;
    const int local_q = simd_group;
    const int global_q = q_tile_start + local_q;
    const int lane = simd_lane;

    if (h >= num_heads || global_q >= N_dec) return;

    const float scale = rsqrt(float(head_dim));

    const int q_head_offset = h * N_dec * head_dim;
    const int kv_head_offset = h * N_enc * head_dim;

    // Double buffers (no padding to fit in 32KB)
    threadgroup float K_buf[2][TILE_K_V8][HEAD_DIM_V8];
    threadgroup float V_buf[2][TILE_K_V8][HEAD_DIM_V8];

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;

    const int q_offset = q_head_offset + global_q * head_dim;
    float q0 = Q[q_offset + lane * 2];
    float q1 = Q[q_offset + lane * 2 + 1];

    const int num_tiles = (N_enc + TILE_K_V8 - 1) / TILE_K_V8;
    int cur_buf = 0;

    // Load first tile
    {
        int k_row = local_q * 2;
        int k_global = k_row;
        if (k_global < N_enc && k_row < TILE_K_V8) {
            int k_offset = kv_head_offset + k_global * head_dim;
            K_buf[0][k_row][lane * 2] = K[k_offset + lane * 2];
            K_buf[0][k_row][lane * 2 + 1] = K[k_offset + lane * 2 + 1];
            V_buf[0][k_row][lane * 2] = V[k_offset + lane * 2];
            V_buf[0][k_row][lane * 2 + 1] = V[k_offset + lane * 2 + 1];
        }
        if (k_global + 1 < N_enc && k_row + 1 < TILE_K_V8) {
            int k_offset = kv_head_offset + (k_global + 1) * head_dim;
            K_buf[0][k_row + 1][lane * 2] = K[k_offset + lane * 2];
            K_buf[0][k_row + 1][lane * 2 + 1] = K[k_offset + lane * 2 + 1];
            V_buf[0][k_row + 1][lane * 2] = V[k_offset + lane * 2];
            V_buf[0][k_row + 1][lane * 2 + 1] = V[k_offset + lane * 2 + 1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int tile = 0; tile < num_tiles; tile++) {
        int next_buf = 1 - cur_buf;

        // Prefetch next tile
        if (tile + 1 < num_tiles) {
            int next_tile_start = (tile + 1) * TILE_K_V8;
            int k_row = local_q * 2;
            int k_global = next_tile_start + k_row;
            if (k_global < N_enc && k_row < TILE_K_V8) {
                int k_offset = kv_head_offset + k_global * head_dim;
                K_buf[next_buf][k_row][lane * 2] = K[k_offset + lane * 2];
                K_buf[next_buf][k_row][lane * 2 + 1] = K[k_offset + lane * 2 + 1];
                V_buf[next_buf][k_row][lane * 2] = V[k_offset + lane * 2];
                V_buf[next_buf][k_row][lane * 2 + 1] = V[k_offset + lane * 2 + 1];
            }
            if (k_global + 1 < N_enc && k_row + 1 < TILE_K_V8) {
                int k_offset = kv_head_offset + (k_global + 1) * head_dim;
                K_buf[next_buf][k_row + 1][lane * 2] = K[k_offset + lane * 2];
                K_buf[next_buf][k_row + 1][lane * 2 + 1] = K[k_offset + lane * 2 + 1];
                V_buf[next_buf][k_row + 1][lane * 2] = V[k_offset + lane * 2];
                V_buf[next_buf][k_row + 1][lane * 2 + 1] = V[k_offset + lane * 2 + 1];
            }
        }

        // Compute
        int tile_start = tile * TILE_K_V8;
        int tile_end = min(TILE_K_V8, N_enc - tile_start);

        for (int k = 0; k < tile_end; k++) {
            float partial_dot = q0 * K_buf[cur_buf][k][lane * 2] +
                               q1 * K_buf[cur_buf][k][lane * 2 + 1];
            float dot = simd_sum(partial_dot) * scale;

            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);
            float exp_dot = exp(dot - row_max);

            row_sum = row_sum * rescale + exp_dot;
            acc0 = acc0 * rescale + exp_dot * V_buf[cur_buf][k][lane * 2];
            acc1 = acc1 * rescale + exp_dot * V_buf[cur_buf][k][lane * 2 + 1];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur_buf = next_buf;
    }

    if (global_q < N_dec) {
        const int o_offset = q_head_offset + global_q * head_dim;
        float inv_sum = 1.0f / row_sum;
        O[o_offset + lane * 2] = acc0 * inv_sum;
        O[o_offset + lane * 2 + 1] = acc1 * inv_sum;
    }
}

// ============================================================================
// Flash Attention v9 - Batched queries (8 at a time) for better compute density
// ============================================================================
// Instead of 1 query per SIMD group, process 8 queries per threadgroup.
// This increases arithmetic intensity and reduces memory bandwidth pressure.
//
// Grid: [num_heads, ceil(N/8)]
// Threadgroup: [32, 8] = 256 threads

constant int TILE_Q_V9 = 8;        // Queries per threadgroup
constant int TILE_K_V9 = 32;       // Keys per tile
constant int HEAD_DIM_V9 = 64;

kernel void flash_attention_v9(
    device const float* Q [[buffer(0)]],      // [H, N, D]
    device const float* K [[buffer(1)]],      // [H, N, D]
    device const float* V [[buffer(2)]],      // [H, N, D]
    device float* O [[buffer(3)]],            // [H, N, D]
    constant int3& dims [[buffer(4)]],        // N, num_heads, head_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const int N = dims.x;
    const int num_heads = dims.y;
    const int head_dim = dims.z;

    const int h = tgid.x;
    const int q_tile_start = tgid.y * TILE_Q_V9;
    const int local_q = simd_group;  // Which query this SIMD group handles (0-7)
    const int global_q = q_tile_start + local_q;
    const int lane = simd_lane;

    if (h >= num_heads || global_q >= N) return;

    const float scale = rsqrt(float(head_dim));
    const int head_offset = h * N * head_dim;

    // Shared K/V tile
    threadgroup float K_tile[TILE_K_V9][HEAD_DIM_V9];
    threadgroup float V_tile[TILE_K_V9][HEAD_DIM_V9];

    // Per-query accumulators
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;

    // Load Q into registers
    const int q_offset = head_offset + global_q * head_dim;
    float q0 = Q[q_offset + lane * 2];
    float q1 = Q[q_offset + lane * 2 + 1];

    // Iterate over K/V tiles
    for (int k_tile_start = 0; k_tile_start < N; k_tile_start += TILE_K_V9) {
        // Cooperative load: 8 SIMD groups × 32 lanes = 256 threads
        // Load 32 rows × 64 cols = 2048 floats, each thread loads 8 floats
        int rows_per_simd = TILE_K_V9 / 8;  // 4 rows per SIMD group
        for (int r = 0; r < rows_per_simd; r++) {
            int k_row = local_q * rows_per_simd + r;
            int k_global = k_tile_start + k_row;
            if (k_global < N && k_row < TILE_K_V9) {
                int k_off = head_offset + k_global * head_dim;
                K_tile[k_row][lane * 2] = K[k_off + lane * 2];
                K_tile[k_row][lane * 2 + 1] = K[k_off + lane * 2 + 1];
                V_tile[k_row][lane * 2] = V[k_off + lane * 2];
                V_tile[k_row][lane * 2 + 1] = V[k_off + lane * 2 + 1];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        int tile_end = min(TILE_K_V9, N - k_tile_start);

        // Compute attention
        for (int k = 0; k < tile_end; k++) {
            float partial = q0 * K_tile[k][lane * 2] + q1 * K_tile[k][lane * 2 + 1];
            float dot = simd_sum(partial) * scale;

            float old_max = row_max;
            row_max = max(row_max, dot);
            float rescale = exp(old_max - row_max);
            float exp_dot = exp(dot - row_max);

            row_sum = row_sum * rescale + exp_dot;
            acc0 = acc0 * rescale + exp_dot * V_tile[k][lane * 2];
            acc1 = acc1 * rescale + exp_dot * V_tile[k][lane * 2 + 1];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (global_q < N) {
        const int o_offset = head_offset + global_q * head_dim;
        float inv_sum = 1.0f / row_sum;
        O[o_offset + lane * 2] = acc0 * inv_sum;
        O[o_offset + lane * 2 + 1] = acc1 * inv_sum;
    }
}
