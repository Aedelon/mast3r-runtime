// MASt3R Runtime - Metal Matching Shaders
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include <metal_stdlib>
using namespace metal;

/**
 * L2 normalize descriptors.
 */
kernel void normalize_descriptors(
    device float* desc [[buffer(0)]],    // [N, D]
    constant int2& dims [[buffer(1)]],   // N, D
    uint gid [[thread_position_in_grid]]
) {
    const int N = dims.x;
    const int D = dims.y;

    if (gid >= uint(N)) return;

    device float* row = desc + gid * D;

    // Compute L2 norm
    float norm_sq = 0.0f;
    for (int i = 0; i < D; ++i) {
        norm_sq += row[i] * row[i];
    }

    float inv_norm = 1.0f / sqrt(max(norm_sq, 1e-8f));

    // Normalize in place
    for (int i = 0; i < D; ++i) {
        row[i] *= inv_norm;
    }
}

/**
 * Compute cosine similarity matrix.
 *
 * Each thread computes one element of the similarity matrix.
 */
kernel void compute_similarity(
    device const float* desc_1 [[buffer(0)]],  // [N1, D]
    device const float* desc_2 [[buffer(1)]],  // [N2, D]
    device float* sim [[buffer(2)]],           // [N1, N2]
    constant int3& dims [[buffer(3)]],         // N1, N2, D
    uint2 gid [[thread_position_in_grid]]
) {
    const int N1 = dims.x;
    const int N2 = dims.y;
    const int D = dims.z;

    int i = gid.y;
    int j = gid.x;

    if (i >= N1 || j >= N2) return;

    // Dot product (assuming normalized descriptors)
    float dot = 0.0f;
    for (int k = 0; k < D; ++k) {
        dot += desc_1[i * D + k] * desc_2[j * D + k];
    }

    sim[i * N2 + j] = dot;
}

/**
 * Find argmax for each row (forward nearest neighbor).
 */
kernel void row_argmax(
    device const float* sim [[buffer(0)]],  // [N1, N2]
    device int* nn_12 [[buffer(1)]],        // [N1] output
    device float* nn_scores [[buffer(2)]],  // [N1] output
    constant int2& dims [[buffer(3)]],      // N1, N2
    uint gid [[thread_position_in_grid]]
) {
    const int N1 = dims.x;
    const int N2 = dims.y;

    if (gid >= uint(N1)) return;

    device const float* row = sim + gid * N2;

    int best_j = 0;
    float best_score = row[0];

    for (int j = 1; j < N2; ++j) {
        if (row[j] > best_score) {
            best_score = row[j];
            best_j = j;
        }
    }

    nn_12[gid] = best_j;
    nn_scores[gid] = best_score;
}

/**
 * Find argmax for each column (backward nearest neighbor).
 */
kernel void col_argmax(
    device const float* sim [[buffer(0)]],  // [N1, N2]
    device int* nn_21 [[buffer(1)]],        // [N2] output
    constant int2& dims [[buffer(2)]],      // N1, N2
    uint gid [[thread_position_in_grid]]
) {
    const int N1 = dims.x;
    const int N2 = dims.y;

    if (gid >= uint(N2)) return;

    int best_i = 0;
    float best_score = sim[gid];

    for (int i = 1; i < N1; ++i) {
        float score = sim[i * N2 + gid];
        if (score > best_score) {
            best_score = score;
            best_i = i;
        }
    }

    nn_21[gid] = best_i;
}

/**
 * Check reciprocity and collect matches.
 */
kernel void check_reciprocity(
    device const int* nn_12 [[buffer(0)]],      // [N1]
    device const int* nn_21 [[buffer(1)]],      // [N2]
    device const float* scores [[buffer(2)]],   // [N1]
    device atomic_int* match_count [[buffer(3)]],
    device int* matches_1 [[buffer(4)]],        // [max_matches] output
    device int* matches_2 [[buffer(5)]],        // [max_matches] output
    device float* match_scores [[buffer(6)]],   // [max_matches] output
    constant int& N1 [[buffer(7)]],
    constant float& threshold [[buffer(8)]],
    constant int& max_matches [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(N1)) return;

    int j = nn_12[gid];
    float score = scores[gid];

    // Check reciprocity: nn_21[nn_12[i]] == i
    if (nn_21[j] == int(gid) && score >= threshold) {
        int idx = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
        if (idx < max_matches) {
            matches_1[idx] = int(gid);
            matches_2[idx] = j;
            match_scores[idx] = score;
        }
    }
}

/**
 * Apply confidence weighting to similarity matrix.
 */
kernel void apply_confidence_weighting(
    device float* sim [[buffer(0)]],           // [N1, N2]
    device const float* conf_1 [[buffer(1)]],  // [N1]
    device const float* conf_2 [[buffer(2)]],  // [N2]
    constant int2& dims [[buffer(3)]],         // N1, N2
    uint2 gid [[thread_position_in_grid]]
) {
    const int N1 = dims.x;
    const int N2 = dims.y;

    int i = gid.y;
    int j = gid.x;

    if (i >= N1 || j >= N2) return;

    sim[i * N2 + j] *= conf_1[i] * conf_2[j];
}

/**
 * Convert flat indices to 2D coordinates.
 */
kernel void indices_to_coords(
    device const int* indices [[buffer(0)]],   // [N]
    device float* coords [[buffer(1)]],        // [N, 2]
    constant int& width [[buffer(2)]],
    constant int& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(count)) return;

    int idx = indices[gid];
    coords[gid * 2 + 0] = float(idx % width);  // x
    coords[gid * 2 + 1] = float(idx / width);  // y
}