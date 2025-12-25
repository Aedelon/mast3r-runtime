// MASt3R Runtime - GPU Top-K Selection with Radix Sort
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
//
// Implements parallel bitonic sort and radix-based top-k selection
// entirely on GPU to avoid CPU roundtrips.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Bitonic Sort (for small K, in-threadgroup)
// ============================================================================

constant int BITONIC_BLOCK_SIZE = 512;

struct ScoreIndex {
    float score;
    int index;
};

/**
 * Bitonic sort step - compare and swap within threadgroup.
 */
inline void compare_and_swap(
    threadgroup ScoreIndex* data,
    int i, int j, bool ascending
) {
    if ((data[i].score < data[j].score) == ascending) {
        ScoreIndex temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

/**
 * Bitonic sort for small arrays entirely in threadgroup memory.
 * Sorts in descending order (highest scores first).
 */
kernel void bitonic_sort_local(
    device ScoreIndex* data [[buffer(0)]],
    constant int& N [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    threadgroup ScoreIndex local_data[BITONIC_BLOCK_SIZE];

    // Load into shared memory
    int global_idx = tgid * BITONIC_BLOCK_SIZE + tid;
    if (global_idx < N) {
        local_data[tid] = data[global_idx];
    } else {
        local_data[tid].score = -INFINITY;
        local_data[tid].index = -1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort
    for (int k = 2; k <= BITONIC_BLOCK_SIZE; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid && ixj < BITONIC_BLOCK_SIZE) {
                bool ascending = ((tid & k) == 0);
                // We want descending order (highest first)
                compare_and_swap(local_data, tid, ixj, !ascending);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write back
    if (global_idx < N) {
        data[global_idx] = local_data[tid];
    }
}

// ============================================================================
// Parallel Top-K Selection (for large arrays)
// ============================================================================

/**
 * Find approximate k-th largest value using histogram.
 * This gives us a threshold for filtering.
 */
kernel void compute_histogram(
    device const float* scores [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant float2& range [[buffer(3)]],  // min, max
    constant int& num_bins [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(N)) return;

    float score = scores[gid];
    float min_val = range.x;
    float max_val = range.y;

    int bin = int((score - min_val) / (max_val - min_val) * float(num_bins));
    bin = clamp(bin, 0, num_bins - 1);

    atomic_fetch_add_explicit(&histogram[bin], 1, memory_order_relaxed);
}

/**
 * Find threshold from histogram for top-k selection.
 * Run on CPU or single thread after histogram computation.
 */
kernel void find_threshold(
    device const uint* histogram [[buffer(0)]],
    device float* threshold [[buffer(1)]],
    constant int& K [[buffer(2)]],
    constant float2& range [[buffer(3)]],
    constant int& num_bins [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    float min_val = range.x;
    float max_val = range.y;
    float bin_width = (max_val - min_val) / float(num_bins);

    // Scan from highest bin to find threshold
    int count = 0;
    for (int i = num_bins - 1; i >= 0; --i) {
        count += histogram[i];
        if (count >= K) {
            // Threshold is the lower bound of this bin
            *threshold = min_val + float(i) * bin_width;
            return;
        }
    }

    *threshold = min_val;
}

/**
 * Filter elements above threshold and compact.
 */
kernel void filter_topk(
    device const float* scores [[buffer(0)]],
    device const int* indices [[buffer(1)]],  // Original indices
    device ScoreIndex* output [[buffer(2)]],
    device atomic_uint* output_count [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant float& threshold [[buffer(5)]],
    constant int& max_output [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(N)) return;

    float score = scores[gid];
    if (score >= threshold) {
        uint idx = atomic_fetch_add_explicit(output_count, 1, memory_order_relaxed);
        if (idx < uint(max_output)) {
            output[idx].score = score;
            output[idx].index = indices ? indices[gid] : int(gid);
        }
    }
}

// ============================================================================
// Top-K for Each Row (Matching)
// ============================================================================

constant int TOPK_MAX = 64;  // Max K for in-register selection

/**
 * Per-row top-k selection.
 * Each thread handles one row and maintains a min-heap of size K.
 */
kernel void row_topk(
    device const float* sim [[buffer(0)]],   // [N1, N2] similarity matrix
    device int* topk_indices [[buffer(1)]],  // [N1, K] output indices
    device float* topk_scores [[buffer(2)]], // [N1, K] output scores
    constant int3& dims [[buffer(3)]],       // N1, N2, K
    uint gid [[thread_position_in_grid]]
) {
    const int N1 = dims.x;
    const int N2 = dims.y;
    const int K = min(dims.z, TOPK_MAX);

    if (gid >= uint(N1)) return;

    device const float* row = sim + gid * N2;

    // Local arrays for top-k (min-heap)
    float heap_scores[TOPK_MAX];
    int heap_indices[TOPK_MAX];

    // Initialize with first K elements
    for (int i = 0; i < K; ++i) {
        heap_scores[i] = row[i];
        heap_indices[i] = i;
    }

    // Build min-heap
    for (int i = K / 2 - 1; i >= 0; --i) {
        // Heapify down
        int parent = i;
        while (true) {
            int smallest = parent;
            int left = 2 * parent + 1;
            int right = 2 * parent + 2;

            if (left < K && heap_scores[left] < heap_scores[smallest])
                smallest = left;
            if (right < K && heap_scores[right] < heap_scores[smallest])
                smallest = right;

            if (smallest == parent) break;

            // Swap
            float ts = heap_scores[parent];
            int ti = heap_indices[parent];
            heap_scores[parent] = heap_scores[smallest];
            heap_indices[parent] = heap_indices[smallest];
            heap_scores[smallest] = ts;
            heap_indices[smallest] = ti;

            parent = smallest;
        }
    }

    // Process remaining elements
    for (int j = K; j < N2; ++j) {
        float score = row[j];

        // If larger than min, replace and heapify
        if (score > heap_scores[0]) {
            heap_scores[0] = score;
            heap_indices[0] = j;

            // Heapify down from root
            int parent = 0;
            while (true) {
                int smallest = parent;
                int left = 2 * parent + 1;
                int right = 2 * parent + 2;

                if (left < K && heap_scores[left] < heap_scores[smallest])
                    smallest = left;
                if (right < K && heap_scores[right] < heap_scores[smallest])
                    smallest = right;

                if (smallest == parent) break;

                float ts = heap_scores[parent];
                int ti = heap_indices[parent];
                heap_scores[parent] = heap_scores[smallest];
                heap_indices[parent] = heap_indices[smallest];
                heap_scores[smallest] = ts;
                heap_indices[smallest] = ti;

                parent = smallest;
            }
        }
    }

    // Write output (unsorted within top-k, but that's often OK)
    device int* out_idx = topk_indices + gid * K;
    device float* out_score = topk_scores + gid * K;

    for (int i = 0; i < K; ++i) {
        out_idx[i] = heap_indices[i];
        out_score[i] = heap_scores[i];
    }
}

/**
 * Mutual nearest neighbor matching with top-k.
 * Finds matches where both i->j and j->i are in respective top-k.
 */
kernel void mutual_topk_match(
    device const int* topk_12 [[buffer(0)]],     // [N1, K]
    device const float* scores_12 [[buffer(1)]],  // [N1, K]
    device const int* topk_21 [[buffer(2)]],     // [N2, K]
    device atomic_int* match_count [[buffer(3)]],
    device int* matches_1 [[buffer(4)]],
    device int* matches_2 [[buffer(5)]],
    device float* match_scores [[buffer(6)]],
    constant int3& dims [[buffer(7)]],           // N1, N2, K
    constant int& max_matches [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    const int N1 = dims.x;
    const int K = dims.z;

    if (gid >= uint(N1)) return;

    device const int* my_topk = topk_12 + gid * K;
    device const float* my_scores = scores_12 + gid * K;

    for (int k = 0; k < K; ++k) {
        int j = my_topk[k];
        if (j < 0) continue;

        // Check if i is in j's top-k
        device const int* their_topk = topk_21 + j * K;
        bool mutual = false;

        for (int l = 0; l < K; ++l) {
            if (their_topk[l] == int(gid)) {
                mutual = true;
                break;
            }
        }

        if (mutual) {
            int idx = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
            if (idx < max_matches) {
                matches_1[idx] = int(gid);
                matches_2[idx] = j;
                match_scores[idx] = my_scores[k];
            }
        }
    }
}
