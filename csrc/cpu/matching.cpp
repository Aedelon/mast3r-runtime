// MASt3R Runtime - CPU Matching Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

#ifdef MAST3R_HAS_OPENMP
#include <omp.h>
#endif

#include "common/types.hpp"

namespace mast3r {
namespace cpu {

/**
 * L2 normalize descriptors in-place.
 */
void normalize_descriptors(float* desc, int N, int D) {
#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i) {
        float* row = desc + i * D;

        // Compute L2 norm
        float norm = 0.0f;
        for (int j = 0; j < D; ++j) {
            norm += row[j] * row[j];
        }
        norm = std::sqrt(norm);

        // Normalize
        if (norm > 1e-8f) {
            const float inv_norm = 1.0f / norm;
            for (int j = 0; j < D; ++j) {
                row[j] *= inv_norm;
            }
        }
    }
}

/**
 * Compute cosine similarity matrix.
 *
 * @param desc_1 [N1, D] normalized descriptors
 * @param desc_2 [N2, D] normalized descriptors
 * @param sim [N1, N2] output similarity matrix
 */
void compute_similarity(
    const float* desc_1, const float* desc_2,
    float* sim,
    int N1, int N2, int D
) {
#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < D; ++k) {
                dot += desc_1[i * D + k] * desc_2[j * D + k];
            }
            sim[i * N2 + j] = dot;
        }
    }
}

/**
 * Reciprocal matching.
 *
 * @param desc_1 [H*W, D] descriptors from view 1
 * @param desc_2 [H*W, D] descriptors from view 2
 * @param conf_1 [H*W] confidence (optional, can be nullptr)
 * @param conf_2 [H*W] confidence (optional, can be nullptr)
 * @param height image height
 * @param width image width
 * @param desc_dim descriptor dimension
 * @param config matching configuration
 * @return MatchResult with correspondences
 */
MatchResult reciprocal_match(
    const float* desc_1, const float* desc_2,
    const float* conf_1, const float* conf_2,
    int height, int width, int desc_dim,
    const MatchingConfig& config
) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    const int N = height * width;

    // Copy and normalize descriptors
    std::vector<float> desc_1_norm(desc_1, desc_1 + N * desc_dim);
    std::vector<float> desc_2_norm(desc_2, desc_2 + N * desc_dim);
    normalize_descriptors(desc_1_norm.data(), N, desc_dim);
    normalize_descriptors(desc_2_norm.data(), N, desc_dim);

    // Compute similarity matrix [N, N]
    std::vector<float> sim(N * N);
    compute_similarity(
        desc_1_norm.data(), desc_2_norm.data(),
        sim.data(), N, N, desc_dim
    );

    // Apply confidence weighting if provided
    if (conf_1 != nullptr && conf_2 != nullptr) {
#ifdef MAST3R_HAS_OPENMP
        #pragma omp parallel for collapse(2)
#endif
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                sim[i * N + j] *= conf_1[i] * conf_2[j];
            }
        }
    }

    // Find nearest neighbors
    std::vector<int> nn_12(N);  // Forward: desc_1 -> desc_2
    std::vector<int> nn_21(N);  // Backward: desc_2 -> desc_1

#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i) {
        // Forward NN
        int best_j = 0;
        float best_sim = sim[i * N];
        for (int j = 1; j < N; ++j) {
            if (sim[i * N + j] > best_sim) {
                best_sim = sim[i * N + j];
                best_j = j;
            }
        }
        nn_12[i] = best_j;
    }

#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int j = 0; j < N; ++j) {
        // Backward NN
        int best_i = 0;
        float best_sim = sim[j];
        for (int i = 1; i < N; ++i) {
            if (sim[i * N + j] > best_sim) {
                best_sim = sim[i * N + j];
                best_i = i;
            }
        }
        nn_21[j] = best_i;
    }

    // Collect reciprocal matches
    std::vector<int64_t> idx_1, idx_2;
    std::vector<float> confidence;

    if (config.reciprocal) {
        for (int i = 0; i < N; ++i) {
            const int j = nn_12[i];
            if (nn_21[j] == i) {
                const float score = sim[i * N + j];
                if (score >= config.confidence_threshold) {
                    idx_1.push_back(i);
                    idx_2.push_back(j);
                    confidence.push_back(score);
                }
            }
        }
    } else {
        for (int i = 0; i < N; ++i) {
            const int j = nn_12[i];
            const float score = sim[i * N + j];
            if (score >= config.confidence_threshold) {
                idx_1.push_back(i);
                idx_2.push_back(j);
                confidence.push_back(score);
            }
        }
    }

    // Keep top-K by confidence
    const int num_matches = static_cast<int>(idx_1.size());
    if (num_matches > config.top_k) {
        // Sort indices by confidence (descending)
        std::vector<int> order(num_matches);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
            order.begin(), order.begin() + config.top_k, order.end(),
            [&confidence](int a, int b) {
                return confidence[a] > confidence[b];
            }
        );

        // Keep top-K
        std::vector<int64_t> new_idx_1(config.top_k);
        std::vector<int64_t> new_idx_2(config.top_k);
        std::vector<float> new_conf(config.top_k);
        for (int k = 0; k < config.top_k; ++k) {
            new_idx_1[k] = idx_1[order[k]];
            new_idx_2[k] = idx_2[order[k]];
            new_conf[k] = confidence[order[k]];
        }
        idx_1 = std::move(new_idx_1);
        idx_2 = std::move(new_idx_2);
        confidence = std::move(new_conf);
    }

    // Convert flat indices to 2D coordinates
    const int final_count = static_cast<int>(idx_1.size());
    std::vector<float> pts2d_1(final_count * 2);
    std::vector<float> pts2d_2(final_count * 2);

    for (int k = 0; k < final_count; ++k) {
        pts2d_1[k * 2 + 0] = static_cast<float>(idx_1[k] % width);  // x
        pts2d_1[k * 2 + 1] = static_cast<float>(idx_1[k] / width);  // y
        pts2d_2[k * 2 + 0] = static_cast<float>(idx_2[k] % width);
        pts2d_2[k * 2 + 1] = static_cast<float>(idx_2[k] / width);
    }

    auto t1 = Clock::now();

    MatchResult result;
    result.idx_1 = std::move(idx_1);
    result.idx_2 = std::move(idx_2);
    result.pts2d_1 = std::move(pts2d_1);
    result.pts2d_2 = std::move(pts2d_2);
    result.pts3d_1.resize(final_count * 3, 0.0f);  // Placeholder
    result.pts3d_2.resize(final_count * 3, 0.0f);
    result.confidence = std::move(confidence);
    result.match_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    return result;
}

}  // namespace cpu
}  // namespace mast3r