// MASt3R Runtime - CPU Attention Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include <cmath>
#include <vector>

#ifdef MAST3R_HAS_OPENMP
#include <omp.h>
#endif

#ifdef MAST3R_HAS_BLAS
extern "C" {
    // BLAS SGEMM declaration
    void sgemm_(
        const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const float* alpha,
        const float* a, const int* lda,
        const float* b, const int* ldb,
        const float* beta,
        float* c, const int* ldc
    );
}
#endif

namespace mast3r {
namespace cpu {

/**
 * Matrix multiplication C = A @ B
 *
 * @param A [M, K]
 * @param B [K, N]
 * @param C [M, N] output
 */
void matmul(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
#ifdef MAST3R_HAS_BLAS
    // Use BLAS SGEMM
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const char trans_n = 'N';

    // BLAS uses column-major, so we compute C^T = B^T @ A^T
    sgemm_(
        &trans_n, &trans_n,
        &N, &M, &K,
        &alpha,
        B, &N,
        A, &K,
        &beta,
        C, &N
    );
#else
    // Naive implementation with OpenMP
#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#endif
}

/**
 * Softmax along last dimension.
 *
 * @param x [N, D] input/output (in-place)
 */
void softmax(float* x, int N, int D) {
#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i) {
        float* row = x + i * D;

        // Find max for numerical stability
        float max_val = row[0];
        for (int j = 1; j < D; ++j) {
            max_val = std::max(max_val, row[j]);
        }

        // Exp and sum
        float sum = 0.0f;
        for (int j = 0; j < D; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }

        // Normalize
        const float inv_sum = 1.0f / sum;
        for (int j = 0; j < D; ++j) {
            row[j] *= inv_sum;
        }
    }
}

/**
 * Multi-head self-attention.
 *
 * @param Q [B, H, N, D] queries
 * @param K [B, H, N, D] keys
 * @param V [B, H, N, D] values
 * @param output [B, H, N, D] output
 * @param B batch size
 * @param H number of heads
 * @param N sequence length
 * @param D head dimension
 */
void multi_head_attention(
    const float* Q, const float* K, const float* V,
    float* output,
    int B, int H, int N, int D
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Workspace for attention scores
    std::vector<float> attn_scores(N * N);
    std::vector<float> K_t(N * D);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const int offset = (b * H + h) * N * D;
            const float* Q_bh = Q + offset;
            const float* K_bh = K + offset;
            const float* V_bh = V + offset;
            float* out_bh = output + offset;

            // Transpose K: [N, D] -> [D, N]
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < D; ++j) {
                    K_t[j * N + i] = K_bh[i * D + j];
                }
            }

            // Attention scores: Q @ K^T, [N, N]
            matmul(Q_bh, K_t.data(), attn_scores.data(), N, N, D);

            // Scale
            for (int i = 0; i < N * N; ++i) {
                attn_scores[i] *= scale;
            }

            // Softmax
            softmax(attn_scores.data(), N, N);

            // Output: attn @ V, [N, D]
            matmul(attn_scores.data(), V_bh, out_bh, N, D, N);
        }
    }
}

/**
 * Layer normalization.
 *
 * @param x [N, D] input/output (in-place)
 * @param gamma [D] scale
 * @param beta [D] bias
 * @param eps epsilon for numerical stability
 */
void layer_norm(
    float* x, int N, int D,
    const float* gamma, const float* beta,
    float eps = 1e-5f
) {
#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i) {
        float* row = x + i * D;

        // Compute mean
        float mean = 0.0f;
        for (int j = 0; j < D; ++j) {
            mean += row[j];
        }
        mean /= D;

        // Compute variance
        float var = 0.0f;
        for (int j = 0; j < D; ++j) {
            const float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= D;

        // Normalize and scale
        const float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < D; ++j) {
            row[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}

/**
 * GELU activation.
 */
void gelu(float* x, int size) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;

#ifdef MAST3R_HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
        const float v = x[i];
        const float cdf = 0.5f * (1.0f + std::tanh(
            sqrt_2_over_pi * (v + coeff * v * v * v)
        ));
        x[i] = v * cdf;
    }
}

}  // namespace cpu
}  // namespace mast3r