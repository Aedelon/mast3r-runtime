/**
 * CUDA matching kernels - STUB.
 *
 * Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

namespace mast3r {
namespace cuda {

/**
 * Top-K selection kernel using CUB.
 */
template <int BLOCK_SIZE>
__global__ void topk_kernel(
    const float* __restrict__ scores,
    int* __restrict__ indices,
    float* __restrict__ values,
    int n_queries,
    int n_keys,
    int k
) {
    // TODO: Implement efficient top-k using CUB radix sort or bitonic sort
}

/**
 * Reciprocal matching filter.
 */
__global__ void reciprocal_filter_kernel(
    const int* __restrict__ matches_1to2,
    const int* __restrict__ matches_2to1,
    int* __restrict__ valid_mask,
    int n_matches
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_matches) return;

    int match_2 = matches_1to2[idx];
    if (match_2 >= 0 && matches_2to1[match_2] == idx) {
        valid_mask[idx] = 1;
    } else {
        valid_mask[idx] = 0;
    }
}

/**
 * Compute cosine similarity matrix using cuBLAS.
 */
void compute_similarity_matrix(
    cublasHandle_t handle,
    const float* desc1,  // [N, D]
    const float* desc2,  // [M, D]
    float* similarity,   // [N, M]
    int n, int m, int d,
    cudaStream_t stream
) {
    // Similarity = desc1 @ desc2^T
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_T,  // desc2 transposed
        CUBLAS_OP_N,  // desc1 not transposed
        m, n, d,
        &alpha,
        desc2, d,
        desc1, d,
        &beta,
        similarity, m
    );
}

}  // namespace cuda
}  // namespace mast3r
