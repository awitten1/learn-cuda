// Module 8: Hello, tensor cores.
// One warp (32 threads) cooperatively computes a 16x16 fp16 matmul tile
// using nvcuda::wmma. Accumulates into fp32.
//
// Build: make 08-tensor-cores/wmma_hello
// Run:   ./08-tensor-cores/wmma_hello

#include "check.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

constexpr int M = 16, N = 16, K = 16;

__global__ void wmma_tile(const half *A, const half *B, float *C) {
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, M, N, K, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  wmma::load_matrix_sync(a, A, K);
  wmma::load_matrix_sync(b, B, K);
  wmma::mma_sync(acc, a, b, acc);
  wmma::store_matrix_sync(C, acc, N, wmma::mem_row_major);
}

int main() {
  half *A, *B; float *C;
  CUDA_CHECK(cudaMallocManaged(&A, M * K * sizeof(half)));
  CUDA_CHECK(cudaMallocManaged(&B, K * N * sizeof(half)));
  CUDA_CHECK(cudaMallocManaged(&C, M * N * sizeof(float)));
  for (int i = 0; i < M * K; i++) A[i] = __float2half(1.0f);
  for (int i = 0; i < K * N; i++) B[i] = __float2half(1.0f);

  wmma_tile<<<1, 32>>>(A, B, C);
  KERNEL_CHECK();

  printf("C[0][0] = %.1f  (expected %d)\n", C[0], K);
  printf("C[15][15] = %.1f\n", C[M * N - 1]);

  // TODO(you): tile this up to a real (e.g. 4096x4096) matmul.
  // TODO(you): rewrite using mma.sync PTX directly. Compare SASS.
  // TODO(you): compare fp16, bf16, fp8 throughput on this card.
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
