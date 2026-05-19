// Module 9, capstone v0: naive SGEMM. C = A * B, N x N, fp32.
// One thread per output element. No shared memory. Awful, on purpose.
//
// Copy this file as you iterate: sgemm_v1_coalesced.cu, sgemm_v2_tiled.cu, ...
//
// Build: make 09-gemm/sgemm_v0_naive
// Run:   ./09-gemm/sgemm_v0_naive 4096
//
// Compare against cuBLAS (link -lcublas) once you have v1+.

#include "check.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void sgemm_naive(int N, const float *A, const float *B, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < N; k++) acc += A[row * N + k] * B[k * N + col];
  C[row * N + col] = acc;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? std::atoi(argv[1]) : 4096;
  size_t bytes = (size_t)N * N * sizeof(float);

  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, bytes));
  CUDA_CHECK(cudaMalloc(&B, bytes));
  CUDA_CHECK(cudaMalloc(&C, bytes));

  // Cheap init -- not random, but enough to not be all-zero.
  CUDA_CHECK(cudaMemset(A, 0x3f, bytes));   // ~0.74
  CUDA_CHECK(cudaMemset(B, 0x3f, bytes));

  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (N + 15) / 16);
  auto run = [&] { sgemm_naive<<<grid, block>>>(N, A, B, C); };
  float ms = time_ms(run, /*warmup=*/2, /*iters=*/5);
  KERNEL_CHECK();

  // FLOPs = 2 * N^3 (one mul + one add per inner iter).
  double gflops = (2.0 * N * N * N) / 1e9 / (ms / 1e3);
  printf("sgemm_naive N=%d:  %.3f ms   %.1f GFLOPS\n", N, ms, gflops);

  // TODO(you): keep this file frozen. For each next version copy to a new
  // file (sgemm_v1_coalesced.cu, sgemm_v2_tiled.cu, ...). Track GFLOPS for
  // each in notes/09.md. Add cuBLAS in v8 as the ceiling.

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
