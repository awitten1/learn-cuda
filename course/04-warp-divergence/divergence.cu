// Module 4: Warp divergence cost.
// A warp executes 32 threads in lockstep. When threads in a warp take
// different branches, the hardware serializes them -- "divergence."
//
// Build: make 04-warp-divergence/divergence
// Run:   ./04-warp-divergence/divergence

#include "check.h"
#include <cstdio>

__global__ void divergent_kernel(const float *__restrict__ in,
                                 float *__restrict__ out, int n, int mod) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = in[i];
  // Threads in the same warp take different branches when `mod` < 32.
  switch (threadIdx.x % mod) {
    case 0:  x = x * 1.0001f + 0.1f; break;
    case 1:  x = x * 1.0002f + 0.2f; break;
    case 2:  x = x * 1.0003f + 0.3f; break;
    case 3:  x = x * 1.0004f + 0.4f; break;
    case 4:  x = x * 1.0005f + 0.5f; break;
    case 5:  x = x * 1.0006f + 0.6f; break;
    case 6:  x = x * 1.0007f + 0.7f; break;
    case 7:  x = x * 1.0008f + 0.8f; break;
    default: x = x * 1.0009f + 0.9f; break;
  }
  out[i] = x;
}

int main() {
  const int N = 1 << 24;
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, N * sizeof(float)));

  const int block = 256, grid = (N + block - 1) / block;
  printf("mod   ms       GB-eq/s\n");
  for (int mod : {1, 2, 4, 8, 16, 32, 64}) {
    auto run = [&] { divergent_kernel<<<grid, block>>>(d_in, d_out, N, mod); };
    float ms = time_ms(run);
    KERNEL_CHECK();
    double gbs = (double)(2.0 * N * sizeof(float)) / 1e9 / (ms / 1e3);
    printf("%3d   %6.3f   %6.1f\n", mod, ms, gbs);
  }
  // TODO(you): mod=1 means no divergence; mod=32 maximum divergence within a
  // warp; mod=64 returns to no divergence (each warp takes one branch).
  // Does the cost curve match your expectation?

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
