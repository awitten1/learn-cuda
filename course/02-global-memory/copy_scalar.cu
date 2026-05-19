// Module 2: Global memory bandwidth via copy kernels.
// You will write three variants and compare achieved GB/s.
//
// Build: make 02-global-memory/copy_scalar
// Run:   ./02-global-memory/copy_scalar

#include "check.h"
#include <cstdio>

__global__ void copy_scalar(const float *__restrict__ in,
                            float *__restrict__ out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

// TODO(you): write copy_strided<int STRIDE> -- each thread reads in[i*STRIDE].
// Predict the bandwidth ratio vs scalar, then measure.

// TODO(you): write copy_vec4 -- reinterpret as float4 and copy 16B per thread.
// You should beat copy_scalar on a memory-bound workload.

int main() {
  const size_t N = 1u << 27;  // 128M floats = 512 MB
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 1, N * sizeof(float)));

  const int block = 256;
  const int grid = (N + block - 1) / block;
  auto run = [&] { copy_scalar<<<grid, block>>>(d_in, d_out, N); };
  float ms = time_ms(run);
  KERNEL_CHECK();

  // bytes moved = read N floats + write N floats = 2 * N * 4
  double gbs = (double)(2.0 * N * sizeof(float)) / 1e9 / (ms / 1e3);
  printf("copy_scalar: %.3f ms  %.1f GB/s\n", ms, gbs);

  // TODO(you): launch + time copy_strided<2>, <4>, <8>, <16>, <32>.
  // TODO(you): launch + time copy_vec4. Compare to D2D peak from Module 0.

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
