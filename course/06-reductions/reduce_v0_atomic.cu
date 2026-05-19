// Module 6: Sum-reduction, 4 ways.
// v0 (provided): each thread atomicAdd's to a single global. Slowest.
// v1 (TODO):     block-level reduction in shared memory + 1 atomic per block.
// v2 (TODO):     warp-shuffle reduction (no shared memory inside the warp).
// v3 (TODO):     grid-level reduction with cooperative groups.
//
// Build: make 06-reductions/reduce_v0_atomic
// Run:   ./06-reductions/reduce_v0_atomic

#include "check.h"
#include <cstdio>
#include <vector>

__global__ void reduce_v0(const float *__restrict__ in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, in[i]);
}

// TODO(you): reduce_v1_shared
// __shared__ float s[BLOCK];
// s[tid] = (i<n) ? in[i] : 0.0f;
// __syncthreads();
// for (int stride = BLOCK/2; stride > 0; stride >>= 1) {
//   if (tid < stride) s[tid] += s[tid + stride];
//   __syncthreads();
// }
// if (tid == 0) atomicAdd(out, s[0]);

// TODO(you): reduce_v2_shuffle
//   float v = (i<n) ? in[i] : 0.0f;
//   for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
//   // warp leaders write to shared, then one final warp reduces shared
//   // (see "warp shuffle reduction" pattern)

int main() {
  const int N = 1 << 24;
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

  // Fill with 1.0 so the exact answer is N.
  {
    std::vector<float> h(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  const int block = 256, grid = (N + block - 1) / block;
  auto run = [&] {
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
    reduce_v0<<<grid, block>>>(d_in, d_out, N);
  };
  float ms = time_ms(run);
  KERNEL_CHECK();

  float h_out = 0;
  CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  double gbs = (double)(N * sizeof(float)) / 1e9 / (ms / 1e3);
  printf("v0 atomic:  sum=%.0f (expected %d)  %.3f ms  %.1f GB/s read\n",
         h_out, N, ms, gbs);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
