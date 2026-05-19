// Module 5: Occupancy -- how many warps the SM can keep "in flight."
// More warps in flight = more latency-hiding. But more warps also means
// fewer registers per thread.
//
// Build: make 05-occupancy/occupancy_sweep
// Run:   ./05-occupancy/occupancy_sweep

#include "check.h"
#include <cstdio>

// A modestly arithmetic-heavy kernel so register pressure matters.
__global__ void busy(const float *__restrict__ in, float *__restrict__ out,
                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = in[i], b = a * 1.0001f, c = b + 0.5f, d = c * a + b;
  float e = d * d + a, f = e - b, g = f * c + d, h = g + e;
  out[i] = a + b + c + d + e + f + g + h;
}

int main() {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  printf("Max threads/SM=%d, regs/SM=%d, shared/SM=%zu KB\n",
         p.maxThreadsPerMultiProcessor, p.regsPerMultiprocessor,
         p.sharedMemPerMultiprocessor / 1024);

  const int N = 1 << 24;
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

  printf("block  active_blocks/SM  occupancy%%   ms\n");
  for (int block : {32, 64, 128, 256, 512, 1024}) {
    int active = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, busy, block, /*dyn smem*/ 0));
    int warps_per_block = block / p.warpSize;
    int active_warps = active * warps_per_block;
    int max_warps = p.maxThreadsPerMultiProcessor / p.warpSize;
    double occ = 100.0 * active_warps / max_warps;

    int grid = (N + block - 1) / block;
    auto run = [&] { busy<<<grid, block>>>(d_in, d_out, N); };
    float ms = time_ms(run);
    KERNEL_CHECK();
    printf("%4d   %14d  %8.1f    %6.3f\n", block, active, occ, ms);
  }

  // TODO(you): rebuild with --maxrregcount=32 (edit the Makefile or
  // export NVCCFLAGS=...). Watch ptxas print "spills" if you go too low.
  // Time the kernels again. Did "max occupancy" win? Often it doesn't.

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
