// Module 1: Map which blocks land on which SMs.
// Uses inline PTX to read %smid (the SM the block is currently running on).
//
// Build: make 01-execution-model/smid_mapping
// Run:   ./01-execution-model/smid_mapping

#include "check.h"
#include <cstdio>
#include <vector>

__device__ __forceinline__ unsigned get_smid() {
  unsigned id;
  asm("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

__global__ void record_smid(unsigned *block_to_sm) {
  if (threadIdx.x == 0) {
    block_to_sm[blockIdx.x] = get_smid();
  }
}

int main() {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));

  const int blocks = p.multiProcessorCount * 4;
  unsigned *d_map;
  CUDA_CHECK(cudaMalloc(&d_map, blocks * sizeof(unsigned)));
  record_smid<<<blocks, 32>>>(d_map);
  KERNEL_CHECK();

  std::vector<unsigned> h(blocks);
  CUDA_CHECK(cudaMemcpy(h.data(), d_map, blocks * sizeof(unsigned),
                        cudaMemcpyDeviceToHost));

  std::vector<int> hist(p.multiProcessorCount, 0);
  for (int b = 0; b < blocks; b++) hist[h[b]]++;

  printf("Launched %d blocks across %d SMs:\n", blocks, p.multiProcessorCount);
  for (int sm = 0; sm < p.multiProcessorCount; sm++) {
    printf("  SM %2d: %d blocks\n", sm, hist[sm]);
  }

  // TODO(you): change `blocks` to 1, then to p.multiProcessorCount, then to
  // 1024. How does the distribution change? Is it perfectly round-robin?

  CUDA_CHECK(cudaFree(d_map));
  return 0;
}
