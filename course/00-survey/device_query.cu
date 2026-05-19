// Module 0: Print a "spec sheet" of YOUR GPU.
// You will reference these numbers from every later module.
//
// Build: make 00-survey/device_query
// Run:   ./00-survey/device_query

#include "check.h"
#include <cstdio>

int main() {
  int n;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  for (int dev = 0; dev < n; ++dev) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("Device %d: %s (sm_%d%d)\n", dev, p.name, p.major, p.minor);
    printf("  SMs:                  %d\n", p.multiProcessorCount);
    printf("  Warp size:            %d\n", p.warpSize);
    printf("  Max threads / block:  %d\n", p.maxThreadsPerBlock);
    printf("  Max threads / SM:     %d\n", p.maxThreadsPerMultiProcessor);
    printf("  Max blocks  / SM:     %d\n", p.maxBlocksPerMultiProcessor);
    printf("  Regs / block:         %d\n", p.regsPerBlock);
    printf("  Regs / SM:            %d\n", p.regsPerMultiprocessor);
    printf("  Shared mem / block:   %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("  Shared mem / SM:      %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("  L2 cache:             %d KB\n", p.l2CacheSize / 1024);
    printf("  Global mem:           %.2f GB\n", p.totalGlobalMem / 1e9);
    printf("  Mem bus width:        %d bits\n", p.memoryBusWidth);

    // memoryClockRate / clockRate fields were removed in newer CUDA; query
    // them via cudaDeviceGetAttribute instead.
    int mem_clock_khz = 0, sm_clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz,
                                      cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_clock_khz,
                                      cudaDevAttrClockRate, dev));
    printf("  Mem clock:            %.2f GHz\n", mem_clock_khz / 1e6);
    printf("  SM clock (base):      %.2f GHz\n", sm_clock_khz / 1e6);

    // Peak DRAM bandwidth (GB/s) = bus_width_bytes * effective_hz.
    // GDDR memory clock rate already accounts for the data rate, so don't
    // double-count -- check against bandwidthTest.
    double bw_gbs = (double)p.memoryBusWidth / 8.0 *
                    ((double)mem_clock_khz * 1e3) * 2.0 / 1e9;
    printf("  Theoretical peak DRAM BW: %.1f GB/s\n", bw_gbs);

    // FP32 cores: SMs * cores/SM. Cores/SM depends on arch -- look it up.
    printf("\n  TODO(you): look up FP32/Tensor TFLOPs for this arch and\n");
    printf("  record alongside your measured numbers from bandwidth_test.\n");
  }
  return 0;
}
