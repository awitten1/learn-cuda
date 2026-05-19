// Module 1: Measure the cost of launching a kernel.
// Why it matters: small kernels are often bottlenecked by launch latency.
//
// Build: make 01-execution-model/launch_overhead
// Run:   ./01-execution-model/launch_overhead

#include "check.h"
#include <cstdio>

__global__ void empty() {}

int main() {
  const int N = 10000;
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);

  // Warmup so the driver caches what it needs.
  for (int i = 0; i < 100; i++) empty<<<1, 1>>>();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEventRecord(s);
  for (int i = 0; i < N; i++) empty<<<1, 1>>>();
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms = 0;
  cudaEventElapsedTime(&ms, s, e);
  printf("Empty kernel launch: %.3f us / launch (N=%d)\n",
         ms * 1000.0 / N, N);

  // Same thing but synchronize each launch -- shows the host<->device
  // round-trip cost on top of pure launch.
  cudaEventRecord(s);
  for (int i = 0; i < N / 10; i++) {
    empty<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  cudaEventElapsedTime(&ms, s, e);
  printf("Launch + sync:       %.3f us / launch\n", ms * 1000.0 / (N / 10));

  // TODO(you): run under `nsys profile --stats=true` and look at the
  // "CUDA Kernel Statistics" / "CUDA API Statistics" tables.
  return 0;
}
