// Module 0: Measure host<->device and device<->device bandwidth.
// Compare against the theoretical peak printed by device_query.
//
// Build: make 00-survey/bandwidth_test
// Run:   ./00-survey/bandwidth_test

#include "check.h"
#include <cstdio>
#include <cstdlib>

static float bench_copy(void *dst, const void *src, size_t bytes,
                        cudaMemcpyKind kind) {
  auto run = [&] { CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind)); };
  return time_ms(run, /*warmup=*/3, /*iters=*/10);
}

int main() {
  const size_t MB = 1 << 20;
  const size_t bytes = 256 * MB;

  void *h_pageable = std::malloc(bytes);
  void *h_pinned = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
  void *d_a = nullptr, *d_b = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));

  float ms;
  double gbs;
  auto report = [&](const char *label, float ms) {
    gbs = (double)bytes / 1e9 / (ms / 1e3);
    printf("  %-30s %8.3f ms   %7.2f GB/s\n", label, ms, gbs);
  };

  printf("Copy size: %zu MB\n", bytes / MB);
  ms = bench_copy(d_a, h_pageable, bytes, cudaMemcpyHostToDevice);
  report("H2D (pageable)", ms);
  ms = bench_copy(d_a, h_pinned, bytes, cudaMemcpyHostToDevice);
  report("H2D (pinned)", ms);
  ms = bench_copy(h_pageable, d_a, bytes, cudaMemcpyDeviceToHost);
  report("D2H (pageable)", ms);
  ms = bench_copy(h_pinned, d_a, bytes, cudaMemcpyDeviceToHost);
  report("D2H (pinned)", ms);
  ms = bench_copy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice);
  report("D2D", ms);

  // TODO(you): why is pinned faster? Look up "pinned memory" + DMA.
  // TODO(you): record D2D bandwidth -- compare to theoretical DRAM peak.

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFreeHost(h_pinned));
  std::free(h_pageable);
  return 0;
}
