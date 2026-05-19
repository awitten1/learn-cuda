#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(_e),    \
              __FILE__, __LINE__, cudaGetErrorString(_e));                     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define KERNEL_CHECK()                                                         \
  do {                                                                         \
    cudaError_t _e = cudaGetLastError();                                       \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(_e));    \
      std::exit(1);                                                            \
    }                                                                          \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

// Time a callable using CUDA events. Returns average milliseconds per iter.
template <typename F>
float time_ms(F &&f, int warmup = 3, int iters = 10) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int i = 0; i < warmup; i++) f();
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) f();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iters;
}
