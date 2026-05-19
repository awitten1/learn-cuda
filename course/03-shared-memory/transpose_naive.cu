// Module 3: Matrix transpose -- the classic shared-memory exercise.
// v0 (provided): naive. Reads coalesced, writes uncoalesced.
// v1 (TODO):     shared-memory tiled.
// v2 (TODO):     shared-memory tiled with +1 padding to kill bank conflicts.
//
// Build: make 03-shared-memory/transpose_naive
// Run:   ./03-shared-memory/transpose_naive

#include "check.h"
#include <cstdio>

constexpr int TILE = 32;

__global__ void transpose_naive(const float *__restrict__ in,
                                float *__restrict__ out, int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < cols && y < rows) {
    out[x * rows + y] = in[y * cols + x];  // <-- write is strided. Hurts.
  }
}

// TODO(you): __global__ void transpose_tiled(...)
//   __shared__ float tile[TILE][TILE];
//   // load coalesced from `in` into tile
//   __syncthreads();
//   // write coalesced from tile (transposed indexing) into `out`
// Measure -- you should see a big jump in BW. Then check NCU for bank
// conflicts; you'll find a LOT.

// TODO(you): __global__ void transpose_tiled_padded(...)
//   __shared__ float tile[TILE][TILE + 1];   // <-- the +1 is the lesson
// Re-measure. Bank conflicts should drop to ~0.

int main() {
  const int N = 8192;  // N x N matrix of floats = 256 MB
  size_t bytes = (size_t)N * N * sizeof(float);
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemset(d_in, 0, bytes));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

  auto run_naive = [&] { transpose_naive<<<grid, block>>>(d_in, d_out, N, N); };
  float ms = time_ms(run_naive);
  KERNEL_CHECK();
  double gbs = (double)(2.0 * bytes) / 1e9 / (ms / 1e3);
  printf("naive:         %.3f ms  %.1f GB/s\n", ms, gbs);

  // TODO(you): time tiled and tiled_padded the same way.

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
