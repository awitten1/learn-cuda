// Module 7: CUDA streams -- overlap compute with H<->D copies.
// v0 (provided): serial (copy in, compute, copy out -- one big chunk).
// v1 (TODO):     N streams, each handling a chunk; copies and compute should
//                overlap on the nsys timeline.
//
// Build: make 07-streams/pipeline
// Run:   ./07-streams/pipeline

#include "check.h"
#include <cstdio>

__global__ void busy(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = in[i];
  // A handful of FMAs so the kernel isn't trivially fast.
  for (int k = 0; k < 64; k++) x = x * 1.0001f + 0.5f;
  out[i] = x;
}

int main() {
  const int N = 1 << 24;             // 16M floats
  const size_t bytes = N * sizeof(float);

  // Pinned host buffers -- required for true async H<->D overlap.
  float *h_in, *h_out;
  CUDA_CHECK(cudaMallocHost(&h_in, bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, bytes));
  for (int i = 0; i < N; i++) h_in[i] = 1.0f;

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  // ----- v0: serial -----
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
  busy<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms_serial = 0;
  cudaEventElapsedTime(&ms_serial, s, e);
  printf("serial:  %.3f ms\n", ms_serial);

  // TODO(you): v1 -- chunk N into K pieces, create K cudaStreams, and for
  // each stream:
  //   cudaMemcpyAsync(d_in + off, h_in + off, chunkBytes, H2D, stream[k]);
  //   busy<<<..., stream[k]>>>(d_in + off, d_out + off, chunkSize);
  //   cudaMemcpyAsync(h_out + off, d_out + off, chunkBytes, D2H, stream[k]);
  // Then cudaStreamSynchronize each.
  // Profile with `nsys profile -o pipe ./pipeline` and open in Nsight Systems.
  // You should see HtoD, kernel, and DtoH bars overlap on different rows.
  // If they don't, common bugs: forgot pinned host memory, default stream,
  // not enough chunks.

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
