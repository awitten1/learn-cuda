#include <cstdlib>
#include <iostream>
#include <math.h>
#include <thread>
#include <unistd.h>
#include <string.h>
#include <string>
#include <chrono>

// Kernel function to add the elements of two arrays
__global__
void add(long n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(int argc, char** argv)
{
 if (argc == 1) {
    fprintf(stderr, "Usage: ./add_cu <pause t|f> <N>");
    exit(EXIT_FAILURE);
 }

 bool p = strcmp(argv[1],"t") == 0;
 int n = std::stoi(std::string(argv[2]));

 long N = 1<<n;
 float *x, *y;

 // Allocate Unified Memory – accessible from CPU or GPU
 cudaMallocManaged(&x, N*sizeof(float));
 cudaMallocManaged(&y, N*sizeof(float));

 // initialize x and y arrays on the host
 for (long i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }

  if (p) {
    std::cout << "finished initializing on host, sleeping for 30 seconds" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(30));
  }

 // Run kernel on 1M elements on the GPU
 // add<<<1, 1>>>(N, x, y);
 add<<<1, 256>>>(N, x, y);

  if (p) {
    std::cout << "just called add, sleeping for 30s, then will call cudaDeviceSynchronize" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(30));
  }

 // Wait for GPU to finish before accessing on host
 cudaDeviceSynchronize();

  if (p) {
    std::cout << "finished adding on GPU and called cudaDeviceSynchronize" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(30));
  }

 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (long i = 0; i < N; i++) {
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 }
 std::cout << "Max error: " << maxError << std::endl;

  if (p) {
    std::cout << "computed max error, now sleeping for 30 seconds" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(30));
  }

 // Free memory
 cudaFree(x);
 cudaFree(y);
  return 0;
}