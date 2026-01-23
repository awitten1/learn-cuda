#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "utils.h"

__global__ void sum_kernel(const float* data, std::size_t n, double* out)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    double local = 0.0;
    for (std::size_t i = idx; i < n; i += stride) {
        local += data[i];
    }
    atomicAdd(out, local);
}

int main(int argc, char** argv)
{
    std::size_t n = 1u << 20;
    if (argc > 1) {
        n = static_cast<std::size_t>(std::stoull(std::string(argv[1])));
    }

    std::uint32_t seed = 0;
    if (argc > 2) {
        seed = static_cast<std::uint32_t>(std::stoul(std::string(argv[2])));
    }

    std::vector<float> host = gen_random_floats(n, 0.0f, 1.0f, seed);

    float* data = nullptr;
    double* device_sum = nullptr;
    cudaMallocManaged(&data, n * sizeof(float));
    cudaMallocManaged(&device_sum, sizeof(double));

    std::copy(host.begin(), host.end(), data);
    *device_sum = 0.0;

    int device = 0;
    cudaGetDevice(&device);
    cudaMemLocation device_loc{};
    device_loc.type = cudaMemLocationTypeDevice;
    device_loc.id = device;
    cudaMemPrefetchAsync(data, n * sizeof(float), device_loc, 0, 0);
    cudaMemPrefetchAsync(device_sum, sizeof(double), device_loc, 0, 0);

    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    if (grid > 1024) {
        grid = 1024;
    }

    float ms = timeit([&]() {
        sum_kernel<<<grid, block>>>(data, n, device_sum);
        cudaDeviceSynchronize();
    });

    cudaMemLocation host_loc{};
    host_loc.type = cudaMemLocationTypeHost;
    host_loc.id = 0;
    cudaMemPrefetchAsync(device_sum, sizeof(double), host_loc, 0, 0);
    cudaDeviceSynchronize();

    std::cout << "sum=" << *device_sum << std::endl;
    std::cout << "time_ms=" << ms << std::endl;

    cudaFree(data);
    cudaFree(device_sum);
    return 0;
}
