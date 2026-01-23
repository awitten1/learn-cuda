
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "utils.h"

double sum(float* buf, size_t n) {
    double ret = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : ret)
#endif
    for (size_t i = 0; i < n; ++i) {
        ret += buf[i];
    }
    return ret;
}

int main(int argc, char** argv) {
    size_t n = 1u << 20;
    if (argc > 1) {
        n = static_cast<size_t>(std::stoull(std::string(argv[1])));
    }

    std::uint32_t seed = 0;
    if (argc > 2) {
        seed = static_cast<std::uint32_t>(std::stoul(std::string(argv[2])));
    }

    std::vector<float> data = gen_random_floats(n, 0.0f, 1.0f, seed);
    double result = 0.0;
    timeit("cpu_sum_ms", [&]() {
        result = sum(data.data(), data.size());
    });

    std::cout << "sum=" << result << std::endl;
    return 0;
}
