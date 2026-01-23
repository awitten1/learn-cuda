#pragma once

#include <vector>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

inline std::vector<float> gen_random_floats(std::size_t n, float min_val, float max_val, std::uint32_t seed)
{
  if (min_val > max_val) {
    std::swap(min_val, max_val);
  }

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(min_val, max_val);

  std::vector<float> values;
  values.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    values.push_back(dist(rng));
  }
  return values;
}

inline std::vector<float> gen_random_floats(std::size_t n, float min_val = 0.0f, float max_val = 1.0f)
{
  std::random_device rd;
  return gen_random_floats(n, min_val, max_val, rd());
}

template <typename Callable>
float timeit(const char* label, Callable c)
{
  constexpr static auto& now = std::chrono::steady_clock::now;
  auto t1 = now();
  c();
  auto t2 = now();
  float ms = std::chrono::duration<float, std::milli>(t2 - t1).count();
  if (label && label[0] != '\0') {
    std::cout << label << "=" << ms << "ms ";
  }
  return ms;
}
