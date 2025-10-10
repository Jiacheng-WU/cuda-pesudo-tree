#pragma once
#include <cstdint>

// tb is short for Triangular Bipyramid

namespace tb {

using data_t = int32_t;

namespace gpu {
data_t tb_naive(const int32_t N, const data_t seed = 42ULL);
data_t tb_pwarp(const int32_t N, const data_t seed = 42ULL);
data_t tb_torch(const int32_t N, const data_t seed = 42ULL);
} // namespace gpu

namespace cpu {
data_t tb_naive(const int32_t N, const data_t seed = 42ULL);
data_t tb_torch(const int32_t N, const data_t seed = 42ULL);
data_t tb_eigen(const int32_t N, const data_t seed = 42ULL);
} // namespace cpu

} // namespace tb
