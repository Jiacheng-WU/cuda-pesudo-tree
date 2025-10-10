#pragma once
#include <cstdint>

// tb is short for Triangular Bipyramid

namespace tb {
namespace gpu {
int64_t tb_naive(const int32_t N, const int64_t seed = 42ULL);
int64_t tb_pwarp(const int32_t N, const int64_t seed = 42ULL);
int64_t tb_torch(const int32_t N, const int64_t seed = 42ULL);
} // namespace gpu

namespace cpu {
int64_t tb_naive(const int32_t N, const int64_t seed = 42ULL);
int64_t tb_torch(const int32_t N, const int64_t seed = 42ULL);
int64_t tb_eigen(const int32_t N, const int64_t seed = 42ULL);
} // namespace cpu

} // namespace tb
