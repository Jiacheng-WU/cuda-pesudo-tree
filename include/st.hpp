#pragma once
#include <cstdint>

// pt is short for Pseudo-Tree
// st is short for Star

namespace st {
using data_t = int64_t;
namespace gpu {
data_t st_naive(const int32_t N, const data_t seed = 42ULL);
data_t st_pwarp(const int32_t N, const data_t seed = 42ULL);
data_t st_torch(const int32_t N, const data_t seed = 42ULL);
} // namespace gpu

namespace cpu {
data_t st_naive(const int32_t N, const data_t seed = 42ULL);
data_t st_torch(const int32_t N, const data_t seed = 42ULL);
data_t st_eigen(const int32_t N, const data_t seed = 42ULL);
} // namespace cpu

} // namespace st
