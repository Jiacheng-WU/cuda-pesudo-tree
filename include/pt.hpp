#pragma once
#include <cstdint>

namespace pt {
namespace gpu {
int64_t pt_naive(const int32_t N, const int64_t seed = 42ULL);
int64_t pt_pwarp(const int32_t N, const int64_t seed = 42ULL);
int64_t pt_torch(const int32_t N, const int64_t seed = 42ULL);
} // namespace gpu

namespace cpu {
int64_t pt_naive(const int32_t N, const int64_t seed = 42ULL);
int64_t pt_torch(const int32_t N, const int64_t seed = 42ULL);
} // namespace cpu

} // namespace pt
