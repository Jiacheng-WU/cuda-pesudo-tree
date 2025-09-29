#pragma once
#include <cstdint>

namespace pt {
namespace cuda {
int64_t cuda_pt_naive(const int N, const unsigned long long seed = 42ULL);
} // namespace cuda
int64_t cpp_pt_torch(const int N, const unsigned long long seed = 42ULL);
} // namespace pt
