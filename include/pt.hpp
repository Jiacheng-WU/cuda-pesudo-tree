#include <span>

namespace pt {
namespace cuda::detail {
int cuda_pt_naive(std::span<int> data, std::span<int> result);
} // namespace cuda::detail
int cpp_pt_naive_wrapper();
} // namespace pt
