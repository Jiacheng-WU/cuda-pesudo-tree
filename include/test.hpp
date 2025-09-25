#include <span>

namespace test {
namespace cuda::detail {
int cuda_test(std::span<int> data, std::span<int> result);
} // namespace cuda::detail
int cpp_test_wrapper();
} // namespace test
