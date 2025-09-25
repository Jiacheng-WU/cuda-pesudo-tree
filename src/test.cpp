#include "test.hpp"

namespace test {

int cpp_test_wrapper() { return cuda::detail::cuda_test(); }

} // namespace test
