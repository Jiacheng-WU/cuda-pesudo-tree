#include <fmt/format.h>
#include <string>

#include "test.hpp"
int main() {
    std::string message = "Hello, World!";
    fmt::print("{}\n", message);
    // test::cuda::detail::cuda_test();
    test::cpp_test_wrapper();
    return 0;
}
