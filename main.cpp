#include <string>

#include <absl/log/check.h>
#include <fmt/format.h>
#include <proxy/proxy.h>

#include "test.hpp"
int main() {
    std::string message = "Hello, World!";
    fmt::print("{}\n", message);
    int res = test::cpp_test_wrapper();
    DCHECK_EQ(res, 0) << "CUDA test failed with error code: " << res;
    return 0;
}
