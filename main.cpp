#include <string>

#include <absl/log/check.h>
#include <fmt/format.h>
#include <proxy/proxy.h>

#include "pt.hpp"
int main() {
    std::string message = "Hello, World!";
    fmt::print("{}\n", message);
    int res = pt::cpp_pt_naive_wrapper();
    DCHECK_EQ(res, 0) << "CUDA test failed with error code: " << res;
    return 0;
}
