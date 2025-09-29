#include <absl/log/check.h>
#include <fmt/format.h>
#include <proxy/proxy.h>

#include "pt.hpp"
int main() {
    int64_t result = pt::cuda::cuda_pt_naive(32768);
    fmt::print("Result: {}\n", result);

    int64_t cpp_result = pt::cpp_pt_torch(10);
    // Due to not init
    DCHECK_EQ(cpp_result, 0);
    return 0;
}
