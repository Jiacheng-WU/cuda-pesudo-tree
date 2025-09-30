#include <absl/log/check.h>
#include <fmt/format.h>
#include <proxy/proxy.h>

#include "pt.hpp"
int main(int argc, char** argv) {

    int32_t N = 32768; // 2<<15 && 2 << 30

    if (argc > 1) {
        // Test if the first argument is a number
        DCHECK(std::all_of(argv[1], argv[1] + std::strlen(argv[1]), ::isdigit))
            << "The first argument must be a number representing the matrix "
               "size N.";
        N = std::stoull(argv[1]);
    }

    fmt::print("Matrix size: {} x {}\n", N, N);

    int64_t gpu_naive_result = pt::gpu::pt_naive(N);
    fmt::print("GPU Naive Result: {}\n", gpu_naive_result);

    int64_t gpu_torch_result = pt::gpu::pt_torch(N);
    fmt::print("GPU Torch Result: {}\n", gpu_torch_result);

    int64_t cpu_naive_result = pt::cpu::pt_naive(N);
    fmt::print("CPU Naive Result: {}\n", cpu_naive_result);

    int64_t cpu_torch_result = pt::cpu::pt_torch(N);
    fmt::print("CPU Torch Result: {}\n", cpu_torch_result);

    return 0;
}
