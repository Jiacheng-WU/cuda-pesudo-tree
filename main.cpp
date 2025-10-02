#include <absl/log/check.h>
#include <charconv>
#include <fmt/format.h>
#include <proxy/proxy.h>
#include <string>
#include <string_view>

#include "pt.hpp"
#include "timer.hpp"
int main(int argc, char** argv) {

    int32_t N = 32768; // 2<<15 && 2 << 30

    if (argc > 1) {
        std::string_view N_str = argv[1];
        auto [ptr, ec] = std::from_chars(N_str.begin(), N_str.end(), N);
        if (ec != std::errc() || ptr != N_str.end()) {
            fmt::println(
                "1st argument must be a number for the matrix size N.");
            return 1;
        }
        if (N <= 0) {
            fmt::println("Matrix size N must be positive.");
            return 1;
        }
    }

    fmt::println("Matrix size: {} x {}", N, N);

    bool use_gpu = true;
    bool use_cpu = false;

    if (argc > 2) {
        std::string_view mode = argv[2];
        if (mode == "cpu") {
            use_gpu = false;
            use_cpu = true;
        } else if (mode == "gpu") {
            use_gpu = true;
            use_cpu = false;
        } else if (mode == "both") {
            use_gpu = true;
            use_cpu = true;
        } else {
            fmt::println("2nd argument is mode in ['cpu', 'gpu', 'both']");
            return 1;
        }
    }

    if (argc > 3) {
        // Test if the third argument is a number
        std::string_view repeat_str = argv[3];
        int repeat_num = 0;
        auto [ptr, ec] =
            std::from_chars(repeat_str.begin(), repeat_str.end(), repeat_num);
        if (ec != std::errc() || ptr != repeat_str.end()) {
            fmt::println("3rd argument must be a number for the repeat count.");
            return 1;
        }
        if (repeat_num <= 0) {
            fmt::println("Repeat count must be positive.");
            return 1;
        }
        Timer::SetRepeatNum(repeat_num);
    }

    fmt::println("Repeat count: {}", Timer::GetRepeatNum());

    if (use_gpu) {
        int64_t gpu_naive_result = pt::gpu::pt_naive(N);
        fmt::println("GPU Naive result: {}", gpu_naive_result);

        int64_t gpu_pwarp_result = pt::gpu::pt_pwarp(N);
        fmt::println("GPU PWarp result: {}", gpu_pwarp_result);

        int64_t gpu_torch_result = pt::gpu::pt_torch(N);
        fmt::println("GPU Torch result: {}", gpu_torch_result);
    }

    if (use_cpu) {
        int64_t cpu_naive_result = pt::cpu::pt_naive(N);
        fmt::println("CPU Naive result: {}", cpu_naive_result);

        int64_t cpu_torch_result = pt::cpu::pt_torch(N);
        fmt::println("CPU Torch result: {}", cpu_torch_result);
    }

    return 0;
}
