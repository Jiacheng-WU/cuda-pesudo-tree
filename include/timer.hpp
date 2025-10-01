#pragma once

#include <chrono>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <fmt/format.h>

class Timer {
  public:
    enum Mode { CPU, GPU };

    inline static auto EmptyResetFunc = []() -> void {};

    template <typename Func>
        requires std::is_invocable_r_v<void, Func>
    inline static void WarmUp(Func&& func) {
        func();
    }

    template <typename Func>
        requires std::is_invocable_r_v<void, Func>
    inline static float TimingCPU(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        return duration.count();
    }

    template <typename Func>
        requires std::is_invocable_r_v<void, Func>
    inline static float TimingGPU(Func&& func) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        func();
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        return milliseconds;
    }

    template <Mode mode, typename CallFunc, typename ResetFunc>
        requires std::is_invocable_r_v<void, CallFunc> &&
                 std::is_invocable_r_v<void, ResetFunc>
    inline static void RepeatTiming(const char* name, CallFunc&& func,
                                    ResetFunc&& reset_func) {
        reset_func();
        Timer::WarmUp(std::forward<decltype(func)>(func));
        float total_time = 0.0f;
        for (uint64_t i = 0; i < repeat_num; ++i) {
            reset_func();
            if constexpr (mode == Mode::GPU) {
                total_time +=
                    Timer::TimingGPU(std::forward<decltype(func)>(func));
            } else if constexpr (mode == Mode::CPU) {
                total_time +=
                    Timer::TimingCPU(std::forward<decltype(func)>(func));
            } else {
                static_assert(mode == Mode::CPU || mode == Mode::GPU,
                              "Unsupported mode");
            }
        }
        fmt::println("{} took: {:.6f} ms (averaged over {} runs)", name,
                     total_time / repeat_num, repeat_num);
    }

    static void SetRepeatNum(uint64_t num) { repeat_num = num; }
    static uint64_t GetRepeatNum() { return repeat_num; }

  private:
    inline static uint64_t repeat_num = 3;
};
