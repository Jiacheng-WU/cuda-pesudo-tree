#pragma once

#include <concepts>
#include <cuda_runtime.h>
#include <fmt/format.h>

class GpuTimer {
  public:
    inline static auto EmptyResetFunc = []() -> void {};
    GpuTimer(const std::string& name) : name_(name) {
        cudaEventCreate(&start_);
        cudaEventCreate(&end_);
        cudaEventRecord(start_);
    }

    ~GpuTimer() {
        cudaEventRecord(end_);
        cudaEventSynchronize(end_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, end_);
        fmt::print("{} took: {} ms\n", name_, milliseconds);
        cudaEventDestroy(start_);
        cudaEventDestroy(end_);
    }

    template <typename Func>
        requires std::is_invocable_r_v<void, Func>
    inline static void WarmUp(Func&& func) {
        func();
    }

    template <typename Func>
        requires std::is_invocable_r_v<void, Func>
    inline static float Timing(Func&& func) {
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

    template <typename CallFunc, typename ResetFunc>
        requires std::is_invocable_r_v<void, CallFunc> &&
                 std::is_invocable_r_v<void, ResetFunc>
    inline static void RepeatTiming(const char* name, int repeat_count,
                                    CallFunc&& func, ResetFunc&& reset_func) {
        reset_func();
        GpuTimer::WarmUp(std::forward<decltype(func)>(func));
        float total_time = 0.0f;
        for (int i = 0; i < repeat_count; ++i) {
            reset_func();
            float elapsed_time =
                GpuTimer::Timing(std::forward<decltype(func)>(func));
            total_time += elapsed_time;
        }
        fmt::print("{} took: {} ms (averaged over {} runs)\n", name,
                   total_time / repeat_count, repeat_count);
    }

  private:
    cudaEvent_t start_, end_;
    std::string name_;
};
