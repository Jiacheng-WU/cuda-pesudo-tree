#pragma once

#include <chrono>
#include <fmt/format.h>

class CpuTimer {
  public:
    inline static auto EmptyResetFunc = []() -> void {};

    CpuTimer(const std::string& name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~CpuTimer() {
        end_ = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end_ - start_;
        fmt::print("{} took: {} ms\n", name_, duration.count());
    }

    template <typename Func>
        requires std::is_invocable_r_v<void, Func>
    inline static void WarmUp(Func&& func) {
        func();
    }

    template <typename Func>
        requires std::is_invocable_r_v<void, Func>
    inline static float Timing(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        return duration.count();
    }

    template <typename CallFunc, typename ResetFunc>
        requires std::is_invocable_r_v<void, CallFunc> &&
                 std::is_invocable_r_v<void, ResetFunc>
    inline static void RepeatTiming(const char* name, int repeat_count,
                                    CallFunc&& func, ResetFunc&& reset_func) {
        reset_func();
        CpuTimer::WarmUp(std::forward<decltype(func)>(func));
        float total_time = 0.0f;
        for (int i = 0; i < repeat_count; ++i) {
            reset_func();
            float elapsed_time =
                CpuTimer::Timing(std::forward<decltype(func)>(func));
            total_time += elapsed_time;
        }
        fmt::print("{} took: {} ms (averaged over {} runs)\n", name,
                   total_time / repeat_count, repeat_count);
    }

  private:
    std::chrono::high_resolution_clock::time_point start_, end_;
    std::string name_;
};
