#include "pt.hpp"

#include <atomic>
#include <execution>
#include <ranges>

#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <torch/torch.h>

#include "cpu_timer.hpp"
#include "random_gen.hpp"

namespace pt::cpu {

int64_t sum_over_2nd_dim(const thrust::host_vector<int64_t>& matrix,
                         const int32_t N, const int32_t i) {
    int64_t total_sum = 0;
    for (int32_t j = 0; j < N; j++) {
        total_sum += matrix[i * N + j];
    }
    return total_sum;
}

int64_t pt_naive(const int32_t N, const int64_t seed) {
    const int64_t num_elements = N * N;

    // 1. Create a Thrust device_vector.
    thrust::host_vector<int64_t> A_matrix(num_elements);
    thrust::host_vector<int64_t> B_matrix(num_elements);
    thrust::host_vector<int64_t> C_matrix(num_elements);

    thrust::tabulate(A_matrix.begin(), A_matrix.end(), RandomGenerator(seed));
    thrust::tabulate(B_matrix.begin(), B_matrix.end(), RandomGenerator(seed));
    thrust::tabulate(C_matrix.begin(), C_matrix.end(), RandomGenerator(seed));

    std::atomic<int64_t> total_sum = 0;
    auto compute_func = [&]() -> void {
        auto indices = std::views::iota(0, N);
        std::for_each(std::execution::par_unseq, std::ranges::begin(indices),
                      std::ranges::end(indices), [&](int32_t i) -> void {
                          int64_t a_row_sum = sum_over_2nd_dim(A_matrix, N, i);
                          int64_t b_row_sum = sum_over_2nd_dim(B_matrix, N, i);
                          int64_t c_row_sum = sum_over_2nd_dim(C_matrix, N, i);
                          total_sum.fetch_add(a_row_sum * b_row_sum *
                                              c_row_sum);
                      });
    };

    auto reset_func = [&]() -> void { total_sum.store(0); };
    CpuTimer::RepeatTiming("CPU Naive", 10, compute_func, reset_func);
    return total_sum.load();
}

int64_t pt_torch(const int32_t N, const int64_t seed) {
    torch::NoGradGuard no_grad;
    const int64_t num_elements = N * N;

    // 1. Create a Thrust device_vector.
    thrust::host_vector<int64_t> A_matrix(num_elements);
    thrust::host_vector<int64_t> B_matrix(num_elements);
    thrust::host_vector<int64_t> C_matrix(num_elements);

    // 2. Use thrust::tabulate to fill the vector.
    // It calls an instance of our RandomGenerator for each index from 0 to
    // num_elements-1.
    thrust::tabulate(A_matrix.begin(), A_matrix.end(), RandomGenerator(seed));
    thrust::tabulate(B_matrix.begin(), B_matrix.end(), RandomGenerator(seed));
    thrust::tabulate(C_matrix.begin(), C_matrix.end(), RandomGenerator(seed));

    int64_t* raw_ptr_A = thrust::raw_pointer_cast(A_matrix.data());
    int64_t* raw_ptr_B = thrust::raw_pointer_cast(B_matrix.data());
    int64_t* raw_ptr_C = thrust::raw_pointer_cast(C_matrix.data());

    torch::Device device = torch::kCPU;
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    int64_t N_int64 = static_cast<int64_t>(N);

    torch::Tensor A = torch::from_blob(raw_ptr_A, {N_int64, N_int64}, options);
    torch::Tensor B = torch::from_blob(raw_ptr_B, {N_int64, N_int64}, options);
    torch::Tensor C = torch::from_blob(raw_ptr_C, {N_int64, N_int64}, options);

    int64_t final_result = 0;
    auto compute_func = [&]() -> void {
        torch::Tensor A_row_sums = A.sum(1);
        torch::Tensor B_row_sums = B.sum(1);
        torch::Tensor C_row_sums = C.sum(1);

        torch::Tensor result = A_row_sums * B_row_sums * C_row_sums;

        final_result = result.sum().item<int64_t>();
    };

    CpuTimer::RepeatTiming("CPU Torch", 10, compute_func,
                           CpuTimer::EmptyResetFunc);
    return final_result;
}

} // namespace pt::cpu
