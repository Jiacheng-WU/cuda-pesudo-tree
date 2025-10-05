#include "pt.hpp"

#include <Eigen/Core>
#include <atomic>
#include <execution>
#include <fmt/base.h>
#include <omp.h>
#include <ranges>

#include <torch/torch.h>

#include <Eigen/Dense>

#include "random_gen.hpp"
#include "timer.hpp"

namespace pt::cpu {

template <std::size_t NumMatrices>
using DataGenerator = CpuDataGenerator<NumMatrices>;

int64_t sum_over_2nd_dim(const int64_t* matrix, const int32_t N,
                         const int32_t i) {
    int64_t total_sum = 0;
    for (int32_t j = 0; j < N; j++) {
        total_sum += matrix[i * N + j];
    }
    return total_sum;
}

int64_t pt_naive(const int32_t N, const int64_t seed) {
    DataGenerator<3> data_gen(N, seed);
    int64_t* A_matrix = data_gen.get_raw_ptr<0>();
    int64_t* B_matrix = data_gen.get_raw_ptr<1>();
    int64_t* C_matrix = data_gen.get_raw_ptr<2>();

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
    Timer::RepeatTiming<Timer::CPU>("CPU Naive", compute_func, reset_func);
    return total_sum.load();
}

int64_t pt_torch(const int32_t N, const int64_t seed) {
    DataGenerator<3> data_gen(N, seed);
    int64_t* raw_ptr_A = data_gen.get_raw_ptr<0>();
    int64_t* raw_ptr_B = data_gen.get_raw_ptr<1>();
    int64_t* raw_ptr_C = data_gen.get_raw_ptr<2>();

    torch::NoGradGuard no_grad;
    torch::Device device = torch::kCPU;
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);

    torch::Tensor A = torch::from_blob(raw_ptr_A, {N, N}, options);
    torch::Tensor B = torch::from_blob(raw_ptr_B, {N, N}, options);
    torch::Tensor C = torch::from_blob(raw_ptr_C, {N, N}, options);

    int64_t final_result = 0;
    auto compute_func = [&]() -> void {
        torch::Tensor A_row_sums = A.sum(1);
        torch::Tensor B_row_sums = B.sum(1);
        torch::Tensor C_row_sums = C.sum(1);

        torch::Tensor result = A_row_sums * B_row_sums * C_row_sums;

        final_result = result.sum().item<int64_t>();
    };

    Timer::RepeatTiming<Timer::CPU>("CPU Torch", compute_func,
                                    Timer::EmptyResetFunc);
    return final_result;
}

int64_t pt_eigen(const int32_t N, const int64_t seed) {
    DataGenerator<3> data_gen(N, seed);
    int64_t* raw_ptr_A = data_gen.get_raw_ptr<0>();
    int64_t* raw_ptr_B = data_gen.get_raw_ptr<1>();
    int64_t* raw_ptr_C = data_gen.get_raw_ptr<2>();

    // Set std::thread::hardware_concurrency() threads which is same
    // for std::execution::par and Intel TBB backend as well
    // We need to set both OpenMP and Eigen threads
    // because eigen may not set OpenMP threads when enable BLAS backend
    omp_set_num_threads(std::thread::hardware_concurrency());
    Eigen::setNbThreads(std::thread::hardware_concurrency());
    using MatrixType =
        Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowVectorType = Eigen::Vector<int64_t, Eigen::Dynamic>;
    Eigen::Map<MatrixType> A(raw_ptr_A, N, N);
    Eigen::Map<MatrixType> B(raw_ptr_B, N, N);
    Eigen::Map<MatrixType> C(raw_ptr_C, N, N);
    RowVectorType A_row_sums(N);
    RowVectorType B_row_sums(N);
    RowVectorType C_row_sums(N);

    int64_t final_result = 0;
    auto compute_func = [&]() -> void {
    // Eigen could do parallelization internally if we write like this:
    // A_row_sums = A.rowwise().sum();
    // B_row_sums = B.rowwise().sum();
    // C_row_sums = C.rowwise().sum();
    // BUT eigen won't parallelize even when N is not so small (32768)
    // decided by their cost model with or without multithreaded BLAS.
    // Also, for C = A * B eigen does parallelize when N = 32768.
    // Thus, here we manually parallelize the row-wise sum by OpenMP
    // Besides, in eigen 5, no need to set Eigen::InitParallel().

#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            A_row_sums(i) = A.row(i).sum();
            B_row_sums(i) = B.row(i).sum();
            C_row_sums(i) = C.row(i).sum();
        }

        final_result =
            (A_row_sums.array() * B_row_sums.array() * C_row_sums.array())
                .sum();
    };

    Timer::RepeatTiming<Timer::CPU>("CPU Eigen", compute_func,
                                    Timer::EmptyResetFunc);
    return final_result;
}

} // namespace pt::cpu
