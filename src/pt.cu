#include "pt.hpp"

#include <array>
#include <cooperative_groups.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>
#include <driver_types.h>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <torch/torch.h>

#include "gpu_timer.hpp"
#include "random_gen.hpp"

#include <concepts>
#include <type_traits>

namespace cg = cooperative_groups;

namespace pt::gpu {

template <std::size_t NumMatrices>
using DataGenerator = GpuDataGenerator<NumMatrices>;

// \sum A_ij * B_ik * C_il
// where A, B, C are (N, N) matrices
// and the output is stored in

constinit const int32_t NUM_THREADS_IN_BLOCK = 1024;

__device__ int64_t sum_over_2nd_dim(const int64_t* Mat, const int32_t N,
                                    int32_t row_idx) {

    int32_t thread_idx = threadIdx.x;
    int32_t num_threads_in_block = blockDim.x;
    int64_t Mat_i_sum_over_part_j = 0;

    // Use loop stride mode for coalesced memory access
    for (int32_t col_idx = thread_idx; col_idx < N;
         col_idx += num_threads_in_block) {
        int64_t Mat_ij = Mat[row_idx * N + col_idx];
        Mat_i_sum_over_part_j += Mat_ij;
    }
    cg::this_thread_block().sync();

    using BlockReduce = cub::BlockReduce<int64_t, NUM_THREADS_IN_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int64_t Mat_i_sum = BlockReduce(temp_storage).Sum(Mat_i_sum_over_part_j);

    return Mat_i_sum;
}

// Assume N is multiple of NUM_THREADS_IN_BLOCK
__global__ void naive_star_kernel(const int64_t* A, const int64_t* B,
                                  const int64_t* C, const int32_t N,
                                  int64_t* g_output) {

    int32_t num_blocks_in_grid = gridDim.x;
    int32_t block_idx = blockIdx.x;
    [[maybe_unused]] int32_t num_threads_in_block = blockDim.x;
    int32_t thread_idx = threadIdx.x;

    assert(num_blocks_in_grid == num_threads_in_block);
    assert(num_threads_in_block == NUM_THREADS_IN_BLOCK);
    assert(N % num_threads_in_block == 0);

    int32_t num_rows_local = N / num_blocks_in_grid;

    int64_t local_part_output = 0;

    auto block = cg::this_thread_block();

    for (int32_t row_idx = block_idx * num_rows_local;
         row_idx < (block_idx + 1) * num_rows_local; ++row_idx) {
        // This might be inefficient since we sync in each iteration
        // Perhaps we could store the intermediate results in shared memory
        // and do the reduction at the end for each row at same time
        // TODO: Perhaps let each warp do the reduction for one row
        int64_t A_i_sum = sum_over_2nd_dim(A, N, row_idx);
        int64_t B_i_sum = sum_over_2nd_dim(B, N, row_idx);
        int64_t C_i_sum = sum_over_2nd_dim(C, N, row_idx);

        if (thread_idx == 0) {
            local_part_output += A_i_sum * B_i_sum * C_i_sum;
        }
        block.sync();
    }

    if (thread_idx == 0) {
        static_assert(sizeof(int64_t) == sizeof(unsigned long long int));
        // atomicAdd only supports unsigned long long int
        // signed and unsigned have the same bit representation
        atomicAdd(reinterpret_cast<unsigned long long int*>(g_output),
                  static_cast<unsigned long long int>(local_part_output));
    }
}

int64_t pt_naive(const int32_t N, const int64_t seed) {
    assert(N % NUM_THREADS_IN_BLOCK == 0);

    DataGenerator<3> data_gen(N, seed);
    int64_t* raw_ptr_A = data_gen.get_raw_ptr<0>();
    int64_t* raw_ptr_B = data_gen.get_raw_ptr<1>();
    int64_t* raw_ptr_C = data_gen.get_raw_ptr<2>();

    // Warm up
    thrust::device_vector<int64_t> d_output(1, 0);
    int64_t* raw_ptr_output = thrust::raw_pointer_cast(d_output.data());

    auto timed_func = [&]() -> void {
        naive_star_kernel<<<NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK>>>(
            raw_ptr_A, raw_ptr_B, raw_ptr_C, N, raw_ptr_output);
    };
    auto reset_func = [&]() -> void { d_output[0] = 0; };
    GpuTimer::RepeatTiming("GPU Naive", 10, timed_func, reset_func);

    thrust::host_vector<int64_t> h_output = d_output;

    return h_output[0];
}

int64_t pt_torch(const int32_t N, const int64_t seed) {

    DataGenerator<3> data_gen(N, seed);
    int64_t* raw_ptr_A = data_gen.get_raw_ptr<0>();
    int64_t* raw_ptr_B = data_gen.get_raw_ptr<1>();
    int64_t* raw_ptr_C = data_gen.get_raw_ptr<2>();

    torch::NoGradGuard no_grad;
    torch::Device device = torch::kCUDA;
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);

    torch::Tensor A = torch::from_blob(raw_ptr_A, {N, N}, options);
    torch::Tensor B = torch::from_blob(raw_ptr_B, {N, N}, options);
    torch::Tensor C = torch::from_blob(raw_ptr_C, {N, N}, options);

    int64_t final_result = 0;
    auto compute_func = [&]() -> void {
        // Warm up
        torch::Tensor A_row_sums = A.sum(1);
        torch::Tensor B_row_sums = B.sum(1);
        torch::Tensor C_row_sums = C.sum(1);

        torch::Tensor result = A_row_sums * B_row_sums * C_row_sums;

        final_result = result.sum().item<int64_t>();
    };
    GpuTimer::RepeatTiming("GPU Torch", 10, compute_func,
                           GpuTimer::EmptyResetFunc);

    return final_result;
}

} // namespace pt::gpu
