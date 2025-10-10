#include "tb.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cub/cub.cuh>
#include <cuda/pipeline>
#include <cuda/ptx>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>
#include <torch/torch.h>

#include "random_gen.hpp"
#include "timer.hpp"

namespace cg = cooperative_groups;

namespace tb::gpu {

template <std::size_t NumMatrices>
using DataGenerator = GpuDataGenerator<NumMatrices, data_t>;

// \sum A_ij * B_ik * C_il
// where A, B, C are (N, N) matrices
// and the output is stored in

constinit const int32_t NUM_THREADS_IN_BLOCK = 1024;
constinit const int32_t WARP_SIZE = 32;

__device__ data_t sum_over_2nd_dim(const data_t* Mat, const int32_t N,
                                   int32_t row_idx) {

    int32_t thread_idx = threadIdx.x;
    int32_t num_threads_in_block = blockDim.x;
    data_t Mat_i_sum_over_part_j = 0;

    // Use loop stride mode for coalesced memory access
    for (int32_t col_idx = thread_idx; col_idx < N;
         col_idx += num_threads_in_block) {
        data_t Mat_ij = Mat[row_idx * N + col_idx];
        Mat_i_sum_over_part_j += Mat_ij;
    }
    cg::this_thread_block().sync();

    using BlockReduce = cub::BlockReduce<data_t, NUM_THREADS_IN_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    data_t Mat_i_sum = BlockReduce(temp_storage).Sum(Mat_i_sum_over_part_j);

    return Mat_i_sum;
}

// Assume N is multiple of NUM_THREADS_IN_BLOCK
__global__ void naive_star_kernel(const data_t* A, const data_t* B,
                                  const data_t* C, const int32_t N,
                                  data_t* g_output) {

    int32_t num_blocks_in_grid = gridDim.x;
    int32_t block_idx = blockIdx.x;
    [[maybe_unused]] int32_t num_threads_in_block = blockDim.x;
    int32_t thread_idx = threadIdx.x;

    assert(num_blocks_in_grid == num_threads_in_block);
    assert(num_threads_in_block == NUM_THREADS_IN_BLOCK);
    assert(N % num_threads_in_block == 0);

    int32_t num_rows_local = N / num_blocks_in_grid;

    int32_t local_part_output = 0;

    auto block = cg::this_thread_block();

    for (int32_t row_idx = block_idx * num_rows_local;
         row_idx < (block_idx + 1) * num_rows_local; ++row_idx) {
        // This might be inefficient since we sync in each iteration
        // Perhaps we could store the intermediate results in shared memory
        // and do the reduction at the end for each row at same time
        // TODO: Perhaps let each warp do the reduction for one row
        data_t A_i_sum = sum_over_2nd_dim(A, N, row_idx);
        data_t B_i_sum = sum_over_2nd_dim(B, N, row_idx);
        data_t C_i_sum = sum_over_2nd_dim(C, N, row_idx);

        if (thread_idx == 0) {
            local_part_output += A_i_sum * B_i_sum * C_i_sum;
        }
        block.sync();
    }

    if (thread_idx == 0) {
        static_assert(sizeof(data_t) == sizeof(int32_t));
        // atomicAdd supports int32_t
        atomicAdd(g_output, local_part_output);
    }
}

data_t tb_naive(const int32_t N, const data_t seed) {
    assert(N % NUM_THREADS_IN_BLOCK == 0);

    DataGenerator<3> data_gen(N, seed);
    data_t* raw_ptr_A = data_gen.get_raw_ptr<0>();
    data_t* raw_ptr_B = data_gen.get_raw_ptr<1>();
    data_t* raw_ptr_C = data_gen.get_raw_ptr<2>();

    thrust::device_vector<data_t> d_output(1, 0);
    data_t* raw_ptr_output = thrust::raw_pointer_cast(d_output.data());

    auto timed_func = [&]() -> void {
        naive_star_kernel<<<NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK>>>(
            raw_ptr_A, raw_ptr_B, raw_ptr_C, N, raw_ptr_output);
    };
    auto reset_func = [&]() -> void { d_output[0] = 0; };
    Timer::RepeatTiming<Timer::GPU>("TB GPU Naive", timed_func, reset_func);

    thrust::host_vector<data_t> h_output = d_output;

    return h_output[0];
}

__device__ data_t sum_over_2nd_dim_warp(const data_t* Mat, const int32_t N,
                                        int32_t row_idx) {

    int32_t thread_idx = threadIdx.x;
    int32_t warp_idx = thread_idx / WARP_SIZE;
    int32_t lane_idx = thread_idx % WARP_SIZE;

    data_t Mat_i_sum_over_part_j = 0;

    cg::thread_block my_block = cg::this_thread_block();
    auto my_warp = cg::tiled_partition<WARP_SIZE>(my_block);
    // Use loop stride mode for coalesced memory access
    for (int32_t col_idx = lane_idx; col_idx < N; col_idx += WARP_SIZE) {
        data_t Mat_ij = Mat[row_idx * N + col_idx];
        Mat_i_sum_over_part_j += Mat_ij;
    }
    my_warp.sync();

    using WarpReduce = cub::WarpReduce<data_t>;

    __shared__ typename WarpReduce::TempStorage
        temp_storage[NUM_THREADS_IN_BLOCK / WARP_SIZE];

    data_t Mat_i_sum =
        WarpReduce(temp_storage[warp_idx]).Sum(Mat_i_sum_over_part_j);
    // now the first lane in each warp has the final result
    return Mat_i_sum;
}

__global__ void pwarp_star_kernel(const data_t* A, const data_t* B,
                                  const data_t* C, const int32_t N,
                                  data_t* g_output) {
    // Each warp handles one row
    int32_t num_warps_in_block = blockDim.x / WARP_SIZE;
    int32_t num_blocks_in_grid = gridDim.x;
    int32_t block_idx = blockIdx.x;
    int32_t thread_idx = threadIdx.x;
    int32_t warp_idx = thread_idx / WARP_SIZE;
    int32_t lane_idx = thread_idx % WARP_SIZE;
    int32_t global_warp_idx = block_idx * num_warps_in_block + warp_idx;

    assert(N % (num_blocks_in_grid * num_warps_in_block) == 0);

    int32_t num_rows_local = N / (num_blocks_in_grid * num_warps_in_block);

    data_t local_part_output = 0;

    // obtain default "current thread block" group
    cg::thread_block my_block = cg::this_thread_block();
    auto my_warp = cg::tiled_partition<WARP_SIZE>(my_block);

    for (int32_t row_idx = global_warp_idx * num_rows_local;
         row_idx < (global_warp_idx + 1) * num_rows_local; ++row_idx) {
        // This might be inefficient since we sync in each iteration
        // Perhaps we could store the intermediate results in shared memory
        // and do the reduction at the end for each row at same time
        data_t A_i_sum = sum_over_2nd_dim_warp(A, N, row_idx);
        data_t B_i_sum = sum_over_2nd_dim_warp(B, N, row_idx);
        data_t C_i_sum = sum_over_2nd_dim_warp(C, N, row_idx);

        if (lane_idx == 0) {
            local_part_output += A_i_sum * B_i_sum * C_i_sum;
        }

        my_warp.sync();
    }

    if (lane_idx == 0) {
        atomicAdd(g_output, local_part_output);
    }
}

data_t tb_pwarp(const int32_t N, const data_t seed) {
    assert(N % NUM_THREADS_IN_BLOCK == 0);
    assert(WARP_SIZE == warpSize);

    DataGenerator<3> data_gen(N, seed);
    data_t* raw_ptr_A = data_gen.get_raw_ptr<0>();
    data_t* raw_ptr_B = data_gen.get_raw_ptr<1>();
    data_t* raw_ptr_C = data_gen.get_raw_ptr<2>();

    thrust::device_vector<data_t> d_output(1, 0);
    data_t* raw_ptr_output = thrust::raw_pointer_cast(d_output.data());

    auto timed_func = [&]() -> void {
        pwarp_star_kernel<<<NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK>>>(
            raw_ptr_A, raw_ptr_B, raw_ptr_C, N, raw_ptr_output);
    };
    auto reset_func = [&]() -> void { d_output[0] = 0; };
    Timer::RepeatTiming<Timer::GPU>("TB GPU PWarp", timed_func, reset_func);
    thrust::host_vector<data_t> h_output = d_output;
    return h_output[0];
}

data_t tb_torch(const int32_t N, const data_t seed) {

    DataGenerator<3> data_gen(N, seed);
    data_t* raw_ptr_A = data_gen.get_raw_ptr<0>();
    data_t* raw_ptr_B = data_gen.get_raw_ptr<1>();
    data_t* raw_ptr_C = data_gen.get_raw_ptr<2>();

    torch::NoGradGuard no_grad;
    torch::Device device = torch::kCUDA;
    auto torch_type = c10::CppTypeToScalarType<data_t>::value;
    auto options = torch::TensorOptions().dtype(torch_type).device(device);

    torch::Tensor A = torch::from_blob(raw_ptr_A, {N, N}, options);
    torch::Tensor B = torch::from_blob(raw_ptr_B, {N, N}, options);
    torch::Tensor C = torch::from_blob(raw_ptr_C, {N, N}, options);

    data_t final_result = 0;
    auto compute_func = [&]() -> void {
        // Warm up
        torch::Tensor A_row_sums = A.sum(1);
        torch::Tensor B_row_sums = B.sum(1);
        torch::Tensor C_row_sums = C.sum(1);

        torch::Tensor result = A_row_sums * B_row_sums * C_row_sums;

        final_result = result.sum().item<data_t>();
    };
    Timer::RepeatTiming<Timer::GPU>("TB GPU Torch", compute_func,
                                    Timer::EmptyResetFunc);

    return final_result;
}

} // namespace tb::gpu
