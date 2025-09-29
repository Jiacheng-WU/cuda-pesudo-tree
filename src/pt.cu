#include "pt.hpp"

#include <cstdint>
#include <cuda_runtime.h>

#include <cooperative_groups.h>

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cub/cub.cuh>

#include <cuda/std/span>

#include <cutlass/cutlass.h>

namespace cg = cooperative_groups;

namespace pt::cuda {

// \sum A_ij * B_ik * C_il
// where A, B, C are (N, N) matrices
// and the output is stored in

constinit const int NUM_THREADS_IN_BLOCK = 1024;

__device__ int64_t sum_over_2nd_dim(const int* Mat, const int N,
                                    const int part_size, int i) {

    int pos_x = i;
    int thread_index_i = threadIdx.x;
    int Mat_i_sum_over_part_j = 0;

    // TODO: need to consider the Memory coalescing
    // change to loop stride mode perhaps at least for warp level
    for (int part_y = 0; part_y < part_size; ++part_y) {
        int pos_y = thread_index_i * part_size + part_y;
        int Mat_ij = Mat[pos_x * N + pos_y];
        Mat_i_sum_over_part_j += Mat_ij;
    }
    cg::this_thread_block().sync();

    using BlockReduce = cub::BlockReduce<int, NUM_THREADS_IN_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int Mat_i_sum = BlockReduce(temp_storage).Sum(Mat_i_sum_over_part_j);

    return Mat_i_sum;
}

// Assume N is multiple of NUM_THREADS_IN_BLOCK
__global__ void naive_star_kernel(const int* A, const int* B, const int* C,
                                  const int N, uint64_t* g_output) {
    int block_index_i = blockIdx.x;
    int thread_index_i = threadIdx.x;
    int block_size = blockDim.x;
    [[maybe_unused]] int grid_size = gridDim.x;
    assert(grid_size == block_size);
    assert(block_size == NUM_THREADS_IN_BLOCK);
    assert(N % block_size == 0);

    int part_size = N / block_size;

    uint64_t local_part_output = 0;

    auto block = cg::this_thread_block();

    for (int part_x = 0; part_x < part_size; ++part_x) {
        // load A_ij
        int pos_x = block_index_i * part_size + part_x;

        // This might be inefficient since we sync in each iteration
        // Perhaps we could store the intermediate results in shared memory
        // and do the reduction at the end for each row at same time
        // TODO: Perhaps let each warp do the reduction for one row
        int64_t A_i_sum = sum_over_2nd_dim(A, N, part_size, pos_x);
        int64_t B_i_sum = sum_over_2nd_dim(B, N, part_size, pos_x);
        int64_t C_i_sum = sum_over_2nd_dim(C, N, part_size, pos_x);

        if (thread_index_i == 0) {
            local_part_output += A_i_sum * B_i_sum * C_i_sum;
        }
        block.sync();
    }

    if (thread_index_i == 0) {
        static_assert(sizeof(uint64_t) == sizeof(unsigned long long int));
        atomicAdd(reinterpret_cast<unsigned long long int*>(g_output),
                  static_cast<unsigned long long int>(local_part_output));
    }
}

// A functor to generate random numbers.
// The operator() is called for each element index 'i'.
struct RandomGenerator {
    unsigned long long seed;

    // Constructor captures the seed
    RandomGenerator(unsigned long long s) : seed(s) {}

    // The __host__ __device__ specifiers allow this functor
    // to be created on the host and run on the device.
    __host__ __device__ int operator()(unsigned int i) {
        // Create a new engine for each thread, but seed it uniquely
        // based on the global seed and the thread's index.
        thrust::default_random_engine rng(seed + i);

        // Define the distribution
        thrust::uniform_int_distribution<int> dist(0, 10);

        // Discard the first value to increase randomness, as the first
        // value from a simple LCG can sometimes be weak.
        rng.discard(1);

        // Return a random number from the distribution
        return dist(rng);
    }
};

int64_t cuda_pt_naive(const int N, const unsigned long long seed) {
    assert(N % NUM_THREADS_IN_BLOCK == 0);
    const int num_elements = N * N;

    // 1. Create a Thrust device_vector.
    thrust::device_vector<int> A_matrix(num_elements);
    thrust::device_vector<int> B_matrix(num_elements);
    thrust::device_vector<int> C_matrix(num_elements);

    // 2. Use thrust::tabulate to fill the vector.
    // It calls an instance of our RandomGenerator for each index from 0 to
    // num_elements-1.
    thrust::tabulate(A_matrix.begin(), A_matrix.end(), RandomGenerator(seed));
    thrust::tabulate(B_matrix.begin(), B_matrix.end(), RandomGenerator(seed));
    thrust::tabulate(C_matrix.begin(), C_matrix.end(), RandomGenerator(seed));

    int* raw_ptr_A = thrust::raw_pointer_cast(A_matrix.data());
    int* raw_ptr_B = thrust::raw_pointer_cast(B_matrix.data());
    int* raw_ptr_C = thrust::raw_pointer_cast(C_matrix.data());

    thrust::device_vector<uint64_t> d_output(1, 0);
    uint64_t* raw_ptr_output = thrust::raw_pointer_cast(d_output.data());
    cudaDeviceSynchronize();

    naive_star_kernel<<<NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK>>>(
        raw_ptr_A, raw_ptr_B, raw_ptr_C, N, raw_ptr_output);
    cudaDeviceSynchronize();
    thrust::host_vector<uint64_t> h_output = d_output;

    return h_output[0];
}

int thrust_test() {
    thrust::host_vector<int> data(100);
    thrust::sequence(data.begin(), data.end());
    thrust::host_vector<int> result(data.size());
    thrust::device_vector<int> d_data(data.begin(), data.end());
    thrust::device_vector<int> d_result(result.size());

    thrust::transform(d_data.begin(), d_data.end(), d_result.begin(),
                      [] __device__(int x) { return x * 2; });

    thrust::copy(d_result.begin(), d_result.end(), result.begin());
    return 0;
}

} // namespace pt::cuda
