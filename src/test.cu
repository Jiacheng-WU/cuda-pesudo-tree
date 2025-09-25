#include "test.hpp"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace test::cuda::detail {

__global__ void kernel(int* data, int* result) {
    // Kernel code here
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    result[idx] = data[idx] * 2; // Example operation
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

int cuda_test(std::span<int> data, std::span<int> result) {
    const int arraySize = static_cast<int>(result.size());

    int *d_data, *d_result;
    cudaMalloc((void**)&d_data, arraySize * sizeof(int));
    cudaMalloc((void**)&d_result, arraySize * sizeof(int));

    cudaMemcpy(d_data, data.data(), arraySize * sizeof(int),
               cudaMemcpyHostToDevice);

    kernel<<<1, arraySize>>>(d_data, d_result);

    cudaMemcpy(result.data(), d_result, arraySize * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}

} // namespace test::cuda::detail
