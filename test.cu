#include "test.h"

__global__ void kernel(int* data, int* result) {
    // Kernel code here
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    result[idx] = data[idx] * 2; // Example operation
}

int test() {
    const int arraySize = 5;
    int data[arraySize] = {1, 2, 3, 4, 5};
    int result[arraySize] = {0};

    int *d_data, *d_result;
    cudaMalloc((void**)&d_data, arraySize * sizeof(int));
    cudaMalloc((void**)&d_result, arraySize * sizeof(int));

    cudaMemcpy(d_data, data, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, arraySize>>>(d_data, d_result);

    cudaMemcpy(result, d_result, arraySize * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_result);

    for (int i = 0; i < arraySize; i++) {
        printf("result[%d] = %d\n", i, result[i]);
    }

    return 0;
}
