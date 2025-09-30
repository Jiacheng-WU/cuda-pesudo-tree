#pragma once

#include <concepts>
#include <cstdint>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/tabulate.h>

// A functor to generate random numbers.
// The operator() is called for each element index 'i'.
template <std::integral T = int64_t>
struct RandomGenerator {
    T seed;

    // Constructor captures the seed
    RandomGenerator(T s) : seed(s) {}

    // The __host__ __device__ specifiers allow this functor
    // to be created on the host and run on the device.
    __host__ __device__ T operator()(int i) {
        // Create a new engine for each thread, but seed it uniquely
        // based on the global seed and the thread's index.
        thrust::default_random_engine rng(seed + i);

        // Define the distribution
        thrust::uniform_int_distribution<T> dist(-10, 10);

        // Discard the first value to increase randomness, as the first
        // value from a simple LCG can sometimes be weak.
        rng.discard(1);

        // Return a random number from the distribution
        return dist(rng);
    }
};

enum class DeviceType { CPU, GPU };

template <std::size_t NumMatrices, typename VectorType>
struct DataGenerator {
  public:
    // Add Var NumMatrices
    const std::size_t num_matrices = NumMatrices;
    using vector_type = VectorType;
    using value_type = typename vector_type::value_type;
    static_assert(std::is_integral_v<value_type>,
                  "VectorType must have integral value_type");

    DataGenerator(const int32_t N, const value_type seed)
        : N(N), num_elements(N * N), seed(seed) {
        for (std::size_t i = 0; i < NumMatrices; ++i) {
            matrices[i].resize(num_elements);
            thrust::tabulate(matrices[i].begin(), matrices[i].end(),
                             RandomGenerator(seed));
        }
    }

    template <std::size_t Idx>
    value_type* get_raw_ptr() {
        static_assert(Idx < NumMatrices, "Index out of bounds");
        return thrust::raw_pointer_cast(matrices[Idx].data());
    }

    template <std::size_t Idx>
    vector_type& get_matrix() {
        static_assert(Idx < NumMatrices, "Index out of bounds");
        return matrices[Idx];
    }

  private:
    const int32_t N;
    const int64_t num_elements;
    const value_type seed;
    std::array<vector_type, NumMatrices> matrices;
};

template <std::size_t NumMatrices, typename ValueType = int64_t>
using CpuDataGenerator =
    DataGenerator<NumMatrices, thrust::host_vector<ValueType>>;
template <std::size_t NumMatrices, typename ValueType = int64_t>
using GpuDataGenerator =
    DataGenerator<NumMatrices, thrust::device_vector<ValueType>>;
