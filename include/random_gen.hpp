#pragma once

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/random.h>
// A functor to generate random numbers.
// The operator() is called for each element index 'i'.
struct RandomGenerator {
    int64_t seed;

    // Constructor captures the seed
    RandomGenerator(int64_t s) : seed(s) {}

    // The __host__ __device__ specifiers allow this functor
    // to be created on the host and run on the device.
    __host__ __device__ int64_t operator()(int i) {
        // Create a new engine for each thread, but seed it uniquely
        // based on the global seed and the thread's index.
        thrust::default_random_engine rng(seed + i);

        // Define the distribution
        thrust::uniform_int_distribution<int64_t> dist(-10, 10);

        // Discard the first value to increase randomness, as the first
        // value from a simple LCG can sometimes be weak.
        rng.discard(1);

        // Return a random number from the distribution
        return dist(rng);
    }
};
