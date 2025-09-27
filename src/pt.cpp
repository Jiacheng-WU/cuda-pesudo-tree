#include "pt.hpp"

#include <format>
#include <iostream>
#include <vector>

namespace pt {

int cpp_pt_naive_wrapper() {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result(data.size(), 0);
    int err = cuda::detail::cuda_pt_naive(data, result);
    for (const auto& val : result) {
        std::cout << std::format("{} ", val);
    }
    std::cout << std::endl;
    return err;
}

} // namespace pt
