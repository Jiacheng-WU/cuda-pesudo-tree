#include "pt.hpp"
#include "torch/torch.h"

namespace pt {

int64_t cpp_pt_torch(const int N, const unsigned long long seed) {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    return 0;
}

} // namespace pt
