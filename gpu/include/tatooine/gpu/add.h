#ifndef TATOOINE_GPU_ADD_H
#define TATOOINE_GPU_ADD_H

#include <vector>
//==============================================================================
namespace tatooine {
namespace gpu {
//==============================================================================

std::vector<float> add(const std::vector<float>&, const std::vector<float>&,
                       const int block_size = 256);

//==============================================================================
}  // namespace gpu
}  // namespace tatooine
//==============================================================================

#endif
