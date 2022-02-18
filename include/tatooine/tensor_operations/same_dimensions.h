#ifndef TATOOINE_TENSOR_OPERATIONS_SAME_DIMENSIONS_H
#define TATOOINE_TENSOR_OPERATIONS_SAME_DIMENSIONS_H
//==============================================================================
#include <tatooine/tensor_concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <static_tensor A, static_tensor B>
auto constexpr same_dimensions() {
  if constexpr (A::rank() != B::rank()) {
    return false;
  } else {
    for (std::size_t i = 0; i < A::rank(); ++i) {
      if (A::dimension(i) != B::dimension(i)) {
        return false;
      }
    }
    return true;
  }
}
//------------------------------------------------------------------------------
auto constexpr same_dimensions(static_tensor auto const& A,
                               static_tensor auto const& B) {
  if constexpr (A.rank() != B.rank()) {
    return false;
  } else {
    for (std::size_t i = 0; i < A.rank(); ++i) {
      if (A.dimension(i) != B.dimension(i)) {
        return false;
      }
    }
    return true;
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
