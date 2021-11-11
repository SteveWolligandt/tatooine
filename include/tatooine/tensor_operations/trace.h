#ifndef TATOOINE_TENSOR_OPERATIONS_TRACE_H
#define TATOOINE_TENSOR_OPERATIONS_TRACE_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename T, std::size_t N>
constexpr auto trace(base_tensor<Tensor, T, N, N> const& A) {
  auto tr = T{};
  for (std::size_t i = 0; i < N; ++i) {
    tr += A(i, i);
  }
  return tr;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
