#ifndef TATOOINE_TENSOR_OPERATIONS_BINARY_OPERATION_H
#define TATOOINE_TENSOR_OPERATIONS_BINARY_OPERATION_H
//==============================================================================
#include <tatooine/tensor_concepts.h>
#include <tatooine/tensor_operations/same_dimensions.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename F, static_tensor Lhs, static_tensor Rhs, std::size_t... Seq>
requires(same_dimensions<Lhs, Rhs>()) constexpr auto binary_operation(
    F&& f, Lhs const& lhs, Rhs const& rhs,
    std::index_sequence<Seq...> /*seq*/) {
  using TOut          = std::invoke_result_t<F, typename Lhs::value_type,
                                    typename Rhs::value_type>;
  auto constexpr rank = Lhs::rank();
  auto t_out          = [&] {
    if constexpr (rank == 1) {
      return vec<TOut, Lhs::dimension(Seq)...>{lhs};
    } else if constexpr (rank == 2) {
      return mat<TOut, Lhs::dimension(Seq)...>{lhs};
    } else {
      return tensor<TOut, Lhs::dimension(Seq)...>{lhs};
    };
  }();
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
//------------------------------------------------------------------------------
template <typename F, static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>()) constexpr auto binary_operation(
    F&& f, Lhs const& lhs, Rhs const& rhs) {
  return binary_operation(std::forward<F>(f), lhs, rhs,
                          std::make_index_sequence<Lhs::rank()>{});
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
