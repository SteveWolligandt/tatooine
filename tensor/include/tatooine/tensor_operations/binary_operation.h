#ifndef TATOOINE_TENSOR_OPERATIONS_BINARY_OPERATION_H
#define TATOOINE_TENSOR_OPERATIONS_BINARY_OPERATION_H
//==============================================================================
#include <tatooine/tensor_concepts.h>
#include <tatooine/tensor_operations/same_dimensions.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename F, static_tensor Lhs, static_tensor Rhs, std::size_t... Seq>
requires(same_dimensions<Lhs, Rhs>())
constexpr auto binary_operation(
    F&& f, Lhs const& lhs, Rhs const& rhs,
    std::index_sequence<Seq...> /*seq*/) {
  using TOut =
      std::invoke_result_t<F, tatooine::value_type<Lhs>, tatooine::value_type<Rhs>>;
  auto constexpr rank = tensor_rank<Lhs>;
  auto t_out          = [&] {
    if constexpr (rank == 1) {
      return vec<TOut, Lhs::dimension(Seq)...>{};
    } else if constexpr (rank == 2) {
      return mat<TOut, Lhs::dimension(Seq)...>{};
    } else {
      return tensor<TOut, Lhs::dimension(Seq)...>{};
    };
  }();
  t_out.for_indices(
      [&](auto const... is) { t_out(is...) = f(lhs(is...), rhs(is...)); });
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
