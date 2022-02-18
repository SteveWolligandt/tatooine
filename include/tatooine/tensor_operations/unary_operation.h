#ifndef TATOOINE_TENSOR_OPERATIONS_UNARY_OPERATION_H
#define TATOOINE_TENSOR_OPERATIONS_UNARY_OPERATION_H
//==============================================================================
#include <tatooine/tensor_concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename F, static_tensor Tensor, std::size_t... Seq>
constexpr auto unary_operation(F&& f, Tensor const& t,
                               std::index_sequence<Seq...> /*seq*/) {
  using TOut          = std::invoke_result_t<F, typename Tensor::value_type>;
  auto constexpr rank = Tensor::rank();
  auto t_out          = [&] {
    if constexpr (rank == 1) {
      return vec<TOut, Tensor::dimension(Seq)...>{t};
    } else if constexpr (rank == 2) {
      return mat<TOut, Tensor::dimension(Seq)...>{t};
    } else {
      return tensor<TOut, Tensor::dimension(Seq)...>{t};
    };
  }();
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
//------------------------------------------------------------------------------
template <typename F, static_tensor Tensor>
constexpr auto unary_operation(F&& f, Tensor const& t) {
  return unary_operation(std::forward<F>(f), t,
                         std::make_index_sequence<Tensor::rank()>{});
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
