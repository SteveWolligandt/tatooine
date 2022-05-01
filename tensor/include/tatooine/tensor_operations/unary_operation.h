#ifndef TATOOINE_TENSOR_OPERATIONS_UNARY_OPERATION_H
#define TATOOINE_TENSOR_OPERATIONS_UNARY_OPERATION_H
//==============================================================================
#include <tatooine/tensor_concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <dynamic_tensor Tensor>
constexpr auto unary_operation(invocable<tensor_value_type<Tensor>> auto&& op,
                               Tensor const& t) {
  using TOut = std::invoke_result_t<decltype(op), tensor_value_type<Tensor>>;
  auto t_out = tensor<TOut>::zeros(t.dimensions());
  for (std::size_t i = 0; i < t.data().size(); ++i) {
    t_out.data()[i] = op(t.data()[i]);
  }
  return t_out;
}
//------------------------------------------------------------------------------
template <typename Op, static_tensor Tensor, std::size_t... Seq>
constexpr auto unary_operation(Op&& op, Tensor const& t,
                               std::index_sequence<Seq...> /*seq*/) {
  using TOut          = std::invoke_result_t<Op, typename Tensor::value_type>;
  auto constexpr rank = Tensor::rank();
  auto t_out          = [&] {
    if constexpr (rank == 1) {
      return vec<TOut, Tensor::dimension(Seq)...>{};
    } else if constexpr (rank == 2) {
      return mat<TOut, Tensor::dimension(Seq)...>{};
    } else {
      return tensor<TOut, Tensor::dimension(Seq)...>{};
    };
  }();
  t_out.for_indices([&](auto const... is) {
    t_out(is...) = op(t(is...));
  });
  return t_out;
}
//------------------------------------------------------------------------------
template <typename Op, static_tensor Tensor>
constexpr auto unary_operation(Op&& op, Tensor const& t) {
  return unary_operation(std::forward<Op>(op), t,
                         std::make_index_sequence<Tensor::rank()>{});
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
