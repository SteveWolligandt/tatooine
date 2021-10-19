#ifndef TATOOINE_TENSOR_OPERATIONS_BINARY_OPERATION_H
#define TATOOINE_TENSOR_OPERATIONS_BINARY_OPERATION_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename F, typename Tensor0, typename T0, typename Tensor1,
          typename T1, size_t N>
constexpr auto binary_operation(F&& f, base_tensor<Tensor0, T0, N> const& lhs,
                                base_tensor<Tensor1, T1, N> const& rhs) {
  using TOut         = typename std::result_of<decltype(f)(T0, T1)>::type;
  vec<TOut, N> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
//------------------------------------------------------------------------------
template <typename F, typename Tensor0, typename T0, typename Tensor1,
          typename T1, size_t M, size_t N>
constexpr auto binary_operation(F&&                                   f,
                                base_tensor<Tensor0, T0, M, N> const& lhs,
                                base_tensor<Tensor1, T1, M, N> const& rhs) {
  using TOut = typename std::result_of<decltype(f)(T0, T1)>::type;
  auto t_out = mat<TOut, M, N>{lhs};
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
//------------------------------------------------------------------------------
template <typename F, typename Tensor0, typename T0, typename Tensor1,
          typename T1, size_t... Dims>
constexpr auto binary_operation(F&&                                      f,
                                base_tensor<Tensor0, T0, Dims...> const& lhs,
                                base_tensor<Tensor1, T1, Dims...> const& rhs) {
  using TOut = typename std::result_of<decltype(f)(T0, T1)>::type;
  tensor<TOut, Dims...> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
