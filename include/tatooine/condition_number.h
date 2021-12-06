#ifndef TATOOINE_TENSOR_OPERATIONS_CONDITION_NUMBER_H
#define TATOOINE_TENSOR_OPERATIONS_CONDITION_NUMBER_H
//==============================================================================
namespace tatooine {
//==============================================================================
/// compute condition number
template <typename T, size_t N, integral  P = int>
auto condition_number(tensor<T, N, N> const& A, P const p = 2) {
  if (p == 1) {
    return 1 / lapack::gecon(tensor{A});
  } else if (p == 2) {
    auto const s = singular_values(A);
    return s(0) / s(N - 1);
  } else {
    throw std::runtime_error{"p = " + std::to_string(p) +
                             " is no valid base. p must be either 1 or 2."};
  }
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N, typename PReal>
auto condition_number(base_tensor<Tensor, T, N, N> const& A, PReal p) {
  return condition_number(tensor{A}, p);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
