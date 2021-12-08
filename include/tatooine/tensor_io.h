#ifndef TATOOINE_TENSOR_IO_H
#define TATOOINE_TENSOR_IO_H
//==============================================================================
#include <tatooine/tensor.h>
#include <ostream>
//==============================================================================
namespace tatooine {
//==============================================================================
/// printing vector
template <typename Tensor, typename T, size_t N>
auto operator<<(std::ostream& out, const base_tensor<Tensor, T, N>& v)
    -> auto& {
  out << "[ ";
  out << std::scientific;
  for (size_t i = 0; i < N; ++i) {
    if constexpr (!is_complex<T>) {}
    out << v(i) << ' ';
  }
  out << "]";
  out << std::defaultfloat;
  return out;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t M, size_t N>
auto operator<<(std::ostream& out, const base_tensor<Tensor, T, M, N>& m)
    -> auto& {
  out << std::scientific;
  for (size_t j = 0; j < M; ++j) {
    out << "[ ";
    for (size_t i = 0; i < N; ++i) {
      if constexpr (!is_complex<T>) {
        if (m(j, i) >= 0) { out << ' '; }
      }
      out << m(j, i) << ' ';
    }
    out << "]\n";
  }
  out << std::defaultfloat;
  return out;
}
//==============================================================================
/// printing dynamic tensors
template <typename DynamicTensor>
requires is_dynamic_tensor<DynamicTensor>
auto operator<<(std::ostream& out, DynamicTensor const& v) -> auto& {
  if (v.num_dimensions() == 1) {
    out << "[ ";
    out << std::scientific;
    for (size_t i = 0; i < v.size(0); ++i) {
      if constexpr (!is_complex<typename DynamicTensor::value_type>) {
      }
      out << v(i) << ' ';
    }
    out << "]";
    out << std::defaultfloat;
  }
  return out;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
