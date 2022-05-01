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
auto operator<<(std::ostream& out, dynamic_tensor auto const& t) -> auto& {
  if (t.rank() == 1) {
    out << "[ ";
    out << std::scientific;
    for (size_t i = 0; i < t.dimension(0); ++i) {
      if constexpr (!is_complex<tensor_value_type<decltype(t)>>) {
        out << t(i) << ' ';
      }
    }
    out << "]";
    out << std::defaultfloat;
  } else if (t.rank() == 2) {
    out << std::scientific;
    for (size_t r = 0; r < t.dimension(0); ++r) {
      out << "[ ";
      for (size_t c = 0; c < t.dimension(1); ++c) {
        if constexpr (!is_complex<tensor_value_type<decltype(t)>>) {
          if (t(r, c) >= 0) {
            out << ' ';
          }
        }
        out << t(r, c) << ' ';
      }
      out << "]\n";
    }
    out << std::defaultfloat;
  }
  return out;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
