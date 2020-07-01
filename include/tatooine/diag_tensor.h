#ifndef TATOOINE_DIAG_TENSOR_H
#define TATOOINE_DIAG_TENSOR_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, size_t VecN, size_t M, size_t N>
struct const_diag_tensor : base_tensor<const_diag_tensor<Tensor, VecN, M, N>,
                                       typename Tensor::real_t, M, N> {
  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  constexpr explicit const_diag_tensor(
      const base_tensor<Tensor, typename Tensor::real_t, VecN>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}
  //----------------------------------------------------------------------------
  constexpr auto operator()(size_t i, size_t j) const { return at(i, j); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(size_t i, size_t j) const -> typename Tensor::real_t {
    assert(i < M);
    assert(j < N);
    if (i == j) {
      return m_internal_tensor(i);
    } else {
      return 0;
    }
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};
//==============================================================================
template <typename Tensor, size_t VecN, size_t M, size_t N>
struct diag_tensor
    : base_tensor<diag_tensor<Tensor, VecN, M, N>, typename Tensor::real_t, M, N> {
  //============================================================================
 private:
  Tensor& m_internal_tensor;
  typename Tensor::real_t zero=0;

  //============================================================================
 public:
  constexpr explicit diag_tensor(
      const base_tensor<Tensor, typename Tensor::real_t, VecN>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  constexpr auto operator()(size_t i, size_t j) const { return at(i, j); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(size_t i, size_t j) -> auto& {
    assert(i < M);
    assert(j < N);
    zero = 0;
    if (i == j) {
      return m_internal_tensor(i);
    } else {
      return zero;
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto at(size_t i, size_t j) const -> typename Tensor::real_t {
    assert(i < M);
    assert(j < N);
    if (i == j) {
      return m_internal_tensor(i);
    } else {
      return 0;
    }
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t VecN>
constexpr auto diag(const base_tensor<Tensor, Real, VecN>& t) {
  return const_diag_tensor<Tensor, VecN, VecN, VecN>{t};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t VecN>
constexpr auto diag(base_tensor<Tensor, Real, VecN>& t) {
  return diag_tensor<Tensor, VecN, VecN, VecN>{t};
}
//------------------------------------------------------------------------------
template <size_t M, size_t N, typename Tensor, typename Real, size_t VecN>
constexpr auto diag_rect(const base_tensor<Tensor, Real, VecN>& t) {
  return const_diag_tensor<Tensor, VecN, M, N>{t};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t M, size_t N, typename Tensor, typename Real, size_t VecN>
constexpr auto diag_rect(base_tensor<Tensor, Real, VecN>& t) {
  return diag_tensor<Tensor, VecN, M, N>{t};
}
//==============================================================================
}
//==============================================================================
#endif
