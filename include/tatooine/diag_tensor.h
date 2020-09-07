#ifndef TATOOINE_DIAG_TENSOR_H
#define TATOOINE_DIAG_TENSOR_H
//==============================================================================
#include <tatooine/base_tensor.h>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, size_t M, size_t N>
struct diag_tensor : base_tensor<diag_tensor<Tensor, M, N>,
                                 typename std::decay_t<Tensor>::value_type, M, N> {
  //============================================================================
  using tensor_t = Tensor;
  using this_t   = diag_tensor<Tensor, M, N>;
  using parent_t =
      base_tensor<this_t, typename std::decay_t<tensor_t>::value_type, M, N>;
  using typename parent_t::value_type;
  //============================================================================
 private:
  tensor_t m_internal_tensor;

  //============================================================================
 public:
  // TODO use concept
  template <typename _Tensor>
  constexpr explicit diag_tensor(_Tensor&& v)
      : m_internal_tensor{std::forward<_Tensor>(v)} {}
  //----------------------------------------------------------------------------
  constexpr auto operator()(size_t i, size_t j) const { return at(i, j); }
  //----------------------------------------------------------------------------
  constexpr auto at(size_t i, size_t j) const -> value_type {
    assert(i < M);
    assert(j < N);
    if (i == j) { return m_internal_tensor(i); }
    return 0;
  }
  //----------------------------------------------------------------------------
  template <typename _Tensor = tensor_t,
            std::enable_if_t<std::is_const_v<std::remove_reference_t<_Tensor>>,
                             bool> = true>
  auto internal_tensor() -> auto& {
    return m_internal_tensor;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};
//==============================================================================
// deduction guides
//==============================================================================
template <typename Tensor>
diag_tensor(Tensor const& t)
    -> diag_tensor<Tensor const&,
                   Tensor::dimension(0),
                   Tensor::dimension(0)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor>
diag_tensor(Tensor& t)
    -> diag_tensor<Tensor&,
                   Tensor::dimension(0),
                   Tensor::dimension(0)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor>
diag_tensor(Tensor&& t)
    -> diag_tensor<std::decay_t<Tensor>,
                   Tensor::dimension(0),
                   Tensor::dimension(0)>;
//==============================================================================
template <real_or_complex_number Real, size_t N>
struct vec;
//==============================================================================
// factory functions
//==============================================================================
template <typename Tensor, typename Real, size_t N>
constexpr auto diag(base_tensor<Tensor, Real, N> const& t) {
  return diag_tensor<Tensor const&, N, N>{t.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t N>
constexpr auto diag(base_tensor<Tensor, Real, N>& t) {
  return diag_tensor<Tensor&, N, N>{t.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t N>
constexpr auto diag(base_tensor<Tensor, Real, N>&& t) {
  return diag_tensor<vec<Real, N>, N, N>{vec<Real, N>{std::move(t)}};
}
//------------------------------------------------------------------------------
template <size_t M, size_t N, typename Tensor, typename Real, size_t VecN>
constexpr auto diag_rect(const base_tensor<Tensor, Real, VecN>& t) {
  return diag_tensor<Tensor const&, M, N>{t.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t M, size_t N, typename Tensor, typename Real, size_t VecN>
constexpr auto diag_rect(base_tensor<Tensor, Real, VecN>& t) {
  return diag_tensor<Tensor&, M, N>{t.as_derived};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t M, size_t N, typename Tensor, typename Real, size_t VecN>
constexpr auto diag_rect(base_tensor<Tensor, Real, VecN>&& t) {
  return diag_tensor<vec<Real, VecN>, M, N>{vec{std::move(t)}};
}
//==============================================================================
// free functions
//==============================================================================
template <typename Tensor, size_t N>
auto inv(diag_tensor<Tensor, N, N> const& A) {
  using value_type = typename std::decay_t<Tensor>::value_type;
  return diag_tensor<vec<value_type, N>, N, N>{value_type(1) / A.internal_tensor()};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
