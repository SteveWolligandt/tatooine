#ifndef TATOOINE_DIAG_TENSOR_H
#define TATOOINE_DIAG_TENSOR_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
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
            enable_if<std::is_const_v<std::remove_reference_t<_Tensor>>> = true>
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
template <typename Real, size_t N>
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
constexpr auto inv(diag_tensor<Tensor, N, N> const& A) -> std::optional<
    diag_tensor<vec<typename std::decay_t<Tensor>::value_type, N>, N, N>> {
  using value_type = typename std::decay_t<Tensor>::value_type;
  for (size_t i = 0; i < N; ++i) {
    if (std::abs(A.internal_tensor()(i)) < 1e-10) {
      return {};
    }
  }
  return diag_tensor<vec<value_type, N>, N, N>{value_type(1) /
                                               A.internal_tensor()};
}
//------------------------------------------------------------------------------
#include <tatooine/vec.h>
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <typename TensorA, typename TensorB, size_t N>
requires is_vec<TensorB> && (std::decay_t<TensorB>::num_dimensions() == N)
#else
template <typename TensorA, typename TensorB, size_t N,
          enable_if_vec<TensorB>                                    = true,
          enable_if<(std::decay_t<TensorB>::num_dimensions() == N)> = true>
#endif
constexpr auto solve(diag_tensor<TensorA, N, N> const& A, TensorB&& b)
    -> std::optional<
        vec<std::common_type_t<typename std::decay_t<TensorA>::value_type,
                               typename std::decay_t<TensorB>::value_type>,
            N>> {
  for (size_t i = 0; i < N; ++i) {
    if (std::abs(A.internal_tensor()(i)) < 1e-10) {
      return {};
    }
  }
  return A.internal_tensor() / b;
}
//------------------------------------------------------------------------------
template <typename TensorA, typename TensorB, typename BReal, size_t N>
constexpr auto operator*(diag_tensor<TensorA, N, N> const&     A,
                         base_tensor<TensorB, BReal, N> const& b)
    -> vec<
        std::common_type_t<typename std::decay_t<TensorA>::value_type, BReal>,
        N> {
  vec<std::common_type_t<typename std::decay_t<TensorA>::value_type, BReal>, N>
      ret = b;
  for (size_t i = 0; i < N; ++i) {
    ret(i) *= A.internal_tensor()(i);
  }
  return ret;
}
//------------------------------------------------------------------------------
template <typename TensorA, typename TensorB, typename BReal, size_t N>
constexpr auto operator*(base_tensor<TensorB, BReal, N> const& b,
                         diag_tensor<TensorA, N, N> const&     A) {
  return A * b;
}
//------------------------------------------------------------------------------
#include <tatooine/mat.h>
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <typename TensorA, typename TensorB, size_t N>
requires is_mat<TensorB> &&
         (std::decay_t<TensorB>::dimension(0) == N)
#else
template <typename TensorA, typename TensorB, size_t N,
          enable_if_mat<TensorB>                                    = true,
          enable_if<(std::decay_t<TensorB>::num_dimension(0) == N)> = true>
#endif
constexpr auto solve(diag_tensor<TensorA, N, N> const& A, TensorB&& B)
    -> std::optional<
        mat<std::common_type_t<typename std::decay_t<TensorA>::value_type,
                               typename std::decay_t<TensorB>::value_type>,
            N, N>> {
  for (size_t i = 0; i < N; ++i) {
    if (std::abs(A.internal_tensor()(i)) < 1e-10) {
      return {};
    }
  }
  mat<std::common_type_t<typename std::decay_t<TensorA>::value_type,
                         typename std::decay_t<TensorB>::value_type>,
      N, N>
      ret = B;
  for (size_t i = 0; i < N; ++i) {
    ret.row(i) /= A.internal_tensor()(i);
  }
  return ret;
}
//------------------------------------------------------------------------------
template <typename TensorA, typename TensorB, typename BReal, size_t M,
          size_t N>
constexpr auto operator*(diag_tensor<TensorA, M, M> const&        A,
                         base_tensor<TensorB, BReal, M, N> const& B) {
  using mat_t =
      mat<std::common_type_t<typename std::decay_t<TensorA>::value_type, BReal>,
          M, N>;
  auto ret = mat_t{B};
  for (size_t i = 0; i < M; ++i) {
    ret.row(i) *= A.internal_tensor()(i);
  }
  return ret;
}
//------------------------------------------------------------------------------
template <typename TensorA, typename TensorB, typename BReal, size_t M,
          size_t N>
constexpr auto operator*(base_tensor<TensorB, BReal, M, N> const& B,
                         diag_tensor<TensorA, N, N> const&        A)
    -> mat<
        std::common_type_t<typename std::decay_t<TensorA>::value_type, BReal>,
        M, N> {
  using common_type =
      common_type<typename std::decay_t<TensorA>::value_type, BReal>;
  auto ret = mat<common_type, M, N>{B};
  for (size_t i = 0; i < N; ++i) {
    ret.col(i) *= A.internal_tensor()(i);
  }
  return ret;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
