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
#ifdef __cpp_concepts
template <arithmetic_or_complex Real, size_t N>
#else
template <typename Real, size_t N>
#endif
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
  auto ret = mat<std::common_type_t<typename std::decay_t<TensorA>::value_type,
                                    typename std::decay_t<TensorB>::value_type>,
                 N, N>{B};
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
// dynamic
//==============================================================================
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
struct const_diag_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
  DynamicTensor const&  m_tensor;
  static constexpr auto zero = typename DynamicTensor::value_type{};
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto at(Is const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[const_diag_dynamic_tensor::at] need exactly two indices"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral R, integral C>
#else
  template <typename R, typename C, enable_if_integral<R, C> = true>
#endif
  auto at(R const r, C const c) const -> value_type const& {
    if (r == c) {
      return m_tensor(r);
    }
    return zero;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto operator()(Indices const& indices) const -> auto const& {
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto operator()(Indices const& indices) -> auto& {
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto at(Indices indices) -> auto& {
    assert(indices.size() == num_dimensions());
    std::reverse(begin(indices), end(indices));
    return m_tensor(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto at(Indices indices) const -> auto const& {
    assert(indices.size() == num_dimensions());
    std::reverse(begin(indices), end(indices));
    return m_tensor(indices);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto operator()(Is const... is) const -> value_type const& {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const /*i*/) const { return m_tensor.size(0); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
struct is_dynamic_tensor_impl<const_diag_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor>
struct diag_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
  DynamicTensor&        m_tensor;
  static constexpr auto zero = typename DynamicTensor::value_type{};
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto at(Is const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[diag_dynamic_tensor::at] need exactly two indices"};
  }
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto at(Is const... /*is*/) -> value_type& {
    throw std::runtime_error{
        "[diag_dynamic_tensor::at] need exactly two indices"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral R, integral C>
#else
  template <typename R, typename C, enable_if_integral<R, C> = true>
#endif
  auto at(R const r, C const c) const -> value_type const& {
    if (r == c) {
      return m_tensor(r);
    }
    return zero;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral R, integral C>
#else
  template <typename R, typename C, enable_if_integral<R, C> = true>
#endif
  auto at(R const r, C const c) -> value_type& {
    static typename DynamicTensor::value_type zero;
    zero = typename DynamicTensor::value_type{};
    if (r == c) {
      return m_tensor(r);
    }
    return zero;
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto operator()(Is const... is) const -> value_type const& {
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto operator()(Is const... is) -> value_type& {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const /*i*/) const { return m_tensor.size(0); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
struct is_dynamic_tensor_impl<diag_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
auto diag(DynamicTensor const& A) {
  assert(A.num_dimensions() == 1);
  return const_diag_dynamic_tensor<DynamicTensor>{A};
}
//------------------------------------------------------------------------------
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
auto diag(DynamicTensor& A) {
  assert(A.num_dimensions() == 1);
  return diag_dynamic_tensor<DynamicTensor>{A};
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <typename LhsTensor, typename RhsTensor>
requires is_dynamic_tensor<LhsTensor>
#else
template <typename LhsTensor, typename RhsTensor,
          enable_if<is_dynamic_tensor<LhsTensor>> = true>
#endif
auto operator*(LhsTensor const& lhs, diag_dynamic_tensor<RhsTensor> const& rhs)
    -> tensor<std::common_type_t<typename LhsTensor::value_type,
                                 typename RhsTensor::value_type>> {
  using out_t = tensor<std::common_type_t<typename LhsTensor::value_type,
                                          typename RhsTensor::value_type>>;
  out_t out;
  // matrix-matrix-multiplication
  if (lhs.num_dimensions() == 2 && lhs.size(1) == rhs.size(0)) {
    auto out = out_t::zeros(lhs.size(0), rhs.size(1));
    for (size_t r = 0; r < lhs.size(0); ++r) {
      for (size_t c = 0; c < rhs.size(1); ++c) {
        out(r, c) = lhs(r, c) * rhs(c, c);
      }
    }
    return out;
  }

  std::stringstream A;
  A << "[ " << lhs.size(0);
  for (size_t i = 1; i < lhs.num_dimensions(); ++i) {
    A << " x " << lhs.size(i);
  }
  A << " ]";
  std::stringstream B;
  B << "[ " << rhs.size(0);
  for (size_t i = 1; i < rhs.num_dimensions(); ++i) {
    B << " x " << rhs.size(i);
  }
  B << " ]";
  throw std::runtime_error{"Cannot contract given dynamic tensors. (A:" +
                           A.str() + "; B" + B.str() + ")"};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
