#ifndef TATOOINE_DIAG_TENSOR_H
#define TATOOINE_DIAG_TENSOR_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <static_vec Tensor, std::size_t M, std::size_t N>
struct diag_static_tensor {
  static auto constexpr is_tensor() { return true; }
  static auto constexpr is_diag() { return true; }
  static auto constexpr is_static() { return true; }
  static auto constexpr rank() { return 2; }
  static auto constexpr dimensions() { return std::array{M, N}; }
  static auto constexpr dimension(std::size_t const i) {
    switch (i) {
      default:
      case 0:
        return M;
      case 1:
        return N;
    }
  }
  //============================================================================
  using tensor_type = Tensor;
  using this_type   = diag_static_tensor<Tensor, M, N>;
  using value_type  = tensor_value_type<Tensor>;
  //============================================================================
 private:
  tensor_type m_internal_tensor;

  //============================================================================
 public:
  constexpr explicit diag_static_tensor(static_vec auto&& v)
      : m_internal_tensor{std::forward<decltype(v)>(v)} {}
  //----------------------------------------------------------------------------
  constexpr auto at(integral auto const... is) const -> value_type {
    if constexpr (sizeof...(is) == 2) {
      auto i = std::array{static_cast<std::size_t>(is)...};
      assert(i[0] < M);
      assert(i[1] < N);
      if (i[0] == i[1]) {
        return internal_tensor()(i[0]);
      }
      return 0;
    } else {
      return value_type(0) / value_type(0);
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is) const {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  constexpr auto at(integral_range auto const& is) const -> value_type {
    assert(is.size() == 2);
    return at(is[0], is[1]);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral_range auto const& is) const {
    assert(is.size() == 2);
    return at(is[0], is[1]);
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
  auto internal_tensor() -> auto& { return m_internal_tensor; }
};
//==============================================================================
// deduction guides
//==============================================================================
template <static_vec Tensor>
diag_static_tensor(Tensor const& t)
    -> diag_static_tensor<Tensor const&, Tensor::dimension(0),
                          Tensor::dimension(0)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_vec Tensor>
diag_static_tensor(Tensor& t)
    -> diag_static_tensor<Tensor&, Tensor::dimension(0), Tensor::dimension(0)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_vec Tensor>
diag_static_tensor(Tensor&& t)
    -> diag_static_tensor<std::decay_t<Tensor>, Tensor::dimension(0),
                          Tensor::dimension(0)>;
//==============================================================================
template <arithmetic_or_complex Real, std::size_t N>
struct vec;
//==============================================================================
// factory functions
//==============================================================================
constexpr auto diag(static_vec auto&& t) {
  return diag_static_tensor{std::forward<decltype(t)>(t)};
}
//------------------------------------------------------------------------------
template <std::size_t M, std::size_t N>
constexpr auto diag_rect(static_vec auto&& t) {
  if constexpr (std::is_rvalue_reference_v<decltype(t)>) {
    return diag_static_tensor<std::decay_t<decltype(t)>, M, N>{
        std::forward<decltype(t)>(t)};
  } else {
    return diag_static_tensor<decltype(t), M, N>{std::forward<decltype(t)>(t)};
  }
}
//==============================================================================
// free functions
//==============================================================================
template <typename Tensor, std::size_t N>
constexpr auto inv(diag_static_tensor<Tensor, N, N> const& A) -> std::optional<
    diag_static_tensor<vec<tensor_value_type<Tensor>, N>, N, N>> {
  using value_type = tensor_value_type<Tensor>;
  for (std::size_t i = 0; i < N; ++i) {
    if (std::abs(A.internal_tensor()(i)) < 1e-10) {
      return {};
    }
  }
  return diag_static_tensor{value_type(1) / A.internal_tensor()};
}
//------------------------------------------------------------------------------
#include <tatooine/vec.h>
//------------------------------------------------------------------------------
template <typename TensorA, static_vec TensorB, std::size_t N>
requires(tensor_dimensions<TensorB>[0] == N)
constexpr auto solve(diag_static_tensor<TensorA, N, N> const& A, TensorB&& b)
    -> std::optional<
        vec<common_type<tensor_value_type<TensorA>, tensor_value_type<TensorB>>,
            N>> {
  return A.internal_tensor() / b;
}
//------------------------------------------------------------------------------
template <typename TensorA, static_vec TensorB, std::size_t N>
requires(tensor_dimensions<TensorB>[0] == N)
constexpr auto solve(diag_static_tensor<TensorA, N, N>&& A, TensorB&& b)
    -> std::optional<
        vec<common_type<tensor_value_type<TensorA>, tensor_value_type<TensorB>>,
            N>> {
  return A.internal_tensor() / b;
}
//------------------------------------------------------------------------------
template <typename TensorA, std::size_t M, std::size_t N>
constexpr auto operator*(diag_static_tensor<TensorA, M, N> const& A,
                         static_vec auto const&                   b)
    -> vec<
        common_type<tensor_value_type<TensorA>, tensor_value_type<decltype(b)>>,
        M>
requires(N == decltype(b)::dimension(0)) {
  vec<common_type<tensor_value_type<TensorA>, tensor_value_type<decltype(b)>>,
      M>
      ret = b;
  for (std::size_t i = 0; i < N; ++i) {
    ret(i) *= A.internal_tensor()(i);
  }
  return ret;
}
////------------------------------------------------------------------------------
// template <typename TensorA, typename TensorB, typename BReal, std::size_t N>
// constexpr auto operator*(base_tensor<TensorB, BReal, N> const& b,
//                          diag_static_tensor<TensorA, N, N> const&     A) {
//   return A * b;
// }
//------------------------------------------------------------------------------
#include <tatooine/mat.h>
//------------------------------------------------------------------------------
template <typename TensorA, std::size_t M, std::size_t N>
constexpr auto operator*(
    diag_static_tensor<TensorA, M, N> const& A,
    static_mat auto const&                   B) requires(N ==
                                       std::decay_t<decltype(B)>::dimension(
                                           0)) {
  using mat_t = mat<
      common_type<tensor_value_type<TensorA>, tensor_value_type<decltype(B)>>,
      M, decltype(B)::dimension(1)>;
  auto ret = mat_t{B};
  for (std::size_t i = 0; i < M; ++i) {
    ret.row(i) *= A.internal_tensor()(i);
  }
  return ret;
}
//------------------------------------------------------------------------------
template <typename TensorA, std::size_t M, std::size_t N>
constexpr auto operator*(
    static_tensor auto const& B,
    diag_static_tensor<TensorA, M, N> const&
        A) requires(std::decay_t<decltype(B)>::dimension(1) == M) {
  auto ret = mat<
      common_type<tensor_value_type<TensorA>, tensor_value_type<decltype(B)>>,
      std::decay_t<decltype(B)>::dimension(0), N>{B};
  for (std::size_t i = 0; i < N; ++i) {
    ret.col(i) *= A.internal_tensor()(i);
  }
  return ret;
}
//------------------------------------------------------------------------------
template <typename TensorA, static_mat TensorB, std::size_t N>
requires(tensor_dimensions<TensorB>[0] == N)
constexpr auto solve(diag_static_tensor<TensorA, N, N>&& A, TensorB&& B)
    -> std::optional<
        mat<common_type<tensor_value_type<TensorA>, tensor_value_type<TensorB>>,
            N, N>> {
  auto ret =
      mat<common_type<tensor_value_type<TensorA>, tensor_value_type<TensorB>>,
          N, N>{B};
  for (std::size_t i = 0; i < N; ++i) {
    ret.row(i) /= A.internal_tensor()(i);
  }
  return ret;
}
//------------------------------------------------------------------------------
template <typename TensorA, static_mat TensorB, std::size_t N>
requires(tensor_dimensions<TensorB>[0] == N)
constexpr auto solve(diag_static_tensor<TensorA, N, N> const& A, TensorB&& B)
    -> std::optional<
        mat<common_type<tensor_value_type<TensorA>, tensor_value_type<TensorB>>,
            N, N>> {
  auto ret =
      mat<common_type<tensor_value_type<TensorA>, tensor_value_type<TensorB>>,
          N, N>{B};
  for (std::size_t i = 0; i < N; ++i) {
    ret.row(i) /= A.internal_tensor()(i);
  }
  return ret;
}
//==============================================================================
// dynamic
//==============================================================================
template <dynamic_tensor Tensor>
struct diag_dynamic_tensor {
  using value_type = tensor_value_type<Tensor>;
  static auto constexpr is_tensor() { return true; }
  static auto constexpr is_diag() { return true; }
  static auto constexpr is_dynamic() { return true; }
  //============================================================================
  Tensor m_internal_tensor;
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> auto const& { return m_internal_tensor; }
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  //----------------------------------------------------------------------------
  static auto constexpr rank() { return 2; }
  auto dimensions() const {
    return std::vector{internal_tensor().dimension(0),
                       internal_tensor().dimension(0)};
  }
  auto dimension(std::size_t const i) const {
    return internal_tensor().dimension(i);
  }
  //============================================================================
  auto at(integral auto const... is) const -> value_type {
    if constexpr (sizeof...(is) == 2) {
      auto i = std::array{is...};
      if (i[0] == i[1]) {
        return internal_tensor()(i[0]);
      }
      return 0;
    } else {
      return value_type(0) / value_type(0);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral auto const... is) const { return at(is...); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(integral_range auto const& is) const -> value_type {
    assert(is.size() == 2);
    return at(is[0], is[1]);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator()(integral_range auto const& is) const {
    assert(is.size() == 2);
    return at(is[0], is[1]);
  }
};
//==============================================================================
// deduction guides
//==============================================================================
template <dynamic_tensor Tensor>
diag_dynamic_tensor(Tensor const& t) -> diag_dynamic_tensor<Tensor const&>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <dynamic_tensor Tensor>
diag_dynamic_tensor(Tensor& t) -> diag_dynamic_tensor<Tensor&>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <dynamic_tensor Tensor>
diag_dynamic_tensor(Tensor&& t) -> diag_dynamic_tensor<std::decay_t<Tensor>>;
//==============================================================================
auto diag(dynamic_tensor auto&& A) {
  assert(A.rank() == 1);
  return diag_dynamic_tensor{std::forward<decltype(A)>(A)};
}
//------------------------------------------------------------------------------
template <typename Lhs, dynamic_tensor Rhs>
requires(dynamic_tensor<Lhs>&& diag_tensor<Lhs>) auto operator*(Lhs const& lhs,
                                                                Rhs const& rhs)
    -> tensor<common_type<tensor_value_type<Lhs>, tensor_value_type<Rhs>>> {
  using out_t =
      tensor<common_type<tensor_value_type<Lhs>, tensor_value_type<Rhs>>>;
  auto out = out_t{};
  // matrix-matrix-multiplication
  if (lhs.rank() == 2 && rhs.rank() == 2 &&
      lhs.internal_tensor().dimension(0) == rhs.dimension(0)) {
    auto out =
        out_t::zeros(lhs.internal_tensor().dimension(0), rhs.dimension(1));
    for (std::size_t r = 0; r < lhs.internal_tensor().dimension(0); ++r) {
      for (std::size_t c = 0; c < rhs.dimension(1); ++c) {
        out(r, c) = lhs.internal_tensor()(r) * rhs(r, c);
      }
    }
    return out;

    // matrix-vector-multiplication
  } else if (lhs.rank() == 2 && rhs.rank() == 1 &&
             lhs.dimension(1) == rhs.dimension(0)) {
    auto out = out_t::zeros(lhs.dimension(0));
    for (std::size_t i = 0; i < rhs.dimension(1); ++i) {
      out(i) = lhs.internal_tensor()(i) * rhs(i);
    }
    return out;
  }

  std::stringstream A;
  A << "[ " << lhs.dimension(0);
  for (std::size_t i = 1; i < lhs.rank(); ++i) {
    A << " x " << lhs.dimension(i);
  }
  A << " ]";
  std::stringstream B;
  B << "[ " << rhs.dimension(0);
  for (std::size_t i = 1; i < rhs.rank(); ++i) {
    B << " x " << rhs.dimension(i);
  }
  B << " ]";
  throw std::runtime_error{"Cannot contract given dynamic tensors. (A:" +
                           A.str() + "; B" + B.str() + ")"};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
