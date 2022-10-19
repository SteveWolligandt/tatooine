#ifndef TATOOINE_INTERPOLATION_H
#define TATOOINE_INTERPOLATION_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/polynomial.h>
#include <tatooine/tensor.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
//==============================================================================
namespace tatooine::interpolation {
//==============================================================================
template <typename Real>
struct linear;
//==============================================================================
template <floating_point Real>
struct linear<Real> {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr std::size_t num_derivatives = 0;
  using real_type                              = Real;
  using interpolant_type                       = tatooine::polynomial<Real, 1>;
  static constexpr std::size_t num_dimensions() { return 1; }

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  interpolant_type m_interpolant;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr linear()                         = default;
  constexpr linear(linear const&)            = default;
  constexpr linear(linear&&)                 = default;
  constexpr linear& operator=(linear const&) = default;
  constexpr linear& operator=(linear&&)      = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr linear(Real const ft0, Real const ft1)
      : m_interpolant{ft0, ft1 - ft0} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr linear(Real const t0, Real const t1, Real const ft0, Real const ft1)
      : m_interpolant{(ft0 * t1 - ft1 * t0) / (t1 - t0),
                      (ft1 - ft0) / (t1 - t0)} {}
  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto evaluate(arithmetic auto const t) const {
    return m_interpolant(t);
  }
  constexpr auto operator()(arithmetic auto const t) const {
    return evaluate(t);
  }
  //----------------------------------------------------------------------------
  constexpr auto polynomial() const -> auto const& { return m_interpolant; }
  constexpr auto polynomial() -> auto& { return m_interpolant; }
};
//==============================================================================
template <typename T>
struct interpolation_value_type_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t... Dims>
struct interpolation_value_type_impl<tensor<Real, Dims...>> {
  using type = tensor<Real, Dims...>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t N>
struct interpolation_value_type_impl<tensor<Real, N>> {
  using type = vec<Real, N>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t N>
struct interpolation_value_type_impl<vec<Real, N>>
    : interpolation_value_type_impl<tensor<Real, N>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t M, std::size_t N>
struct interpolation_value_type_impl<tensor<Real, M, N>> {
  using type = mat<Real, M, N>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t M, std::size_t N>
struct interpolation_value_type_impl<mat<Real, M, N>>
    : interpolation_value_type_impl<tensor<Real, M, N>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
using interpolation_value_type =
    typename interpolation_value_type_impl<T>::type;
//==============================================================================
template <typename Real>
struct interpolation_tensor_type_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t... Dims>
struct interpolation_tensor_type_impl<tensor<Real, Dims...>> {
  using type = tensor<Real, Dims...>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t N>
struct interpolation_tensor_type_impl<tensor<Real, N>> {
  using type = tensor<Real, N>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t N>
struct interpolation_tensor_type_impl<vec<Real, N>> {
  using type = tensor<Real, N>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t M, std::size_t N>
struct interpolation_tensor_type_impl<mat<Real, M, N>> {
  using type = tensor<Real, M, N>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
using interpolation_tensor_type =
    typename interpolation_tensor_type_impl<T>::type;
//==============================================================================
template <floating_point Float>
linear(Float, Float) -> linear<Float>;
template <floating_point Float>
linear(Float, Float, Float, Float) -> linear<Float>;
//==============================================================================
template <floating_point Real, std::size_t... Dims>
struct linear<tensor<Real, Dims...>> {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr std::size_t num_derivatives = 0;
  using real_type        = Real;
  using value_type       = interpolation_value_type<tensor<Real, Dims...>>;
  using tensor_type      = interpolation_tensor_type<tensor<Real, Dims...>>;
  using interpolant_type = tatooine::polynomial<Real, 1>;
  using polynomial_array_type =
      static_multidim_array<interpolant_type, x_fastest, stack, Dims...>;
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_array_type m_interpolants = {};
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr linear()                         = default;
  constexpr linear(linear const&)            = default;
  constexpr linear(linear&&)                 = default;
  constexpr linear& operator=(linear const&) = default;
  constexpr linear& operator=(linear&&)      = default;
  // - - - - - - - - - - - - - - - - - - -j- - - - - - - - - - - - - - - - - - -
  constexpr linear(Real const t0, Real const t1,
                   tensor_type const& ft0, tensor_type const& ft1) {
    auto it = [&](auto const... is) {
      m_interpolants(is...) =
          interpolant_type{(ft0(is...) * t1 - ft1(is...) * t0) / (t1 - t0),
                           (ft1(is...) - ft0(is...)) / (t1 - t0)};
    };
    for_loop(it, Dims...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr linear(tensor_type const& ft0, tensor_type const& ft1) {
    auto it = [&](auto const... is) {
      m_interpolants(is...) =
          interpolant_type{ft0(is...), ft1(is...) - ft0(is...)};
    };
    for_loop(it, Dims...);
  }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto evaluate(arithmetic auto const t) const {
    auto interpolation = value_type{};
    auto it            = [&](auto const... is) {
      interpolation(is...) = m_interpolants(is...)(t);
    };
    for_loop(it, Dims...);
    return interpolation;
  }
  constexpr auto operator()(arithmetic auto const t) const {
    return evaluate(t);
  }
  //----------------------------------------------------------------------------
  auto polynomial() const -> auto const& { return m_interpolants; }
  auto polynomial() -> auto& { return m_interpolants; }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t N>
struct linear<vec<Real, N>> : linear<tensor<Real, N>> {
  using linear<tensor<Real, N>>::linear;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t M, std::size_t N>
struct linear<mat<Real, M, N>> : linear<tensor<Real, N>> {
  using linear<tensor<Real, N>>::linear;
};
template <floating_point Float, std::size_t... Dims>
linear(Float, Float, tensor<Float, Dims...>, tensor<Float, Dims...>)
    -> linear<tensor<Float, Dims...>>;
template <floating_point Float, std::size_t... Dims>
linear(tensor<Float, Dims...>, tensor<Float, Dims...>)
    -> linear<tensor<Float, Dims...>>;
template <floating_point Float, std::size_t N>
linear(Float, Float, vec<Float, N>, vec<Float, N>) -> linear<tensor<Float, N>>;
template <floating_point Float, std::size_t N>
linear(vec<Float, N>, vec<Float, N>) -> linear<tensor<Float, N>>;
template <floating_point Float, std::size_t M, std::size_t N>
linear(Float, Float, mat<Float, M, N>, mat<Float, M, N>)
    -> linear<tensor<Float, N>>;
template <floating_point Float, std::size_t M, std::size_t N>
linear(mat<Float, M, N>, mat<Float, M, N>) -> linear<tensor<Float, M, N>>;
//==============================================================================
template <typename Real>
struct cubic;
//==============================================================================
template <floating_point Real>
struct cubic<Real> : polynomial<Real, 3> {
  static constexpr std::size_t num_derivatives = 1;
  using real_type                              = Real;
  using parent_type                            = tatooine::polynomial<Real, 3>;
  static constexpr std::size_t num_dimensions() { return 1; }

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  constexpr cubic()                        = default;
  constexpr cubic(cubic const&)            = default;
  constexpr cubic(cubic&&)                 = default;
  constexpr cubic& operator=(cubic const&) = default;
  constexpr cubic& operator=(cubic&&)      = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr cubic(Real const t0, Real const t1, Real const ft0, Real const ft1,
                  Real const dft0_dt, Real const dft1_dt)
      : parent_type{0, 0, 0, 0} {
    constexpr Real zero = 0;
    constexpr Real one  = 1;
    auto const     A    = mat<Real, 4, 4>{{one, t0, t0 * t0, t0 * t0 * t0},
                                          {one, t1, t1 * t1, t1 * t1 * t1},
                                          {zero, one, 2 * t0, 3 * t0 * t0},
                                          {zero, one, 2 * t1, 3 * t1 * t1}};
    auto const     b    = vec<Real, 4>{ft0, ft1, dft0_dt, dft1_dt};
    this->set_coefficients(solve(A, b)->data());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr cubic(Real const ft0, Real const ft1, Real const dft0_dt,
                  Real const dft1_dt)
      : parent_type{ft0, dft0_dt, 3 * ft1 - 3 * ft0 - dft1_dt - 2 * dft0_dt,
                    -2 * ft1 + 2 * ft0 + dft1_dt + dft0_dt} {}
};
//------------------------------------------------------------------------------
template <floating_point Float>
cubic(Float, Float, Float, Float) -> cubic<Float>;
//------------------------------------------------------------------------------
template <floating_point Float>
cubic(Float, Float, Float, Float, Float, Float) -> cubic<Float>;
//------------------------------------------------------------------------------
template <arithmetic Real, std::size_t N>
struct cubic<tensor<Real, N>> {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr std::size_t num_derivatives = 1;
  using real_type                              = Real;
  using vec_type                               = vec<Real, N>;
  using interpolant_type                       = tatooine::polynomial<Real, 3>;
  using polynomial_array_type = std::array<interpolant_type, N>;
  static constexpr std::size_t num_dimensions() { return N; }
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_array_type m_interpolants;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr cubic()                        = default;
  constexpr cubic(cubic const&)            = default;
  constexpr cubic(cubic&&)                 = default;
  constexpr cubic& operator=(cubic const&) = default;
  constexpr cubic& operator=(cubic&&)      = default;
  //-----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  constexpr cubic(vec_type const& ft0, vec_type const& ft1,
                  vec_type const& dft0_dt, vec_type const& dft1_dt,
                  std::index_sequence<Is...> /*seq*/)
      : m_interpolants{interpolant_type{
            ft0(Is), dft0_dt(Is),
            3 * ft1(Is) - 3 * ft0(Is) - dft1_dt(Is) - 2 * dft0_dt(Is),
            -2 * ft1(Is) + 2 * ft0(Is) + dft1_dt(Is) + dft0_dt(Is)}...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr cubic(vec_type const& ft0, vec_type const& ft1,
                  vec_type const& dft0_dt, vec_type const& dft1_dt)
      : cubic{ft0, ft1, dft0_dt, dft1_dt, std::make_index_sequence<N>{}} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr cubic(real_type const t0, real_type const t1, vec_type const& ft0,
                  vec_type const& ft1, vec_type const& dft0_dt,
                  vec_type const& dft1_dt) {
    mat<Real, 4, N> B;
    B.row(0)     = ft0;
    B.row(1)     = ft1;
    B.row(2)     = dft0_dt;
    B.row(3)     = dft1_dt;
    auto const A = Mat4<Real>{{1.0, t0, t0 * t0, t0 * t0 * t0},
                              {1.0, t1, t1 * t1, t1 * t1 * t1},
                              {0.0, 1.0, 2 * t0, 3 * t0 * t0},
                              {0.0, 1.0, 2 * t1, 3 * t1 * t1}};
    auto const C = *solve(A, B);
    for (std::size_t i = 0; i < N; ++i) {
      m_interpolants[i].set_coefficients(C(0, i), C(1, i), C(2, i), C(3, i));
    }
  }
  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  constexpr auto evaluate(arithmetic auto const t,
                          std::index_sequence<Is...> /*seq*/) const {
    return vec{m_interpolants[Is](t)...};
  }
  constexpr auto evaluate(arithmetic auto const t) const {
    return evaluate(t, std::make_index_sequence<N>{});
  }
  constexpr auto operator()(arithmetic auto const t) const {
    return evaluate(t, std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  auto polynomial() const -> auto const& { return m_interpolants; }
  auto polynomial() -> auto& { return m_interpolants; }
};
//==============================================================================
template <typename Real, std::size_t N>
struct cubic<vec<Real, N>> : cubic<tensor<Real, N>> {
  using cubic<tensor<Real, N>>::cubic;
};
template <floating_point Float, std::size_t N>
cubic(tensor<Float, N>, tensor<Float, N>, tensor<Float, N>, tensor<Float, N>)
    -> cubic<tensor<Float, N>>;
//------------------------------------------------------------------------------
template <floating_point Float, std::size_t N>
cubic(Float, Float, tensor<Float, N>, tensor<Float, N>, tensor<Float, N>,
      tensor<Float, N>) -> cubic<tensor<Float, N>>;
//------------------------------------------------------------------------------
template <floating_point Float, std::size_t N>
cubic(vec<Float, N>, vec<Float, N>, vec<Float, N>, vec<Float, N>)
    -> cubic<tensor<Float, N>>;
//------------------------------------------------------------------------------
template <floating_point Float, std::size_t N>
cubic(Float, Float, vec<Float, N>, vec<Float, N>, vec<Float, N>, vec<Float, N>)
    -> cubic<tensor<Float, N>>;
//==============================================================================
template <typename Real>
struct quintic;
//==============================================================================
template <floating_point Real>
struct quintic<Real> {
  static constexpr std::size_t num_derivatives = 2;
  using real_type                              = Real;
  using interpolant_type                       = tatooine::polynomial<Real, 5>;
  static constexpr std::size_t num_dimensions() { return 1; }
  static_assert(std::is_arithmetic<Real>::value);

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  interpolant_type m_interpolant;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  constexpr quintic()                          = default;
  constexpr quintic(quintic const&)            = default;
  constexpr quintic(quintic&&)                 = default;
  constexpr quintic& operator=(quintic const&) = default;
  constexpr quintic& operator=(quintic&&)      = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr quintic(Real const t0, Real const t1, Real const ft0,
                    Real const ft1, Real const dft0_dt, Real const dft1_dt,
                    Real const ddft0_dtt, Real const ddft1_dtt)
      : m_interpolant{0, 0, 0, 0, 0, 0} {
    auto const A =
        Mat4<Real>{{1.0, t0, t0 * t0, t0 * t0 * t0, t0 * t0 * t0 * t0,
                    t0 * t0 * t0 * t0 * t0},
                   {1.0, t1, t1 * t1, t1 * t1 * t1, t1 * t1 * t1 * t1,
                    t1 * t1 * t1 * t1 * t1},
                   {0.0, 1.0, 2 * t0, 3 * t0 * t0, 4 * t0 * t0 * t0,
                    5 * t0 * t0 * t0 * t0},
                   {0.0, 1.0, 2 * t1, 3 * t1 * t1, 4 * t1 * t1 * t1,
                    5 * t1 * t1 * t1 * t1},
                   {0.0, 0.0, 2.0, 6 * t0, 12 * t0 * t0, 20 * t0 * t0 * t0},
                   {0.0, 0.0, 2.0, 6 * t1, 12 * t1 * t1, 20 * t1 * t1 * t1}};
    vec<Real, 6> b{ft0, ft1, dft0_dt, dft1_dt, ddft0_dtt, ddft1_dtt};
    m_interpolant.set_coefficients(solve(A, b)->data());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr quintic(Real const ft0, Real const ft1, Real const dft0_dt,
                    Real const dft1_dt, Real const ddft0_dtt,
                    Real const ddft1_dtt)
      : m_interpolant{ft0,
                      dft0_dt,
                      ddft0_dtt / 2,
                      (20 * ft1 - 20 * ft0 - 8 * dft1_dt - 12 * dft0_dt +
                       ddft1_dtt - 3 * ddft0_dtt) /
                          2,
                      -(30 * ft1 - 30 * ft0 - 14 * dft1_dt - 16 * dft0_dt +
                        2 * ddft1_dtt - 3 * ddft0_dtt) /
                          2,
                      (12 * ft1 - 12 * ft0 - 6 * dft1_dt - 6 * dft0_dt +
                       ddft1_dtt - ddft0_dtt) /
                          2} {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto evaluate(arithmetic auto const t) const {
    return m_interpolant(t);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(arithmetic auto const t) const {
    return evaluate(t);
  }
  //----------------------------------------------------------------------------
  constexpr auto polynomial() const -> auto const& { return m_interpolant; }
  constexpr auto polynomial() -> auto& { return m_interpolant; }
};
//------------------------------------------------------------------------------
template <floating_point Float>
quintic(Float, Float, Float, Float, Float, Float) -> quintic<Float>;
//==============================================================================
}  // namespace tatooine::interpolation
//==============================================================================
#endif
