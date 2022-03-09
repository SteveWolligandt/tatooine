#ifndef TATOOINE_INTERPOLATION_H
#define TATOOINE_INTERPOLATION_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/polynomial.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
//#include <tatooine/polynomial_line.h>
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine::interpolation {
//==============================================================================
template <typename Real>
struct linear {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr size_t num_derivatives = 0;
  using real_type                            = Real;
  using polynomial_t                      = tatooine::polynomial<Real, 1>;
  static constexpr size_t num_dimensions() { return 1; }
  static_assert(is_floating_point<Real>);

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_t m_polynomial;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr linear()              = default;
  constexpr linear(const linear&) = default;
  constexpr linear(linear&&)      = default;
  constexpr linear& operator=(const linear&) = default;
  constexpr linear& operator=(linear&&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr linear(const Real ft0, const Real ft1)
      : m_polynomial{ft0, ft1 - ft0} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr linear(Real const t0, Real const t1, const Real ft0, const Real ft1)
      : m_polynomial{(ft0 * t1 - ft1 * t0) / (t1 - t0),
                     (ft1 - ft0) / (t1 - t0)} {}
  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto evaluate(Real t) const { return m_polynomial(t); }
  constexpr auto operator()(Real t) const { return evaluate(t); }
  //----------------------------------------------------------------------------
  constexpr auto polynomial() const -> auto const& { return m_polynomial; }
  constexpr auto polynomial() -> auto& { return m_polynomial; }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Real, size_t N>
struct linear<tensor<Real, N>> {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr size_t num_derivatives = 0;
  using real_type                            = Real;
  using vec_t                             = vec<Real, N>;
  using polynomial_t                      = tatooine::polynomial<Real, 1>;
  using polynomial_array_t                = std::array<polynomial_t, N>;
  static constexpr size_t num_dimensions() { return N; }
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_array_t m_polynomials;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr linear()              = default;
  constexpr linear(const linear&) = default;
  constexpr linear(linear&&)      = default;
  constexpr linear& operator=(const linear&) = default;
  constexpr linear& operator=(linear&&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t... Is>
  constexpr linear(Real const t0, Real const t1, const vec_t& ft0,
                   const vec_t& ft1, std::index_sequence<Is...> /*seq*/)
      : m_polynomials{polynomial_t{(ft0(Is) * t1 - ft1(Is) * t0) / (t1 - t0),
                                   (ft1(Is) - ft0(Is)) / (t1 - t0)}...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr linear(Real const t0, Real const t1, const vec_t& ft0,
                   const vec_t& ft1)
      : linear{t0, t1, ft0, ft1, std::make_index_sequence<N>{}} {}

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t... Is>
  constexpr linear(const vec_t& ft0, const vec_t& ft1,
                   std::index_sequence<Is...> /*seq*/)
      : m_polynomials{polynomial_t{ft0(Is), ft1(Is) - ft0(Is)}...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr linear(const vec_t& ft0, const vec_t& ft1)
      : linear{ft0, ft1, std::make_index_sequence<N>{}} {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto evaluate(Real t, std::index_sequence<Is...> /*seq*/) const {
    return vec{m_polynomials[Is](t)...};
  }
  constexpr auto evaluate(Real t) const {
    return evaluate(t, std::make_index_sequence<N>{});
  }
  constexpr auto operator()(Real t) const {
    return evaluate(t, std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  auto polynomial() const -> auto const& { return m_polynomials; }
  auto polynomial() -> auto& { return m_polynomials; }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
struct linear<vec<Real, N>> : linear<tensor<Real, N>> {
  using linear<tensor<Real, N>>::linear;
};
//==============================================================================
template <typename Real>
struct cubic : polynomial<Real, 3> {
  static constexpr size_t num_derivatives = 1;
  using real_type                            = Real;
  using parent_type                          = tatooine::polynomial<Real, 3>;
  static constexpr size_t num_dimensions() { return 1; }
  static_assert(is_floating_point<Real>);

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  constexpr cubic()             = default;
  constexpr cubic(const cubic&) = default;
  constexpr cubic(cubic&&)      = default;
  constexpr cubic& operator=(const cubic&) = default;
  constexpr cubic& operator=(cubic&&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr cubic(Real const t0, Real const t1, Real const ft0, Real const ft1,
                  Real const dft0_dt, Real const dft1_dt)
      : parent_type{0, 0, 0, 0} {
    constexpr Real        zero = 0;
    constexpr Real        one  = 1;
    mat<Real, 4, 4> const A{{one, t0, t0 * t0, t0 * t0 * t0},
                            {one, t1, t1 * t1, t1 * t1 * t1},
                            {zero, one, 2 * t0, 3 * t0 * t0},
                            {zero, one, 2 * t1, 3 * t1 * t1}};
    vec<Real, 4>          b{ft0, ft1, dft0_dt, dft1_dt};
    this->set_coefficients(solve(A, b)->data());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr cubic(const Real ft0, const Real ft1, const Real dft0_dt,
                  const Real dft1_dt)
      : parent_type{ft0, dft0_dt, 3 * ft1 - 3 * ft0 - dft1_dt - 2 * dft0_dt,
                 -2 * ft1 + 2 * ft0 + dft1_dt + dft0_dt} {}
};
//------------------------------------------------------------------------------
template <arithmetic Real, size_t N>
struct cubic<tensor<Real, N>> {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr size_t num_derivatives = 1;
  using real_type                            = Real;
  using vec_t                             = vec<Real, N>;
  using polynomial_t                      = tatooine::polynomial<Real, 3>;
  using polynomial_array_t                = std::array<polynomial_t, N>;
  static constexpr size_t num_dimensions() { return N; }
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_array_t m_polynomials;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr cubic()             = default;
  constexpr cubic(const cubic&) = default;
  constexpr cubic(cubic&&)      = default;
  constexpr cubic& operator=(const cubic&) = default;
  constexpr cubic& operator=(cubic&&) = default;
  //-----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr cubic(const vec_t& ft0, const vec_t& ft1, const vec_t& dft0_dt,
                  const vec_t& dft1_dt, std::index_sequence<Is...> /*seq*/)
      : m_polynomials{polynomial_t{
            ft0(Is), dft0_dt(Is),
            3 * ft1(Is) - 3 * ft0(Is) - dft1_dt(Is) - 2 * dft0_dt(Is),
            -2 * ft1(Is) + 2 * ft0(Is) + dft1_dt(Is) + dft0_dt(Is)}...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr cubic(const vec_t& ft0, const vec_t& ft1, const vec_t& dft0_dt,
                  const vec_t& dft1_dt)
      : cubic{ft0, ft1, dft0_dt, dft1_dt, std::make_index_sequence<N>{}} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr cubic(real_type const t0, real_type const t1, const vec_t& ft0,
                  const vec_t& ft1, const vec_t& dft0_dt,
                  const vec_t& dft1_dt) {
    mat<Real, 4, N> B;
    B.row(0) = ft0;
    B.row(1) = ft1;
    B.row(2) = dft0_dt;
    B.row(3) = dft1_dt;
    mat<Real, 4, 4> const A{{1.0, t0, t0 * t0, t0 * t0 * t0},
                            {1.0, t1, t1 * t1, t1 * t1 * t1},
                            {0.0, 1.0, 2 * t0, 3 * t0 * t0},
                            {0.0, 1.0, 2 * t1, 3 * t1 * t1}};
    const auto            C = *solve(A, B);
    for (size_t i = 0; i < N; ++i) {
      m_polynomials[i].set_coefficients(C(0, i), C(1, i), C(2, i), C(3, i));
    }
  }
  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto evaluate(Real t, std::index_sequence<Is...> /*seq*/) const {
    return vec{m_polynomials[Is](t)...};
  }
  constexpr auto evaluate(Real t) const {
    return evaluate(t, std::make_index_sequence<N>{});
  }
  constexpr auto operator()(Real t) const {
    return evaluate(t, std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  auto polynomial() const -> const auto& { return m_polynomials; }
  auto polynomial() -> auto& { return m_polynomials; }
};
//------------------------------------------------------------------------------
template <typename Real, size_t N>
struct cubic<vec<Real, N>> : cubic<tensor<Real, N>> {
  using cubic<tensor<Real, N>>::cubic;
};
//==============================================================================
template <typename Real>
struct quintic {
  static constexpr size_t num_derivatives = 2;
  using real_type                            = Real;
  using polynomial_t                      = tatooine::polynomial<Real, 5>;
  static constexpr size_t num_dimensions() { return 1; }
  static_assert(std::is_arithmetic<Real>::value);

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  polynomial_t m_polynomial;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  constexpr quintic()               = default;
  constexpr quintic(const quintic&) = default;
  constexpr quintic(quintic&&)      = default;
  constexpr quintic& operator=(const quintic&) = default;
  constexpr quintic& operator=(quintic&&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr quintic(Real const t0, Real const t1, Real const ft0,
                    Real const ft1, Real const dft0_dt, Real const dft1_dt,
                    Real const ddft0_dtt, Real const ddft1_dtt)
      : m_polynomial{0, 0, 0, 0, 0, 0} {
    mat<Real, 4, 4> const A{
        {1.0, t0, t0 * t0, t0 * t0 * t0, t0 * t0 * t0 * t0,
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
    m_polynomial.set_coefficients(solve(A, b)->data());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr quintic(const Real ft0, const Real ft1, const Real dft0_dt,
                    const Real dft1_dt, const Real ddft0_dtt,
                    const Real ddft1_dtt)
      : m_polynomial{ft0,
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
  constexpr auto evaluate(Real t) const { return m_polynomial(t); }
  constexpr auto operator()(Real t) const { return evaluate(t); }
  //----------------------------------------------------------------------------
  constexpr const auto& polynomial() const { return m_polynomial; }
  constexpr auto&       polynomial() { return m_polynomial; }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// template <arithmetic Real, size_t N>
// struct quintic<tensor<Real, N>> {
//  //----------------------------------------------------------------------------
//  // traits
//  //----------------------------------------------------------------------------
// public:
//  static constexpr size_t num_derivatives = 1;
//  using real_type                            = Real;
//  using vec_t                             = vec<Real, N>;
//  using polynomial_line_t                 = tatooine::polynomial_line<Real, N, 3>;
//  static constexpr size_t num_dimensions() {
//    return N;
//  }
//  //----------------------------------------------------------------------------
//  // members
//  //----------------------------------------------------------------------------
// public:
//  polynomial_line_t m_curve;
//  //----------------------------------------------------------------------------
//  // ctors
//  //----------------------------------------------------------------------------
//  constexpr quintic()               = default;
//  constexpr quintic(const quintic&) = default;
//  constexpr quintic(quintic&&)      = default;
//  constexpr quintic& operator=(const quintic&) = default;
//  constexpr quintic& operator=(quintic&&) = default;
//  //-----------------------------------------------------------------------------
// private:
//  template <size_t... Is>
//  constexpr quintic(const vec_t& ft0, const vec_t& ft1, const vec_t& dft0_dt,
//                    const vec_t& dft1_dt, const Real ddft0_dtt,
//                    const Real ddft1_dtt, std::index_sequence<Is...> [>seq<])
//      : m_curve{polynomial{
//            ft0(Is), dft0_dt(Is), ddft0_dtt(Is) / 2,
//            (20 * ft1(Is) - 20 * ft0(Is) - 8 * dft1_dt(Is) - 12 * dft0_dt(Is)
//            +
//             ddft1_dtt(Is) - 3 * ddft0_dtt(Is)) /
//                2,
//            -(30 * ft1(Is) - 30 * ft0(Is) - 14 * dft1_dt(Is) -
//              16 * dft0_dt(Is) + 2 * ddft1_dtt(Is) - 3 * ddft0_dtt(Is)) /
//                2,
//            (12 * ft1(Is) - 12 * ft0(Is) - 6 * dft1_dt(Is) - 6 * dft0_dt(Is) +
//             ddft1_dtt(Is) - ddft0_dtt(Is)) /
//                2}...} {}
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  -
// public:
//  constexpr quintic(const Real ft0, const Real ft1, const Real dft0_dt,
//                    const Real dft1_dt, const Real ddft0_dtt,
//                    const Real ddft1_dtt)
//      : quintic{ft0,
//                ft1,
//                dft0_dt,
//                dft1_dt,
//                ddft0_dtt,
//                ddft1_dtt,
//                std::make_index_sequence<N>{}} {}
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  - constexpr quintic(real_type const t0, real_type const t1, const vec_t& ft0,
//                    const vec_t& ft1, const vec_t& dft0_dt,
//                    const vec_t& dft1_dt, const vec_t& ddft0_dtt,
//                    const vec_t& ddft1_dtt) {
//    mat<Real, 4, N> B;
//    B.row(0) = ft0;
//    B.row(1) = ft1;
//    B.row(2) = dft0_dt;
//    B.row(3) = dft1_dt;
//    B.row(4) = ddft0_dtt;
//    B.row(5) = ddft1_dtt;
//    mat<Real, 4, 4> const A{
//        {1.0, t0, t0 * t0, t0 * t0 * t0, t0 * t0 * t0 * t0,
//         t0 * t0 * t0 * t0 * t0},
//        {1.0, t1, t1 * t1, t1 * t1 * t1, t1 * t1 * t1 * t1,
//         t1 * t1 * t1 * t1 * t1},
//        {0.0, 1.0, 2 * t0, 3 * t0 * t0, 4 * t0 * t0 * t0,
//         5 * t0 * t0 * t0 * t0},
//        {0.0, 1.0, 2 * t1, 3 * t1 * t1, 4 * t1 * t1 * t1,
//         5 * t1 * t1 * t1 * t1},
//        {0.0, 0.0, 2.0, 6 * t0, 12 * t0 * t0, 20 * t0 * t0 * t0},
//        {0.0, 0.0, 2.0, 6 * t1, 12 * t1 * t1, 20 * t1 * t1 * t1}};
//    const auto C = *solve(A, B);
//    for (size_t i = 0; i < N; ++i) {
//      m_curve.polynomial(i).set_coefficients(C(0, i), C(1, i), C(2, i), C(3,
//      i),
//                                             C(4, i), C(5, i));
//    }
//  }
//  //----------------------------------------------------------------------------
//  // methods
//  //----------------------------------------------------------------------------
//  constexpr auto evaluate(Real t) const {
//    return m_curve(t);
//  }
//  constexpr auto operator()(Real t) const {
//    return evaluate(t);
//  }
//  //----------------------------------------------------------------------------
//  const auto& curve() const {
//    return m_curve;
//  }
//  auto& curve() {
//    return m_curve;
//  }
//};
////------------------------------------------------------------------------------
// template <typename Real, size_t N>
// struct quintic<vec<Real, N>> : quintic<tensor<Real, N>> {
//  using quintic<tensor<Real, N>>::quintic;
//};
//==============================================================================
}  // namespace tatooine::interpolation
//==============================================================================
#endif
