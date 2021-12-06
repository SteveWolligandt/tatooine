#ifndef TATOOINE_ODE_VCLIBS_VECTOROPERATIONS_H
#define TATOOINE_ODE_VCLIBS_VECTOROPERATIONS_H
//==============================================================================
#include <boost/range/algorithm/transform.hpp>
#include <vcode/odeint.hh>
#include <tatooine/vec.h>
//==============================================================================
template <typename Real, std::size_t N>
struct VC::odeint::vector_operations_t<tatooine::vec<Real, N>> {
  using vec_t = tatooine::vec<Real, N>;

  //----------------------------------------------------------------------------
  static constexpr auto isfinitenorm(vec_t const& v) {
    for (auto c : v) {
      if (!std::isfinite(c)) {
        return false;
      }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  static constexpr auto norm2(vec_t const& x) {
    static_assert(std::is_floating_point<Real>(), "require floating point");
    return std::sqrt(sqr(x));
  }

  //----------------------------------------------------------------------------
  static constexpr auto norm1(vec_t const& x) { return tatooine::norm1(x); }
  //----------------------------------------------------------------------------
  static constexpr auto norminf(vec_t const& x) { return norm_inf(x); }
  //----------------------------------------------------------------------------
  static constexpr auto abs(vec_t v) {
    boost::transform(v, v.begin(), [](auto const c) { return std::abs(c); });
    return v;
  }

  //----------------------------------------------------------------------------
  static constexpr auto max(vec_t const& x, vec_t const& y) {
    vec_t v;
    for (std::size_t i = 0; i < N; ++i) {
      v(i) = std::max(x(i), y(i));
    }
    return v;
  }
};
//==============================================================================
#endif
