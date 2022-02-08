#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_MODIFIED_DOUBLEGYRE_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_MODIFIED_DOUBLEGYRE_H
//==============================================================================
#include <tatooine/algorithm.h>
#include <tatooine/field.h>
#include <tatooine/line.h>
#include <tatooine/linspace.h>
#include <tatooine/type_traits.h>

#include <algorithm>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
template <typename Real>
struct modified_doublegyre : vectorfield<modified_doublegyre<Real>, Real, 2> {
  using this_type   = modified_doublegyre<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  //============================================================================
  static constexpr Real pi      = M_PI;
  static constexpr Real epsilon = 0.25;
  static constexpr Real omega   = 2 * pi * 0.1;
  static constexpr Real A       = 0.1;
  static constexpr Real c       = -0.2040811331;
  static constexpr Real cc      = c * c;
  static constexpr Real d       = 9.964223388;
  //============================================================================
  constexpr tensor_type evaluate(const pos_type& x, Real t) const {
    const Real a  = epsilon * std::sin(omega * (t + timeoffset(t)));
    const Real b  = 1 - 2 * a;
    const Real f  = a * x(0) * x(0) + b * x(0);
    const Real df = 2 * a * x(0) + b;

    return tensor_type{-pi * A * std::sin(pi * f) * std::cos(pi * x(1)),
                    pi * A * std::cos(pi * f) * std::sin(pi * x(1)) * df};
  }

  //----------------------------------------------------------------------------
  constexpr static auto timeoffset(const Real t) {
    const Real r = pi / 5 * t + d;

    const Real q =
        std::clamp<Real>((4 * pi * c * sin(r) - 4 * std::asin(2 * c * cos(r))) /
                        (pi * (1 - cc * sin(r) * sin(r))),
                    -1, 1);

    const Real p = 5 / pi * std::asin(q) - t;
    return p;
    // Real       min_p       = p;
    // auto       closer_to_0 = [&min_p](Real p) -> Real {
    //  if (std::abs(p) < std::abs(min_p)) { return p; }
    //  return min_p;
    //};
    //
    // for (int i = 0; i <= 1; ++i) {
    //  min_p = closer_to_0(5 + i * 10 - 2 * t - p);
    //  min_p = closer_to_0(5 - i * 10 - 2 * t - p);
    //}
    //
    // for (int i = 1; i <= 1; ++i) {
    //  min_p = closer_to_0(p + i * 10);
    //  min_p = closer_to_0(p - i * 10);
    //}
    //
    // return min_p;
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_type& x, Real) const {
    return x(0) >= 0 && x(0) <= 2 && x(1) >= 0 && x(1) <= 1;
  }

  //----------------------------------------------------------------------------
  struct hyperbolic_trajectory_t {
    auto at(Real t) const {
      return vec<Real, 2>{c * std::sin(pi / 5 * t + d) + 1, 0};
    }
    auto operator()(Real t) const { return at(t); }
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto hyperbolic_trajectory() const {
    return hyperbolic_trajectory_t{};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto hyperbolic_trajectory(Real t) const {
    return hyperbolic_trajectory_t{}(t);
  }

  //----------------------------------------------------------------------------
  struct hyperbolic_trajectory_spacetime_t {
    auto at(Real t) const {
      return vec<Real, 3>{c * std::sin(pi / 5 * t + d) + 1, 0, t};
    }
    auto operator()(Real t) const { return at(t); }
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto hyperbolic_trajectory_spacetime() const {
    return hyperbolic_trajectory_spacetime_t{};
  }
};
//==============================================================================
modified_doublegyre()->modified_doublegyre<double>;
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
#endif
