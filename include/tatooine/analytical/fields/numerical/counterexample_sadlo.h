#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_COUNTEREXAMPLE_SADLO_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_COUNTEREXAMPLE_SADLO_H
//==============================================================================
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
template <typename Real>
struct counterexample_sadlo
    : field<counterexample_sadlo<Real>, Real, 2, 2> {
  using this_t   = counterexample_sadlo<Real>;
  using parent_t = field<this_t, Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using vec_t = vec<Real, 2>;

  //============================================================================
  constexpr counterexample_sadlo() noexcept {}
  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& p, Real t) const {
    const Real x    = p(0);
    const Real y    = p(1);
    const Real xx   = x * x;
    const Real yy   = y * y;
    const Real xxyy = xx + yy;
    const Real r    = (-(1.0 / 80.0) * xxyy * xxyy + 81.0 / 80.0);
    return tensor_t{r * (-0.5 * x + 0.5 * std::cos(t) - std::sin(t)),
                    r * (0.5 * y - 0.5 * std::sin(t) + std::cos(t))};
  }

  struct bifurcationline_t{
    vec<Real, 2> at(Real t) const { return {std::cos(t), std::sin(t)}; }
    auto         operator()(Real t) const { return at(t); }
  };

  struct bifurcationline_spacetime_t {
    vec<Real, 3> at(Real t) const {
      return {std::cos(t), std::sin(t), t};
    }
    auto         operator()(Real t) const { return at(t); }
  };

  auto bifurcationline() const { return bifurcationline_t{};}
  auto bifurcationline_spacetime() const { return bifurcationline_spacetime_t{};}

  constexpr bool in_domain(const pos_t& x, Real /*t*/) const {
    return length(x) <= 3;
  }
};
//==============================================================================
counterexample_sadlo()->counterexample_sadlo<double>;
//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#endif