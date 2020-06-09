#ifndef TATOOINE_ANALYTICAL_FIELDS_SYMBOLIC_DOUBLEGYRE_H
#define TATOOINE_ANALYTICAL_FIELDS_SYMBOLIC_DOUBLEGYRE_H
//==============================================================================
#include <tatooine/packages.h>
#if TATOOINE_GINAC_AVAILABLE
#include <tatooine/symbolic_field.h>
//==============================================================================
namespace tatooine::analytical::fields::symbolic {
//==============================================================================
template <typename Real>
struct doublegyre : field<Real, 2, 2> {
  using this_t   = doublegyre<Real>;
  using parent_t = field<Real, 2, 2>;
  using parent_t::t;
  using parent_t::x;
  using typename parent_t::pos_t;
  using typename parent_t::symtensor_t;
  using typename parent_t::tensor_t;

  explicit doublegyre(const GiNaC::ex& eps   = GiNaC::numeric{1, 4},
                      const GiNaC::ex& omega = 2 * GiNaC::Pi *
                                               GiNaC::numeric{1, 10},
                      const GiNaC::ex& A = GiNaC::numeric{1, 10}) {
    using GiNaC::Pi;
    auto a = eps * sin(omega * t());
    auto b = 1 - 2 * a;
    auto f = a * pow(x(0), 2) + b * x(0);
    this->set_expr(vec<GiNaC::ex, 2>{
        -Pi * A * sin(Pi * f) * cos(Pi * x(1)),
        Pi * A * cos(Pi * f) * sin(Pi * x(1)) * f.diff(x(0))});
  }

  //----------------------------------------------------------------------------
  constexpr auto in_domain(const pos_t& x, Real) const -> bool {
    return x(0) >= 0 && x(0) <= 2 && x(1) >= 0 && x(1) <= 1;
  }
};
doublegyre()->doublegyre<double>;


//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct is_field<symbolic::doublegyre<Real>> : std::true_type {};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
#endif
