#ifndef TATOOINE_ANALYTICAL_FIELDS_SYMBOLIC_COUNTEREXAMPLE_SADLO_H
#define TATOOINE_ANALYTICAL_FIELDS_SYMBOLIC_COUNTEREXAMPLE_SADLO_H
#include <tatooine/packages.h>
#if TATOOINE_GINAC_AVAILABLE
#include <tatooine/symbolic_field.h>
//==============================================================================
namespace tatooine::analytical::fields::symbolic {
//==============================================================================
template <typename Real>
struct counterexample_sadlo : field<Real, 2, 2> {
  using this_type   = counterexample_sadlo<Real>;
  using parent_type = field<Real, 2, 2>;
  using parent_type::t;
  using parent_type::x;
  using typename parent_type::pos_type;
  using typename parent_type::symtensor_type;
  using typename parent_type::tensor_type;

  counterexample_sadlo() {
    auto r =
        -GiNaC::numeric{1, 80} *
            GiNaC::power(GiNaC::power(x(0), 2) + GiNaC::power(x(1), 2), 2) +
        GiNaC::numeric{81, 80};

    this->set_expr(vec{r * (-GiNaC::numeric{1, 2} * x(0) +
                            GiNaC::numeric{1, 2} * cos(t()) - sin(t())),
                       r * (GiNaC::numeric{1, 2} * x(1) -
                            GiNaC::numeric{1, 2} * sin(t()) + cos(t()))});
  }

  constexpr bool in_domain(const pos_type& x, Real /*t*/) const {
    return length(x) <= 3;
  }
};
//==============================================================================
counterexample_sadlo()->counterexample_sadlo<double>;
//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================
#endif
#endif
