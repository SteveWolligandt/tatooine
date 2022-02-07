#ifndef TATOOINE_ORBIT_H
#define TATOOINE_ORBIT_H

#include "field.h"

//==============================================================================
namespace tatooine::numerical {
//==============================================================================
template <typename Real>
struct orbit : field<orbit<Real>, Real, 3, 3> {
  using this_t   = orbit<Real>;
  using parent_type = field<this_t, Real, 3, 3>;
  using typename parent_type::pos_t;
  using typename parent_type::tensor_t;
  using vec_t = tensor_t;

  //============================================================================
  constexpr tensor_t evaluate(const pos_t& x, Real /*t*/) const {
    Real r = 1;
    return vec_t{-x(1), x(0), -x(2)} +
           (vec_t{x(0), x(1), 0} * (length(vec{x(0), x(1)}) - r));
  }
};

orbit()->orbit<double>;

//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#endif
