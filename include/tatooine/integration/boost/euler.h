#ifndef __TATOOINE_EULER_INTEGRATOR_H__
#define __TATOOINE_EULER_INTEGRATOR_H__

#include "boostintegrator.h"

#include "../../concept_defines.h"

//==============================================================================
namespace tatooine::integration::boost {
//==============================================================================
//
template <typename real_t, size_t n>
struct euler_helper {
  using type = ::boost::numeric::odeint::euler<Vec<real_t, n>>;
};
template <typename real_t, size_t n>
using euler_t = typename euler_helper<real_t, n>::type;

template <size_t n, typename real_t>
struct euler : BoostIntegrator<n, real_t,euler_t<real_t, n>> {
 public:
  euler(const real_t stepsize = 0.01)
      : BoostIntegrator<n, real_t, euler_t<real_t, n>>{euler_t<real_t, n>{},
                                                       stepsize} {}
};

//==============================================================================
}  // namespace tatooine::integration::boost
//==============================================================================

#endif
