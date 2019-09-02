#ifndef __TATOOINE_RB4_INTEGRATOR_H__
#define __TATOOINE_RB4_INTEGRATOR_H__

#define ARMA_DONT_PRINT_ERRORS

#include "BaseBoostIntegrator.h"

#include <armadillo>
#include <boost/numeric/odeint.hpp>
#include <iostream>
#include "boostresizer.h"

namespace tatooine {
namespace integration {
namespace boost {
template <typename T>
struct rb4_type {
  using stepper_t = ::boost::numeric::odeint::rosenbrock4<arma::Col<T>>;
  using t         = ::boost::numeric::odeint::rosenbrock4_controller<stepper_t>;
};
template <unsigned n, template <typename, typename> typename interpolator_t = HermiteInterpolator, typename T = double>
class Rosenbrock4 : public BaseBoostIntegrator<typename rb4_type<T>::t, n, interpolator_t, T> {
 public:
  Rosenbrock4(const T initial_stepsize = 0.01, const T absolute_error_tolerance = 1e-10,
              const T relative_error_tolerance = 1e-6, const T a_x = 1, const T a_dxdt = 1)
      : BaseBoostIntegrator<typename rb4_type<T>::t, n, interpolator_t, T>(
            typename rb4_type<T>::t(                              //
                ::boost::numeric::odeint::default_error_checker<  //
                    T,                                            //
                    ::boost::numeric::odeint::range_algebra,      //
                    ::boost::numeric::odeint::default_operations  //
                    >(absolute_error_tolerance, relative_error_tolerance, a_x, a_dxdt)),
            initial_stepsize) {}
};

}  // namespace boost
}  // namespace integration

template <unsigned n, template <typename, typename> typename interpolator_t = HermiteInterpolator, class T = double>
auto make_rb4_integrator(const T initial_stepsize = 0.01, const T absolute_error_tolerance = 1e-10,
                         const T relative_error_tolerance = 1e-6, const T a_x = 1, const T a_dxdt = 1) {
  return integration::boost::Rosenbrock4<n, interpolator_t, T>(initial_stepsize, absolute_error_tolerance,
                                                               relative_error_tolerance, a_x, a_dxdt);
}
}  // namespace tatooine

#endif