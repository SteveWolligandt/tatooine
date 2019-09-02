#ifndef __TATOOINE_BASE_BOOST_INTEGRATOR_H__
#define __TATOOINE_BASE_BOOST_INTEGRATOR_H__

#include "../../geometry/HermiteInterpolator.h"
#include "../../line.h"
#include "../integrator.h"

#include <boost/numeric/odeint.hpp>
#include "../../vecmat.h"

#include "../../concept_defines.h"

//==============================================================================
namespace tatooine::integration::boost {
//==============================================================================

template <size_t n, typename real_t, typename stepper_t>
struct BoostIntegrator
    : Integrator<n, real_t, BoostIntegrator<n, real_t, stepper_t>> {
 public:
  using this_t     = BoostIntegrator<n, real_t, stepper_t>;
  using parent_t   = Integrator<n, real_t, this_t>;
  using pos_t      = typename parent_t::pos_t;
  using integral_t = typename parent_t::integral_t;

  //============================================================================
  BoostIntegrator(const stepper_t &stepper, const real_t stepsize)
      : m_stepper(stepper), m_stepsize(stepsize) {}
  BoostIntegrator(stepper_t &&stepper, const real_t stepsize)
      : m_stepper(std::move(stepper)), m_stepsize(stepsize) {}

 private:
  friend parent_t;

  template <typename vf_t>
  auto &calc(const vf_t &vf, integral_t &integral, const pos_t &x, real_t t0,
             real_t tau) const {
    pos_t x_copy(x);

    ::boost::numeric::odeint::integrate_adaptive(
        m_stepper,
        [&vf, tau, t0](const pos_t &x, pos_t &sample, real_t t) {
          if (tau >= 0)
            sample = vf(x, t);
          else {
            sample = vf(x, 2 * t0 - t);
            sample = -sample;
          }
        },
        x_copy, t0, t0 + std::abs(tau), m_stepsize,
        [&integral, tau, t0](const pos_t &x, real_t t) {
          if (tau >= 0)
            integral.push_back(x, t);
          else
            integral.push_back(x, 2 * t0 - t);
        });
    return integral;
  }

  //----------------------------------------------------------------------------

  real_t &      stepsize() { return m_stepsize; }
  const real_t &stepsize() const { return m_stepsize; }
  void          set_stepsize(const real_t stepsize) { m_stepsize = stepsize; }

 protected:
  //============================================================================
  stepper_t m_stepper;
  real_t    m_stepsize;
};

//==============================================================================
}  // namespace tatooine::integration::boost
//==============================================================================

#include "../../concept_undefs.h"

#endif
