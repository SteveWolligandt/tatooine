#ifndef TATOOINE_BASE_BOOST_INTEGRATOR_H
#define TATOOINE_BASE_BOOST_INTEGRATOR_H

#include "../../line.h"
#include "../integrator.h"

#include <boost/numeric/odeint.hpp>
#include "../../tensor.h"


//==============================================================================
namespace boost::numeric::odeint {
//==============================================================================
template <typename Real, size_t N>
struct is_resizeable<tatooine::vec<Real, N>> {
  using type              = boost::false_type;
  static const bool value = type::value;
};
//==============================================================================
}  // namespace boost::numeric::odeint
//==============================================================================
namespace tatooine::integration::boost {
//==============================================================================

template <typename Real, size_t N, typename Stepper>
struct integrator
    : tatooine::integration::integrator<Real, N, integrator<Real, N, Stepper>> {
 public:
  using this_t     = integrator<Real, N, Stepper>;
  using parent_t   = tatooine::integration::integrator<Real, N, this_t>;
  using pos_t      = typename parent_t::pos_t;
  using integral_t = typename parent_t::integral_t;

  //============================================================================
  integrator(const Stepper &stepper, const Real stepsize)
      : m_stepper{stepper}, m_stepsize{stepsize} {}
  integrator(Stepper &&stepper, const Real stepsize)
      : m_stepper{std::move(stepper)}, m_stepsize{stepsize} {}

 private:
  friend parent_t;

  template <typename V>
  auto &calc_forward(const V &v, integral_t &integral, const pos_t &x, Real t0,
             Real tau) const {
    pos_t x_copy(x);

    ::boost::numeric::odeint::integrate_adaptive(
        m_stepper,
        [&v, tau, t0](const pos_t &x, pos_t &sample, Real t) {
          if (tau >= 0) {
            sample = v(x, t);
          } else {
            sample = v(x, 2 * t0 - t);
            sample = -sample;
          }
        },
        x_copy, t0, t0 + std::abs(tau), m_stepsize,
        [&integral, tau, t0](const pos_t &x, Real t) {
          if (tau >= 0) {
            integral.push_back(x, t);
          } else {
            integral.push_back(x, 2 * t0 - t);
          }
        });
    return integral;
  }

  //----------------------------------------------------------------------------
  template <typename V>
  auto &calc_backward(const V &v, integral_t &integral, const pos_t &x, Real t0,
             Real tau) const {
    pos_t x_copy(x);

    ::boost::numeric::odeint::integrate_adaptive(
        m_stepper,
        [&v, tau, t0](const pos_t &x, pos_t &sample, Real t) {
          if (tau >= 0)
            sample = v(x, t);
          else {
            sample = v(x, 2 * t0 - t);
            sample = -sample;
          }
        },
        x_copy, t0, t0 + std::abs(tau), m_stepsize,
        [&integral, tau, t0](const pos_t &x, Real t) {
          if (tau >= 0)
            integral.push_front(x, t);
          else
            integral.push_front(x, 2 * t0 - t);
        });
    return integral;
  }

  //----------------------------------------------------------------------------

  Real &      stepsize() { return m_stepsize; }
  const Real &stepsize() const { return m_stepsize; }
  void          set_stepsize(const Real stepsize) { m_stepsize = stepsize; }

 protected:
  //============================================================================
  Stepper m_stepper;
  Real    m_stepsize;
};

//==============================================================================
}  // namespace tatooine::integration::boost
//==============================================================================


#endif
