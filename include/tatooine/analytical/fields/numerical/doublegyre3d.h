#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_DOUBLEGYRE3D_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_DOUBLEGYRE3D_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
template <floating_point Real>
struct doublegyre3d : vectorfield<doublegyre3d<Real>, Real, 3> {
  using this_t   = doublegyre3d<Real>;
  using parent_t = vectorfield<this_t, Real, 3>;
  using real_t   = typename parent_t::real_t;
  using pos_t    = typename parent_t::pos_t;
  using tensor_t = typename parent_t::tensor_t;
  //============================================================================
  real_t m_eps = 0.25, m_omega = 2.0 * M_PI / 10.0, m_A = 0.1;
  //============================================================================
  virtual ~doublegyre3d() = default;
  //============================================================================
  auto evaluate(pos_t const& p, real_t const t) const -> tensor_t final {
    real_t const a = m_eps * sin(m_omega * t);
    real_t const b = 1.0 - 2.0 * a;
    real_t const f = a * p.x() * p.x() + b * p.x();
    return {-M_PI * m_A * std::sin(M_PI * f) * std::cos(M_PI * p.y()),
            M_PI * m_A * std::cos(M_PI * f) * std::sin(M_PI * p.y()) *
                (2.0 * a * p.x() + b),
            m_omega / M_PI * p.z() * (1.0 - p.z()) *
                (p.z() - 0.5 - m_eps * sin(2.0 * m_omega * t))};
  }
  //----------------------------------------------------------------------------
  auto in_domain(pos_t const&, real_t const) const -> bool final {
    return true;
  }
};
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
#endif
