#ifndef TATOOINE_ANALYTICAL_NUMERICAL_DOUBLEGYRE3D_H
#define TATOOINE_ANALYTICAL_NUMERICAL_DOUBLEGYRE3D_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
template <floating_point Real>
struct doublegyre3d : vectorfield<doublegyre3d<Real>, Real, 3> {
  using this_type   = doublegyre3d<Real>;
  using parent_type = vectorfield<this_type, Real, 3>;
  using real_type   = typename parent_type::real_type;
  using pos_type    = typename parent_type::pos_type;
  using tensor_type = typename parent_type::tensor_type;
  //============================================================================
  real_type m_eps   = real_type(1) / real_type(4);
  real_type m_omega = 2 * M_PI / 10;
  real_type m_A     = real_type(1) / real_type(10);
  //============================================================================
  virtual ~doublegyre3d() = default;
  //============================================================================
  auto evaluate(pos_type const& p, real_type const t) const -> tensor_type final {
    real_type const a = m_eps * sin(m_omega * t);
    real_type const b = 1.0 - 2.0 * a;
    real_type const f = a * p.x() * p.x() + b * p.x();
    return {-M_PI * m_A * std::sin(M_PI * f) * std::cos(M_PI * p.y()),
             M_PI * m_A * std::cos(M_PI * f) * std::sin(M_PI * p.y()) *
                (2.0 * a * p.x() + b),
            m_omega / M_PI * p.z() * (1.0 - p.z()) *
                (p.z() - 0.5 - m_eps * sin(2.0 * m_omega * t))};
  }
  //----------------------------------------------------------------------------
  auto in_domain(pos_type const&, real_type const) const -> bool final {
    return true;
  }
};
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
#endif
