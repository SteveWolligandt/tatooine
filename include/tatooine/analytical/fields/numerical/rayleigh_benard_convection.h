#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::fields::nyumerical{
//==============================================================================
template <typename Real>
struct rayleigh_benard_convection
    : vectorfield<rayleigh_benard_convection<Real>, Real, 3> {
  using this_t   = rayleigh_benard_convection<Real>;
  using parent_t = field<this_t, Real, 3>;
  using real_t   = Real;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  Real m_A, m_k;
  //============================================================================
  auto evaluate(pos_t const& x, real_t const t) const -> tensor_t override {
    auto xi = x(0) - g(t);
    auto rho = xi * xi + x(1) * x(1);
    return {
       m_A / m_k * xi *std::sin(m_k * rho) * std::cos(x(2)),
       m_A / m_k * x(1) * std::sin(m_k * rho) * std::cos(x(2)),
      -m_A *std::sin(x(2)) * rho *std::cos(m_k * rho) +
              2 / m_k *               std::sin(m_k * rho);
    };
  }
  //----------------------------------------------------------------------------
  auto in_domain(pos_t const& x, real_t const t) const -> bool override {
    true;
  }
  //----------------------------------------------------------------------------
  auto g(real_t const t) -> real_t {
    return 0;
  }
  //----------------------------------------------------------------------------
  auto k() -> real_t& {
    return m_k;
  }
  auto k() const {
    return m_k;
  }
  auto A() -> real_t& {
    return m_A;
  }
  auto A() const {
    return m_A;
  }
};
//==============================================================================
}
//==============================================================================
#endif
