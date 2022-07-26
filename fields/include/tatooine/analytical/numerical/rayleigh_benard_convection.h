#ifndef TATOOINE_ANALYTICAL_NUMERICAL_H
#define TATOOINE_ANALYTICAL_NUMERICAL_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
template <typename Real>
struct rayleigh_benard_convection
    : vectorfield<rayleigh_benard_convection<Real>, Real, 3> {
  using this_type   = rayleigh_benard_convection<Real>;
  using parent_type = vectorfield<this_type, Real, 3>;
  using real_type   = Real;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  //============================================================================
 private:
  Real m_A, m_k;
  //============================================================================
 public:
  rayleigh_benard_convection() : m_A{1}, m_k{1} {}
  //----------------------------------------------------------------------------
  rayleigh_benard_convection(real_type A, real_type k) : m_A{A}, m_k{k} {}
  //----------------------------------------------------------------------------
  rayleigh_benard_convection(rayleigh_benard_convection const&) = default;
  rayleigh_benard_convection(rayleigh_benard_convection&&)noexcept      = default;
  auto operator=(rayleigh_benard_convection const&)
      -> rayleigh_benard_convection<real_type>&  = default;
  auto operator=(rayleigh_benard_convection&&) noexcept
      -> rayleigh_benard_convection<real_type>& = default;
  //============================================================================
  auto evaluate(pos_type const& x, real_type const t) const -> tensor_type override {
    auto xi = x(0) - g(t);
    auto rho = xi * xi + x(1) * x(1);
    return {m_A / m_k * xi * std::sin(m_k * rho) * std::cos(x(2)),
            m_A / m_k * x(1) * std::sin(m_k * rho) * std::cos(x(2)),
            -m_A * std::sin(x(2)) * rho * std::cos(m_k * rho) +
                2 / m_k * std::sin(m_k * rho)};
  }
  //----------------------------------------------------------------------------
  auto in_domain(pos_type const& /*x*/, real_type const /*t*/) const
      -> bool override {
    return true;
  }
  //----------------------------------------------------------------------------
  auto g(real_type const /*t*/) const -> real_type { return 0; }
  //----------------------------------------------------------------------------
  auto k() -> real_type& {
    return m_k;
  }
  auto k() const {
    return m_k;
  }
  auto A() -> real_type& {
    return m_A;
  }
  auto A() const {
    return m_A;
  }
};
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
#endif
