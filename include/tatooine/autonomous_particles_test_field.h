#ifndef TATOOINE_AUTONOMOUS_PARTICLES_TEST_FIELD_H
#define TATOOINE_AUTONOMOUS_PARTICLES_TEST_FIELD_H
//==============================================================================
#include "field.h"
//==============================================================================
namespace tatooine::numerical {
//==============================================================================
template <typename Real>
struct autonomous_particles_test_field
    : vectorfield<autonomous_particles_test_field<Real>, Real, 2> {
  using this_t   = autonomous_particles_test_field<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  struct flowmap_t {
    using this_t   = flowmap_t;
    using parent_t = vectorfield<this_t, Real, 2>;
    //==========================================================================
    constexpr auto evaluate(pos_t const& x, Real const /*t*/,
                            Real const   tau) const -> pos_t {
      return {x(0) / std::sqrt(-std::exp(2 * tau) * x(0) * x(0) + x(0) * x(0) +
                               std::exp(2 * tau)),
              std::exp(-2 * tau) *
                  std::pow(-std::exp(2 * tau) * x(0) * x(0) + x(0) * x(0) +
                               std::exp(2 * tau),
                           Real(3) / Real(2)) *
                  x(1)};
    }
    //--------------------------------------------------------------------------
    constexpr auto operator()(pos_t const& x, Real const t,
                              Real const tau) const -> pos_t {
      return evaluate(x, t, tau);
    }
  };
  //----------------------------------------------------------------------------
  constexpr auto flowmap() const { return flowmap_t{}; }
  //============================================================================
  constexpr autonomous_particles_test_field() noexcept {}
  constexpr autonomous_particles_test_field(
      autonomous_particles_test_field const&) = default;
  constexpr autonomous_particles_test_field(
      autonomous_particles_test_field&&) noexcept = default;
  constexpr auto operator=(autonomous_particles_test_field const&)
      -> autonomous_particles_test_field& = default;
  constexpr auto operator=(autonomous_particles_test_field&&) noexcept
      -> autonomous_particles_test_field& = default;
  //----------------------------------------------------------------------------
  ~autonomous_particles_test_field() override = default;
  //==============================================================================
  constexpr auto evaluate(pos_t const& x, Real const /*t*/) const
      -> tensor_t final {
    return {x(0) * x(0) * x(0) - x(0), (1 - 3 * x(0) * x(0)) * x(1)};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& /*x*/, Real const /*t*/) const
      -> bool final {
    return true;
  }
};
autonomous_particles_test_field()->autonomous_particles_test_field<double>;
//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct has_analytical_flowmap<numerical::autonomous_particles_test_field<Real>>
    : std::true_type {};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
