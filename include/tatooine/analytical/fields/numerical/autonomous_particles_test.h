#ifndef TATOOINE_AUTONOMOUS_PARTICLES_FIELD_H
#define TATOOINE_AUTONOMOUS_PARTICLES_FIELD_H
//==============================================================================
#include <tatooine/differentiated_field.h>
#include <tatooine/field.h>
#include <tatooine/flowmap_gradient_central_differences.h>
#include <tatooine/numerical_flowmap.h>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
template <std::floating_point Real>
struct autonomous_particles_test
    : vectorfield<autonomous_particles_test<Real>, Real, 2> {
  using this_t   = autonomous_particles_test<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  using typename parent_t::real_t;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  constexpr autonomous_particles_test() noexcept {}
  constexpr autonomous_particles_test(autonomous_particles_test const&) =
      default;
  constexpr autonomous_particles_test(autonomous_particles_test&&) noexcept =
      default;
  constexpr auto operator           =(autonomous_particles_test const&)
      -> autonomous_particles_test& = default;
  constexpr auto operator           =(autonomous_particles_test&&) noexcept
      -> autonomous_particles_test& = default;
  //----------------------------------------------------------------------------
  ~autonomous_particles_test() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(pos_t const& x, real_t const /*t*/) const
      -> tensor_t final {
    return {x(0) * x(0) * x(0) - x(0),
            (1 - 3 * x(0) * x(0)) * x(1)};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(pos_t const& /*x*/,
                                         real_t const /*t*/) const -> bool final {
    return true;
  }
};
//==============================================================================
autonomous_particles_test()->autonomous_particles_test<double>;
//==============================================================================
template <typename Real>
struct autonomous_particles_test_flowmap {
  using real_t = Real;
  using vec_t  = vec<real_t, 2>;
  using pos_t  = vec_t;
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& x, real_t const /*t*/,
                          real_t const   tau) const -> pos_t {
    return {x(0) / std::sqrt(-std::exp(2 * tau) * x(0) * x(0) + x(0) * x(0) +
                             std::exp(2 * tau)),
            std::exp(-2 * tau) *
                std::pow(-std::exp(2 * tau) * x(0) * x(0) + x(0) * x(0) +
                             std::exp(2 * tau),
                         real_t(3) / real_t(2)) *
                x(1)};
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(pos_t const& x, real_t const t, real_t const tau) const
      -> pos_t {
    return evaluate(x, t, tau);
  }
};
//------------------------------------------------------------------------------
template <
    template <typename, size_t> typename ODESolver = ode::vclibs::rungekutta43,
    template <typename> typename InterpolationKernel = interpolation::hermite,
    std::floating_point Real>
constexpr auto flowmap(
    vectorfield<analytical::fields::numerical::autonomous_particles_test<Real>,
                Real, 2> const& v,
    tag::numerical_t /*tag*/) {
  return numerical_flowmap<
      analytical::fields::numerical::autonomous_particles_test<Real>, ODESolver,
      InterpolationKernel>{v};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
constexpr auto flowmap(
    vectorfield<analytical::fields::numerical::autonomous_particles_test<Real>,
                Real, 2> const&,
    tag::analytical_t /*tag*/) {
  return analytical::fields::numerical::autonomous_particles_test_flowmap<
      Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
constexpr auto flowmap(
    vectorfield<analytical::fields::numerical::autonomous_particles_test<Real>,
                Real, 2> const& v) {
  return flowmap(v, tag::analytical);
}
//==============================================================================
template <std::floating_point Real>
struct autonomous_particles_test_flowmap_gradient {
  using real_t     = Real;
  using vec_t      = vec<real_t, 2>;
  using pos_t      = vec_t;
  using mat_t      = mat<real_t, 2, 2>;
  using gradient_t = mat_t;
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& x, real_t const /*t*/,
                          real_t const   tau) const -> gradient_t {
    auto const a = std::exp(2 * tau);
    auto const b = std::exp(-2 * tau);
    auto const c = std::sqrt(-a * x(0) * x(0) + x(0) * x(0) + a);
    auto const d =
        std::pow((-a * x(0) * x(0) + x(0) * x(0) + a), real_t(3) / real_t(2));
    return {{1 / c - (x(0) * (2 * x(0) - 2 * a * x(0))) / (2 * d), real_t(0)},
            {(3 * b * (2 * x(0) - 2 * a * x(0)) * c * x(1)) / 2  , b * d}};
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(pos_t const& x, Real const t,
                            real_t const tau) const {
    return evaluate(x, t, tau);
  }
};
//------------------------------------------------------------------------------
template <std::floating_point Real>
auto diff(analytical::fields::numerical::autonomous_particles_test_flowmap<
              Real> const&,
          tag::analytical_t /*tag*/) {
  return typename analytical::fields::numerical::
      autonomous_particles_test_flowmap_gradient<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point VReal, std::floating_point EpsReal = VReal>
auto diff(analytical::fields::numerical::autonomous_particles_test_flowmap<
              VReal> const& flowmap,
          tag::central_t /*tag*/, EpsReal epsilon = 1e-7) {
  return flowmap_gradient_central_differences<
      analytical::fields::numerical::autonomous_particles_test_flowmap<VReal>>{
      flowmap, epsilon};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point VReal, std::floating_point EpsReal>
constexpr auto diff(analytical::fields::numerical::
                        autonomous_particles_test_flowmap<VReal> const& flowmap,
                    tag::central_t /*tag*/, vec<EpsReal, 2> epsilon) {
  return flowmap_gradient_central_differences<
      analytical::fields::numerical::autonomous_particles_test_flowmap<VReal>>{
      flowmap, epsilon};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
auto diff(analytical::fields::numerical::autonomous_particles_test_flowmap<
          Real> const& flowmap) {
  return diff(flowmap, tag::analytical);
}
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
namespace tatooine {
//==============================================================================
template <std::floating_point Real>
struct differentiated_field<
    analytical::fields::numerical::autonomous_particles_test<Real>, 2, 2>
    : field<analytical::fields::numerical::autonomous_particles_test<Real>,
            Real, 2, 2, 2> {
  using this_t = differentiated_field<
      analytical::fields::numerical::autonomous_particles_test<Real>>;
  using parent_t = field<this_t, Real, 2, 2, 2>;
  using typename parent_t::real_t;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
 public:
  constexpr auto evaluate(pos_t const& x, real_t const /*t*/) const
      -> tensor_t final {
    return {{3 * x(0) * x(0) - 1, 0},
            {-6 * x(0) * x(1)   , 1 - 3 * x(0) * x(0)}};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& /*x*/, real_t const /*t*/) const
      -> bool final {
    return true;
  }
};
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto diff(
    analytical::fields::numerical::autonomous_particles_test<Real>&) {
  return differentiated_field<
      analytical::fields::numerical::autonomous_particles_test<Real>, 2, 2>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real>
constexpr auto diff(
    vectorfield<analytical::fields::numerical::autonomous_particles_test<Real>,
                Real, 2> const&) {
  return differentiated_field<
      analytical::fields::numerical::autonomous_particles_test<Real>, 2, 2>{};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
