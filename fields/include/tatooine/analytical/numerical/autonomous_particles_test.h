#ifndef TATOOINE_AUTONOMOUS_PARTICLES_FIELD_H
#define TATOOINE_AUTONOMOUS_PARTICLES_FIELD_H
//==============================================================================
#include <tatooine/differentiated_field.h>
#include <tatooine/field.h>
#include <tatooine/flowmap_gradient_central_differences.h>
#include <tatooine/numerical_flowmap.h>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
template <std::floating_point Real>
struct autonomous_particles_test
    : vectorfield<autonomous_particles_test<Real>, Real, 2> {
  using this_type   = autonomous_particles_test<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::real_type;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
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
  [[nodiscard]] constexpr auto evaluate(pos_type const& x, real_type const /*t*/) const
      -> tensor_type final {
    return {x(0) * x(0) * x(0) - x(0),
            (1 - 3 * x(0) * x(0)) * x(1)};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(pos_type const& /*x*/,
                                         real_type const /*t*/) const -> bool final {
    return true;
  }
};
//==============================================================================
autonomous_particles_test()->autonomous_particles_test<double>;
//==============================================================================
template <std::floating_point Real>
struct autonomous_particles_test_flowmap {
  using real_type = Real;
  using vec_t  = vec<real_type, 2>;
  using pos_type  = vec_t;
  static constexpr auto num_dimensions() -> std::size_t { return 2; }
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_type const& x, real_type const /*t*/,
                          real_type const   tau) const -> pos_type {
    auto const a = std::exp(2 * tau);
    auto const b = std::exp(-2 * tau);
    auto const c = std::sqrt((1 - a) * x(0) * x(0) + a);
    return {x(0) / c, -b * c * ((a - 1) * x(0) * x(0) - a) * x(1)

    };
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(pos_type const& x, real_type const t, real_type const tau) const
      -> pos_type {
    return evaluate(x, t, tau);
  }
};
//------------------------------------------------------------------------------
template <
    template <typename, size_t> typename ODESolver = ode::vclibs::rungekutta43,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    std::floating_point Real>
constexpr auto flowmap(
    vectorfield<autonomous_particles_test<Real>, Real, 2> const& v,
    tag::numerical_t /*tag*/) {
  return numerical_flowmap<autonomous_particles_test<Real>, ODESolver,
                           InterpolationKernel>{v};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
constexpr auto flowmap(
    vectorfield<autonomous_particles_test<Real>, Real, 2> const&,
    tag::analytical_t /*tag*/) {
  return autonomous_particles_test_flowmap<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
constexpr auto flowmap(
    vectorfield<autonomous_particles_test<Real>, Real, 2> const& v) {
  return flowmap(v, tag::analytical);
}
//==============================================================================
template <std::floating_point Real>
struct autonomous_particles_test_flowmap_gradient {
  using real_type     = Real;
  using vec_t      = vec<real_type, 2>;
  using pos_type      = vec_t;
  using mat_t      = mat<real_type, 2, 2>;
  using gradient_t = mat_t;
  static constexpr auto num_dimensions() -> std::size_t { return 2; }
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_type const& x, real_type const /*t*/,
                          real_type const   tau) const -> gradient_t {
    auto const a = std::exp(2 * tau);
    auto const b = std::exp(-2 * tau);
    auto const c = std::sqrt(-a * x(0) * x(0) + x(0) * x(0) + a);
    auto const d =
        std::pow((-a * x(0) * x(0) + x(0) * x(0) + a), real_type(3) / real_type(2));
    return {{1 / c - (x(0) * (2 * x(0) - 2 * a * x(0))) / (2 * d), real_type(0)},
            {(3 * b * (2 * x(0) - 2 * a * x(0)) * c * x(1)) / 2  , b * d}};
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(pos_type const& x, Real const t,
                            real_type const tau) const {
    return evaluate(x, t, tau);
  }
};
//------------------------------------------------------------------------------
template <std::floating_point Real>
auto diff(autonomous_particles_test_flowmap<Real> const&,
          tag::analytical_t /*tag*/) {
  return autonomous_particles_test_flowmap_gradient<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point VReal, std::floating_point EpsReal = VReal>
auto diff(autonomous_particles_test_flowmap<VReal> const& flowmap,
          tag::central_t /*tag*/, EpsReal epsilon = 1e-7) {
  return flowmap_gradient_central_differences<
      autonomous_particles_test_flowmap<VReal>>{flowmap, epsilon};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point VReal, std::floating_point EpsReal>
constexpr auto diff(autonomous_particles_test_flowmap<VReal> const& flowmap,
                    tag::central_t /*tag*/, vec<EpsReal, 2> epsilon) {
  return flowmap_gradient_central_differences<
      autonomous_particles_test_flowmap<VReal>>{flowmap, epsilon};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
auto diff(autonomous_particles_test_flowmap<Real> const& flowmap) {
  return diff(flowmap, tag::analytical);
}
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
namespace tatooine {
//==============================================================================
template <std::floating_point Real>
struct differentiated_field<
    analytical::numerical::autonomous_particles_test<Real>, 2, 2>
    : field<analytical::numerical::autonomous_particles_test<Real>,
            Real, 2, 2, 2> {
  using this_type = differentiated_field<
      analytical::numerical::autonomous_particles_test<Real>>;
  using parent_type = field<this_type, Real, 2, 2, 2>;
  using typename parent_type::real_type;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;

  //============================================================================
 public:
  constexpr auto evaluate(pos_type const& x, real_type const /*t*/) const
      -> tensor_type final {
    return {{3 * x(0) * x(0) - 1, 0                  },
            {-6 * x(0) * x(1)   , 1 - 3 * x(0) * x(0)}};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_type const& /*x*/, real_type const /*t*/) const
      -> bool final {
    return true;
  }
};
//------------------------------------------------------------------------------
template <std::floating_point Real>
constexpr auto diff(
    analytical::numerical::autonomous_particles_test<Real>&) {
  return differentiated_field<
      analytical::numerical::autonomous_particles_test<Real>, 2, 2>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
constexpr auto diff(
    vectorfield<analytical::numerical::autonomous_particles_test<Real>,
                Real, 2> const&) {
  return differentiated_field<
      analytical::numerical::autonomous_particles_test<Real>, 2, 2>{};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
