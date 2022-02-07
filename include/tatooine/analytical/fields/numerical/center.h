#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_CENTER_FIELD_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_CENTER_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
template <typename Real>
struct center : vectorfield<center<Real>, Real, 2> {
  using this_t   = center<Real>;
  using parent_type = vectorfield<this_t, Real, 2>;
  using typename parent_type::pos_t;
  using typename parent_type::tensor_t;
  //============================================================================
  constexpr center() noexcept {}
  constexpr center(center const&)     = default;
  constexpr center(center&&) noexcept = default;
  constexpr auto operator=(center const&) -> center& = default;
  constexpr auto operator=(center&&) noexcept -> center& = default;
  //----------------------------------------------------------------------------
  ~center() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(pos_t const& x, Real const /*t*/) const
      -> tensor_t final {
    return {x(1), -x(0)};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(pos_t const& /*x*/,
                                         Real const /*t*/) const -> bool final {
    return true;
  }
};
  //============================================================================
center()->center<double>;
  //============================================================================
template <typename Real>
struct center_flowmap {
  using real_t = Real;
  using vec_t  = vec<Real, 2>;
  using pos_t  = vec_t;
  static constexpr auto num_dimensions() { return 2; }
  //==============================================================================
  constexpr auto evaluate(pos_t const& x, Real const /*t*/,
                          Real const   tau) const -> pos_t {
    return {std::cos(tau) * x(0) + std::sin(tau) * x(1),
            -std::sin(tau) * x(0) + std::cos(tau) * x(1)};
  }
  //--------------------------------------------------------------------------
  constexpr auto operator()(pos_t const& x, Real const t, Real const tau) const
      -> pos_t {
    return evaluate(x, t, tau);
  }
};
//------------------------------------------------------------------------------
template <
    template <typename, size_t> typename ODESolver = ode::vclibs::rungekutta43,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    std::floating_point Real>
constexpr auto flowmap(
    vectorfield<analytical::fields::numerical::center<Real>, Real, 2> const& v,
    tag::numerical_t /*tag*/) {
  return numerical_flowmap<analytical::fields::numerical::center<Real>,
                           ODESolver, InterpolationKernel>{v};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
constexpr auto flowmap(
    vectorfield<analytical::fields::numerical::center<Real>, Real, 2> const&,
    tag::analytical_t /*tag*/) {
  return analytical::fields::numerical::center_flowmap<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
constexpr auto flowmap(vectorfield<analytical::fields::numerical::center<Real>,
                                   Real, 2> const& v) {
  return flowmap(v, tag::analytical);
}
//==============================================================================
template <std::floating_point Real>
struct center_flowmap_gradient {
  using real_t     = Real;
  using vec_t      = vec<Real, 2>;
  using pos_t      = vec_t;
  using mat_t      = mat<real_t, 2, 2>;
  using gradient_t = mat_t;
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& /*x*/, Real const /*t*/,
                          Real const tau) const -> gradient_t {
    auto const ctau = std::cos(tau);
    auto const stau = std::sin(tau);
    return {{ ctau, stau},
            {-stau, ctau}};
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(pos_t const& x, Real const t,
                            Real const tau) const {
    return evaluate(x, t, tau);
  }
};
//------------------------------------------------------------------------------
template <std::floating_point Real>
auto diff(analytical::fields::numerical::center_flowmap<Real> const&,
          tag::analytical_t /*tag*/) {
  return
      typename analytical::fields::numerical::center_flowmap_gradient<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point VReal, std::floating_point EpsReal = VReal>
auto diff(analytical::fields::numerical::center_flowmap<VReal> const& flowmap,
          tag::central_t /*tag*/, EpsReal epsilon = 1e-7) {
  return flowmap_gradient_central_differences<
      analytical::fields::numerical::center_flowmap<VReal>>{flowmap, epsilon};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point VReal, std::floating_point EpsReal>
constexpr auto diff(
    analytical::fields::numerical::center_flowmap<VReal> const& flowmap,
    tag::central_t /*tag*/, vec<EpsReal, 2> epsilon) {
  return flowmap_gradient_central_differences<
      analytical::fields::numerical::center_flowmap<VReal>>{flowmap, epsilon};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
auto diff(analytical::fields::numerical::center_flowmap<Real> const& flowmap) {
  return diff(flowmap, tag::analytical);
}
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
#include <tatooine/differentiated_field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct differentiated_field<analytical::fields::numerical::center<Real>>
    : matrixfield<analytical::fields::numerical::center<Real>, Real, 2> {
  using this_t   = differentiated_field<analytical::fields::numerical::center<Real>>;
  using parent_type = matrixfield<this_t, Real, 2>;
  using typename parent_type::pos_t;
  using typename parent_type::tensor_t;

  //============================================================================
 private:
  analytical::fields::numerical::center<Real> m_internal_field;

  //============================================================================
 public:
  differentiated_field(analytical::fields::numerical::center<Real> const& f)
      : m_internal_field{f.as_derived()} {}
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& /*x*/, Real const /*t*/) const
      -> tensor_t final {
    return {{ 0, 1},
            {-1, 0}};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& /*x*/, Real const /*t*/) const
      -> bool final {
    return true;
  }
  //----------------------------------------------------------------------------
  constexpr auto internal_field() -> auto& { return m_internal_field; }
  constexpr auto internal_field() const -> auto const& {
    return m_internal_field;
  }
};
//==============================================================================
template <typename Real>
constexpr auto diff(analytical::fields::numerical::center<Real>&) {
  return differentiated_field<analytical::fields::numerical::center<Real>>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real>
constexpr auto diff(
    vectorfield<analytical::fields::numerical::center<Real>, Real, 2> const&) {
  return differentiated_field<analytical::fields::numerical::center<Real>>{};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
