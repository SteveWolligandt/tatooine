#ifndef TATOOINE_ANALYTICAL_NUMERICAL_CENTER_FIELD_H
#define TATOOINE_ANALYTICAL_NUMERICAL_CENTER_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
template <floating_point Real>
struct center : vectorfield<center<Real>, Real, 2> {
  using this_type   = center<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  //============================================================================
  constexpr center() noexcept {}
  constexpr center(center const&)     = default;
  constexpr center(center&&) noexcept = default;
  constexpr auto operator=(center const&) -> center& = default;
  constexpr auto operator=(center&&) noexcept -> center& = default;
  //----------------------------------------------------------------------------
  ~center() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(pos_type const& x,
                                        Real const /*t*/) const -> tensor_type {
    return {x(1), -x(0)};
  }
};
//============================================================================
center()->center<double>;
//============================================================================
template <floating_point Real>
struct center_flowmap {
  using real_type = Real;
  using vec_t     = vec<Real, 2>;
  using pos_type  = vec_t;
  static constexpr auto num_dimensions() { return 2; }
  //============================================================================
  constexpr auto evaluate(pos_type const& x, Real const /*t*/,
                          Real const      tau) const -> pos_type {
    return {gcem::cos(tau) * x(0) + gcem::sin(tau) * x(1),
            -gcem::sin(tau) * x(0) + gcem::cos(tau) * x(1)};
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(pos_type const& x, Real const t,
                            Real const tau) const -> pos_type {
    return evaluate(x, t, tau);
  }
};
//------------------------------------------------------------------------------
template <
    template <typename, size_t> typename ODESolver = ode::vclibs::rungekutta43,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    floating_point Real>
constexpr auto flowmap(
    vectorfield<analytical::numerical::center<Real>, Real, 2> const& v,
    tag::numerical_t /*tag*/) {
  return numerical_flowmap<analytical::numerical::center<Real>,
                           ODESolver, InterpolationKernel>{v};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <floating_point Real>
constexpr auto flowmap(
    vectorfield<analytical::numerical::center<Real>, Real, 2> const&,
    tag::analytical_t /*tag*/) {
  return analytical::numerical::center_flowmap<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <floating_point Real>
constexpr auto flowmap(vectorfield<analytical::numerical::center<Real>,
                                   Real, 2> const& v) {
  return flowmap(v, tag::analytical);
}
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
#include <tatooine/differentiated_field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <floating_point Real>
struct differentiated_field<analytical::numerical::center<Real>>
    : matrixfield<analytical::numerical::center<Real>, Real, 2> {
  using this_type =
      differentiated_field<analytical::numerical::center<Real>>;
  using parent_type = matrixfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;

  //============================================================================
 private:
  analytical::numerical::center<Real> m_internal_field;

  //============================================================================
 public:
  differentiated_field(analytical::numerical::center<Real> const& f)
      : m_internal_field{f.as_derived()} {}
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_type const& /*x*/, Real const /*t*/) const
      -> tensor_type {
    return {{0, 1}, {-1, 0}};
  }
  //----------------------------------------------------------------------------
  constexpr auto internal_field() -> auto& { return m_internal_field; }
  constexpr auto internal_field() const -> auto const& {
    return m_internal_field;
  }
};
//==============================================================================
template <floating_point Real>
constexpr auto diff(analytical::numerical::center<Real>&) {
  return differentiated_field<analytical::numerical::center<Real>>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <floating_point Real>
constexpr auto diff(
    vectorfield<analytical::numerical::center<Real>, Real, 2> const&) {
  return differentiated_field<analytical::numerical::center<Real>>{};
}
//==============================================================================
template <floating_point Real>
struct differentiated_flowmap<analytical::numerical::center<Real>> {
  using real_type  = Real;
  using vec_t      = vec<Real, 2>;
  using pos_type   = vec_t;
  using mat_t      = mat<real_type, 2, 2>;
  using gradient_t = mat_t;
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_type const& /*x*/, Real const /*t*/,
                          Real const tau) const -> gradient_t {
    auto const ctau = gcem::cos(tau);
    auto const stau = gcem::sin(tau);
    return {{ctau, stau}, {-stau, ctau}};
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(pos_type const& x, Real const t,
                            Real const tau) const {
    return evaluate(x, t, tau);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
