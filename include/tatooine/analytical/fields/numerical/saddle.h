#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_SADDLE_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_SADDLE_H
//==============================================================================
#include <tatooine/differentiated_field.h>
#include <tatooine/field.h>
#include <tatooine/flowmap_gradient_central_differences.h>
#include <tatooine/numerical_flowmap.h>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
template <std::floating_point Real>
struct saddle : vectorfield<saddle<Real>, Real, 2> {
  using this_t   = saddle<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  constexpr saddle() noexcept {}
  constexpr saddle(saddle const&)     = default;
  constexpr saddle(saddle&&) noexcept = default;
  constexpr auto operator=(saddle const&) -> saddle& = default;
  constexpr auto operator=(saddle&&) noexcept -> saddle& = default;
  //----------------------------------------------------------------------------
  ~saddle() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(pos_t const& x, Real const /*t*/) const
      -> tensor_t {
    return tensor_t{-x(0), x(1)};
  }
  ////----------------------------------------------------------------------------
  //[[nodiscard]] constexpr auto in_domain(pos_t const& [>x<],
  //                                       Real const [>t<]) const -> bool final {
  //  return true;
  //}
};
//==============================================================================
saddle()->saddle<double>;
//==============================================================================
template <typename Real>
struct saddle_flowmap {
  using real_t = Real;
  using vec_t  = vec<Real, 2>;
  using pos_t  = vec_t;
  saddle_flowmap() = default;
  saddle_flowmap(saddle<Real> const&) {}
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& x, Real const /*t*/,
                          Real const   tau) const -> pos_t {
    return {std::exp(-tau) * x(0), std::exp(tau) * x(1)};
  }
  //----------------------------------------------------------------------------
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
constexpr auto flowmap(vectorfield<saddle<Real>, Real, 2> const& v,
                       tag::numerical_t /*tag*/) {
  return numerical_flowmap<saddle<Real>, ODESolver, InterpolationKernel>{v};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
constexpr auto flowmap(vectorfield<saddle<Real>, Real, 2> const& /*v*/,
                       tag::analytical_t /*tag*/) {
  return analytical::fields::numerical::saddle_flowmap<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
constexpr auto flowmap(vectorfield<saddle<Real>, Real, 2> const& v) {
  return flowmap(v, tag::analytical);
}
//==============================================================================
template <std::floating_point Real>
struct saddle_flowmap_gradient {
  using real_t     = Real;
  using vec_t      = vec<Real, 2>;
  using pos_t      = vec_t;
  using mat_t      = mat<real_t, 2, 2>;
  using gradient_t = mat_t;
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& /*x*/, Real const /*t*/,
                          Real const tau) const -> gradient_t {
    return {{std::exp(-tau), real_t(0)},
            {real_t(0), std::exp(tau)}};
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(pos_t const& x, Real const t,
                            Real const tau) const {
    return evaluate(x, t, tau);
  }
};
//------------------------------------------------------------------------------
template <std::floating_point Real>
auto diff(saddle_flowmap<Real> const&, tag::analytical_t /*tag*/) {
  return saddle_flowmap_gradient<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <floating_point Real>
auto diff(saddle_flowmap<Real> const& flowmap, tag::central_t /*tag*/,
          Real const                  epsilon) {
  return flowmap_gradient_central_differences<saddle_flowmap<Real>>{flowmap,
                                                                    epsilon};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <floating_point Real>
constexpr auto diff(saddle_flowmap<Real> const& flowmap, tag::central_t /*tag*/,
                    vec<Real, 2>                epsilon) {
  return flowmap_gradient_central_differences<saddle_flowmap<Real>>{flowmap,
                                                                    epsilon};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Real>
auto diff(saddle_flowmap<Real> const& flowmap) {
  return diff(flowmap, tag::analytical);
}
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
namespace tatooine {
//==============================================================================
template <std::floating_point Real>
struct differentiated_field<analytical::fields::numerical::saddle<Real>>
    : matrixfield<analytical::fields::numerical::saddle<Real>, Real, 2> {
  using this_t =
      differentiated_field<analytical::fields::numerical::saddle<Real>>;
  using parent_t = matrixfield<this_t, Real, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
 public:
  constexpr auto evaluate(pos_t const& /*x*/, Real const /*t*/) const
      -> tensor_t {
    return {{-1, 0}, {0, 1}};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& /*x*/, Real const /*t*/) const -> bool {
    return true;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
