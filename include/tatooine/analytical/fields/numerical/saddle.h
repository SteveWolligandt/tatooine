#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_SADDLE_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_SADDLE_H
//==============================================================================
#include <tatooine/differentiated_field.h>
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
template <typename Real>
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
      -> tensor_t final {
    return tensor_t{-x(0), x(1)};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(pos_t const& /*x*/,
                                         Real const /*t*/) const -> bool final {
    return true;
  }
};
//==============================================================================
saddle()->saddle<double>;
//==============================================================================
template <typename Real>
struct saddle_flowmap {
  using real_t = Real;
  using vec_t  = vec<Real, 2>;
  using pos_t  = vec_t;
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
//==============================================================================
template <typename Real>
struct saddle_flowmap_gradient {
  using real_t = Real;
  using vec_t  = vec<Real, 2>;
  using pos_t  = vec_t;
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& /*x*/, Real const /*t*/,
                          Real const tau) const {
    return mat<real_t, 2, 2>{{std::exp(-tau), real_t(0)},
                             {real_t(0), std::exp(tau)}};
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(pos_t const& x, Real const t,
                            Real const tau) const {
    return evaluate(x, t, tau);
  }
};
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct differentiated_field<analytical::fields::numerical::saddle<Real>, 2, 2>
    : field<analytical::fields::numerical::saddle<Real>, Real, 2, 2, 2> {
  using this_t =
      differentiated_field<analytical::fields::numerical::saddle<Real>>;
  using parent_t = field<this_t, Real, 2, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
 public:
  constexpr auto evaluate(pos_t const& /*x*/, Real const /*t*/) const
      -> tensor_t final {
    return {{-1, 0}, {0, 1}};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& /*x*/, Real const /*t*/) const
      -> bool final {
    return true;
  }
};
//==============================================================================
template <typename Real>
constexpr auto flowmap(analytical::fields::numerical::saddle<Real> const&) {
  return typename analytical::fields::numerical::saddle_flowmap<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real>
constexpr auto flowmap(
    vectorfield<analytical::fields::numerical::saddle<Real>, Real, 2> const&) {
  return typename analytical::fields::numerical::saddle_flowmap<Real>{};
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto diff(
    analytical::fields::numerical::saddle_flowmap<Real> const&) {
  return
      typename analytical::fields::numerical::saddle_flowmap_gradient<Real>{};
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto diff(analytical::fields::numerical::saddle<Real>&) {
  return differentiated_field<analytical::fields::numerical::saddle<Real>, 2,
                              2>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real>
constexpr auto diff(
    vectorfield<analytical::fields::numerical::saddle<Real>, Real, 2> const&) {
  return differentiated_field<analytical::fields::numerical::saddle<Real>, 2,
                              2>{};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
