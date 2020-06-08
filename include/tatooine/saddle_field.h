#ifndef TATOOINE_SADDLE_FIELD_H
#define TATOOINE_SADDLE_FIELD_H
//==============================================================================
#include "field.h"
#include "differentiated_field.h"
//==============================================================================
namespace tatooine::numerical {
//==============================================================================
template <typename Real>
struct saddle_field : vectorfield<saddle_field<Real>, Real, 2> {
  using this_t   = saddle_field<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //==============================================================================
  constexpr saddle_field() noexcept {}
  constexpr saddle_field(saddle_field const&)     = default;
  constexpr saddle_field(saddle_field&&) noexcept = default;
  constexpr auto operator=(saddle_field const&) -> saddle_field& = default;
  constexpr auto operator=(saddle_field&&) noexcept -> saddle_field& = default;
  //----------------------------------------------------------------------------
  ~saddle_field() override = default;
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
saddle_field()->saddle_field<double>;
//============================================================================
template <typename Real>
struct saddle_field_flowmap {
  using real_t = Real;
  using vec_t  = vec<Real, 2>;
  using pos_t  = vec_t;
  static constexpr auto num_dimensions() { return 2; }
  //--------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& x, Real const /*t*/,
                          Real const   tau) const -> pos_t {
        std::cerr << "anal\n";
    return {std::exp(-tau) * x(0), std::exp(tau) * x(1)};
  }
  //--------------------------------------------------------------------------
  constexpr auto operator()(pos_t const& x, Real const t, Real const tau) const
      -> pos_t {
    return evaluate(x, t, tau);
  }
};
//============================================================================
template <typename Real>
struct saddle_field_flowmap_gradient {
  using real_t = Real;
  using vec_t  = vec<Real, 2>;
  using pos_t  = vec_t;
  static constexpr auto num_dimensions() { return 2; }
  //--------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& /*x*/, Real const /*t*/,
                          Real const tau) const {
    return mat<real_t, 2, 2>{{std::exp(-tau), real_t(0)},
                             {real_t(0), std::exp(-tau)}};
  }
  //--------------------------------------------------------------------------
  constexpr auto operator()(pos_t const& x, Real const t,
                            Real const tau) const {
    return evaluate(x, t, tau);
  }
};
//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct differentiated_field<numerical::saddle_field<Real>, 2, 2>
    : field<numerical::saddle_field<Real>, Real, 2, 2, 2> {
  using this_t   = differentiated_field<numerical::saddle_field<Real>>;
  using parent_t = field<this_t, Real, 2, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
 public:
  constexpr auto evaluate(pos_t const& /*x*/, Real const /*t*/) const
      -> tensor_t final {
    return {{-1, 0},
            { 0, 1}};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& /*x*/, Real const /*t*/) const
      -> bool final {
    return true;
  }
};
//==============================================================================
template <typename Real>
constexpr auto flowmap(numerical::saddle_field<Real> const&) {
  return typename numerical::saddle_field_flowmap<Real>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real>
constexpr auto flowmap(vectorfield<numerical::saddle_field<Real>, Real, 2> const&) {
  std::cerr << "flowmap\n";
  return typename numerical::saddle_field_flowmap<Real>{};
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto diff(numerical::saddle_field_flowmap<Real> const&) {
  return typename numerical::saddle_field_flowmap_gradient<Real>{};
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto diff(numerical::saddle_field<Real>&) {
  return differentiated_field<numerical::saddle_field<Real>, 2, 2>{};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real>
constexpr auto diff(
    vectorfield<numerical::saddle_field<Real>, Real, 2> const&) {
  return differentiated_field<numerical::saddle_field<Real>, 2, 2>{};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
