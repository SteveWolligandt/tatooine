#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_CENTER_FIELD_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_CENTER_FIELD_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
template <typename Real>
struct center_field : vectorfield<center_field<Real>, Real, 2> {
  using this_t   = center_field<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  struct flowmap_t {
    constexpr auto evaluate(pos_t const& x, Real const /*t*/,
                            Real const   tau) const -> pos_t {
      return {std::cos(tau) * x(0) + std::sin(tau) * x(1),
              -std::sin(tau) * x(0) + std::cos(tau) * x(1)};
    }
    //--------------------------------------------------------------------------
    constexpr auto operator()(pos_t const& x, Real const t,
                              Real const tau) const -> pos_t {
      return evaluate(x, t, tau);
    }
  };
  //============================================================================
  constexpr center_field() noexcept {}
  constexpr center_field(center_field const&)     = default;
  constexpr center_field(center_field&&) noexcept = default;
  constexpr auto operator=(center_field const&) -> center_field& = default;
  constexpr auto operator=(center_field&&) noexcept -> center_field& = default;
  //----------------------------------------------------------------------------
  ~center_field() override = default;
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
  //----------------------------------------------------------------------------
  auto flowmap() const { return flowmap_t{}; }
};
center_field()->center_field<double>;
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
#include "diff.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct has_analytical_flowmap<numerical::center_field<Real>> : std::true_type {
};
//==============================================================================
template <typename Real>
struct derived_field<numerical::center_field<Real>>
    : field<numerical::center_field<Real>, Real, 2, 2, 2> {
  using this_t   = derived_field<numerical::center_field<Real>>;
  using parent_t = field<this_t, Real, 2, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
 private:
  numerical::center_field<Real> m_internal_field;

  //============================================================================
 public:
  derived_field(numerical::center_field<Real> const& f)
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
}  // namespace tatooine
//==============================================================================
#endif
