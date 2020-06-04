#ifndef TATOOINE_CIRCLE_FIELD_H
#define TATOOINE_CIRCLE_FIELD_H
//==============================================================================
#include "field.h"
//==============================================================================
namespace tatooine::numerical {
//==============================================================================
template <typename Real>
struct circle_field : vectorfield<circle_field<Real>, Real, 2> {
  using this_t   = circle_field<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  constexpr circle_field() noexcept {}
  constexpr circle_field(const circle_field&)     = default;
  constexpr circle_field(circle_field&&) noexcept = default;
  constexpr auto operator=(const circle_field&) -> circle_field& = default;
  constexpr auto operator=(circle_field&&) noexcept -> circle_field& = default;
  //----------------------------------------------------------------------------
  ~circle_field() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(const pos_t& x, Real /*t*/) const
      -> tensor_t final {
    return {x(1), -x(0)};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(const pos_t& /*x*/, Real /*t*/) const
      -> bool final {
    return true;
  }
};
circle_field()->circle_field<double>;
//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#include "diff.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct derived_field<numerical::circle_field<Real>>
    : field<numerical::circle_field<Real>, Real, 2, 2, 2> {
  using this_t   = derived_field<numerical::circle_field<Real>>;
  using parent_t = field<this_t, Real, 2, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
 private:
  numerical::circle_field<Real> m_internal_field;

  //============================================================================
 public:
  derived_field(const numerical::circle_field<Real>& f)
      : m_internal_field{f.as_derived()} {}
  //----------------------------------------------------------------------------
  constexpr auto evaluate(const pos_t& /*x*/, const Real /*t*/) const
      -> tensor_t final {
    return {{0, 1}, {-1, 0}};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(const pos_t& /*x*/, Real /*t*/) const -> bool final {
    return true;
  }
  //----------------------------------------------------------------------------
  constexpr auto internal_field() -> auto& { return m_internal_field; }
  constexpr auto internal_field() const -> const auto& {
    return m_internal_field;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
