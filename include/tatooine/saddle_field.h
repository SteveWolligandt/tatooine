#ifndef TATOOINE_SADDLE_FIELD_H
#define TATOOINE_SADDLE_FIELD_H
//==============================================================================
#include "field.h"
//==============================================================================
namespace tatooine::numerical {
//==============================================================================
template <typename Real>
struct saddle_field : vectorfield<saddle_field<Real>, Real, 2> {
  using this_t   = saddle_field<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
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
saddle_field()->saddle_field<double>;
//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#include "diff.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct derived_field<numerical::saddle_field<Real>>
    : field<numerical::saddle_field<Real>, Real, 2, 2, 2> {
  using this_t   = derived_field<numerical::saddle_field<Real>>;
  using parent_t = field<this_t, Real, 2, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
 private:
  numerical::saddle_field<Real> m_internal_field;

  //============================================================================
 public:
  derived_field(numerical::saddle_field<Real> const& f)
      : m_internal_field{f.as_derived()} {}
  //----------------------------------------------------------------------------
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
