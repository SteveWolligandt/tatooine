#ifndef TATOOINE_FIELDS_ANALYTICAL_NUMERICAL_HARMONIC_OSCILATOR_H
#define TATOOINE_FIELDS_ANALYTICAL_NUMERICAL_HARMONIC_OSCILATOR_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
/// Harmonic Oscilator
template <typename Real>
struct harmonic_oscilator : vectorfield<harmonic_oscilator<Real>, Real, 2> {
  using this_type   = harmonic_oscilator<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::tensor_type;
  //============================================================================
  Real m_gamma;
  //============================================================================
  explicit constexpr harmonic_oscilator(Real const gamma = 0.15) noexcept
      : m_gamma{gamma} {}
  //------------------------------------------------------------------------------
  constexpr harmonic_oscilator(harmonic_oscilator const&)     = default;
  constexpr harmonic_oscilator(harmonic_oscilator&&) noexcept = default;
  //------------------------------------------------------------------------------
  constexpr auto operator=(harmonic_oscilator const&)
      -> harmonic_oscilator& = default;
  constexpr auto operator=(harmonic_oscilator&&) noexcept
      -> harmonic_oscilator& = default;
  //------------------------------------------------------------------------------
  virtual ~harmonic_oscilator() = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(fixed_size_vec<2> auto const& p,
                                        Real const t) const -> tensor_type {
    return tensor_type{p.y(), -p.x() - m_gamma * p.y()};
  }
};
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
#endif
