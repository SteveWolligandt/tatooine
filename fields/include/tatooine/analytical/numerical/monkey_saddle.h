#ifndef TATOOINE_FIELDS_ANALYTICAL_NUMERICAL_MONKEY_SADDLE_H
#define TATOOINE_FIELDS_ANALYTICAL_NUMERICAL_MONKEY_SADDLE_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
/// Monkey Saddle scalar field.
/// from \cite eberly2010ridges
template <typename Real>
struct monkey_saddle : scalarfield<monkey_saddle<Real>, Real, 2> {
  using this_type   = monkey_saddle<Real>;
  using parent_type = scalarfield<this_type, Real, 2>;
  using typename parent_type::tensor_type;
  //============================================================================
  constexpr monkey_saddle() noexcept = default;
  //------------------------------------------------------------------------------
  constexpr monkey_saddle(monkey_saddle const&)     = default;
  constexpr monkey_saddle(monkey_saddle&&) noexcept = default;
  //------------------------------------------------------------------------------
  constexpr auto operator=(monkey_saddle const&) -> monkey_saddle& = default;
  constexpr auto operator=(monkey_saddle&&) noexcept
      -> monkey_saddle& = default;
  //------------------------------------------------------------------------------
  virtual ~monkey_saddle() = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(fixed_size_vec<2> auto const& p,
                                        Real const t) const -> tensor_type {
    return p.x() * p.x() * p.y();
  }
};
monkey_saddle() -> monkey_saddle<real_number>;
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
#endif
