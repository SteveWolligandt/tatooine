#ifndef TATOOINE_SADDLE_H
#define TATOOINE_SADDLE_H
//==============================================================================
#include "field.h"
//==============================================================================
namespace tatooine::numerical {
//==============================================================================
template <typename Real>
struct saddle : vectorfield<saddle<Real>, Real, 2> {
  using this_t   = saddle<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  constexpr saddle() noexcept {}
  constexpr saddle(const saddle&)     = default;
  constexpr saddle(saddle&&) noexcept = default;
  constexpr auto operator=(const saddle&) -> saddle& = default;
  constexpr auto operator=(saddle&&) noexcept -> saddle& = default;
  //----------------------------------------------------------------------------
  ~saddle() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(const pos_t& x, Real /*t*/) const
      -> tensor_t final {
    return tensor_t{-x(0), x(1)};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(const pos_t& /*x*/, Real /*t*/) const
      -> bool final {
    return true;
  }
};
saddle()->saddle<double>;
//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#endif
