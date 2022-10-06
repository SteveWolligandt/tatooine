#ifndef TATOOINE_ANALYTICAL_NUMERICAL_ABCFLOW_H
#define TATOOINE_ANALYTICAL_NUMERICAL_ABCFLOW_H
//==============================================================================
#include <cmath>
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
/// \brief The Arnold–Beltrami–Childress (ABC) flow is a three-dimensional
///        incompressible velocity field which is an exact solution of Euler's
///        equation.
template <typename real_type>
struct abcflow : vectorfield<abcflow<real_type>, real_type, 3> {
  using this_type   = abcflow<real_type>;
  using parent_type = vectorfield<this_type, real_type, 3>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;

  //============================================================================
 private:
  real_type m_a, m_b, m_c;

  //============================================================================
 public:
  explicit constexpr abcflow(real_type const a = 1, real_type const b = 1,
                             real_type const c = 1)
      : m_a{a}, m_b{b}, m_c{c} {}
  constexpr abcflow(const abcflow& other)            = default;
  constexpr abcflow(abcflow&& other)                 = default;
  constexpr abcflow& operator=(const abcflow& other) = default;
  constexpr abcflow& operator=(abcflow&& other)      = default;
  ~abcflow() override = default;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  [[nodiscard]] constexpr auto evaluate(pos_type const& x,
                                        real_type const /*t*/) const -> tensor_type {
    return {m_a * gcem::sin(x(2)) + m_c * gcem::cos(x(1)),
            m_b * gcem::sin(x(0)) + m_a * gcem::cos(x(2)),
            m_c * gcem::sin(x(1)) + m_b * gcem::cos(x(0))};
  }
  [[nodiscard]] constexpr auto in_domain(pos_type const& /*x*/,
                                         real_type const /*t*/) const -> bool {
    return true;
  }
};

abcflow()->abcflow<double>;

//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================

#if TATOOINE_GINAC_AVAILABLE
#include "symbolic_field.h"
//==============================================================================
namespace tatooine::symbolic {
//==============================================================================
/// \brief The Arnold–Beltrami–Childress (ABC) flow is a three-dimensional
///        incompressible velocity field which is an exact solution of Euler's
///        equation.
template <typename real_type>
struct abcflow : field<real_type, 3, 3> {
  using this_type   = abcflow<real_type>;
  using parent_type = field<real_type, 3, 3>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  using typename parent_type::symtensor_type;

  //============================================================================
 public:
  explicit constexpr abcflow(const real_type a = 1, const real_type b = 1,
                    const real_type c = 1) {
    this->set_expr(vec{a * sin(symbol::x(2)) + c * cos(symbol::x(1)),
                       b * sin(symbol::x(0)) + a * cos(symbol::x(2)),
                       c * sin(symbol::x(1)) + b * cos(symbol::x(0))});
  }
  constexpr abcflow(const abcflow& other)            = default;
  constexpr abcflow(abcflow&& other)                 = default;
  constexpr abcflow& operator=(const abcflow& other) = default;
  constexpr abcflow& operator=(abcflow&& other)      = default;
  ~abcflow() override                                = default;
};

abcflow()->abcflow<double>;

//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================
#endif
#endif
