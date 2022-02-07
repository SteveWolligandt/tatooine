#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_ABCFLOW_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_ABCFLOW_H
//==============================================================================
#include <cmath>
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
/// \brief The Arnold–Beltrami–Childress (ABC) flow is a three-dimensional
///        incompressible velocity field which is an exact solution of Euler's
///        equation.
template <typename real_t>
struct abcflow : vectorfield<abcflow<real_t>, real_t, 3> {
  using this_t   = abcflow<real_t>;
  using parent_type = vectorfield<this_t, real_t, 3>;
  using typename parent_type::pos_t;
  using typename parent_type::tensor_t;

  //============================================================================
 private:
  real_t m_a, m_b, m_c;

  //============================================================================
 public:
  explicit constexpr abcflow(real_t const a = 1, real_t const b = 1,
                             real_t const c = 1)
      : m_a{a}, m_b{b}, m_c{c} {}
  constexpr abcflow(const abcflow& other)            = default;
  constexpr abcflow(abcflow&& other)                 = default;
  constexpr abcflow& operator=(const abcflow& other) = default;
  constexpr abcflow& operator=(abcflow&& other)      = default;
  ~abcflow() override = default;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  [[nodiscard]] constexpr auto evaluate(pos_t const& x,
                                        real_t const /*t*/) const -> tensor_t {
    return tensor_t{m_a * std::sin(x(2)) + m_c * std::cos(x(1)),
                    m_b * std::sin(x(0)) + m_a * std::cos(x(2)),
                    m_c * std::sin(x(1)) + m_b * std::cos(x(0))};
  }
  [[nodiscard]] constexpr auto in_domain(pos_t const& /*x*/,
                                         real_t const /*t*/) const -> bool {
    return true;
  }
};

abcflow()->abcflow<double>;

//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================

#if TATOOINE_GINAC_AVAILABLE
#include "symbolic_field.h"
//==============================================================================
namespace tatooine::symbolic {
//==============================================================================
/// \brief The Arnold–Beltrami–Childress (ABC) flow is a three-dimensional
///        incompressible velocity field which is an exact solution of Euler's
///        equation.
template <typename real_t>
struct abcflow : field<real_t, 3, 3> {
  using this_t   = abcflow<real_t>;
  using parent_type = field<real_t, 3, 3>;
  using typename parent_type::pos_t;
  using typename parent_type::tensor_t;
  using typename parent_type::symtensor_t;

  //============================================================================
 public:
  explicit constexpr abcflow(const real_t a = 1, const real_t b = 1,
                    const real_t c = 1) {
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
