#ifndef TATOOINE_ABCFLOW_H
#define TATOOINE_ABCFLOW_H
//==============================================================================
#include <cmath>
#include <tatooine/packages.h>
#include "field.h"
//==============================================================================
namespace tatooine::numerical {
//==============================================================================
/// \brief The Arnold–Beltrami–Childress (ABC) flow is a three-dimensional
///        incompressible velocity field which is an exact solution of Euler's
///        equation.
template <typename real_t>
struct abcflow : field<abcflow<real_t>, real_t, 3, 3> {
  using this_t   = abcflow<real_t>;
  using parent_t = field<this_t, real_t, 3, 3>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
 private:
  real_t m_a, m_b, m_c;

  //============================================================================
 public:
  explicit constexpr abcflow(const real_t a = 1, const real_t b = 1, const real_t c = 1)
      : m_a{a}, m_b{b}, m_c{c} {}
  constexpr abcflow(const abcflow& other)            = default;
  constexpr abcflow(abcflow&& other)                 = default;
  constexpr abcflow& operator=(const abcflow& other) = default;
  constexpr abcflow& operator=(abcflow&& other)      = default;
  ~abcflow() override = default;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 [[nodiscard]]  constexpr tensor_t evaluate(const pos_t& x, real_t) const override {
    return tensor_t{m_a * std::sin(x(2)) + m_c * std::cos(x(1)),
                    m_b * std::sin(x(0)) + m_a * std::cos(x(2)),
                    m_c * std::sin(x(1)) + m_b * std::cos(x(0))};
  }
  [[nodiscard]] constexpr bool in_domain(const pos_t& /*x*/, real_t /*t*/) const override {
    return true;
  }
};

#if has_cxx17_support()
abcflow()->abcflow<double>;
#endif

//==============================================================================
}  // namespace tatooine::numerical
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
  using parent_t = field<real_t, 3, 3>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using typename parent_t::symtensor_t;

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
