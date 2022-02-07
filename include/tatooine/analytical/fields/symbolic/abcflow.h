#ifndef TATOOINE_ANALYTICAL_FIELDS_SYMBOLIC_ABCFLOW_H
#define TATOOINE_ANALYTICAL_FIELDS_SYMBOLIC_ABCFLOW_H
//==============================================================================
#include <tatooine/packages.h>
#if TATOOINE_GINAC_AVAILABLE
#include <tatooine/symbolic_field.h>
//==============================================================================
namespace tatooine::analytical::fields::symbolic {
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
