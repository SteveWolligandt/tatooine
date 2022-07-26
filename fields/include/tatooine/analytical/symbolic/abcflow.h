#ifndef TATOOINE_ANALYTICAL_SYMBOLIC_ABCFLOW_H
#define TATOOINE_ANALYTICAL_SYMBOLIC_ABCFLOW_H
//==============================================================================
#include <tatooine/available_libraries.h>
#if TATOOINE_GINAC_AVAILABLE
#include <tatooine/symbolic_field.h>
//==============================================================================
namespace tatooine::analytical::symbolic {
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
