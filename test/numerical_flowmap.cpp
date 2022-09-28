#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/numerical_flowmap.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::ode::test {
//==============================================================================
TEST_CASE("numerical_flowmap_doublegyre",
          "[vc][rk43][rungekutta43][ode][integrator][integration][flowmap][dg]["
          "doublegyre]") {
  analytical::numerical::doublegyre v;
  auto                                      fm    = flowmap(v);
  [[maybe_unused]] auto const x010 = fm(vec{0.1, 0.1}, 0, 10);
  [[maybe_unused]] auto const x015 = fm(vec{0.1, 0.1}, 0, 15);
  [[maybe_unused]] auto const x0n5 = fm(vec{0.1, 0.1}, 0, -5);
  [[maybe_unused]] auto const x110 = fm(vec{0.1, 0.1}, 1, 10);
  [[maybe_unused]] auto const x115 = fm(vec{0.1, 0.1}, 1, 15);
  [[maybe_unused]] auto const x1n5 = fm(vec{0.1, 0.1}, 1, -5);
  [[maybe_unused]] auto const xn110 = fm(vec{0.1, 0.1}, 0, 10);
  [[maybe_unused]] auto const xn115 = fm(vec{0.1, 0.1}, 0, 15);
  [[maybe_unused]] auto const xn1n5 = fm(vec{0.1, 0.1}, 0, -5);

  auto fmgrad = diff(fm);
}
//==============================================================================
TEST_CASE("numerical_flowmap_internal_pointer",
          "[vc][rk43][rungekutta43][ode][integrator][integration][flowmap][dg]["
          "doublegyre][pointer]") {
  numerical_flowmap_field_pointer<double, 2, ode::boost::rungekuttafehlberg78,
                                  interpolation::cubic> fm;
  analytical::numerical::doublegyre             v;
  polymorphic::vectorfield<double, 2>*                  polymorphic_field = &v;
  fm.set_vectorfield(polymorphic_field);
  [[maybe_unused]] auto const x010  = fm(vec{0.1, 0.1}, 0, 10);
  [[maybe_unused]] auto const x015 = fm(vec{0.1, 0.1}, 0, 15);
  [[maybe_unused]] auto const x0n5 = fm(vec{0.1, 0.1}, 0, -5);
  [[maybe_unused]] auto const x110 = fm(vec{0.1, 0.1}, 1, 10);
  [[maybe_unused]] auto const x115 = fm(vec{0.1, 0.1}, 1, 15);
  [[maybe_unused]] auto const x1n5 = fm(vec{0.1, 0.1}, 1, -5);
  [[maybe_unused]] auto const xn110 = fm(vec{0.1, 0.1}, 0, 10);
  [[maybe_unused]] auto const xn115 = fm(vec{0.1, 0.1}, 0, 15);
  [[maybe_unused]] auto const xn1n5 = fm(vec{0.1, 0.1}, 0, -5);
}
//==============================================================================
}
//==============================================================================

