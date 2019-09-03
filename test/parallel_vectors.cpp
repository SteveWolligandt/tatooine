#include <catch2/catch.hpp>

#include <tatooine/diff.h>
#include <tatooine/doublegyre.h>
#include <tatooine/parallel_vectors.h>
#include <tatooine/spacetime_field.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("parallel_vectors_symbolic_doublegyre_acceleration",
          "[parallel_vectors][pv][symbolic][doublegyre][dg][spacetime_field]"
          "[differentiate][acceleration]") {
  symbolic::doublegyre dg;
  spacetime_field      stdg{dg};
  auto                 jstdg   = diff(stdg);
  auto                 accstdg = jstdg * stdg;
  parallel_vectors     pv{stdg, accstdg, linspace{0.0, 2.0, 41},
                      linspace{-1.0, 1.0, 41}, linspace{0.0, 10.0, 201}};
  write_vtk(pv(), "symbolic_spacetime_doublegyre_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE("parallel_vectors_numerical_doublegyre_acceleration",
          "[parallel_vectors][pv][numerical][doublegyre][dg][spacetime_field]"
          "[differentiate][acceleration]") {
  numerical::doublegyre dg;
  spacetime_field       stdg{dg};
  auto                  jstdg   = diff(stdg);
  auto                  accstdg = jstdg * stdg;
  parallel_vectors      pv{stdg, accstdg, linspace{1e-3, 2.0 - 1e-3, 41},
                      linspace{-1 + 1e-4, 1.0 - 1e-3, 41},
                      linspace{0.0, 10.0, 201}};
  write_vtk(pv(), "numerical_spacetime_doublegyre_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE("parallel_vectors_symbolic_doublegyre_jerk",
          "[parallel_vectors][pv][symbolic][doublegyre][dg][spacetime_field]"
          "[differentiate][jerk]") {
  symbolic::doublegyre dg;
  spacetime_field      stdg{dg};
  auto                 jstdg    = diff(stdg);
  auto                 accstdg  = jstdg * stdg;
  auto                 jaccstdg = diff(accstdg);
  auto                 jerkstdg = jaccstdg * stdg;
  parallel_vectors     pv{stdg, jerkstdg, linspace{0.0, 2.0, 41},
                      linspace{-1.0, 1.0, 41}, linspace{0.0, 10.0, 201}};
  write_vtk(pv(), "symbolic_spacetime_doublegyre_pv_lines_jerk.vtk");
}

//==============================================================================
TEST_CASE("parallel_vectors_numerical_doublegyre_jerk",
          "[parallel_vectors][pv][numerical][doublegyre][dg][spacetime_field]"
          "[differentiate][jerk]") {
  numerical::doublegyre dg;
  spacetime_field       stdg{dg};
  auto                  jstdg    = diff(stdg);
  auto                  accstdg  = jstdg * stdg;
  auto                  jaccstdg = diff(accstdg);
  auto                  jerkstdg = jaccstdg * stdg;
  parallel_vectors      pv{stdg, jerkstdg, linspace{1e-3, 2.0 - 1e-3, 41},
                      linspace{-1 + 1e-4, 1.0 - 1e-3, 41},
                      linspace{0.0, 10.0, 201}};
  write_vtk(pv(), "numerical_spacetime_doublegyre_pv_lines_jerk.vtk");
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
