#include <catch2/catch.hpp>

#include <tatooine/diff.h>
#include <tatooine/doublegyre.h>
#include <tatooine/newdoublegyre.h>
#include <tatooine/counterexample_sadlo.h>
#include <tatooine/parallel_vectors.h>
#include <tatooine/spacetime_field.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================

template <typename Field, typename Real>
auto pv_acceleration(const field<Field, Real, 2, 2>& v, linspace<Real> x,
                     linspace<Real> y, linspace<Real> z) {
  spacetime_field  stv{v};
  auto             grad_stv = diff(stv);
  auto             sta    = grad_stv * stv;
  if constexpr (is_symbolic_field(v)) {
    std::cerr << v.as_derived().expr() << "\n\n";
    std::cerr << sta.expr() << '\n';
  }
  parallel_vectors pv{stv, sta, x, y, z};
  return pv();
}
//------------------------------------------------------------------------------
template <typename Field, typename Real>
auto pv_jerk(const field<Field, Real, 2, 2>& v, linspace<Real> x, linspace<Real> y,
             linspace<Real> z) {
  spacetime_field  stv{v};
  auto             grad_stv = diff(stv);
  auto             sta      = grad_stv * stv;
  auto             grad_sta = diff(sta);
  auto             stb      = grad_sta * stv;
  if constexpr (is_symbolic_field(v)) {
    std::cerr << stb.as_derived().expr() << '\n';
  }
  parallel_vectors pv{stv, stb, x, y, z};
  return pv();
}

//==============================================================================
TEST_CASE("parallel_vectors_numerical_doublegyre_acceleration",
          "[parallel_vectors][pv][numerical][doublegyre][dg][spacetime_field]"
          "[differentiate][acceleration]") {
  write_vtk(pv_acceleration(numerical::doublegyre{}, linspace{1e-3, 2.0 - 1e-3, 81},
               linspace{-1e-4, 1e-3, 11}, linspace{-10.0, 20.0, 401}),
            "numerical_spacetime_doublegyre_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE("parallel_vectors_numerical_doublegyre_jerk",
          "[parallel_vectors][pv][numerical][doublegyre][dg][spacetime_field]"
          "[differentiate][jerk]") {
  write_vtk(pv_jerk(numerical::doublegyre{}, linspace{1e-3, 2.0 - 1e-3, 81},
               linspace{-1e-4, 1e-3, 41}, linspace{-10.0, 20.0, 401}),
            "numerical_spacetime_doublegyre_pv_lines_jerk.vtk");
}

//==============================================================================
TEST_CASE("parallel_vectors_symbolic_doublegyre_acceleration",
          "[parallel_vectors][pv][symbolic][doublegyre][dg][spacetime_field]"
          "[differentiate][acceleration]") {
  write_vtk(pv_acceleration(symbolic::doublegyre{}, linspace{0.0, 2.0, 41},
               linspace{-1.0, 1.0, 41}, linspace{0.0, 10.0, 201}),
            "symbolic_spacetime_doublegyre_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE("parallel_vectors_symbolic_doublegyre_jerk",
          "[parallel_vectors][pv][symbolic][doublegyre][dg][spacetime_field]"
          "[differentiate][jerk]") {
  write_vtk(pv_jerk(symbolic::doublegyre{}, linspace{0.0, 2.0, 41},
               linspace{-1.0, 1.0, 41}, linspace{0.0, 10.0, 201}),
            "symbolic_spacetime_doublegyre_pv_lines_jerk.vtk");
}

//==============================================================================
//==============================================================================
TEST_CASE("parallel_vectors_numerical_newdoublegyre_acceleration",
          "[parallel_vectors][pv][numerical][newdoublegyre][ndg][spacetime_field]"
          "[differentiate][acceleration]") {
  write_vtk(pv_acceleration(numerical::newdoublegyre{}, linspace{1e-3, 2.0 - 1e-3, 81},
               linspace{-1e-4, 1e-3, 11}, linspace{-10.0, 20.0, 401}),
            "numerical_spacetime_newdoublegyre_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE("parallel_vectors_numerical_newdoublegyre_jerk",
          "[parallel_vectors][pv][numerical][newdoublegyre][ndg][spacetime_field]"
          "[differentiate][jerk]") {
  write_vtk(pv_jerk(numerical::newdoublegyre{}, linspace{1e-3, 2.0 - 1e-3, 81},
               linspace{-1e-4, 1e-3, 41}, linspace{-10.0, 20.0, 401}),
            "numerical_spacetime_newdoublegyre_pv_lines_jerk.vtk");
}

//==============================================================================
TEST_CASE("parallel_vectors_symbolic_newdoublegyre_acceleration",
          "[parallel_vectors][pv][symbolic][newdoublegyre][ndg][spacetime_field]"
          "[differentiate][acceleration]") {
  write_vtk(pv_acceleration(symbolic::newdoublegyre{}, linspace{0.0, 2.0, 41},
               linspace{-1.0, 1.0, 41}, linspace{0.0, 10.0, 201}),
            "symbolic_spacetime_newdoublegyre_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE("parallel_vectors_symbolic_newdoublegyre_jerk",
          "[parallel_vectors][pv][symbolic][newdoublegyre][ndg][spacetime_field]"
          "[differentiate][jerk]") {
  write_vtk(pv_jerk(symbolic::newdoublegyre{}, linspace{0.0, 2.0, 41},
               linspace{-1.0, 1.0, 41}, linspace{0.0, 10.0, 201}),
            "symbolic_spacetime_newdoublegyre_pv_lines_jerk.vtk");
}

//==============================================================================
//==============================================================================
TEST_CASE(
    "parallel_vectors_numerical_counterexample_sadlo_acceleration",
    "[parallel_vectors][pv][numerical][counterexample_sadlo][spacetime_field]"
    "[differentiate][acceleration]") {
  write_vtk(
      pv_acceleration(numerical::counterexample_sadlo{},
                      linspace{-5.0 + 1e-3, 5.0 - 1e-4, 151},
                      linspace{-5.0 + 1e-4, 5.0 - 1e-3, 151},
                      linspace{-10.0, 10.0, 401}),
      "numerical_spacetime_counterexample_sadlo_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE(
    "parallel_vectors_numerical_counterexample_sadlo_jerk",
    "[parallel_vectors][pv][numerical][counterexample_sadlo][spacetime_field]"
    "[differentiate][jerk]") {
  write_vtk(pv_jerk(numerical::counterexample_sadlo{},
                      linspace{-5.0 + 1e-3, 5.0 - 1e-4, 151},
                      linspace{-5.0 + 1e-4, 5.0 - 1e-3, 151},
                      linspace{-10.0, 10.0, 401}),
            "numerical_spacetime_counterexample_sadlo_pv_lines_jerk.vtk");
}

//==============================================================================
TEST_CASE(
    "parallel_vectors_symbolic_counterexample_sadlo_acceleration",
    "[parallel_vectors][pv][symbolic][counterexample_sadlo][spacetime_field]"
    "[differentiate][acceleration]") {
  auto pv_lines = pv_acceleration(
      symbolic::counterexample_sadlo{}, linspace{-3.0 + 1e-3, 3.0 - 1e-4, 51},
      linspace{-3.0 + 1e-4, 3.0 - 1e-3, 51}, linspace{-5.0, 5.0, 101});
  write_vtk(
      pv_lines,
      "symbolic_spacetime_counterexample_sadlo_pv_lines_acceleration.vtk");
  //std::vector<size_t> to_delete;
  //for (size_t i = 0; i < pv_lines.size(); ++i) {
  //  if (pv_lines[i].length() < 2) { to_delete.push_back(i); }
  //}
  //for (auto i : to_delete) {
  //  pv_lines.erase(begin(pv_lines) + i);
  //  for (auto& i2 : to_delete) { --i2; }
  //}
  //write_vtk(
  //    pv_lines,
  //    "symbolic_spacetime_counterexample_sadlo_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE(
    "parallel_vectors_symbolic_counterexample_sadlo_jerk",
    "[parallel_vectors][pv][symbolic][counterexample_sadlo][spacetime_field]"
    "[differentiate][jerk]") {
  write_vtk(pv_jerk(symbolic::counterexample_sadlo{},
                      linspace{-3.0 + 1e-3, 3.0 - 1e-4, 101},
                      linspace{-3.0 + 1e-4, 3.0 - 1e-3, 101},
                      linspace{-5.0, 5.0, 201}),
            "symbolic_spacetime_counterexample_sadlo_pv_lines_jerk.vtk");
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
