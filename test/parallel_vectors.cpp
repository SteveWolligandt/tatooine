#include <catch2/catch.hpp>

#include <tatooine/counterexample_sadlo.h>
#include <tatooine/diff.h>
#include <tatooine/doublegyre.h>
#include <tatooine/abcflow.h>
#include <tatooine/modified_doublegyre.h>
#include <tatooine/parallel_vectors.h>
#include <tatooine/spacetime_field.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================
template <typename Field, typename Real, typename... Preds>
auto pv_acceleration(const field<Field, Real, 2, 2>& vf, linspace<Real> x,
                     linspace<Real> y, linspace<Real> z, Preds&&... preds) {
  spacetime_field stvf{vf};
  auto            grad_stvf = diff(stvf);
  auto            staf      = grad_stvf * stvf;
  return parallel_vectors(stvf, staf, x, y, z, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, typename... Preds>
auto pv_acceleration(const field<Field, Real, 3, 3>& vf, linspace<Real> x,
                     linspace<Real> y, linspace<Real> z, Preds&&... preds) {
  auto Jvf = diff(vf);
  auto af  = Jvf * vf;
  return parallel_vectors(vf, af, x, y, z, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, typename... Preds>
auto pv_jerk(const field<Field, Real, 2, 2>& vf, linspace<Real> x,
             linspace<Real> y, linspace<Real> z, Preds&&... preds) {
  spacetime_field stvf{vf};
  auto            Jstvf = diff(stvf);
  auto            staf  = Jstvf * stvf;
  auto            Jstaf = diff(staf);
  auto            stbf  = Jstaf * stvf;
  return parallel_vectors(stvf, stbf, x, y, z, std::forward<Preds>(preds)...);
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, typename... Preds>
auto pv_jerk(const field<Field, Real, 3, 3>& vf, linspace<Real> x,
             linspace<Real> y, linspace<Real> z, Preds&&... preds) {
  auto Jvf = diff(vf);
  auto af  = Jvf * vf;
  auto Jaf = diff(af);
  auto bf  = Jaf * vf;
  return parallel_vectors(vf, bf, x, y, z, std::forward<Preds>(preds)...);
}
//==============================================================================
TEST_CASE("parallel_vectors_numerical_doublegyre_acceleration",
          "[parallel_vectors][pv][numerical][doublegyre][dg][spacetime_field]"
          "[differentiate][acceleration]") {
  write_vtk(
      pv_acceleration(numerical::doublegyre{}, linspace{1e-3, 2.0 - 1e-4, 51},
                      linspace{1e-4, 1 - 1e-3, 26}, linspace{0.0, 10.0, 31}),
      "numerical_spacetime_doublegyre_pv_lines_acceleration.vtk");
}
//==============================================================================
TEST_CASE("parallel_vectors_numerical_doublegyre_jerk",
          "[parallel_vectors][pv][numerical][doublegyre][dg][spacetime_field]"
          "[differentiate][jerk]") {
  write_vtk(pv_jerk(numerical::doublegyre{}, linspace{1e-3, 2.0 - 1e-3, 201},
                    linspace{-1e-4, 1e-3, 101}, linspace{-10.0, 20.0, 401}),
            "numerical_spacetime_doublegyre_pv_lines_jerk.vtk");
}
//==============================================================================
TEST_CASE("parallel_vectors_symbolic_doublegyre_acceleration",
          "[parallel_vectors][pv][symbolic][doublegyre][dg][spacetime_field]"
          "[differentiate][acceleration]") {
  write_vtk(
      pv_acceleration(symbolic::doublegyre{}, linspace{1e-3, 2.0 - 1e-4, 51},
                      linspace{1e-4, 1 - 1e-3, 26}, linspace{0.0, 10.0, 31}),
      "symbolic_spacetime_doublegyre_pv_lines_acceleration.vtk");
}
//==============================================================================
TEST_CASE("parallel_vectors_symbolic_doublegyre_jerk",
          "[parallel_vectors][pv][symbolic][doublegyre][dg][spacetime_field]"
          "[differentiate][jerk]") {
  write_vtk(pv_jerk(symbolic::doublegyre{}, linspace{0.0, 2.0, 201},
                    linspace{-1.0, 1.0, 101}, linspace{0.0, 10.0, 401}),
            "symbolic_spacetime_doublegyre_pv_lines_jerk.vtk");
}
//==============================================================================
//==============================================================================
TEST_CASE("parallel_vectors_numerical_modified_doublegyre_acceleration",
          "[parallel_vectors][pv][numerical][modified_doublegyre][mdg]["
          "spacetime_field]"
          "[differentiate][acceleration]") {
  write_vtk(
      pv_acceleration(numerical::modified_doublegyre{},
                      linspace{1e-3, 2.0 - 1e-3, 201},
                      linspace{-1e-4, 1e-3, 101}, linspace{-10.0, 20.0, 401}),
      "numerical_spacetime_modified_doublegyre_pv_lines_acceleration.vtk");
}
//==============================================================================
TEST_CASE("parallel_vectors_numerical_modified_doublegyre_jerk",
          "[parallel_vectors][pv][numerical][modified_doublegyre][mdg]["
          "spacetime_field]"
          "[differentiate][jerk]") {
  write_vtk(
      pv_jerk(numerical::modified_doublegyre{}, linspace{1e-3, 2.0 - 1e-3, 201},
              linspace{-1e-4, 1e-3, 101}, linspace{-10.0, 20.0, 401}),
      "numerical_spacetime_modified_doublegyre_pv_lines_jerk.vtk");
}
//==============================================================================
TEST_CASE("parallel_vectors_symbolic_modified_doublegyre_acceleration",
          "[parallel_vectors][pv][symbolic][modified_doublegyre][mdg]["
          "spacetime_field]"
          "[differentiate][acceleration]") {
  write_vtk(
      pv_acceleration(symbolic::modified_doublegyre{}, linspace{0.0, 2.0, 201},
                      linspace{-1.0, 1.0, 101}, linspace{0.0, 10.0, 401}),
      "symbolic_spacetime_modified_doublegyre_pv_lines_acceleration.vtk");
}
//==============================================================================
TEST_CASE("parallel_vectors_symbolic_modified_doublegyre_jerk",
          "[parallel_vectors][pv][symbolic][modified_doublegyre][mdg]["
          "spacetime_field]"
          "[differentiate][jerk]") {
  write_vtk(pv_jerk(symbolic::modified_doublegyre{}, linspace{0.0, 2.0, 201},
                    linspace{-1.0, 1.0, 101}, linspace{0.0, 10.0, 401}),
            "symbolic_spacetime_modified_doublegyre_pv_lines_jerk.vtk");
}
//==============================================================================
//==============================================================================
TEST_CASE(
    "parallel_vectors_numerical_counterexample_sadlo_acceleration",
    "[parallel_vectors][pv][numerical][counterexample_sadlo][spacetime_field]"
    "[differentiate][acceleration]") {
  write_vtk(
      pv_acceleration(
          numerical::counterexample_sadlo{},
          linspace{-5.0 + 1e-3, 5.0 - 1e-4, 401},
          linspace{-5.0 + 1e-4, 5.0 - 1e-3, 401}, linspace{-10.0, 10.0, 401},
          [](const auto& x) {
            return std::abs(3 - std::sqrt((x(0) * x(0) + x(1) * x(1)))) > 1e-2;
          }),
      "numerical_spacetime_counterexample_sadlo_pv_lines_acceleration.vtk");
}
//==============================================================================
TEST_CASE(
    "parallel_vectors_numerical_counterexample_sadlo_jerk",
    "[parallel_vectors][pv][numerical][counterexample_sadlo][spacetime_field]"
    "[differentiate][jerk]") {
  write_vtk(
      pv_jerk(
          numerical::counterexample_sadlo{},
          linspace{-5.0 + 1e-3, 5.0 - 1e-4, 401},
          linspace{-5.0 + 1e-4, 5.0 - 1e-3, 401}, linspace{-10.0, 10.0, 401},
          [](const auto& x) {
            return std::abs(3 - std::sqrt((x(0) * x(0) + x(1) * x(1)))) > 1e-2;
          }),
      "numerical_spacetime_counterexample_sadlo_pv_lines_jerk.vtk");
}
//==============================================================================
TEST_CASE(
    "parallel_vectors_symbolic_counterexample_sadlo_acceleration",
    "[parallel_vectors][pv][symbolic][counterexample_sadlo][spacetime_field]"
    "[differentiate][acceleration]") {
  write_vtk(
      pv_acceleration(
          symbolic::counterexample_sadlo{},
          linspace{-5.0 + 1e-3, 5.0 - 1e-4, 401},
          linspace{-5.0 + 1e-4, 5.0 - 1e-3, 401}, linspace{-10.0, 10.0, 401},
          [](const auto& x) {
            return std::abs(3 - std::sqrt((x(0) * x(0) + x(1) * x(1)))) > 1e-2;
          }),
      "symbolic_spacetime_counterexample_sadlo_pv_lines_acceleration.vtk");
}
//==============================================================================
TEST_CASE(
    "parallel_vectors_symbolic_counterexample_sadlo_jerk",
    "[parallel_vectors][pv][symbolic][counterexample_sadlo][spacetime_field]"
    "[differentiate][jerk]") {
  write_vtk(
      pv_jerk(symbolic::counterexample_sadlo{},
              linspace{-3.0 + 1e-3, 3.0 - 1e-4, 401},
              linspace{-3.0 + 1e-4, 3.0 - 1e-3, 401}, linspace{-5.0, 5.0, 401},
              [](const auto& x) {
                return std::abs(3 - std::sqrt((x(0) * x(0) + x(1) * x(1)))) >
                       1e-2;
              }),
      "symbolic_spacetime_counterexample_sadlo_pv_lines_jerk.vtk");
}
//==============================================================================
TEST_CASE("parallel_vectors_numerical_abcflow_acceleration",
          "[parallel_vectors][pv][numerical][abcflow][abc]"
          "[differentiate][acceleration]") {
  write_vtk(pv_acceleration(numerical::abcflow{},
                            linspace{-10.0 + 1e-3, 10.0 + 1e-5, 51},
                            linspace{-10.0 + 1e-4, 10.0 + 1e-4, 51},
                            linspace{-10.0 + 1e-5, 10.0 + 1e-3, 51}),
            "numerical_abcflow_pv_lines_acceleration.vtk");
}
//==============================================================================
TEST_CASE("parallel_vectors_numerical_abcflow_jerk",
          "[parallel_vectors][pv][numerical][abcflow][abc]"
          "[differentiate][jerk]") {
  write_vtk(
      pv_jerk(numerical::abcflow{}, linspace{-10.0 + 1e-3, 10.0 + 1e-5, 51},
              linspace{-10.0 + 1e-4, 10.0 + 1e-4, 51},
              linspace{-10.0 + 1e-5, 10.0 + 1e-3, 51}),
      "numerical_abcflow_pv_lines_jerk.vtk");
}
//==============================================================================
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
