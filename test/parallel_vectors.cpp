#include <catch2/catch.hpp>

#include <tatooine/counterexample_sadlo.h>
#include <tatooine/diff.h>
#include <tatooine/doublegyre.h>
#include <tatooine/newdoublegyre.h>
#include <tatooine/parallel_vectors.h>
#include <tatooine/spacetime_field.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================

template <typename Field, typename Real, typename... Preds>
auto pv_acceleration(const field<Field, Real, 2, 2>& v, linspace<Real> x,
                     linspace<Real> y, linspace<Real> z, Preds&&... preds) {
  spacetime_field stv{v};
  auto            grad_stv = diff(stv);
  auto            sta      = grad_stv * stv;
  if constexpr (is_symbolic_field_v<Field>) {
    std::cerr << v.as_derived().expr() << "\n\n";
    std::cerr << sta.expr() << '\n';
  }
  parallel_vectors pv{stv, sta, x, y, z};
  std::thread watcher([&](){
      auto p = pv.progress();
      while(p < 1) {
        std::cerr << p << "                             \r";
        std::this_thread::sleep_for(std::chrono::milliseconds{500});
        p = pv.progress();
      }
      });
  auto             ls = pv(0, std::forward<Preds>(preds)...);
  watcher.join();
  return ls;
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, typename... Preds>
auto pv_jerk(const field<Field, Real, 2, 2>& v, linspace<Real> x,
             linspace<Real> y, linspace<Real> z, Preds&&... preds) {
  spacetime_field stv{v};
  auto            grad_stv = diff(stv);
  auto            sta      = grad_stv * stv;
  auto            grad_sta = diff(sta);
  auto            stb      = grad_sta * stv;
  if constexpr (is_symbolic_field(v)) {
    std::cerr << stb.as_derived().expr() << '\n';
  }
  parallel_vectors pv{stv, stb, x, y, z};
  std::thread watcher([&](){
      auto p = pv.progress();
      while(p < 1) {
        std::cerr << p << "                             \r";
        std::this_thread::sleep_for(std::chrono::milliseconds{500});
        p = pv.progress();
      }
      });
  auto             ls = pv(0, std::forward<Preds>(preds)...);
  watcher.join();
  return ls;
}

//==============================================================================
TEST_CASE("parallel_vectors_numerical_doublegyre_acceleration",
          "[parallel_vectors][pv][numerical][doublegyre][dg][spacetime_field]"
          "[differentiate][acceleration]") {
  write_vtk(
      pv_acceleration(numerical::doublegyre{}, linspace{1e-3, 2.0 - 1e-3, 81},
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
TEST_CASE(
    "parallel_vectors_numerical_newdoublegyre_acceleration",
    "[parallel_vectors][pv][numerical][newdoublegyre][ndg][spacetime_field]"
    "[differentiate][acceleration]") {
  write_vtk(pv_acceleration(
                numerical::newdoublegyre{}, linspace{1e-3, 2.0 - 1e-3, 81},
                linspace{-1e-4, 1e-3, 11}, linspace{-10.0, 20.0, 401}),
            "numerical_spacetime_newdoublegyre_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE(
    "parallel_vectors_numerical_newdoublegyre_jerk",
    "[parallel_vectors][pv][numerical][newdoublegyre][ndg][spacetime_field]"
    "[differentiate][jerk]") {
  write_vtk(pv_jerk(numerical::newdoublegyre{}, linspace{1e-3, 2.0 - 1e-3, 81},
                    linspace{-1e-4, 1e-3, 41}, linspace{-10.0, 20.0, 401}),
            "numerical_spacetime_newdoublegyre_pv_lines_jerk.vtk");
}

//==============================================================================
TEST_CASE(
    "parallel_vectors_symbolic_newdoublegyre_acceleration",
    "[parallel_vectors][pv][symbolic][newdoublegyre][ndg][spacetime_field]"
    "[differentiate][acceleration]") {
  write_vtk(pv_acceleration(symbolic::newdoublegyre{}, linspace{0.0, 2.0, 41},
                            linspace{-1.0, 1.0, 41}, linspace{0.0, 10.0, 201}),
            "symbolic_spacetime_newdoublegyre_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE(
    "parallel_vectors_symbolic_newdoublegyre_jerk",
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
  auto pv_lines = pv_acceleration(
      numerical::counterexample_sadlo{}, linspace{-5.0 + 1e-3, 5.0 - 1e-4, 401},
      linspace{-5.0 + 1e-4, 5.0 - 1e-3, 401}, linspace{-10.0, 10.0, 401},
      [](const auto& x) {
        return std::abs(3 - std::sqrt((x(0) * x(0) + x(1) * x(1)))) > 1e-3;
      });
  write_vtk(
      pv_lines,
      "numerical_spacetime_counterexample_sadlo_pv_lines_acceleration.vtk");
  // std::vector<size_t> to_delete;
  // for (size_t i = 0; i < pv_lines.size(); ++i) {
  //  if (pv_lines[i].length() < 2) { to_delete.push_back(i); }
  //}
  // for (auto i : to_delete) {
  //  pv_lines.erase(begin(pv_lines) + i);
  //  for (auto& i2 : to_delete) { --i2; }
  //}
  // write_vtk(
  //    pv_lines,
  //    "symbolic_spacetime_counterexample_sadlo_pv_lines_acceleration.vtk");
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
  write_vtk(
      pv_acceleration(symbolic::counterexample_sadlo{},
                      linspace{-5.0 + 1e-3, 5.0 - 1e-4, 151},
                      linspace{-5.0 + 1e-4, 5.0 - 1e-3, 151},
                      linspace{-10.0, 10.0, 401}),
      "symbolic_spacetime_counterexample_sadlo_pv_lines_acceleration.vtk");
}

//==============================================================================
TEST_CASE(
    "parallel_vectors_symbolic_counterexample_sadlo_jerk",
    "[parallel_vectors][pv][symbolic][counterexample_sadlo][spacetime_field]"
    "[differentiate][jerk]") {
  write_vtk(
      pv_jerk(symbolic::counterexample_sadlo{},
              linspace{-3.0 + 1e-3, 3.0 - 1e-4, 101},
              linspace{-3.0 + 1e-4, 3.0 - 1e-3, 101}, linspace{-5.0, 5.0, 201}),
      "symbolic_spacetime_counterexample_sadlo_pv_lines_jerk.vtk");
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
