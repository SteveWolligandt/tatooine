#include <tatooine/interpolation.h>
#include <tatooine/linspace.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("interpolation_linear_scalar", "[interpolation][linear][scalar]") {
  auto const f = interpolation::linear{1.0, 2.0};
  REQUIRE(f(0) == 1);
  REQUIRE(f(1) == 2);
  REQUIRE(f(0.5) == 1.5);
  REQUIRE(f(0.25) == 1.25);
  REQUIRE(f(0.75) == 1.75);
}
//==============================================================================
TEST_CASE("interpolation_linear_vec", "[interpolation][linear][vec]") {
  auto const f = interpolation::linear{vec{1.0, 2.0}, vec{2.0, 4.0}};
  REQUIRE(f(0)(0) == 1);
  REQUIRE(f(0)(1) == 2);
  REQUIRE(f(1)(0) == 2);
  REQUIRE(f(1)(1) == 4);
  REQUIRE(f(0.5)(0) == 1.5);
  REQUIRE(f(0.5)(1) == 3);
  REQUIRE(f(0.25)(0) == 1.25);
  REQUIRE(f(0.25)(1) == 2.5);
  REQUIRE(f(0.75)(0) == 1.75);
  REQUIRE(f(0.75)(1) == 3.5);
}
//==============================================================================
TEST_CASE("interpolation_linear_mat", "[interpolation][linear][mat]") {
  auto const f = interpolation::linear{mat{{1.0, 2.0},
                                           {3.0, 4.0}},
                                       mat{{2.0, 3.0},
                                           {4.0, 5.0}}};
  REQUIRE(f(0)(0, 0) == 1);
  REQUIRE(f(1)(0, 0) == 2);
  REQUIRE(f(.25)(0, 0) == 1.25);
  REQUIRE(f(0)(0, 1) == 2);
  REQUIRE(f(1)(0, 1) == 3);
  REQUIRE(f(.25)(0, 1) == 2.25);
  REQUIRE(f(0)(1, 0) == 3);
  REQUIRE(f(1)(1, 0) == 4);
  REQUIRE(f(.25)(1, 0) == 3.25);
  REQUIRE(f(0)(1, 1) == 4);
  REQUIRE(f(1)(1, 1) == 5);
  REQUIRE(f(.25)(1, 1) == 4.25);
}
//==============================================================================
TEST_CASE("interpolation_cubic_scalar", "[interpolation][cubic][scalar]") {
  auto const f = interpolation::cubic{0.0, 1.0, 1.0, -2.0};
  REQUIRE(f(0) == 0);
  REQUIRE(f(1) == 1);
  REQUIRE(diff(f)(0) == 1);
  REQUIRE(diff(f)(1) == -2);
}
////==============================================================================
// TEST_CASE("interpolation_cubic_vector", "[interpolation][cubic][vector]") {
//   interpolation::cubic interp{vec{0.0, 0.0}, vec{1.0, 0.0}, vec{0.0, 2.0},
//                               vec{0.0, -2.0}};
//   std::cerr << "[";
//   for (auto t : linspace(0.0, 1.0, 10)) {
//     std::cerr << interp(t)(0) << ' ' << interp(t)(1) << ';';
//   }
//   std::cerr << "]\n";
// }
////==============================================================================
// TEST_CASE("interpolation_cubic_example", "[interpolation][cubic][example0]")
// {
//   const vec fx0{0.76129106859416196, 0.68170915153208544};
//   const vec fx1{1.0, 1.0};
//   const vec fx0dx{1.0, 2.0};
//   const vec fx1dx{1.0, 0.0};
//
//   interpolation::cubic interp{fx0, fx1, fx0dx, fx1dx};
//   auto                 curve = interp.curve();
//
//   REQUIRE(approx_equal(fx0dx, curve.tangent(0)));
//   REQUIRE(approx_equal(fx1dx, curve.tangent(1)));
// }
////==============================================================================
// TEST_CASE("interpolation_linear_0", "[interpolation][linear]") {
//   double const ft0       = 3;
//   double const ft1       = 2;
//
//   interpolation::linear interp{ft0, ft1};
//   auto const&          f     = interp.polynomial();
//
//   INFO(f);
//   REQUIRE(ft0 == Approx(f(0)));
//   REQUIRE(ft1 == Approx(f(1)));
// }
////==============================================================================
// TEST_CASE("interpolation_cubic_0", "[interpolation][cubic]") {
//   double const ft0       = 3;
//   double const ft1       = 2;
//   double const dft0_dt   = -1;
//   double const dft1_dt   = 2;
//
//   interpolation::cubic interp{ft0, ft1, dft0_dt, dft1_dt};
//   auto const&          f     = interp.polynomial();
//   auto const           df_dt = diff(f);
//
//   INFO(f);
//   REQUIRE(ft0 == Approx(f(0)));
//   REQUIRE(ft1 == Approx(f(1)));
//   REQUIRE(dft0_dt == Approx(df_dt(0)));
//   REQUIRE(dft1_dt == Approx(df_dt(1)));
// }
////==============================================================================
// TEST_CASE("interpolation_quintic", "[interpolation][quintic]") {
//   double const ft0       = 3;
//   double const ft1       = 2;
//   double const dft0_dt   = -1;
//   double const dft1_dt   = 2;
//   double const ddft0_dtt = 1;
//   double const ddft1_dtt = -2;
//
//   interpolation::quintic interp{ft0,     ft1,       dft0_dt,
//                                 dft1_dt, ddft0_dtt, ddft1_dtt};
//   auto const&            f       = interp.polynomial();
//   auto const             df_dt   = diff(f);
//   auto const             ddf_dtt = diff(df_dt);
//
//   INFO(f);
//   REQUIRE(ft0 == Approx(f(0)));
//   REQUIRE(ft1 == Approx(f(1)));
//   REQUIRE(dft0_dt == Approx(df_dt(0)));
//   REQUIRE(dft1_dt == Approx(df_dt(1)));
//   REQUIRE(ddft0_dtt == Approx(ddf_dtt(0)));
//   REQUIRE(ddft1_dtt == Approx(ddf_dtt(1)));
// }
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
