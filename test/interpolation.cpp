#include <tatooine/interpolation.h>
#include <tatooine/linspace.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("interpolation_hermite_scalar", "[interpolation][hermite][scalar]") {
  interpolation::hermite interp{0.0, 1.0, 1.0, -2.0};

  std::cerr << "[";
  for (auto t : linspace(0.0, 1.0, 10)) {
    std::cerr << t << ' ' << interp(t) << ';';
  }
  std::cerr << "]\n";
}
//==============================================================================
TEST_CASE("interpolation_hermite_vector", "[interpolation][hermite][vector]") {
  interpolation::hermite interp{vec{0.0, 0.0}, vec{1.0, 0.0}, vec{0.0, 2.0},
                                vec{0.0, -2.0}};
  std::cerr << "[";
  for (auto t : linspace(0.0, 1.0, 10)) {
    std::cerr << interp(t)(0) << ' ' << interp(t)(1) << ';';
  }
  std::cerr << "]\n";
}
//==============================================================================
TEST_CASE("interpolation_hermite_example",
          "[interpolation][hermite][example0]") {
  const vec fx0{0.76129106859416196, 0.68170915153208544};
  const vec fx1{1.0, 1.0};
  const vec fx0dx{1.0, 2.0};
  const vec fx1dx{1.0, 0.0};

  interpolation::hermite interp{fx0, fx1, fx0dx, fx1dx};
  auto curve = interp.curve();

  REQUIRE(approx_equal(fx0dx, curve.tangent(0)));
  REQUIRE(approx_equal(fx1dx, curve.tangent(1)));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
