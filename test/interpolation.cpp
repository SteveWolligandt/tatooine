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
}  // namespace tatooine::test
//==============================================================================
