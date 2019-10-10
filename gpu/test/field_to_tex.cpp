#include <tatooine/doublegyre.h>
#include <tatooine/gpu/field_to_tex.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine {
namespace gpu {
namespace test {
//==============================================================================

TEST_CASE("field_to_tex",
          "[cuda][field_to_tex][dg]") {
  doublegyre<double> v;
  auto               tex =
      field_to_tex(v, linspace<double>{0, 2, 21}, linspace<double>{0, 1, 11});
}

//==============================================================================
}  // namespace test
}  // namespace gpu
}  // namespace tatooine
//==============================================================================
