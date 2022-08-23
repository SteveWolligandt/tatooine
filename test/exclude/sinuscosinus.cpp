#include <tatooine/sinuscosinus.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/diff.h>
#include <catch2/catch_test_macros.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("sinuscosinus", "[sinuscosinus][sincos][symbolic][diff]") {
  spacetime_field stsincos{symbolic::sinuscosinus{}*symbolic::cosinussinus{}};
  auto dstsincos = diff(stsincos);
  std::cerr << stsincos.expr() << '\n';
  std::cerr << dstsincos.expr() << '\n';
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
