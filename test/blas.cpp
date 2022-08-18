#include <catch2/catch.hpp>
#include <tatooine/blas.h>
#include <tatooine/mat.h>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("blas_gemm", "[blas][gemm]") {
  auto A = mat24{{1, 2, 3, 4},
                 {2, 3, 4, 5}};

  auto B = mat43{{1, 2, 3},
                 {2, 3, 4},
                 {3, 4, 5},
                 {4, 5, 6}};

  auto C = mat23::zeros();
  blas::gemm(real_number(1), A, B, real_number(0), C);
  REQUIRE(C(0, 0) == Approx(30));
  REQUIRE(C(1, 0) == Approx(40));

  REQUIRE(C(0, 1) == Approx(40));
  REQUIRE(C(1, 1) == Approx(54));

  REQUIRE(C(0, 2) == Approx(50));
  REQUIRE(C(1, 2) == Approx(68));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
