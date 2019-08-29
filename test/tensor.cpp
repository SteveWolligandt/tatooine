#include "../tensor.h"
#include "../symbolic.h"
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("tensor_slice", "[tensor][slice]") {
  vec v{1.0, 2.0, 3.0};
  REQUIRE(v(0) == 1);
  REQUIRE(v(1) == 2);
  REQUIRE(v(2) == 3);
  REQUIRE(v.num_dimensions() == 1);
  REQUIRE(v.dimension(0) == 3);
  REQUIRE(v.num_components() == 3);
  
  vec v2{1.0, 2.0};
  REQUIRE(v(0) == 1);
  REQUIRE(v(1) == 2);
  
  mat m{{1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}};
  REQUIRE(m(0,0) == 1); REQUIRE(m(0,1) == 2);
  REQUIRE(m(1,0) == 3); REQUIRE(m(1,1) == 4);
  REQUIRE(m(2,0) == 5); REQUIRE(m(2,1) == 6);
  REQUIRE(m.num_dimensions() == 2);
  REQUIRE(m.dimension(0) == 3);
  REQUIRE(m.dimension(1) == 2);
  REQUIRE(m.num_components() == 6);

  auto c0 = m.col(0);
  REQUIRE(c0(0) == 1);
  REQUIRE(c0(1) == 3);
  REQUIRE(c0(2) == 5);
  REQUIRE(c0.num_dimensions() == 1);
  REQUIRE(c0.dimension(0) == 3);
  REQUIRE(c0.num_components() == 3);

  auto c1 = m.col(1);
  REQUIRE(c1(0) == 2);
  REQUIRE(c1(1) == 4);
  REQUIRE(c1(2) == 6);
  REQUIRE(c1.num_dimensions() == 1);
  REQUIRE(c1.dimension(0) == 3);
  REQUIRE(c1.num_components() == 3);

  auto r0 = m.row(0);
  REQUIRE(r0(0) == 1);
  REQUIRE(r0(1) == 2);
  REQUIRE(r0.num_dimensions() == 1);
  REQUIRE(r0.dimension(0) == 2);
  REQUIRE(r0.num_components() == 2);

  auto r1 = m.row(1);
  REQUIRE(r1(0) == 3);
  REQUIRE(r1(1) == 4);
  REQUIRE(r1.num_dimensions() == 1);
  REQUIRE(r1.dimension(0) == 2);
  REQUIRE(r1.num_components() == 2);

  auto r2 = m.row(2);
  REQUIRE(r2(0) == 5);
  REQUIRE(r2(1) == 6);
  REQUIRE(r2.num_dimensions() == 1);
  REQUIRE(r2.dimension(0) == 2);
  REQUIRE(r2.num_components() == 2);

  REQUIRE(dot(m.col(0), m.col(1)) == (1 * 2 + 3 * 4 + 5 * 6));
  auto prod = m * v2;
  REQUIRE(prod(0) == (1 * 1 + 2 * 2));
  REQUIRE(prod(1) == (1 * 3 + 2 * 4));
  REQUIRE(prod(2) == (1 * 5 + 2 * 6));
  REQUIRE(prod.num_dimensions() == 1);
  REQUIRE(prod.dimension(0) == 3);
  REQUIRE(prod.num_components() == 3);

  c1 = v;
  REQUIRE(c1(0) == 1);
  REQUIRE(c1(1) == 2);
  REQUIRE(c1(2) == 3);
  REQUIRE(r0(1) == 1);
  REQUIRE(r1(1) == 2);
  REQUIRE(r2(1) == 3);

  tensor<double, 3, 3, 3> t;
  auto                    slice = t.slice<1>(0);
  slice(2, 1);
  auto slice2 = slice.slice<1>(2);
  slice2(1);
}

//==============================================================================
TEST_CASE("tensor_negate", "[tensor][operation][negate]") {
  auto m     = mat4::rand();
  auto m_neg = -m;
  m.for_indices([&](const auto... is) { CHECK(m(is...) == -m_neg(is...)); });
}

//==============================================================================
TEST_CASE("tensor_addition", "[tensor][operation][addition]") {
  auto m0 = mat4::rand();
  auto m1 = mat4::rand();
  auto added = m0 + m1;
  m0.for_indices([&](const auto... is) {
    CHECK((added(is...) == m0(is...) + m1(is...)));
  });
  std::cerr << added << '\n';
}

//==============================================================================
TEST_CASE("tensor_symbolic", "[tensor][symbolic]") {
  vec  v{symbolic::symbol::x(0),
         symbolic::symbol::x(0) * symbolic::symbol::x(1)};
  auto m = mat2::rand();
  std::cerr << v << '\n';
  std::cerr << dot(v, vec{3, 2}) << '\n';
  std::cerr << m * v << '\n';
  std::cerr << evtod(v, symbolic::symbol::x(0) == 2,
                     symbolic::symbol::x(1) == 3)
            << '\n';
  auto vdfx0 = diff(v, symbolic::symbol::x(0));
  auto vdfx1 = diff(v, symbolic::symbol::x(1));
  std::cerr << vdfx0 << '\n';
  std::cerr << vdfx1 << '\n';
}

//==============================================================================
TEST_CASE("tensor_eigenvalue", "[tensor][eigenvalue]") {
  const mat m{{1.0,  2.0,  3.0},
              {4.0,  6.0,  8.0},
              {9.0, 12.0, 15.0}};
  const auto eps = 1e-4;
  const auto [eigvecs, eigvals] = eigenvectors(m);
  const auto ve = real(eigvecs);
  const auto va = real(eigvals);

  REQUIRE(va(0) == Approx(2.2874e+01).epsilon(eps));
  REQUIRE(va(1) == Approx(-8.7434e-01).epsilon(eps));
  REQUIRE(va(2) == Approx(2.0305e-16).epsilon(eps));

  REQUIRE(ve(0,0) == Approx(0.16168).epsilon(eps));
  REQUIRE(ve(1,0) == Approx(0.45378).epsilon(eps));
  REQUIRE(ve(2,0) == Approx(0.87632).epsilon(eps));

  REQUIRE(ve(0,1) == Approx(0.62794).epsilon(eps));
  REQUIRE(ve(1,1) == Approx(0.40677).epsilon(eps));
  REQUIRE(ve(2,1) == Approx(-0.66350).epsilon(eps));

  REQUIRE(ve(0,2) == Approx(0.40825).epsilon(eps));
  REQUIRE(ve(1,2) == Approx(-0.81650).epsilon(eps));
  REQUIRE(ve(2,2) == Approx(0.40825).epsilon(eps));
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
