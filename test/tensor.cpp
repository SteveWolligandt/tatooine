#include <tatooine/tensor.h>
#include <tatooine/symbolic.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("tensor_print_matrix", "[tensor][print][matrix]") {
  std::cerr << mat<int, 3, 3>{random_uniform{0, 9}} << '\n';
}

//==============================================================================
TEST_CASE("tensor_initializers", "[tensor][initializers]") {
  SECTION("constructors") {
    auto m3z = mat3::zeros();
    m3z.for_indices([&m3z](const auto... is) { CHECK(m3z(is...) == 0); });
    auto m3o = mat3::ones();
    m3o.for_indices([&m3o](const auto... is) { CHECK(m3o(is...) == 1); });
    auto m3ru = mat3::randu();
    auto m3rn = mat3::randn();
  }
  SECTION("factory functions") {
    mat3 m3z{zeros};
    m3z.for_indices([&m3z](const auto... is) { CHECK(m3z(is...) == 0); });
    mat3 m3o{ones};
    m3o.for_indices([&m3o](const auto... is) { CHECK(m3o(is...) == 1); });
    mat3 m3f{fill{3}};
    m3f.for_indices([&m3f](const auto... is) { CHECK(m3f(is...) == 3); });
    mat3 m3ru{random_uniform{}};
    mat3 m3rn{random_normal{}};
  }
}


//==============================================================================
TEST_CASE("tensor_assignment", "[tensor][assignment]") {
  vec v{1.0, 2.0, 3.0};

  mat m{{1.0,  2.0,  3.0},
        {4.0,  5.0,  7.0},
        {8.0,  9.0, 10.0}};

  v(1) = 4.0;
  CHECK(approx_equal(v, vec{1.0, 4.0, 3.0}, 1e-6));

  m.row(1) = v;
  CHECK(m(1, 0) == v(0));
  CHECK(m(1, 1) == v(1));
  CHECK(m(1, 2) == v(2));

  m.col(2) = v;
  CHECK(m(0, 2) == v(0));
  CHECK(m(1, 2) == v(1));
  CHECK(m(2, 2) == v(2));

  vec v2{10.0, 11.0, 12.0};
  vec v3{20, 21, 22};
  v = v2;
  CHECK(v(0) == v2(0));
  CHECK(v(1) == v2(1));
  CHECK(v(2) == v2(2));

  v2 = v3;
  CHECK(v2(0) == v3(0));
  CHECK(v2(1) == v3(1));
  CHECK(v2(2) == v3(2));
}

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
  auto m     = mat4::randu();
  auto m_neg = -m;
  m.for_indices([&](const auto... is) { CHECK(m(is...) == -m_neg(is...)); });
}

//==============================================================================
TEST_CASE("tensor_addition", "[tensor][operation][addition]") {
  auto m0 = mat4::randu();
  auto m1 = mat4::randu();
  auto added = m0 + m1;
  m0.for_indices([&](const auto... is) {
    CHECK((added(is...) == m0(is...) + m1(is...)));
  });
}

//==============================================================================
TEST_CASE("tensor_symbolic", "[tensor][symbolic]") {
  vec  v{symbolic::symbol::x(0),
         symbolic::symbol::x(0) * symbolic::symbol::x(1)};
  auto m = mat2::randu();
  auto vdfx0 = diff(v, symbolic::symbol::x(0));
  auto vdfx1 = diff(v, symbolic::symbol::x(1));
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
  REQUIRE(va(2) == Approx(2.0305e-16).margin(eps));

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
TEST_CASE("tensor_compare", "[tensor][compare]") {
  vec v1{0.1, 0.1};
  vec v2{0.2, 0.2};
  REQUIRE(v1 < v2);
  REQUIRE_FALSE(v1 < v1);
  REQUIRE(v1 == v1);
  REQUIRE_FALSE(v2 < v2);
  REQUIRE(v2 == v2);
  REQUIRE_FALSE(v2 < v1);

  REQUIRE(std::pair{0.0, vec{0.1, 0.1}} < std::pair{0.0, vec{0.2, 0.2}});
  REQUIRE(std::array{0.1, 0.1, 0.0} < std::array{0.2, 0.2, 0.0});
  REQUIRE(vec{0.1, 0.1, 0.0} < vec{0.2, 0.2, 0.0});
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
