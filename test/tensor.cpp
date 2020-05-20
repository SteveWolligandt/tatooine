#include <tatooine/tensor.h>
#include <tatooine/symbolic.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("tensor_initializers", "[tensor][initializers]") {
  SECTION("constructors") {
    auto m3z = mat3::zeros();
    m3z.for_indices([&m3z](const auto... is) { CHECK(m3z(is...) == 0); });
    auto m3o = mat3::ones();
    m3o.for_indices([&m3o](const auto... is) { CHECK(m3o(is...) == 1); });
    [[maybe_unused]] auto m3ru = mat3::randu();
    [[maybe_unused]] auto m3rn = mat3::randn();
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
TEST_CASE("tensor_print_matrix", "[tensor][print][matrix]") {
  std::cerr << mat<int, 3, 3>{random_uniform{0, 9}} << '\n';
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
TEST_CASE("tensor_eigenvalue", "[tensor][eigenvalue]") {
  const mat m{{1.0,  2.0,  3.0},
              {4.0,  6.0,  8.0},
              {9.0, 12.0, 15.0}};
  const auto eps = 1e-4;
  const auto [eigvecs, eigvals] = eigenvectors(m);
  const auto eigvals2 = eigenvalues(m);
  const auto ve = real(eigvecs);
  const auto va = real(eigvals);
  const auto va2 = real(eigvals2);

  REQUIRE(va(0) == Approx(2.2874e+01).epsilon(eps));
  REQUIRE(va(1) == Approx(-8.7434e-01).epsilon(eps));
  REQUIRE(va(2) == Approx(2.0305e-16).margin(eps));
  REQUIRE(va2(0) == Approx(2.2874e+01).epsilon(eps));
  REQUIRE(va2(1) == Approx(-8.7434e-01).epsilon(eps));
  REQUIRE(va2(2) == Approx(2.0305e-16).margin(eps));

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
TEST_CASE("tensor_complex", "[tensor][complex][view]") {
  vec<std::complex<double>, 3> v{std::complex<double>{1, 2},
                                 std::complex<double>{2, 3},
                                 std::complex<double>{3, 4}};
  auto                         imag_v = imag(v);
  for (size_t i = 0; i < v.dimension(0); ++i) {
    REQUIRE(v(i).imag() == imag_v(i));
  }
  auto real_v = real(v);
  for (size_t i = 0; i < v.dimension(0); ++i) {
    REQUIRE(v(i).real() == real_v(i));
  }
  std::max(std::abs(min(real(v))), max(real(v)));
}

//==============================================================================
TEST_CASE("tensor_matrix_transpose", "[tensor][matrix][mat][transpose][view]") {
  auto A = mat<double, 2, 3>::randu();
  auto At = transpose(A);
  auto& A2 = transpose(At);

  REQUIRE(&A == &A2);

  REQUIRE(A.dimension(0) == At.dimension(1));
  REQUIRE(A.dimension(1) == At.dimension(0));

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) { REQUIRE(A(i, j) == At(j, i)); }
  }
}
//==============================================================================
TEST_CASE("tensor_abs", "[tensor][abs]") {
  SECTION("vec") {
    vec  x{-1, 2, -3};
    auto ax = abs(x);
    CAPTURE(x, ax);
    REQUIRE(ax(0) == 1);
    REQUIRE(ax(1) == 2);
    REQUIRE(ax(2) == 3);
  }
  SECTION("mat") {
    mat  A{{-1.79222, -7.94109,  3.67540},
           { 2.38520,  0.82284,  8.53506},
           {-1.37601, -6.15705, -0.71982}};
    auto aA = abs(A);
    CAPTURE(A, aA);
    REQUIRE(aA(0, 0) == -A(0, 0));
    REQUIRE(aA(1, 0) ==  A(1, 0));
    REQUIRE(aA(2, 0) == -A(2, 0));
    REQUIRE(aA(0, 1) == -A(0, 1));
    REQUIRE(aA(1, 1) ==  A(1, 1));
    REQUIRE(aA(2, 1) == -A(2, 1));
    REQUIRE(aA(0, 2) ==  A(0, 2));
    REQUIRE(aA(1, 2) ==  A(1, 2));
    REQUIRE(aA(2, 2) == -A(2, 2));
  }
}
TEST_CASE("tensor_sum", "[tensor][sum]") {
  SECTION("vec") {
    vec  x{-1, 2, -3};
    REQUIRE(sum(x) == Approx(-1+2-3));
  }
  SECTION("mat") {
    mat A{{-1.79222, -7.94109, 3.67540},
          {2.38520, 0.82284, 8.53506},
          {-1.37601, -6.15705, -0.71982}};

    CAPTURE(A.col(0), abs(A).slice<1>(0));
    REQUIRE(sum(A.col(0)) == Approx(-1.79222 + 2.38520 - 1.37601));
    REQUIRE(sum(abs(A).slice<1>(0)) == Approx(1.79222 + 2.38520 + 1.37601));
  }
}
//==============================================================================
TEST_CASE("tensor_matrix_norm1",
          "[tensor][norm1][1-norm][matrix]") {
    mat  A{{-1.79222, -7.94109,  3.67540},
           { 2.38520,  0.82284,  8.53506},
           {-1.37601, -6.15705, -0.71982}};
  REQUIRE(norm1(A) == Approx(14.921));
}
//==============================================================================
TEST_CASE("tensor_gesvd", "[tensor][gesvd]") {
  SECTION("3x3") {
    const mat A{{-1.79222, -7.94109, 3.67540},
                {2.38520, 0.82284, 8.53506},
                {-1.37601, -6.15705, -0.71982}};
    const auto [U, s, VT] = gesvd(A, lapack_job::S, lapack_job::S);
   auto diff = U * diag(s) * VT - A;
   REQUIRE(s(0) == Approx(1.073340050125074e+01));
   REQUIRE(s(1) == Approx(9.148458171897648e+00));
   REQUIRE(s(2) == Approx(6.447152840514361e-01));

   diff.for_indices([&diff](auto... is) {
     REQUIRE(diff(is...) == Approx(0).margin(1e-10));
   });
  }
  SECTION("4x3") {
    const mat A{
        {6.405871813e+00, -4.344670595e+00, 9.471184691e+00, 5.850792157e+00},
        {3.049605906e+00, 1.018629735e+00, 5.535464761e+00, 2.691779530e+00},
        {9.002176872e+00, 3.332492228e-01, -2.365651229e+00, 9.458283935e+00}};
    const auto [U, s, VT] = gesvd(A, lapack_job::S, lapack_job::S);
    CAPTURE(A,U,s,VT);
    auto diff             = U * diag(s) * VT - A;
    REQUIRE(s(0) == Approx(1.733194199066472e+01));
    REQUIRE(s(1) == Approx(9.963856829919013e+00));
    REQUIRE(s(2) == Approx(2.932957998778161e+00));
    // diff.for_indices([&diff](auto... is) {
    //  REQUIRE(diff(is...) == Approx(0).margin(1e-10));
    //});
  }
}
//==============================================================================
TEST_CASE("tensor_condition_number",
          "[tensor][cond][condition][condition_number]") {
  mat A{{-1.79222, -7.94109,  3.67540},
        { 2.38520,  0.82284,  8.53506},
        {-1.37601, -6.15705, -0.71982}};
  CAPTURE(A);
  REQUIRE(condition_number(A, 2) == Approx(16.64827989465566));
  REQUIRE(condition_number(A, 1) == Approx(26.47561499847143));
}
//==============================================================================
#if TATOOINE_GINAC_AVAILABLE
//==============================================================================
TEST_CASE("tensor_symbolic", "[tensor][symbolic]") {
  vec  v{symbolic::symbol::x(0),
         symbolic::symbol::x(0) * symbolic::symbol::x(1)};
  //auto m = mat2::randu();
  auto vdfx0 = diff(v, symbolic::symbol::x(0));
  auto vdfx1 = diff(v, symbolic::symbol::x(1));
}
TEST_CASE("tensor_ginac_matrix_conversion",
          "[tensor][symbolic][conversion][matrix]") {
  using namespace symbolic;
  SECTION("to ginac") {
    mat  m{{symbol::x(0), symbol::x(1)},
          {symbol::x(1) * symbol::t(), GiNaC::ex{symbol::x(0)}}};
    auto mg = to_ginac_matrix(m);
    for (size_t r = 0; r < m.dimension(0); ++r) {
      for (size_t c = 0; c < m.dimension(1); ++c) {
        REQUIRE(mg(r, c) == m(r, c));
      }
    }
  }
  SECTION("to tensor") {
    GiNaC::matrix m{{symbol::x(0), symbol::x(1)},
                    {symbol::x(1) * symbol::t(), GiNaC::ex{symbol::x(0)}}};
    auto          t = to_mat<2, 2>(m);
    for (size_t r = 0; r < m.rows(); ++r) {
      for (size_t c = 0; c < m.cols(); ++c) {
        REQUIRE(t(r, c) == m(r, c));
      }
    }
  }
}
//==============================================================================
TEST_CASE("tensor_symbolic_inverse", "[tensor][symbolic][inverse][matrix]") {
  using namespace symbolic;
  mat m{{symbol::x(0), symbol::x(1)},
        {symbol::x(1) * symbol::t(), GiNaC::ex{symbol::x(0)}}};
  auto inv = (inverse(m));
  auto eye = m * inv;
  expand(eye);
  //eval(eye);
  //normal(eye);
  std::cerr << eye << '\n';
}
#endif
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
