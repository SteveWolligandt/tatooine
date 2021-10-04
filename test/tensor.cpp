#include <tatooine/symbolic.h>
#include <tatooine/tensor.h>
#include <tatooine/utility.h>

#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("tensor_initializers", "[tensor][initializers]") {
  SECTION("constructors") {
    auto m3z = mat3::zeros();
    m3z.for_indices([&m3z](auto const... is) { CHECK(m3z(is...) == 0); });
    auto m3o = mat3::ones();
    m3o.for_indices([&m3o](auto const... is) { CHECK(m3o(is...) == 1); });
    [[maybe_unused]] auto m3ru = mat3::randu();
    [[maybe_unused]] auto m3rn = mat3::randn();
  }
  SECTION("factory functions") {
    mat3 m3z{tag::zeros};
    m3z.for_indices([&m3z](auto const... is) { CHECK(m3z(is...) == 0); });
    mat3 m3o{tag::ones};
    m3o.for_indices([&m3o](auto const... is) { CHECK(m3o(is...) == 1); });
    mat3 m3f{tag::fill{3}};
    m3f.for_indices([&m3f](auto const... is) { CHECK(m3f(is...) == 3); });
    mat3 m3ru{random::uniform{}};
    mat3 m3rn{random::normal{}};
  }
}
//==============================================================================
TEST_CASE("tensor_print_matrix", "[tensor][print][matrix]") {
  // std::cerr << mat<int, 3, 3>{random::uniform{0, 9}} << '\n';
}
//==============================================================================
TEST_CASE("tensor_assignment", "[tensor][assignment]") {
  vec v{1.0, 2.0, 3.0};

  mat m{{1.0, 2.0, 3.0}, {4.0, 5.0, 7.0}, {8.0, 9.0, 10.0}};

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
  REQUIRE(v.rank() == 1);
  REQUIRE(v.dimension(0) == 3);
  REQUIRE(v.num_components() == 3);

  vec v2{1.0, 2.0};
  REQUIRE(v(0) == 1);
  REQUIRE(v(1) == 2);

  mat m{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  REQUIRE(m(0, 0) == 1);
  REQUIRE(m(0, 1) == 2);
  REQUIRE(m(1, 0) == 3);
  REQUIRE(m(1, 1) == 4);
  REQUIRE(m(2, 0) == 5);
  REQUIRE(m(2, 1) == 6);
  REQUIRE(m.rank() == 2);
  REQUIRE(m.dimension(0) == 3);
  REQUIRE(m.dimension(1) == 2);
  REQUIRE(m.num_components() == 6);

  auto c0 = m.col(0);
  REQUIRE(c0(0) == 1);
  REQUIRE(c0(1) == 3);
  REQUIRE(c0(2) == 5);
  REQUIRE(c0.rank() == 1);
  REQUIRE(c0.dimension(0) == 3);
  REQUIRE(c0.num_components() == 3);

  auto c1 = m.col(1);
  REQUIRE(c1(0) == 2);
  REQUIRE(c1(1) == 4);
  REQUIRE(c1(2) == 6);
  REQUIRE(c1.rank() == 1);
  REQUIRE(c1.dimension(0) == 3);
  REQUIRE(c1.num_components() == 3);

  auto r0 = m.row(0);
  REQUIRE(r0(0) == 1);
  REQUIRE(r0(1) == 2);
  REQUIRE(r0.rank() == 1);
  REQUIRE(r0.dimension(0) == 2);
  REQUIRE(r0.num_components() == 2);

  auto r1 = m.row(1);
  REQUIRE(r1(0) == 3);
  REQUIRE(r1(1) == 4);
  REQUIRE(r1.rank() == 1);
  REQUIRE(r1.dimension(0) == 2);
  REQUIRE(r1.num_components() == 2);

  auto r2 = m.row(2);
  REQUIRE(r2(0) == 5);
  REQUIRE(r2(1) == 6);
  REQUIRE(r2.rank() == 1);
  REQUIRE(r2.dimension(0) == 2);
  REQUIRE(r2.num_components() == 2);

  REQUIRE(dot(m.col(0), m.col(1)) == (1 * 2 + 3 * 4 + 5 * 6));
  auto prod = m * v2;
  REQUIRE(prod(0) == (1 * 1 + 2 * 2));
  REQUIRE(prod(1) == (1 * 3 + 2 * 4));
  REQUIRE(prod(2) == (1 * 5 + 2 * 6));
  REQUIRE(prod.rank() == 1);
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
  m.for_indices([&](auto const... is) { CHECK(m(is...) == -m_neg(is...)); });
}
//==============================================================================
TEST_CASE("tensor_addition", "[tensor][operation][addition]") {
  auto m0    = mat4::randu();
  auto m1    = mat4::randu();
  auto added = m0 + m1;
  m0.for_indices([&](auto const... is) {
    CHECK((added(is...) == m0(is...) + m1(is...)));
  });
}
//==============================================================================
TEST_CASE("tensor_eigenvalue", "[tensor][eigenvalue]") {
  auto const eps = 1e-4;
  SECTION("2x2") {
    SECTION("test 1") {
      mat2 const A{{1, 2}, {3, 4}};
      auto const eigvals   = eigenvalues(A);
      auto const real_vals = real(eigvals);

      REQUIRE(real_vals(0) == Approx(-0.372281323269014).epsilon(eps));
      REQUIRE(real_vals(1) == Approx(5.372281323269014).epsilon(eps));
    }
    SECTION("test 2") {
      mat2 const A{{0, 1}, {-1, 0}};
      auto const eigvals   = eigenvalues(A);
      auto const real_vals = real(eigvals);
      auto const imag_vals = imag(eigvals);

      REQUIRE(real_vals(0) == Approx(0).epsilon(eps));
      REQUIRE(real_vals(1) == Approx(0).epsilon(eps));
      REQUIRE(imag_vals(0) == Approx(1).epsilon(eps));
      REQUIRE(imag_vals(1) == Approx(-1).epsilon(eps));
    }
    SECTION("test 3") {
      mat2 const A{{1, 5}, {-3, 3}};
      auto const eigvals   = eigenvalues(A);
      auto const real_vals = real(eigvals);
      auto const imag_vals = imag(eigvals);

      REQUIRE(real_vals(0) == Approx(2).epsilon(eps));
      REQUIRE(real_vals(1) == Approx(2).epsilon(eps));
      REQUIRE(imag_vals(0) == Approx(3.741657386773942).epsilon(eps));
      REQUIRE(imag_vals(1) == Approx(-3.741657386773942).epsilon(eps));
    }
    SECTION("test 4") {
      mat2 const A{{0.908712465200650, 1.095797002691362},
                   {1.095797002691362, 1.340019245906524}};
      auto const [eigvectors, eigvals] = eigenvectors_sym(A);
      mat2 const expected_vecs{{-0.772365411077248, 0.635178456633625},
                                {0.635178456633625, 0.772365411077248}};
      vec2 const expected_vals{7.550089243812286e-03, 2.241181621863362};
      CAPTURE(A, expected_vals, eigvals, expected_vecs, eigvectors);
      REQUIRE(approx_equal(eigvals, expected_vals));
      REQUIRE(approx_equal(eigvectors, expected_vecs));
    }
    SECTION("test 5") {
      mat2 const A{{1.122370425420169e-02, 2.264844840657990e-02},
                   {2.264844840657990e-02, 4.369530401129955e-01}};
      auto const [eigvectors, eigvals] = eigenvectors_sym(A);
      mat2 const expected_vecs{{-9.985958405945158e-01, 5.297496717632329e-02},
                                {5.297496717632329e-02, 9.985958405945158e-01}};
      vec2 const expected_vals{1.002221636274510e-02, 4.381545280044520e-01};
      CAPTURE(A, expected_vals, eigvals, expected_vecs, eigvectors);
      REQUIRE(approx_equal(eigvals, expected_vals));
      REQUIRE(approx_equal(eigvectors, expected_vecs));
    }
  }
  SECTION("3x3") {
    auto const A = mat{{1.0,  2.0,  3.0},
                       {4.0,  6.0,  8.0},
                       {9.0, 12.0, 15.0}};
    auto const [eigvecs, eigvals] = eigenvectors(A);
    auto const eigvals2           = eigenvalues(A);
    auto const ve                 = real(eigvecs);
    auto const va                 = real(eigvals);
    auto const va2                = real(eigvals2);

    REQUIRE(va(0) == Approx(2.2874e+01).epsilon(eps));
    REQUIRE(va(1) == Approx(-8.7434e-01).epsilon(eps));
    REQUIRE(va(2) == Approx(2.0305e-16).margin(eps));
    REQUIRE(va2(0) == Approx(2.2874e+01).epsilon(eps));
    REQUIRE(va2(1) == Approx(-8.7434e-01).epsilon(eps));
    REQUIRE(va2(2) == Approx(2.0305e-16).margin(eps));

    REQUIRE(ve(0, 0) == Approx(0.16168).epsilon(eps));
    REQUIRE(ve(1, 0) == Approx(0.45378).epsilon(eps));
    REQUIRE(ve(2, 0) == Approx(0.87632).epsilon(eps));

    REQUIRE(ve(0, 1) == Approx(0.62794).epsilon(eps));
    REQUIRE(ve(1, 1) == Approx(0.40677).epsilon(eps));
    REQUIRE(ve(2, 1) == Approx(-0.66350).epsilon(eps));

    REQUIRE(ve(0, 2) == Approx(0.40825).epsilon(eps));
    REQUIRE(ve(1, 2) == Approx(-0.81650).epsilon(eps));
    REQUIRE(ve(2, 2) == Approx(0.40825).epsilon(eps));
  }
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
  auto  A  = mat<double, 2, 3>::randu();
  auto  At = transposed(A);
  auto& A2 = transposed(At);

  REQUIRE(&A == &A2);

  REQUIRE(A.dimension(0) == At.dimension(1));
  REQUIRE(A.dimension(1) == At.dimension(0));

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      REQUIRE(A(i, j) == At(j, i));
    }
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
    mat  A{{-1.79222, -7.94109, 3.67540},
          {2.38520, 0.82284, 8.53506},
          {-1.37601, -6.15705, -0.71982}};
    auto aA = abs(A);
    CAPTURE(A, aA);
    REQUIRE(aA(0, 0) == -A(0, 0));
    REQUIRE(aA(1, 0) == A(1, 0));
    REQUIRE(aA(2, 0) == -A(2, 0));
    REQUIRE(aA(0, 1) == -A(0, 1));
    REQUIRE(aA(1, 1) == A(1, 1));
    REQUIRE(aA(2, 1) == -A(2, 1));
    REQUIRE(aA(0, 2) == A(0, 2));
    REQUIRE(aA(1, 2) == A(1, 2));
    REQUIRE(aA(2, 2) == -A(2, 2));
  }
}
TEST_CASE("tensor_sum", "[tensor][sum]") {
  SECTION("vec") {
    vec x{-1, 2, -3};
    REQUIRE(sum(x) == Approx(-1 + 2 - 3));
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
TEST_CASE("tensor_matrix_norm1", "[tensor][norm1][1-norm][matrix]") {
  mat A{{-1.79222, -7.94109, 3.67540},
        {2.38520, 0.82284, 8.53506},
        {-1.37601, -6.15705, -0.71982}};
  REQUIRE(norm1(A) == Approx(14.921));
}
//==============================================================================
//TEST_CASE("tensor_svd", "[tensor][svd]") {
//  //----------------------------------------------------------------------------
//  SECTION("2x2") {
//    mat  A{{-1.79222, -7.94109}, {2.38520, 0.82284}};
//    auto s = singular_values(A);
//    INFO("A =\n" << A);
//    INFO("s = " << s);
//    REQUIRE(approx_equal(s, vec{8.256124221718464, 2.115566226250951}));
//  }
//  SECTION("3x3") {
//    mat A{{-1.79222, -7.94109, 3.67540},
//          {2.38520, 0.82284, 8.53506},
//          {-1.37601, -6.15705, -0.71982}};
//    auto const [U, s, VT] = svd(A);
//    auto const S          = diag(s);
//    INFO("A =\n" << A);
//    INFO("U*S*VT =\n" << U * S * VT);
//    INFO("U*S =\n" << U * S);
//    INFO("S*VT =\n" << S * VT);
//    INFO("U =\n" << U);
//    INFO("S =\n" << S);
//    INFO("V =\n" << transposed(VT));
//    REQUIRE(approx_equal(s, vec{1.073340050125074e+01, 9.148458171897648e+00,
//                                6.447152840514361e-01}));
//    REQUIRE(approx_equal(U * S * VT, A));
//  }
//  //----------------------------------------------------------------------------
//  SECTION("3x4") {
//    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//    SECTION("full") {
//      mat A{
//          {6.405871813e+00, -4.344670595e+00, 9.471184691e+00, 5.850792157e+00},
//          {3.049605906e+00, 1.018629735e+00, 5.535464761e+00, 2.691779530e+00},
//          {9.002176872e+00, 3.332492228e-01, -2.365651229e+00,
//           9.458283935e+00}};
//      auto const [U, s, VT] = svd(A, tag::full);
//      auto const S          = diag_rect<3, 4>(s);
//      INFO("A =\n" << A);
//      INFO("U*S*VT =\n" << U * S * VT);
//      INFO("U =\n" << U);
//      INFO("S =\n" << S);
//      INFO("V =\n" << transposed(VT));
//      REQUIRE(approx_equal(s, vec{1.733194199066472e+01, 9.963856829919013e+00,
//                                  2.932957998778161e+00}));
//      REQUIRE(approx_equal(U * S * VT, A));
//    }
//    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//    SECTION("economy") {
//      mat A{
//          {6.405871813e+00, -4.344670595e+00, 9.471184691e+00, 5.850792157e+00},
//          {3.049605906e+00, 1.018629735e+00, 5.535464761e+00, 2.691779530e+00},
//          {9.002176872e+00, 3.332492228e-01, -2.365651229e+00,
//           9.458283935e+00}};
//      auto const [U, s, VT] = svd(A, tag::economy);
//      auto const S          = diag(s);
//      INFO("A =\n" << A);
//      INFO("U*S*VT =\n" << U * S * VT);
//      INFO("U =\n" << U);
//      INFO("S =\n" << S);
//      INFO("V =\n" << transposed(VT));
//      REQUIRE(approx_equal(s, vec{1.733194199066472e+01, 9.963856829919013e+00,
//                                  2.932957998778161e+00}));
//      REQUIRE(approx_equal(U * S * VT, A));
//    }
//  }
//  //----------------------------------------------------------------------------
//  SECTION("4x3") {
//    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//    SECTION("full") {
//      mat A{{6.405871813e+00, -4.344670595e+00, 9.471184691e+00},
//            {5.850792157e+00, 3.049605906e+00, 1.018629735e+00},
//            {5.535464761e+00, 2.691779530e+00, 9.002176872e+00},
//            {3.332492228e-01, -2.365651229e+00, 9.458283935e+00}};
//      auto const [U, s, VT] = svd(A, tag::full);
//      auto const S          = diag_rect<4, 3>(s);
//      INFO("A =\n" << A);
//      INFO("U*S*VT =\n" << U * S * VT);
//      INFO("U =\n" << U);
//      INFO("S =\n" << S);
//      INFO("V =\n" << transposed(VT));
//      REQUIRE(approx_equal(s, vec{1.814712763774386e+01, 7.766671714721034e+00,
//                                  4.317113346117844e+00}));
//      REQUIRE(approx_equal(U * S * VT, A));
//    }
//    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//    SECTION("economy") {
//      mat A{
//          {6.405871813e+00, -4.344670595e+00, 9.471184691e+00, 5.850792157e+00},
//          {3.049605906e+00, 1.018629735e+00, 5.535464761e+00, 2.691779530e+00},
//          {9.002176872e+00, 3.332492228e-01, -2.365651229e+00,
//           9.458283935e+00}};
//      auto const [U, s, VT] = svd(A, tag::economy);
//      auto const S          = diag(s);
//      INFO("A =\n" << A);
//      INFO("U*S*VT =\n" << U * S * VT);
//      INFO("U =\n" << U);
//      INFO("S =\n" << S);
//      INFO("V =\n" << transposed(VT));
//      REQUIRE(approx_equal(s, vec{1.733194199066472e+01, 9.963856829919013e+00,
//                                  2.932957998778161e+00}));
//      REQUIRE(approx_equal(U * S * VT, A));
//    }
//  }
//  //----------------------------------------------------------------------------
//  SECTION("left") {
//    // just check if it compiles
//    mat A{{-1.79222, -7.94109, 3.67540},
//          {2.38520, 0.82284, 8.53506},
//          {2.38520, 0.82284, 8.53506},
//          {-1.37601, -6.15705, -0.71982}};
//    [[maybe_unused]] std::tuple<mat<double, 4, 4>, vec<double, 3>> LF =
//        svd_left(A, tag::full);
//    [[maybe_unused]] std::tuple<mat<double, 4, 3>, vec<double, 3>> LE =
//        svd_left(A, tag::economy);
//  }
//  //----------------------------------------------------------------------------
//  SECTION("right") {
//    // just check if it compiles
//    mat A{
//        {6.405871813e+00, -4.344670595e+00, 9.471184691e+00, 5.850792157e+00},
//        {3.049605906e+00, 1.018629735e+00, 5.535464761e+00, 2.691779530e+00},
//        {9.002176872e+00, 3.332492228e-01, -2.365651229e+00, 9.458283935e+00}};
//    [[maybe_unused]] std::tuple<vec<double, 3>, mat<double, 4, 4>> RF =
//        svd_right(A, tag::full);
//    [[maybe_unused]] std::tuple<vec<double, 3>, mat<double, 3, 4>> RE =
//        svd_right(A, tag::economy);
//  }
//}
//==============================================================================
//TEST_CASE("tensor_condition_number",
//          "[tensor][cond][condition][condition_number]") {
//  mat A{{-1.79222, -7.94109, 3.67540},
//        {2.38520, 0.82284, 8.53506},
//        {-1.37601, -6.15705, -0.71982}};
//  CAPTURE(A);
//  REQUIRE(condition_number(A, 2) == Approx(16.64827989465566));
//  REQUIRE(condition_number(A, 1) == Approx(26.47561499847143));
//}
//==============================================================================
TEST_CASE("tensor_inverse", "[tensor][inverse]") {
  SECTION("2x2") {
    SECTION("symmetric") {
      mat const A{{-8.440076080242212, -2.928428126618154},
                  {-2.928428126618154, 6.910592837797260}};
      mat const invA{{-1.032948519056287e-01, -4.377215335591266e-02},
                     {-4.377215335591266e-02, 1.261565252782578e-01}};
      INFO("A = \n" << A);
      INFO("inv(A) = \n" << *inv(A));
      INFO("inv_sym(A) = \n" << *inv_sym(A));
      INFO("invA = \n" << invA);
      REQUIRE(approx_equal(*inv(A), invA));
      REQUIRE(approx_equal(*inv_sym(A), invA));
    }
    SECTION("non-symmetric") {
      mat const A{{-2.885334385073613, 1.078956733870776},
                  {9.080364391292385, 4.875068575051886}};
      mat const invA{{-2.042895558527066e-01, 4.521363926545416e-02},
                     {3.805123107336210e-01, 1.209098233058101e-01}};
      INFO("A = \n" << A);
      INFO("inv(A) = \n" << *inv(A));
      INFO("invA = \n" << invA);
      REQUIRE(approx_equal(*inv(A), invA));
    }
  }
  SECTION("3x3"){} {
    SECTION("symmetric") {
      mat const A{{1.064928271071055e+02, 5.634642453472562e+01,
                   -3.402565790227180e+01},
                  {5.634642453472562e+01, 4.267363152922029e+01,
                   -3.444091660187971e+01},
                  {-3.402565790227180e+01, -3.444091660187971e+01,
                   1.901133286147969e+02}};
      mat const invA{{3.196413521586861e-02, -4.402539061060532e-02,
                      -2.254834416457464e-03},
                     {-4.402539061060532e-02, 8.808445663148774e-02,
                      8.077900460878503e-03},
                     {-2.254834416457464e-03, 8.077900460878503e-03,
                      6.319851850288617e-03}};
      INFO("A = \n" << A);
      INFO("inv(A) = \n" << *inv(A));
      INFO("inv_sym(A) = \n" << *inv_sym(A));
      INFO("invA = \n" << invA);
      REQUIRE(approx_equal(*inv(A), invA));
      REQUIRE(approx_equal(*inv_sym(A), invA));
    }
    SECTION("non-symmetric") {
      mat const A{
          {3.515752719640549e+00, 7.115778381209886e+00, 6.595301960558867e+00},
          {1.576618127685059e-02, 2.214604090099829e+00, 6.145641681782411e+00},
          {-8.665651707796259e+00, 7.017073536451740e+00,
           -8.110517127655505e+00}};
      mat const invA{{1.312234927387698e-01, -2.233939031857682e-01,
                      -6.256571756846976e-02},
                     {1.141285533158541e-01, -6.151950324683048e-02,
                      4.619131457920062e-02},
                     {-4.146327878179570e-02, 1.854588131396796e-01,
                      -1.648469874994442e-02}};
      INFO("A = \n" << A);
      INFO("inv(A) = \n" << *inv(A));
      INFO("invA = \n" << invA);
      REQUIRE(approx_equal(*inv(A), invA));
    }
  }
  SECTION("4x4") {
    SECTION("symmetric") {
      mat const A{{1.350341311909890e+02, 8.567365169041685e+01,
                   -1.132381081472242e+02, 8.432808625069161e+01},
                  {8.567365169041685e+01, 1.293144351930369e+02,
                   -1.356057585851566e+02, 4.648840935750652e+01},
                  {-1.132381081472242e+02, -1.356057585851566e+02,
                   1.538237973045502e+02, -5.851341430431881e+01},
                  {8.432808625069161e+01, 4.648840935750652e+01,
                   -5.851341430431881e+01, 6.591531362592717e+01}};
      mat const invA{{3.138763030506230e-01, 3.194962322369723e-01,
                      4.140805922642098e-01, -2.593053024761461e-01},
                     {3.194962322369723e-01, 4.342150031830025e-01,
                      5.224225592275986e-01, -2.512275537360439e-01},
                     {4.140805922642098e-01, 5.224225592275986e-01,
                      6.495453284994243e-01, -3.215960238699798e-01},
                     {-2.593053024761461e-01, -2.512275537360439e-01,
                      -3.215960238699798e-01, 2.386123500582727e-01}};
      INFO("A = \n" << A);
      INFO("inv(A) = \n" << *inv(A));
      INFO("inv_sym(A) = \n" << inv_sym(A));
      INFO("invA = \n" << invA);
      REQUIRE(approx_equal(*inv(A), invA));
      REQUIRE(approx_equal(inv_sym(A), invA));
    }
    SECTION("non-symmetric") {
      mat const A{{2.819414707372358e+00, 5.092058185719120e+00,
                   6.206984260584314e+00, -7.913868947673507e+00},
                  {8.802209197395886e+00, 1.068140649177005e+00,
                   -1.477683418398215e-01, -7.118482177335404e+00},
                  {-8.904260165719547e+00, -4.197369384960997e+00,
                   -1.274681167278112e+00, 7.436076006450509e+00},
                  {-6.157685777237223e-01, 1.546786636375565e+00,
                   3.502618062398810e+00, -7.132689569517438e+00}};
      mat const invA{{1.698108732523950e-01, 2.257553204291699e-01,
                      1.802456994608172e-01, -2.258019902590027e-01},
                     {-1.995958663243526e-01, -4.906997788721548e-01,
                      -5.572788670770507e-01, 1.301956488485441e-01},
                     {4.649456755205916e-01, 3.730684886917653e-01,
                      5.386029639462839e-01, -3.266801297913252e-01},
                     {1.703748201372928e-01, 5.729893752058061e-02,
                      1.280778195586736e-01, -2.528932523294680e-01}};
      INFO("A = \n" << A);
      INFO("inv(A) = \n" << *inv(A));
      INFO("invA = \n" << invA);
      REQUIRE(approx_equal(*inv(A), invA));
    }
  }
  SECTION("diagonal") {
    SECTION("2x2") {
      auto const A  = diag(vec2::randu());
      auto const Ai = *inv(A);
      for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
          if (i == j) {
            REQUIRE(Ai(i, j) == 1 / A(i, j));
          } else {
            REQUIRE(Ai(i, j) == 0);
          }
        }
      }
    }
    SECTION("3x3") {
      auto const A  = diag(vec3::randu());
      auto const Ai = *inv(A);
      for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
          if (i == j) {
            REQUIRE(Ai(i, j) == 1 / A(i, j));
          } else {
            REQUIRE(Ai(i, j) == 0);
          }
        }
      }
    }
    SECTION("4x4") {
      auto const A  = diag(vec4::randu());
      auto const Ai = *inv(A);
      for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
          if (i == j) {
            REQUIRE(Ai(i, j) == 1 / A(i, j));
          } else {
            REQUIRE(Ai(i, j) == 0);
          }
        }
      }
    }
  }
}
//==============================================================================
TEST_CASE("tensor_vandermonde_matrix", "[tensor][matrix][vander]") {
  auto V0 = mat3::vander(1, 2, 3);
  for (size_t i = 0; i < 3; ++i) {
    REQUIRE(V0(i, 0) == 1);
  }
  REQUIRE(V0(0, 1) == 1);
  REQUIRE(V0(1, 1) == 2);
  REQUIRE(V0(2, 1) == 3);
  REQUIRE(V0(0, 2) == 1);
  REQUIRE(V0(1, 2) == 4);
  REQUIRE(V0(2, 2) == 9);

  auto V1 = mat5::vander(-2, -1, 0, 1, 2);
  for (size_t i = 0; i < 5; ++i) {
    REQUIRE(V1(i, 0) == 1);
  }
  // REQUIRE(V1(0, 1) == 1);
  // REQUIRE(V1(1, 1) == 2);
  // REQUIRE(V1(2, 1) == 3);
  // REQUIRE(V1(0, 2) == 1);
  // REQUIRE(V1(1, 2) == 4);
  // REQUIRE(V1(2, 2) == 9);
}
//==============================================================================
TEST_CASE("tensor_diag", "[tensor][diag]") {
  vec       v{1.0, 2.0, 3.0};
  vec const cv{1.0, 2.0, 3.0};
  {
    auto V = diag(v);
    CAPTURE(type_name(V));
    REQUIRE(std::is_reference_v<decltype(V)::tensor_t>);
    REQUIRE_FALSE(
        std::is_const_v<std::remove_reference_t<decltype(V)::tensor_t>>);

    auto IV = *inv(V);
    REQUIRE_FALSE(std::is_reference_v<decltype(IV)::tensor_t>);
    REQUIRE_FALSE(std::is_const_v<decltype(IV)::tensor_t>);
  }

  {
    auto CV = diag(cv);
    CAPTURE(type_name(CV));
    REQUIRE(std::is_reference_v<decltype(CV)::tensor_t>);
    REQUIRE(std::is_const_v<std::remove_reference_t<decltype(CV)::tensor_t>>);

    auto ICV = *inv(CV);
    REQUIRE_FALSE(std::is_reference_v<decltype(ICV)::tensor_t>);
    REQUIRE_FALSE(std::is_const_v<decltype(ICV)::tensor_t>);
  }
  {
    auto MV = diag(vec{1.0, 2.0, 3.0});
    CAPTURE(type_name(MV));
    REQUIRE_FALSE(std::is_reference_v<decltype(MV)::tensor_t>);
    REQUIRE_FALSE(std::is_const_v<decltype(MV)::tensor_t>);

    auto IMV = *inv(MV);
    REQUIRE_FALSE(std::is_reference_v<decltype(IMV)::tensor_t>);
    REQUIRE_FALSE(std::is_const_v<decltype(IMV)::tensor_t>);
  }
}
//==============================================================================
TEST_CASE("tensor_diag_inverse", "[tensor][diag][inverse]") {
  vec const v{1.0, 2.0, 3.0};
  auto      V  = diag(v);
  auto      IV = *inv(V);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        REQUIRE(IV(i, j) == 1 / V(i, j));
        REQUIRE(IV(i, j) == 1 / v(i));
      } else {
        REQUIRE(IV(i, j) == 0);
      }
    }
  }
}
//==============================================================================
TEST_CASE("tensor_solve_qr", "[tensor][solve][qr]") {
  SECTION("[3x2] * x = [3]") {
    auto const A = mat{{0.558452565029228, 0.588181218880352},
                       {0.310060713289403, 0.745229548770475},
                       {0.574182859934675, 0.672548212017124}};
    auto const b = vec{0.959108917701253, 0.824089907371562, 0.216504922237786};
    auto const x = solve_qr(A, b);
    REQUIRE(x(0) == Approx(-0.122835608990910));
    REQUIRE(x(1) == Approx(1.063825231297154));
  }
  SECTION("[10x3] * x = [3]") {
    auto const A = mat{
        {8.965093625403926e-01, 3.893250293791598e-01, 3.305594961369532e-01},
        {2.403050248090894e-01, 8.836745625824101e-01, 5.722894106761662e-01},
        {7.146705532712633e-01, 6.140332082043635e-01, 3.042693218568352e-01},
        {7.164723388996441e-02, 2.104420571805208e-01, 7.758014534737182e-01},
        {6.234541574707730e-02, 4.253903093209775e-01, 6.951198154185299e-01},
        {9.298071959976453e-01, 5.365139016381471e-03, 8.049938496151908e-01},
        {3.654880261090755e-01, 2.834181302915960e-01, 1.459213249563707e-02},
        {9.266321323108422e-01, 4.255710690609775e-01, 5.555195811338909e-01},
        {7.201109603362171e-01, 3.773362170594952e-02, 5.093482205989774e-02},
        {9.222061381020611e-01, 4.398406997561723e-01, 3.150885879908960e-02}};
    auto const b =
        vec{8.833326321826099e-01, 1.096405426335730e-01, 4.346577071588996e-01,
            5.566271839555079e-01, 3.988973169837728e-01, 6.473914675875588e-01,
            6.781564356703795e-01, 1.596150228791643e-01, 6.935348727541041e-01,
            5.778721894369010e-02

        };
    auto const x = solve_qr(A, b);
    REQUIRE(x.dimension(0) == 3);
    REQUIRE(x(0) == Approx(0.457193947639988));
    REQUIRE(x(1) == Approx(-0.124366910738489));
    REQUIRE(x(2) == Approx(0.423926150017107));
  }
  SECTION("[3x2] * X = [3x3]") {
    auto const A = mat{{5.585869366509278e-01, 2.214462772401433e-01},
                       {2.918793314368302e-01, 2.659715516929950e-02},
                       {2.146386150607548e-01, 8.039619985765174e-01}};
    auto const B =
        mat{{0.279979694746482, 0.315199822433918, 0.111821780748876},
            {0.758387692091753, 0.230406806948115, 0.161407203444348},
            {0.744379569825457, 0.520073661017721, 0.922036927885906}};

    auto const X = solve_qr(A, B);
    auto const S = mat{
        {7.737842084745754e-01, 4.482121340398817e-01, -9.328850644791921e-02},
        {6.398173008803022e-01, 5.139792582646610e-01, 1.147391148379995e+00}};

    REQUIRE(X.dimension(0) == 2);
    REQUIRE(X.dimension(1) == 3);
    for_loop(
        [&](auto const... is) {
          CAPTURE(std::array{is...}, X, S);
          REQUIRE(X(is...) == Approx(S(is...)));
        },
        2, 3);
  }
}
//==============================================================================
#if TATOOINE_GINAC_AVAILABLE
//==============================================================================
TEST_CASE("tensor_symbolic", "[tensor][symbolic]") {
  vec v{symbolic::symbol::x(0),
        symbolic::symbol::x(0) * symbolic::symbol::x(1)};
  // auto m = mat2::randu();
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
  mat  m{{symbol::x(0), symbol::x(1)},
        {symbol::x(1) * symbol::t(), GiNaC::ex{symbol::x(0)}}};
  auto inv = (*inv(m));
  auto eye = m * inv;
  expand(eye);
  // eval(eye);
  // normal(eye);
}
#endif
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
