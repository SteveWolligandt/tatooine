#include <tatooine/constant_vectorfield.h>
#include <tatooine/doublegyre.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/line.h>
#include <tatooine/sinuscosinus.h>

#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("line_push_back", "[line][push_back]") {
  line<double, 2> l;
  l.push_back({0, 0});
  l.push_back({1, 1});
  l.push_back({2, 0});
  REQUIRE(approx_equal(l.vertex_at(0), vec{0, 0}, 0));
  REQUIRE(approx_equal(l.vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(l.vertex_at(2), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE("line_initializer_list", "[line][initializer_list]") {
  line<double, 2> l{{0, 0}, {1, 1}, {2, 0}};
  REQUIRE(approx_equal(l.vertex_at(0), vec{0, 0}, 0));
  REQUIRE(approx_equal(l.vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(l.vertex_at(2), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE("line_push_front", "[line][push_front]") {
  line<double, 2> l;
  l.push_front({0, 0});
  l.push_front({1, 1});
  l.push_front({2, 0});
  REQUIRE(approx_equal(l.vertex_at(2), vec{0, 0}, 0));
  REQUIRE(approx_equal(l.vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(l.vertex_at(0), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE("line_tangent_container", "[line][tangents][container]") {
  line<double, 2> l;
  l.push_front({0, 0});
  l.push_front({1, 1});
  l.push_front({2, 0});
  std::vector<vec<double, 2>> tangents(3);
  boost::copy(l.tangents(), begin(tangents));
}
//==============================================================================
TEST_CASE("line_second_derivative_container", "[line][second_derivative][container]") {
  line<double, 2> l;
  l.push_front({0, 0});
  l.push_front({1, 1});
  l.push_front({2, 0});
  std::vector<vec<double, 2>> second_derivative(3);
  boost::copy(l.second_derivatives(), begin(second_derivative));
}
//==============================================================================
TEST_CASE("line_curvature_container", "[line][curvature][container]") {
  line<double, 2> l;
  l.push_front({0, 0});
  l.push_front({1, 1});
  l.push_front({2, 0});
  std::vector<double> curvature(3);
  boost::copy(l.curvatures(), begin(curvature));
}
//==============================================================================
TEST_CASE(
    "line_property",
    "[line][tangent][second_derivative][curvature][property][container]") {
  line<double, 2> l;
  l.push_front(0, 0);
  l.push_front(1, 1);
  l.push_front(2, 0);
  l.curvatures_to_property();
  //l.write_vtk("line_property.vtk");
}
//==============================================================================
TEST_CASE("line_parameterized_initialization",
          "[line][parameterization][initialization]") {
  parameterized_line<double, 2> l{
      {vec{1, 2}, 0}, {vec{2, 3}, 1}, {vec{3, 4}, 2}};
  auto [x0, t0] = l[0];
  auto [x1, t1] = l[1];
  auto [x2, t2] = l[2];
  REQUIRE(x0(0) == 1);
  REQUIRE(x0(1) == 2);
  REQUIRE(t0 == 0);

  REQUIRE(x1(0) == 2);
  REQUIRE(x1(1) == 3);
  REQUIRE(t1 == 1);

  REQUIRE(x2(0) == 3);
  REQUIRE(x2(1) == 4);
  REQUIRE(t2 == 2);
}

//==============================================================================
TEST_CASE("line_sampling_linear",
          "[line][parameterization][linear][sampling]") {
  vec                           v0{0.1, 0.2};
  vec                           v1{0.5, 0.9};
  vec                           v2{0.9, 0.2};
  parameterized_line<double, 2> l;
  l.push_back(v0, 0);
  l.push_back(v1, 0.5);
  l.push_back(v2, 1);

  REQUIRE(approx_equal(l.sample(0), v0));
  REQUIRE(approx_equal(l.sample(0.5), v1));
  REQUIRE(approx_equal(l.sample(1), v2));

  REQUIRE(approx_equal(l.sample(0.1/2),
                       (v0 * 0.9 + v1 * 0.1)));
  REQUIRE(approx_equal(l.sample(0.9/2),
                       (v0 * 0.1 + v1 * 0.9)));
  REQUIRE(approx_equal(l.sample(1.1/2),
                       (v1 * 0.9 + v2 * 0.1)));
  REQUIRE(approx_equal(l.sample(1.9/2),
                       (v1 * 0.1 + v2 * 0.9)));
  REQUIRE_NOTHROW(l.sample(0.3));
  REQUIRE_NOTHROW(l.sample(0.7));
  REQUIRE_THROWS(l.sample(-0.01));
  REQUIRE_THROWS(l.sample(1.01));
}

//==============================================================================
TEST_CASE("line_sampling_hermite",
          "[line][parameterization][hermite][sampling]") {
  vec                           v0{0.0, 0.0};
  vec                           v1{1.0, 1.0};
  vec                           v2{2.0, 0.0};
  parameterized_line<double, 2> l;
  l.push_back(v0, 0);
  l.push_back(v1, 1);
  l.push_back(v2, 2);

  REQUIRE(approx_equal(l.sample(0), v0));
  REQUIRE(approx_equal(l.sample(1), v1));
  REQUIRE(approx_equal(l.sample(2), v2));
  REQUIRE_NOTHROW(l.sample(0.5));
  REQUIRE_NOTHROW(l.sample(1.5));
  REQUIRE_THROWS(l.sample(-0.01));
  REQUIRE_THROWS(l.sample(2.01));
}

//==============================================================================
TEST_CASE("line_paramaterization_quadratic_tangent",
          "[line][parameterization][quadratic][tangent]") {
  SECTION("simple") {
    vec v0{1.0, 1.0}; double t0 = 0;
    vec v1{3.0, 2.0}; double t1 = 3;
    vec v2{2.0, 3.0}; double t2 = 4;
    parameterized_line<double, 2> l{{v0, t0}, {v1, t1}, {v2, t2}};

    l.tangents_to_property();
    //l.write_vtk("simple_quadratic_tangents.vtk");
  }
}

//==============================================================================
TEST_CASE("line_paramaterization_uniform",
          "[line][parameterization][uniform]") {
  vec                           v0{0.0, 0.0};
  vec                           v1{1.0, 1.0};
  vec                           v2{2.0, 0.0};
  parameterized_line<double, 2> l;
  l.push_back(v0, 0);
  l.push_back(v1, 0);
  l.push_back(v2, 0);
  l.uniform_parameterization();
}

//==============================================================================
TEST_CASE("line_paramaterization_chordal",
          "[line][parameterization][chordal]") {
  vec                           v0{0.0, 0.0};
  vec                           v1{1.0, 1.0};
  vec                           v2{2.0, 0.0};
  parameterized_line<double, 2> l;
  l.push_back(v0, 0);
  l.push_back(v1, 0);
  l.push_back(v2, 0);
  l.chordal_parameterization();
}

//==============================================================================
TEST_CASE("line_paramaterization_centripetal",
          "[line][parameterization][centripetal]") {
  vec                           v0{0.0, 0.0};
  vec                           v1{1.0, 1.0};
  vec                           v2{2.0, 0.0};
  parameterized_line<double, 2> l;
  l.push_back(v0, 0);
  l.push_back(v1, 0);
  l.push_back(v2, 0);
  l.centripetal_parameterization();
}
//==============================================================================
TEST_CASE("line_curvature", "[line][curvature]") {
  SECTION("circle") {
    // create a closed circle
    auto            radius      = GENERATE(1.0, 2.0, 3.0, 4.0);
    const size_t    num_samples = 1000;
    line<double, 2> circle;
    circle.set_closed(true);

    for (size_t i = 0; i < num_samples; ++i) {
      const double angle = M_PI * 2 / static_cast<double>(num_samples) * i;
      circle.push_back(std::cos(angle) * radius, std::sin(angle) * radius);
    }

    for (size_t i = 0; i < num_samples; ++i) {
      REQUIRE(circle.curvature_at(i) == Approx(1 / radius).margin(1e-6));
    }
  }
  SECTION("streamline") {
    SECTION("constant_vectorfield") {
      constant_vectorfield<double, 3>              v;
      integration::vclibs::rungekutta43<double, 3> rk43;
      auto integral_curve = rk43.integrate(v, {0.0, 0.0, 0.0}, 0, 10);
      for (size_t i = 0; i < integral_curve.num_vertices(); ++i) {
        REQUIRE(integral_curve.curvature_at(i) == Approx(0).margin(1e-6));
      }
    }
    //SECTION("doublegyre") {
    //  numerical::doublegyre                        v;
    //  integration::vclibs::rungekutta43<double, 2> rk43;
    //  auto integral_curve = rk43.integrate(v, {0.1, 0.1}, 0, 10);
    //  double curv_add = 0;
    //  for (size_t i = 0; i < integral_curve.num_vertices(); ++i) {
    //    curv_add += integral_curve.curvature_at(i);
    //  }
    //  auto curv_mean = curv_add / integral_curve.num_vertices();
    //  std::cerr << "curvature mean doublegyre pathline x_0 = {0.1, 0.1}, t_0 = "
    //               "0, tau = 10: "
    //            << curv_mean << '\n';
    //}
  }
}
//==============================================================================
TEST_CASE("line_integrated_curvature", "[line][integrated_curvature]") {
  SECTION("pathlines") {
    SECTION("cosinus sinus vectorfield") {
      auto                    radius = GENERATE(1.0, 2.0, 3.0, 4.0);
      numerical::cosinussinus v{radius};
      integration::vclibs::rungekutta43<double, 2> rk43;
      auto full_circle = rk43.integrate(v, {0.0, 0.0}, 0, 2 * M_PI);
      auto half_circle = rk43.integrate(v, {0.0, 0.0}, 0, 2 * M_PI);
      auto kappa_dt_full = full_circle.integrated_curvature();
      auto kappa_dt_half = half_circle.integrated_curvature();
      CAPTURE(kappa_dt_full);
      CAPTURE(kappa_dt_half);
      REQUIRE(kappa_dt_full == Approx(kappa_dt_half).margin(1e-6));
    }
    SECTION("double gyre pathline") {
      numerical::doublegyre                        v;
      integration::vclibs::rungekutta43<double, 2> rk43;
      size_t                                       cnt = 0;
      for (auto t : linspace(0.0, 10.0, 100)) {
        auto integral_curve = rk43.integrate(v, {0.2, 0.2}, t, t + 10);
        auto kappa_dt       = integral_curve.integrated_curvature();
        CAPTURE(cnt, kappa_dt);
        //integral_curve.write_vtk("doublegyre_quadratic_tangents_" +
        //                         std::to_string(cnt++) + ".vtk");
      }
    }
  }
}
//==============================================================================
TEST_CASE("line_resample", "[line][parameterization][resample]") {
  SECTION("double gyre pathline") {
    numerical::doublegyre                        v;
    integration::vclibs::rungekutta43<double, 2> rk43;
    auto integral_curve = rk43.integrate(v, {0.2, 0.2}, 0, 10);
    integral_curve.write_vtk("original_dg_pathline.vtk");
    size_t i = 0;
    for (auto n : std::array{10000, 20000}) {
      integral_curve.resample(linspace(0.0, 10.0, n))
          .write_vtk("resampled_dg_pathline" + std::to_string(i++) + ".vtk");
    }
  }
  SECTION("simple line") {
    parameterized_line<double, 2> l{
        {{0.0, 0.0}, 0}, {{1.0, 1.0}, 1}, {{2.0, 0.0}, 2}};
    l.write_vtk("original_line.vtk");
    l.resample(linspace(0.0, 2.0, 10000))
        .write_vtk("resampled_line.vtk");
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
