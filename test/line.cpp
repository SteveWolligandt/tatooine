#include <tatooine/line.h>
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

  REQUIRE(approx_equal(l.sample<interpolation::linear>(0), v0));
  REQUIRE(approx_equal(l.sample<interpolation::linear>(0.5), v1));
  REQUIRE(approx_equal(l.sample<interpolation::linear>(1), v2));

  REQUIRE(approx_equal(l.sample<interpolation::linear>(0.1/2),
                       (v0 * 0.9 + v1 * 0.1)));
  REQUIRE(approx_equal(l.sample<interpolation::linear>(0.9/2),
                       (v0 * 0.1 + v1 * 0.9)));
  REQUIRE(approx_equal(l.sample<interpolation::linear>(1.1/2),
                       (v1 * 0.9 + v2 * 0.1)));
  REQUIRE(approx_equal(l.sample<interpolation::linear>(1.9/2),
                       (v1 * 0.1 + v2 * 0.9)));
  REQUIRE_NOTHROW(l.sample<interpolation::linear>(0.3));
  REQUIRE_NOTHROW(l.sample<interpolation::linear>(0.7));
  REQUIRE_THROWS(l.sample<interpolation::linear>(-0.01));
  REQUIRE_THROWS(l.sample<interpolation::linear>(1.01));
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

  REQUIRE(approx_equal(l.sample<interpolation::hermite>(0), v0));
  REQUIRE(approx_equal(l.sample<interpolation::hermite>(1), v1));
  REQUIRE(approx_equal(l.sample<interpolation::hermite>(2), v2));
  REQUIRE_NOTHROW(l.sample<interpolation::hermite>(0.5));
  REQUIRE_NOTHROW(l.sample<interpolation::hermite>(1.5));
  REQUIRE_THROWS(l.sample<interpolation::hermite>(-0.01));
  REQUIRE_THROWS(l.sample<interpolation::hermite>(2.01));
}

//==============================================================================
TEST_CASE("line_paramaterization_resample",
          "[line][parameterization][resample]") {
  vec                           v0{0.1, 0.2};
  vec                           v1{0.5, 0.9};
  vec                           v2{0.9, 0.2};
  parameterized_line<double, 2> l;
  l.push_back(v0, 0);
  l.push_back(v1, 1);
  l.push_back(v2, 2);

  l.resample<interpolation::linear>(linspace(0.0, 2.0, 101))
      .write_vtk("resampled_line_linear.vtk");
  l.resample<interpolation::hermite>(linspace(0.0, 2.0, 101))
      .write_vtk("resampled_line_hermite.vtk");
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
  // create a closed circle
  auto radius = GENERATE(1.0, 2.0, 3.0, 4.0);
  const size_t    num_samples = 1000;
  line<double, 2> circle;
  circle.set_closed(true);

  for (size_t i = 0; i < num_samples; ++i) {
    const double angle = M_PI * 2 / static_cast<double>(num_samples) * i;
    circle.push_back(std::cos(angle) * radius, std::sin(angle) * radius);
  }

  for (size_t i = 0; i < num_samples; ++i) {
    REQUIRE(circle.curvature(i) == Approx(1 / radius).margin(1e-6));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
