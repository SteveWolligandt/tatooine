#include <tatooine/line.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("line_push_back", "[line][push_back]") {
  line<double, 2> l;
  l.push_back({0,0});
  l.push_back({1,1});
  l.push_back({2,0});
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
  l.push_front({0,0});
  l.push_front({1,1});
  l.push_front({2,0});
  REQUIRE(approx_equal(l.vertex_at(2), vec{0, 0}, 0));
  REQUIRE(approx_equal(l.vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(l.vertex_at(0), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE("line_parameterized_initialization",
          "[line][parameterization][initialization]") {
  parameterized_line<double, 2> l{{vec{1, 2}, 0}, {vec{2, 3}, 1}, {vec{3, 4}, 2}};
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
TEST_CASE("line_sampling_linear", "[line][parameterization][linear]") {
  vec v0{0.0,0.0};
  vec v1{1.0,1.0};
  vec v2{2.0,0.0};
  parameterized_line<double, 2> l;
  l.push_back(v0, 0);
  l.push_back(v1, 1);
  l.push_back(v2, 2);

  REQUIRE(approx_equal(l.sample<interpolation::linear>(0), v0));
  REQUIRE(approx_equal(l.sample<interpolation::linear>(1), v1));
  REQUIRE(approx_equal(l.sample<interpolation::linear>(2), v2));

  REQUIRE(approx_equal(l.sample<interpolation::linear>(0.1),
                       (v0 * 0.9 + v1 * 0.1)));
  REQUIRE(approx_equal(l.sample<interpolation::linear>(0.9),
                       (v0 * 0.1 + v1 * 0.9)));
  REQUIRE(approx_equal(l.sample<interpolation::linear>(1.1),
                       (v1 * 0.9 + v2 * 0.1)));
  REQUIRE(approx_equal(l.sample<interpolation::linear>(1.9),
                       (v1 * 0.1 + v2 * 0.9)));
}

//==============================================================================
TEST_CASE("line_sampling_hermite", "[line][parameterization][hermite]") {
  vec v0{0.0,0.0};
  vec v1{1.0,1.0};
  vec v2{2.0,0.0};
  parameterized_line<double, 2> l;
  l.push_back(v0, 0);
  l.push_back(v1, 1);
  l.push_back(v2, 2);

  REQUIRE(approx_equal(l(0), v0));
  REQUIRE(approx_equal(l(1), v1));
  REQUIRE(approx_equal(l(2), v2));
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
  vec v0{0.0,0.0};
  vec v1{1.0,1.0};
  vec v2{2.0,0.0};
  parameterized_line<double, 2> l;
  l.push_back(v0, 0);
  l.push_back(v1, 0);
  l.push_back(v2, 0);
  l.centripetal_parameterization();
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
