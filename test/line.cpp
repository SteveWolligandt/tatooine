#include <tatooine/line.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(line2, "line_push_back", "[line][push_back]") {
  push_back(vec{0.0, 0.0});
  push_back(vec{1.0, 1.0});
  push_back(vec{2.0, 0.0});
  REQUIRE(approx_equal(vertex_at(0), vec{0, 0}, 0));
  REQUIRE(approx_equal(vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(vertex_at(2), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE_METHOD(line2, "line_push_front", "[line][push_front]") {
  push_front(vec{0, 0});
  push_front(vec{1, 1});
  push_front(vec{2, 0});
  REQUIRE(approx_equal(vertex_at(2), vec{0, 0}, 0));
  REQUIRE(approx_equal(vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(vertex_at(0), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE("line_initializer_list", "[line][initializer_list]") {
  auto const l = line2{vec{0, 0}, vec{1, 1}, vec{2, 0}};
  REQUIRE(approx_equal(l.vertex_at(0), vec{0, 0}, 0));
  REQUIRE(approx_equal(l.vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(l.vertex_at(2), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE_METHOD(line2, "line_property", "[line][property]") {
  push_front(vec{0, 0});
  push_front(vec{1, 1});
  push_front(vec{2, 0});
  {
    auto &prop = scalar_vertex_property("prop");
    prop[0]    = 1;
    prop[1]    = 2;
    prop[2]    = 3;
  }
  REQUIRE(scalar_vertex_property("prop")[0] == 1);
  REQUIRE(scalar_vertex_property("prop")[1] == 2);
  REQUIRE(scalar_vertex_property("prop")[2] == 3);
}
//==============================================================================
TEST_CASE_METHOD(line2, "line_tangents", "[line][tangents]") {
  push_back(vec2{0, 0});
  push_back(vec2{1, 1});
  push_back(vec2{2, 0});
  compute_parameterization();
  normalize_parameterization();
  compute_tangents();
}
//==============================================================================
TEST_CASE_METHOD(line2, "line_linear_sampler", "[line][linear][sampler]") {
  push_back(vec2{0, 0});
  push_back(vec2{1, 1});
  push_back(vec2{2, 0});
  compute_uniform_parameterization();
  auto const x = linear_sampler();
  REQUIRE(approx_equal(x(0.0), vec2{0, 0}));
  REQUIRE(approx_equal(x(0.5), vec2{0.5, 0.5}));
  REQUIRE(approx_equal(x(0.75), vec2{0.75, 0.75}));
  REQUIRE(approx_equal(x(1.0), vec2{1, 1}));
  REQUIRE(approx_equal(x(2.0), vec2{2, 0}));
}
//==============================================================================
TEST_CASE_METHOD(line2, "line_resample", "[line][parameterization][resample]") {
  push_back(vec2{0, 0});
  push_back(vec2{1, 1});
  push_back(vec2{2, 0});
  compute_normalized_parameterization();
  auto &prop      = scalar_vertex_property("prop");
  prop[0] = 1;
  prop[1] = 2;
  prop[2] = 5;

  resample<interpolation::linear>(linspace{0.0, 1.0, 101});
  resample<interpolation::cubic>(linspace{0.0, 1.0, 101});
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
