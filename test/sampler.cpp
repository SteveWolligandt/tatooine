#include<tatooine/sampler.h>
#include<tatooine/grid.h>
#include<catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("sampler_bilinear_scalar", "[sampler][bilinear][scalar]") {
  grid<linspace<double>, linspace<double>> g{linspace{0.0, 10.0, 11},
                                             linspace{0.0, 10.0, 11}};
  auto& prop = g.add_scalar_vertex_property("prop");
  auto  s    = prop.linear_sampler();
  prop(0, 0) = 1;
  prop(1, 0) = 3;
  prop(0, 1) = 2;
  prop(1, 1) = 4;
  SECTION("evaluation") {
    REQUIRE(s.sample(0.1, 0.1) == Approx(1.3));
    REQUIRE(s.sample(0.3, 0.1) == Approx(1.7));
    REQUIRE(s.sample(0.5, 0.1) == Approx(2.1));
    REQUIRE(s.sample(0.7, 0.1) == Approx(2.5));
    REQUIRE(s.sample(0.9, 0.1) == Approx(2.9));
    REQUIRE(s.sample(0.1, 0.3) == Approx(1.5));
    REQUIRE(s.sample(0.3, 0.3) == Approx(1.9));
    REQUIRE(s.sample(0.5, 0.3) == Approx(2.3));
    REQUIRE(s.sample(0.7, 0.3) == Approx(2.7));
    REQUIRE(s.sample(0.9, 0.3) == Approx(3.1));
  }
  SECTION("derivative") {
    auto ds = diff(s);
    for (auto const u : linspace{0.1, 0.9, 20}){
      for (auto const v : linspace{0.1, 0.9, 20}) {
        auto const d = ds(u, v);
        REQUIRE(d(0) == 2);
        REQUIRE(d(1) == 1);
      }
    }
  }
}
//==============================================================================
TEST_CASE("sampler_trilinear_scalar", "[sampler][trilinear][diff]") {
  auto  g    = grid{linspace{0.0, 10.0, 11},
                    linspace{0.0, 10.0, 11},
                    linspace{0.0, 10.0, 11}};
  auto& prop = g.add_scalar_vertex_property("prop");
  auto  s    = prop.linear_sampler();
  prop(0, 0, 0) = 1;
  prop(1, 0, 0) = 3;
  prop(0, 1, 0) = 2;
  prop(1, 1, 0) = 4;
  prop(0, 0, 1) = 3;
  prop(1, 0, 1) = 8;
  prop(0, 1, 1) = 5;
  prop(1, 1, 1) = 11;
  
  SECTION("evaluation") {
    REQUIRE(s(0.1, 0.1, 0.1) == Approx(1.541));
  }

  SECTION("derivative") {
    auto ds = diff(s);
    {auto const d = ds(0.1, 0.1, 0.1);
     REQUIRE(d(0) == Approx(2.31));
     REQUIRE(d(1) == Approx(1.11));
     REQUIRE(d(2) == Approx(2.41));}
    {auto const d = ds(0.3, 0.5, 0.7);
      REQUIRE(d(0) == Approx(4.449999999999999));
      REQUIRE(d(1) == Approx(1.91));
      REQUIRE(d(2) == Approx(3.55));}

  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
