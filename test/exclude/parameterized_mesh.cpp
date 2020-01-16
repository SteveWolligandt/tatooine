#include <tatooine/parameterized_mesh.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

struct parameterized_surface_fixture : parameterized_mesh<double, 3> {
  parameterized_surface_fixture() {
    std::vector vs{
        insert_vertex({0, 0, 0}, {0, 0}), insert_vertex({1, 0, 1}, {1, 0}),
        insert_vertex({1, 1, 2}, {1, 1}), insert_vertex({0, 1, 3}, {0, 1})};

    insert_face(vs[0], vs[1], vs[2]);
    insert_face(vs[0], vs[2], vs[3]);
  }
};

TEST_CASE_METHOD(parameterized_surface_fixture, "parameterized_mesh") {
  SECTION("Sampling") {
    REQUIRE(sample(0, 0)(0) == 0);
    REQUIRE(sample(0, 0)(1) == 0);
    REQUIRE(sample(0, 0)(2) == 0);

    REQUIRE(sample(1, 0)(0) == 1);
    REQUIRE(sample(1, 0)(1) == 0);
    REQUIRE(sample(1, 0)(2) == 1);

    REQUIRE(sample(1, 1)(0) == 1);
    REQUIRE(sample(1, 1)(1) == 1);
    REQUIRE(sample(1, 1)(2) == 2);

    REQUIRE(sample(0, 1)(0) == 0);
    REQUIRE(sample(0, 1)(1) == 1);
    REQUIRE(sample(0, 1)(2) == 3);

    REQUIRE(sample(0.5, 0.5)(0) == 0.5);
    REQUIRE(sample(0.5, 0.5)(1) == 0.5);
    REQUIRE(sample(0.5, 0.5)(2) == 1);

    REQUIRE_THROWS(sample(-1, 0));
    REQUIRE_THROWS(sample(2, 0));
    REQUIRE_THROWS(sample(0, -1));
    REQUIRE_THROWS(sample(0, 2));
  }
  SECTION("Copy Constructor") {
    parameterized_mesh<double, 3> other{*this};
    REQUIRE(other.num_vertices() == num_vertices());
    REQUIRE(sample(0, 0)(0) == other(0, 0)(0));
    REQUIRE(sample(0, 0)(1) == other(0, 0)(1));
    REQUIRE(sample(0, 0)(2) == other(0, 0)(2));

    REQUIRE(sample(1, 0)(0) == other(1, 0)(0));
    REQUIRE(sample(1, 0)(1) == other(1, 0)(1));
    REQUIRE(sample(1, 0)(2) == other(1, 0)(2));

    REQUIRE(sample(1, 1)(0) == other(1, 1)(0));
    REQUIRE(sample(1, 1)(1) == other(1, 1)(1));
    REQUIRE(sample(1, 1)(2) == other(1, 1)(2));

    REQUIRE(sample(0, 1)(0) == other(0, 1)(0));
    REQUIRE(sample(0, 1)(1) == other(0, 1)(1));
    REQUIRE(sample(0, 1)(2) == other(0, 1)(2));

    REQUIRE(sample(0.5, 0.5)(0) == other(0.5, 0.5)(0));
    REQUIRE(sample(0.5, 0.5)(1) == other(0.5, 0.5)(1));
    REQUIRE(sample(0.5, 0.5)(2) == other(0.5, 0.5)(2));
  }
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
