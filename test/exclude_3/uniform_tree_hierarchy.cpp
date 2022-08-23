#include <tatooine/uniform_tree_hierarchy.h>
#include <tatooine/geometry/ellipse.h>
#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
TEST_CASE("uniform_tree_hierarchy_ellipses",
          "[uniform_tree_hierarchy][ellipse]") {
  auto ellipses =
      std::vector<geometry::ellipse<real_number>>{{0.4, vec2{-0.5, -0.5}},
                                                  {0.4, vec2{0.5, -0.5}},
                                                  {0.4, vec2{-0.5, 0.5}},
                                                  {0.4, vec2{0.5, 0.5}}};
  auto hierarchy =
      uniform_tree_hierarchy<std::vector<geometry::ellipse<real_number>>>{
          vec2{-1, -1}, vec2{1, 1}, ellipses, 3};
  REQUIRE(hierarchy.is_splitted());
  REQUIRE_FALSE(hierarchy.child_at(0, 0)->is_splitted());
  REQUIRE_FALSE(hierarchy.child_at(1, 0)->is_splitted());
  REQUIRE_FALSE(hierarchy.child_at(0, 1)->is_splitted());
  REQUIRE_FALSE(hierarchy.child_at(1, 1)->is_splitted());
  REQUIRE(hierarchy.child_at(0, 0)->holds_ellipses());
  REQUIRE(hierarchy.child_at(1, 0)->holds_ellipses());
  REQUIRE(hierarchy.child_at(0, 1)->holds_ellipses());
  REQUIRE(hierarchy.child_at(1, 1)->holds_ellipses());
  REQUIRE(size(hierarchy.child_at(0, 0)->ellipses_in_node()) == 1);
  REQUIRE(size(hierarchy.child_at(1, 0)->ellipses_in_node()) == 1);
  REQUIRE(size(hierarchy.child_at(0, 1)->ellipses_in_node()) == 1);
  REQUIRE(size(hierarchy.child_at(1, 1)->ellipses_in_node()) == 1);
  REQUIRE(hierarchy.child_at(0, 0)->ellipses_in_node()[0] == 0);
  REQUIRE(hierarchy.child_at(1, 0)->ellipses_in_node()[0] == 1);
  REQUIRE(hierarchy.child_at(0, 1)->ellipses_in_node()[0] == 2);
  REQUIRE(hierarchy.child_at(1, 1)->ellipses_in_node()[0] == 3);
  ellipses.emplace_back(vec2{0.999, 0.999}, 0.0001);
  hierarchy.insert_ellipse(4);

  REQUIRE_FALSE(hierarchy.child_at(0, 0)->is_splitted());
  REQUIRE_FALSE(hierarchy.child_at(1, 0)->is_splitted());
  REQUIRE_FALSE(hierarchy.child_at(0, 1)->is_splitted());
  REQUIRE(hierarchy.child_at(1, 1)->is_splitted());

  REQUIRE(hierarchy.child_at(0, 0)->ellipses_in_node()[0] == 0);
  REQUIRE(hierarchy.child_at(1, 0)->ellipses_in_node()[0] == 1);
  REQUIRE(hierarchy.child_at(0, 1)->ellipses_in_node()[0] == 2);
  REQUIRE_FALSE(hierarchy.child_at(1, 1)->holds_ellipses());
  REQUIRE(size(hierarchy.child_at(1, 1)->child_at(0, 0)->ellipses_in_node()) ==
          1);
  REQUIRE(size(hierarchy.child_at(1, 1)->child_at(0, 1)->ellipses_in_node()) ==
          1);
  REQUIRE(size(hierarchy.child_at(1, 1)->child_at(1, 0)->ellipses_in_node()) ==
          1);
  REQUIRE(size(hierarchy.child_at(1, 1)->child_at(1, 1)->ellipses_in_node()) ==
          2);
  REQUIRE(hierarchy.child_at(1, 1)->child_at(0, 0)->ellipses_in_node()[0] == 3);
  REQUIRE(hierarchy.child_at(1, 1)->child_at(0, 1)->ellipses_in_node()[0] == 3);
  REQUIRE(hierarchy.child_at(1, 1)->child_at(1, 0)->ellipses_in_node()[0] == 3);
  REQUIRE(hierarchy.child_at(1, 1)->child_at(1, 1)->ellipses_in_node()[0] == 3);
  REQUIRE(hierarchy.child_at(1, 1)->child_at(1, 1)->ellipses_in_node()[1] == 4);

  REQUIRE(size(hierarchy.nearby_ellipses(vec2{0.75, 0.75})) == 2);
  REQUIRE(size(hierarchy.nearby_ellipses(vec2{0.25, 0.75})) == 1);
  REQUIRE(size(hierarchy.nearby_ellipses(vec2{0.75, 0.25})) == 1);
  REQUIRE(size(hierarchy.nearby_ellipses(vec2{0.25, 0.25})) == 1);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
