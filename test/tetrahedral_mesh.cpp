#include <tatooine/geometry/sphere.h>
#include <tatooine/unstructured_tetrahedral_grid.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("unstructured_tetrahedral_grid_copy",
          "[unstructured_tetrahedral_grid][copy]") {
  auto       mesh = unstructured_tetrahedral_grid3{};
  auto const v0   = mesh.insert_vertex(0.0, 0.0, 0.0);
  auto const v1   = mesh.insert_vertex(1.0, 0.0, 0.0);
  auto const v2   = mesh.insert_vertex(0.0, 1.0, 0.0);
  auto const v3   = mesh.insert_vertex(0.0, 0.0, 1.0);
  auto const t0   = mesh.insert_simplex(v0, v1, v2, v3);

  auto& vertex_prop = mesh.scalar_vertex_property("vertex_prop");
  vertex_prop[v0]   = 0;
  vertex_prop[v1]   = 1;
  vertex_prop[v2]   = 2;
  vertex_prop[v3]   = 3;
  auto& tet_prop    = mesh.scalar_simplex_property("tet_prop");
  tet_prop[t0]      = 4;

  auto copied_mesh = mesh;

  REQUIRE(mesh[v0] == copied_mesh[v0]);
  REQUIRE(mesh[v1] == copied_mesh[v1]);
  REQUIRE(mesh[v2] == copied_mesh[v2]);
  REQUIRE(mesh[v3] == copied_mesh[v3]);
  mesh[v0](0) = 2;
  REQUIRE_FALSE(mesh[v0] == copied_mesh[v0]);

  {
    auto& copied_vertex_prop =
        copied_mesh.vertex_property<double>("vertex_prop");
    REQUIRE(vertex_prop[v0] == copied_vertex_prop[v0]);
    REQUIRE(vertex_prop[v1] == copied_vertex_prop[v1]);
    REQUIRE(vertex_prop[v2] == copied_vertex_prop[v2]);
    REQUIRE(vertex_prop[v3] == copied_vertex_prop[v3]);

    vertex_prop[v0] = 100;
    REQUIRE_FALSE(vertex_prop[v0] == copied_vertex_prop[v0]);

    auto& copied_tet_prop =
        copied_mesh.simplex_property<double>("tet_prop");
    REQUIRE(tet_prop[t0] == copied_tet_prop[t0]);

    tet_prop[t0] = 10;
    REQUIRE_FALSE(tet_prop[t0] == copied_tet_prop[t0]);
  }

  copied_mesh = mesh;
  {
    auto& copied_vertex_prop =
        copied_mesh.vertex_property<double>("vertex_prop");
    REQUIRE(mesh[v0] == copied_mesh[v0]);
    REQUIRE(vertex_prop[v0] == copied_vertex_prop[v0]);

    auto& copied_tet_prop =
        copied_mesh.simplex_property<double>("tet_prop");
    auto const [v0, v1, v2, v3]     = mesh[t0];
    auto const [cv0, cv1, cv2, cv3] = copied_mesh[t0];
    REQUIRE(v0.index() == cv0.index());
    REQUIRE(v1.index() == cv1.index());
    REQUIRE(v2.index() == cv2.index());
    REQUIRE(v3.index() == cv3.index());
    REQUIRE(tet_prop[t0] == copied_tet_prop[t0]);
  }
}
//==============================================================================
TEST_CASE("unstructured_tetrahedral_grid_from_grid",
          "[unstructured_tetrahedral_grid][grid]") {
  auto const g = rectilinear_grid{linspace{0.0, 1.0, 5}, linspace{0.0, 1.0, 5},
                                  linspace{0.0, 1.0, 5}};
  auto       mesh = unstructured_tetrahedral_grid3{g};
  mesh.write("unstructured_tetrahedral_grid_from_3d_grid.vtu");
}
//==============================================================================
#if TATOOINE_CGAL_AVAILABLE
TEST_CASE("unstructured_tetrahedral_grid_vertex_property_sampler",
          "[unstructured_tetrahedral_grid][vertex_property][sampler]") {
  std::size_t const    num_points  = 100;
  auto const           radius      = real_number{1};
  auto const           s           = geometry::sphere3{radius};
  auto mesh = unstructured_tetrahedral_grid3{s.random_points(num_points)};
  using v = decltype(mesh)::vertex_handle;

  mesh.build_delaunay_mesh();
  auto& prop = mesh.scalar_vertex_property("prop");
  for (std::size_t i = 0; i < num_points; ++i) {
    prop[v{i}] = i;
  }
  auto prop_sampler = mesh.sampler(prop);
  SECTION("Vertex Identities") {
    for (auto v : mesh.vertices()) {
      REQUIRE(prop_sampler(mesh[v]) == Approx(prop[v]));
    }
  }
  SECTION("Interpolated Property") {
    for (auto tet : mesh.simplices()) {
      auto const [v0, v1, v2, v3] = mesh[tet];
      REQUIRE(prop_sampler(mesh[v0] * 0.1 + mesh[v1] * 0.2 + mesh[v2] * 0.3 +
                           mesh[v3] * 0.4) ==
              Approx(prop[v0] * 0.1 + prop[v1] * 0.2 + prop[v2] * 0.3 +
                     prop[v3] * 0.4)
                  .margin(1e-6));
    }
  }
}
#endif
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
