#include <tatooine/geometry/sphere.h>
#include <tatooine/unstructured_tetrahedral_grid.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("unstructured_tetrahedral_grid_copy",
          "[unstructured_tetrahedral_grid][copy]") {
  unstructured_tetrahedral_grid mesh;
  auto const                    v0 = mesh.insert_vertex(0.0, 0.0, 0.0);
  auto const                    v1 = mesh.insert_vertex(1.0, 0.0, 0.0);
  auto const                    v2 = mesh.insert_vertex(0.0, 1.0, 0.0);
  auto const                    v3 = mesh.insert_vertex(0.0, 0.0, 1.0);
  auto const                    t0 = mesh.insert_tetrahedron(v0, v1, v2, v3);

  auto& vertex_prop = mesh.add_vertex_property<double>("vertex_prop");
  vertex_prop[v0]   = 0;
  vertex_prop[v1]   = 1;
  vertex_prop[v2]   = 2;
  vertex_prop[v3]   = 3;
  auto& tet_prop    = mesh.add_tetrahedron_property<double>("tet_prop");
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
        copied_mesh.tetrahedron_property<double>("tet_prop");
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
        copied_mesh.tetrahedron_property<double>("tet_prop");
    auto const [v0, v1, v2, v3]     = mesh[t0];
    auto const [cv0, cv1, cv2, cv3] = copied_mesh[t0];
    REQUIRE(v0.i == cv0.i);
    REQUIRE(v1.i == cv1.i);
    REQUIRE(v2.i == cv2.i);
    REQUIRE(v3.i == cv3.i);
    REQUIRE(tet_prop[t0] == copied_tet_prop[t0]);
  }
}
//==============================================================================
TEST_CASE("unstructured_tetrahedral_grid_from_grid",
          "[unstructured_tetrahedral_grid][grid]") {
  auto const g =
      grid{linspace{0.0, 1.0, 5}, linspace{0.0, 1.0, 5}, linspace{0.0, 1.0, 5}};
  unstructured_tetrahedral_grid mesh{g};
  mesh.write_vtk("unstructured_tetrahedral_grid_from_3d_grid.vtk");
}
//==============================================================================
#ifdef TATOOINE_HAS_CGAL_SUPPORT
TEST_CASE("unstructured_tetrahedral_grid_vertex_property_sampler",
          "[unstructured_tetrahedral_grid][vertex_property][sampler]") {
  size_t const num_points  = 100;
  size_t const random_seed = 1234;
  real_t const radius      = 1;
  auto const   s           = geometry::sphere<real_t, 3>{radius};
  auto         mesh        = unstructured_tetrahedral_grid{
      s.random_points(num_points, std::mt19937_64{random_seed})};
  using v = decltype(mesh)::vertex_handle;

  mesh.build_delaunay_mesh();
  auto& prop = mesh.add_vertex_property<double>("prop");
  for (size_t i = 0; i < num_points; ++i) {
    prop[v{i}] = i;
  }
  auto prop_sampler = mesh.sampler(prop);
  SECTION("Vertex Identities") {
    for (auto v : mesh.vertices()) {
      REQUIRE(prop_sampler(mesh[v]) == Approx(prop[v]));
    }
  }
  SECTION("Interpolated Property") {
    for (auto tet : mesh.tetrahedrons()) {
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
