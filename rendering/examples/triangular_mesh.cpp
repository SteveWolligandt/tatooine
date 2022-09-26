#include <tatooine/rendering/interactive.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/field_operations.h>
#include <tatooine/random.h>
#include <tatooine/streamsurface.h>
#include <tatooine/analytical/abcflow.h>
#include <tatooine/isosurface.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  //auto grid =
  //    rectilinear_grid{linspace{-10.0, 10.0, 256},
  //                     linspace{-10.0, 10.0, 256},
  //                     linspace{-10.0, 10.0, 256}};
  //auto const& prop = grid.sample_to_vertex_property(
  //    euclidean_length(analytical::numerical::abcflow{}), "length");
  //auto const mesh = isosurface(prop, 1);
  auto mesh = unstructured_triangular_grid2{};
  mesh.insert_vertex(0,0);
  mesh.insert_vertex(1,0);
  mesh.insert_vertex(0,1);
  mesh.insert_vertex(1,1);
  mesh.build_delaunay_mesh();
  rendering::interactive::show(mesh);
}
