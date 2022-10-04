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
  auto grid =
      rectilinear_grid{linspace{-10.0, 10.0, 100},
                       linspace{-10.0, 10.0, 100},
                       linspace{-10.0, 10.0, 100}};
  auto const& prop = grid.sample_to_vertex_property(
      euclidean_length(analytical::numerical::abcflow{}), "length");
  auto mesh = isosurface(prop, 1);
  mesh.remove_duplicate_vertices();
  rendering::interactive::show(mesh);
}
