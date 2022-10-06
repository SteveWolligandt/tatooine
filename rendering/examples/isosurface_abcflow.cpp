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
      rectilinear_grid{linspace{-5.0, 5.0, 10},
                       linspace{-5.0, 5.0, 10},
                       linspace{-5.0, 5.0, 10}};
  auto const& prop = grid.sample_to_vertex_property(
      euclidean_length(analytical::numerical::abcflow{}), "length");
  auto mesh = isosurface(prop, 1);
  //mesh.remove_duplicate_vertices(execution_policy::sequential);
  //mesh.tidy_up();
  rendering::interactive::show(mesh);
}
