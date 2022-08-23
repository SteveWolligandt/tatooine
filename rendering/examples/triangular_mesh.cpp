#include <tatooine/rendering/interactive.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/field_operations.h>
#include <tatooine/random.h>
#include <tatooine/streamsurface.h>
#include <tatooine/analytical/abcflow.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto const mesh = isosurface(
      euclidean_length(analytical::numerical::abcflow{}),
      rectilinear_grid{linspace{-10.0, 10.0, 256},
                       linspace{-10.0, 10.0, 256},
                       linspace{-10.0, 10.0, 256}},
      1)
  mesh.sample_to_vertex_property(v, "velocity");

  rendering::interactive::show(mesh);
}
