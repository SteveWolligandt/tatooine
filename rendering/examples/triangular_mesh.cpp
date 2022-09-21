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
  std::cout << "foo\n";
  auto grid =
      rectilinear_grid{linspace{-10.0, 10.0, 10},
                       linspace{-10.0, 10.0, 10},
                       linspace{-10.0, 10.0, 10}};
  auto const& prop = grid.sample_to_vertex_property(
      euclidean_length(analytical::numerical::abcflow{}), "length");
  auto const mesh = isosurface(prop, 1);
  std::cout << mesh.vertices().size() <<"\n";
  std::cout << mesh.simplices().size() <<"\n";
  rendering::interactive::show(mesh);
}
