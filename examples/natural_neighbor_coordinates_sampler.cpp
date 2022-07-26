#include <tatooine/pointset.h>
#include <tatooine/rectilinear_grid.h>

using namespace tatooine;
auto main() -> int {
  auto ps = pointset2{};
  auto v0 = ps.insert_vertex(0,1);
  auto v1 = ps.insert_vertex(1,0);
  auto v2 = ps.insert_vertex(-1,0);
  auto & foo = ps.scalar_vertex_property("foo");
  foo[v0] = 0;
  foo[v1] = 1;
  foo[v2] = -1;
  auto foo_sampler = ps.natural_neighbor_coordinates_sampler(foo);
  auto g = rectilinear_grid{linspace{-2.0, 2.0, 500}, linspace{-2.0, 2.0, 500}};
  g.sample_to_vertex_property(foo_sampler, "foo");
  g.write("natural_neighbor_coordinates_sampler.example.vtr");
}
