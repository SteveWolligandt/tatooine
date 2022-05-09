#include <tatooine/unstructured_triangular_grid.h>
using namespace tatooine;
auto main() -> int {
  auto mesh =  unstructured_triangular_grid2 {};
  auto const v0 = mesh.insert_vertex(0,0);
  auto const v1 = mesh.insert_vertex(1,0);
  auto const v2 = mesh.insert_vertex(0,1);
  auto const v3 = mesh.insert_vertex(1,1);
  mesh.insert_triangle(v0, v1, v2);
  mesh.insert_triangle(v1, v3, v2);
  auto& scalar_prop = mesh.scalar_vertex_property("scalar");
  auto& vec2_prop = mesh.vec2_vertex_property("vec2");
  auto& vec3_prop = mesh.vec3_vertex_property("vec3");
  auto& mat2_prop = mesh.mat2_vertex_property("mat2");
  scalar_prop[v0] = 1;
  scalar_prop[v3] = 2;
  mat2_prop[v0] = mat2::randu();
  mat2_prop[v1] = mat2::randu();
  mat2_prop[v2] = mat2::randu();
  mat2_prop[v3] = mat2::randu();
  std::cout << mat2_prop[v1]<<'\n';
  mesh.write("geometry.example.write_vtu.vtu");
}
