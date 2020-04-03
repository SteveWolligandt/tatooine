#include <tatooine/simple_tri_mesh.h>
int main() {
  const std::string file =
      "boussinesq/"
      "pathsurfaces_20_60_10.000000_-5.000000_5.000000_2_0.100000_28.vtk";
  tatooine::simple_tri_mesh<double, 2> mesh{file};
}
