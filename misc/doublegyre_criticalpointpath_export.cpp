#include <tatooine/line.h>
#include <tatooine/linspace.h>
#include <tatooine/doublegyre.h>
#include <tatooine/newton_raphson.h>

using namespace tatooine;

int main() {
  symbolic::doublegyre v;
  line<double, 3> path;
  vec2                 x{1, 0};
  for (auto t : linspace(-10.0, 20.0, 601)) {
    x = newton_raphson(v, x, t, 1000);
    path.push_back({x(0), x(1), t});
  }
  path.write_vtk("doublegyre_criticalpointpath.vtk");
}
