#include <tatooine/line.h>
#include <tatooine/linspace.h>

using namespace tatooine;

int main (int argc, char** argv) {
  double tmin = -10;
  double tmax = 20;
  size_t n    = 300;
  std::string out_path = "straight_line_for_dg.vtk";
  if (argc > 1) { tmin = std::stof(argv[1]); }
  if (argc > 2) { tmax = std::stof(argv[2]); }
  if (argc > 3) { n = std::stoi(argv[3]); }
  if (argc > 4) { out_path = argv[4]; }
  line<double, 3> path;
  vec2            x{1, 0};
  for (auto t : linspace(tmin, tmax, n)) {
    std::cerr << ((t - tmin) / (tmax - tmin) * 100) << "%        \r";
    path.push_back({1, 0, t});
  }
  std::cerr << std::endl;
  path.write_vtk(out_path);
}
