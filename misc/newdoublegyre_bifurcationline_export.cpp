#include <tatooine/newdoublegyre.h>
#include <tatooine/linspace.h>
#include <tatooine/line.h>
#include <string>

int main (int argc, char** argv) {
  tatooine::numerical::newdoublegyre v;
  auto                               b = v.bifurcationline_spacetime();

  double tmin = -10;
  double tmax = 20;
  size_t n    = 300;
  std::string out_path = "newdoublegyre_bifurcationline.vtk";
  tatooine::line<double, 3> l;

  if (argc > 1) { tmin = std::stof(argv[1]); }
  if (argc > 2) { tmax = std::stof(argv[2]); }
  if (argc > 3) { n = std::stoi(argv[3]); }
  if (argc > 4) { out_path = argv[4]; }

  for (auto t : tatooine::linspace(tmin, tmax, n)) {
    std::cerr << ((t - tmin) / (tmax - tmin) * 100) << "%        \r";
    l.push_back(b(t));
  }
  std::cerr << std::endl;
  l.write_vtk(out_path);
}
