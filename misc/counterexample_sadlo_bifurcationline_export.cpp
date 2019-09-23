#include <tatooine/counterexample_sadlo.h>
#include <tatooine/linspace.h>
#include <tatooine/line.h>
#include <string>

int main (int argc, char** argv) {
  tatooine::numerical::counterexample_sadlo v;
  auto                                      b = v.bifurcationline_spacetime();

  double tmin = -10;
  double tmax = 10;
  size_t n    = 200;
  std::string out_path = "counterexample_sadlo_bifurcationline.vtk";
  tatooine::line<double, 3> l;

  if (argc > 1) { tmin = std::stof(argv[1]); }
  if (argc > 2) { tmax = std::stof(argv[2]); }
  if (argc > 3) { n = std::stoi(argv[3]); }
  if (argc > 4) { out_path = argv[4]; }

  for (auto t : tatooine::linspace(tmin, tmax, n)) { l.push_back(b(t)); }
  l.write_vtk(out_path);
}
