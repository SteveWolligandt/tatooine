#include <tatooine/fields/scivis_contest_2020_ensemble_member.h>
#include <tatooine/parallel_vectors.h>
#include <filesystem>
//==============================================================================
int main(int argc, char** argv) {
  using namespace tatooine;
  namespace fs = std::filesystem;
  if (argc < 2) { throw std::runtime_error{"specify ensemble file path"}; }

  fields::scivis_contest_2020_ensemble_member v{argv[1]};
  auto            J = diff(v, 1e-7);
  auto            a      = J * v;
  double t = std::stod(argv[2]);
  auto x_domain = v.xc_axis;
  x_domain.pop_back();
  x_domain.pop_back();
  x_domain.pop_front();
  x_domain.pop_front();
  auto y_domain = v.yc_axis;
  y_domain.pop_back();
  y_domain.pop_back();
  y_domain.pop_front();
  y_domain.pop_front();
  size_t   zn = static_cast<size_t>((v.z_axis.back() - v.z_axis.front()) /
                                    (v.z_axis[1] - v.z_axis[0]));
  zn          = v.z_axis.size() * 2;
  zn = 100;
  linspace z_domain{v.z_axis.front(), v.z_axis.back(), zn};
  z_domain.pop_back();
  z_domain.pop_back();
  z_domain.pop_front();
  z_domain.pop_front();

  grid  pv_grid{x_domain, y_domain, z_domain};

  auto const pv_lines = parallel_vectors(v, a, pv_grid, t);
  fs::path    p        = argv[1];
  std::string outpath =
      fs::path{p.filename()}.replace_extension("pv_lines_" + std::to_string(t) + ".vtk");
  write_vtk(pv_lines, outpath);
}
