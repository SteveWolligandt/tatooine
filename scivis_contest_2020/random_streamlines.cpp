#include <tatooine/fields/scivis_contest_2020_ensemble_member.h>
#include <tatooine/filesystem.h>
#include <tatooine/fixed_time_field.h>
//==============================================================================
int main(int, char** argv) {
  using namespace tatooine;
  namespace fs = filesystem;
  using V = fields::scivis_contest_2020_ensemble_member;
  V v{argv[1]};

  double t = std::stod(argv[2]);
  fixed_time_field v_fixed{v, t};

  std::vector<parameterized_line<V::real_type, 3, interpolation::linear>>
               pathlines;
  size_t const num_pathlines = 100;
  pathlines.reserve(2 * num_pathlines);
  ode::vclibs::rungekutta43<V::real_type, 3> solver;

  auto              bb = v.m_w_grid.boundingbox();
  vec<V::real_type, 4> xt;
  bool              in_domain = false;
  for (size_t i = 0; i < num_pathlines; ++i) {
    do {
      xt        = bb.random_point();
      in_domain = v_fixed.in_domain(vec{xt(0), xt(1), xt(2)}, t);
    } while (!in_domain );

    std::cerr << "found position! (" << i << ")\n";
    pathlines.emplace_back();
    solver.solve(v_fixed, vec{xt(0), xt(1), xt(2)}, xt(3), 1000000,
                 [&pathlines](auto t, const auto& y) {
                   pathlines.back().push_back(y, t);
                 });
    pathlines.emplace_back();
    solver.solve(v_fixed, vec{xt(0), xt(1), xt(2)}, xt(3), -1000000,
                 [&pathlines](auto t, const auto& y) {
                   pathlines.back().push_back(y, t);
                 });
  }
  fs::path    p       = argv[1];
  std::string outpath = fs::path{p.filename()}.replace_extension(
      "random_streamlines_" + std::to_string(t) + ".vtk");
  write_vtk(pathlines, outpath);
}
