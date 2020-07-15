#include <tatooine/fields/scivis_contest_2020_ensemble_member.h>
#include <filesystem>
//==============================================================================
int main(int, char** argv) {
  using namespace tatooine;
  namespace fs = std::filesystem;
  using V      = fields::scivis_contest_2020_ensemble_member;
  V v{argv[1]};

  std::vector<parameterized_line<V::real_t, 3, interpolation::linear>>
               pathlines;
  size_t const num_pathlines = 100;
  pathlines.reserve(2 * num_pathlines);
  ode::vclibs::rungekutta43<V::real_t, 3> solver;

  auto              bb = v.m_w_grid.boundingbox();
  vec<V::real_t, 4> xt;
  bool              in_domain = false;
  for (size_t i = 0; i < num_pathlines; ++i) {
    do {
      xt        = bb.random_point();
      in_domain = v.in_domain(vec{xt(0), xt(1), xt(2)}, xt(3));
    } while (!in_domain );

    std::cerr << "found position! (" << i << ")\n";
    pathlines.emplace_back();
    solver.solve(v, vec{xt(0), xt(1), xt(2)}, xt(3), 1000000,
                 [&pathlines](auto t, const auto& y) {
                   pathlines.back().push_back(y, t);
                 });
    pathlines.emplace_back();
    solver.solve(v, vec{xt(0), xt(1), xt(2)}, xt(3), -1000000,
                 [&pathlines](auto t, const auto& y) {
                   pathlines.back().push_back(y, t);
                 });
  }
  fs::path    p = argv[1];
  std::string outpath =
      fs::path{p.filename()}.replace_extension("random_pathlines.vtk");
  write_vtk(pathlines, outpath);
}
