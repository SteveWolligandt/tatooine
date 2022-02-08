#include <tatooine/fields/scivis_contest_2020_ensemble_member.h>

#include <tatooine/filesystem.h>
#include <mutex>
//==============================================================================
int main(int, char** argv) {
  using namespace tatooine;
  namespace fs = filesystem;
  using V      = fields::scivis_contest_2020_ensemble_member;
  V v{argv[1]};

  std::vector<parameterized_line<V::real_type, 3, interpolation::linear>>
               pathlines;
  size_t const num_pathlines = 100;
  pathlines.reserve(num_pathlines);

  auto              bb = v.m_w_grid.boundingbox();
  vec<V::real_type, 4> xt;
  bool              in_domain = false;
  std::mutex        mutex;
//#pragma omp parallel for
  for (size_t i = 0; i < num_pathlines; ++i) {
    do {
      std::lock_guard lock{mutex};
      xt        = bb.random_point();
      in_domain = v.in_domain(vec{xt(0), xt(1), xt(2)}, xt(3));
    } while (!in_domain);

    std::cerr << "found position! (" << i << ")\n";
    auto& pathline = [&mutex, &pathlines]() -> decltype(auto) {
      std::lock_guard lock{mutex};
      return pathlines.emplace_back();
    }();
    ode::vclibs::rungekutta43<V::real_type, 3> solver;
    solver.solve(v, vec{xt(0), xt(1), xt(2)}, xt(3), 1000000,
                 [&mutex, &pathline](auto t, const auto& y) {
                   if (pathline.empty()) {
                     pathline.push_back(y, t);
                     return;
                   } else if (distance(pathline.back_vertex(), y) > 1e-6) {
                     pathline.push_back(y, t);
                     return;
                   }
                 });
    solver.solve(v, vec{xt(0), xt(1), xt(2)}, xt(3), -1000000,
                 [&mutex, &pathline](auto t, const auto& y) {
                   if (pathline.empty()) {
                     pathline.push_back(y, t);
                     return;
                   } else if (distance(pathline.front_vertex(), y) > 1e-6) {
                     pathline.push_front(y, t);
                     return;
                   }
                 });
  }
  fs::path    p = argv[1];
  std::string outpath =
      fs::path{p.filename()}.replace_extension("random_pathlines.vtk");
  write_vtk(pathlines, outpath);
}
