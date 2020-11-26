#include <tatooine/grid.h>
#include <tatooine/geometry/sphere.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
template <size_t N>
auto smear(auto& scalar_field, auto& smeared_scalar_field,
           geometry::sphere<double, N> const& s, double const r1,
           double const r_t, double const t, double const t0,
           vec<double, N> const& dir) {
  auto sampler = scalar_field.template sampler<interpolation::cubic>();
  scalar_field.grid().parallel_loop_over_vertex_indices([&](auto const... is) {
    auto const current_pos               = scalar_field.grid()(is...);
    auto const smear_origin              = current_pos - dir;
    auto const sqr_distance_to_sphere_origin =
        sqr_distance(smear_origin, s.center());
    if (sqr_distance_to_sphere_origin < s.radius() * s.radius()) {
      auto const r      = std::sqrt(sqr_distance_to_sphere_origin);
      auto const s_x    = [&]() -> double {
        if (r < r1) {
          return 1;
        }
        if (r > s.radius()) {
          return 0;
        }
        return (r - s.radius()) / (r1 - s.radius());
      }();
      auto const lambda_s = s_x * s_x * s_x + 3 * s_x * s_x * (1 - s_x);
      auto const s_t      = [&]() -> double {
        if (auto const t_diff = std::abs(t - t0); t_diff < r_t) {
          return 1 - t_diff / r_t;
        }
        return 0;
      }();
      auto const lambda_t = s_t * s_t * s_t + 3 * s_t * s_t * (1 - s_t);
      assert(lambda_s >= 0 && lambda_s <= 1);
      auto const lambda = lambda_s * lambda_t;
      smeared_scalar_field(is...) =
          sampler(current_pos) * (1 - lambda) + sampler(smear_origin) * lambda;
    }
  });
}
//==============================================================================
int main() {
  std::ifstream f{"drift_piped_301_2.00_1.00.df"};
  size_t res_x{}, res_y{}, res_t{};
  double min_x{}, min_y{}, min_t{};
  double extent_x{}, extent_y{}, extent_t{};
  if (f.is_open()) {
    f >> res_x >> res_y >> res_t >>
         min_x >> min_y >> min_t >>
         extent_x >> extent_y >> extent_t;
    std::cerr << res_x << ", " << res_y << ", " << res_t << '\n';
    std::cerr << min_x << ", " << min_y << ", " << min_t << '\n';
    std::cerr << extent_x << ", " << extent_y << ", " << extent_t << '\n';
  }
  std::vector<uniform_grid_2d<double>> grids(
      res_t, uniform_grid_2d<double>{linspace{min_x, min_x + extent_x, res_x},
                                     linspace{min_y, min_y + extent_y, res_x}});
  std::vector<double> ts;
  double const        spacing = extent_t / (res_t - 1);
  for (size_t i = 0; i < res_t; ++i) {
    ts.push_back(min_t + spacing * i);
  }
  double val;
  for (size_t i = 0; i < res_t; ++i) {
    auto& a = grids[i].add_vertex_property<double>("a");
    grids[i].loop_over_vertex_indices([&](auto const... is) {
      f >> val;
      a(is...) = val;
    });
  }
  for (size_t i = 0; i < res_t; ++i) {
    auto& b = grids[i].add_vertex_property<double>("b");
    grids[i].loop_over_vertex_indices([&](auto const... is) {
      f >> val;
      b(is...) = val;
    });
  }
  grids[0].write_vtk("a.vtk");
  grids[0].write_vtk("b.vtk");
  //auto& scalar_field = g.add_vertex_property<double>("scalar_field");
  //auto& smeared_scalar_field =
  //    g.add_vertex_property<double>("smeared_scalar_field");
  //
  //for (size_t x = 0; x < g.size(0); x++) {
  //  auto const v = static_cast<double>(x) / static_cast<double>(g.size(0) - 1);
  //  for (size_t y = 0; y < g.size(1); y++) {
  //    scalar_field(x, y)         = v;
  //    smeared_scalar_field(x, y) = v;
  //  }
  //}
  //g.write_vtk("original.vtk");
  //
  //double           r1 = 0.03, r2 = 0.1;
  //double           r_t = 0.1, t0 = 0, t = 0;
  //geometry::sphere s{r2, vec2{0.5, 0.5}};
  //vec2             dir{0.1, 0.0};
  //dir = normalize(dir);
  //dir *= (r2 - r1) * 0.1;
  //bool ping = true;
  //for (size_t i = 0; i < 20; ++i) {
  //  if (ping) {
  //    smear(scalar_field, smeared_scalar_field, s, r1, r_t, t, t0, dir);
  //  } else {
  //    smear(smeared_scalar_field, scalar_field, s, r1, r_t, t, t0, dir);
  //  }
  //  s.center() += dir;
  //  ping = !ping;
  //}
  //g.write_vtk("smeared.vtk");
}
