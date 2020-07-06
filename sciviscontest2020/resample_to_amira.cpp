#include <tatooine/fields/scivis_contest_2020_ensemble_member.h>
#include <tatooine/grid.h>
#include <tatooine/linspace.h>

#include <filesystem>
int main(int argc, char** argv) {
  if (argc < 2) { throw std::runtime_error{"specify ensemble file path"}; }

  using namespace tatooine;
  fields::scivis_contest_2020_ensemble_member v{argv[1]};
  auto x_domain = v.xc_axis;
  x_domain.pop_back();
  auto y_domain = v.yc_axis;
  y_domain.pop_back();
  linspace z_domain{v.z_axis.front(), v.z_axis.back(), v.z_axis.size() * 2};

  grid  resample_grid{x_domain, y_domain, z_domain};
  auto& vel =
      resample_grid.add_contiguous_vertex_property<vec<double, 3>, x_fastest>(
          "velocity");
  for (auto t : v.t_axis) {
    auto apply_data = [&](auto const... is) {
      vel.data_at(is...) = v(resample_grid.vertex_at(is...), t);
    };
    namespace fs = std::filesystem;
 
    fs::path p = argv[1];
    p.replace_extension("");
    std::cerr << p.string() << "_" << t << ".am" << '\n';

    for_loop(apply_data,
             resample_grid.size<0>(),
             resample_grid.size<1>(),
             resample_grid.size<2>());

    resample_grid.write_amira(p.string(), vel);
  }
}
