#include <tatooine/render_topological_skeleton.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/doublegyre.h>

//==============================================================================
using namespace tatooine::gl;
context ctx;
//==============================================================================
static const std::array<std::string_view, 4> filenames{
    "texture_case1.vtk", "texture_case2.vtk", "texture_case3.vtk",
    "texture_case4.vtk"};
//==============================================================================
int main() {
  using namespace tatooine;
  using namespace interpolation;
  using sampler_t = grid_sampler<double, 2, vec<double, 2>, linear, linear>;
  using V         = sampled_field<sampler_t, double, 2, 2>;
  using integrator_t =
      integration::vclibs::rungekutta43<double, 2, linear>;

  for (const auto& filename : filenames) {
    V v;
    v.sampler().read_vtk_scalars(std::string{filename}, "vector_field");

    integrator_t rk43{integration::vclibs::max_step     = 0.001,
                      integration::vclibs::initial_step = 0.001};
    auto         image = render_topological_skeleton(v, rk43, {1500, 1500});
    std::string  out_filename{filename};
    auto         dotpos = out_filename.find_last_of(".");
    out_filename.replace(dotpos, 4, "_skeleton.png");
    image.write_png(out_filename);
  }

  //size_t tcnt = 0;
  //for (auto t : linspace(0.0, 10.0, 100)) {
  //  numerical::doublegyre v;
  //  auto                  vd = resample<linear, linear>(
  //      v, grid{linspace{-0.1, 2.1, 11}, linspace{-0.1, 1.1, 11}}, t);
  //  integrator_t rk43{integration::vclibs::max_step     = 0.001,
  //                    integration::vclibs::initial_step = 0.001};
  //  render_topological_skeleton(vd, rk43, {2000, 1000})
  //      .write_png("dg_skeleton" + std::to_string(tcnt++) + ".png");
  //}
}
