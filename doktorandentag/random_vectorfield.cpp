#include <tatooine/grid.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
#include <tatooine/random.h>
#include <tatooine/tensor_operations.h>
#include <tatooine/gpu/lic.h>
#include <tatooine/gl/context.h>
#include <sstream>

#include <string>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto ctx = gl::context{};
  auto  seed     = argc > 1 ? std::mt19937_64::result_type{std::stoul(argv[1])}
                            : std::mt19937_64::result_type{124653231};
  auto  rand_x   = random::uniform{0.0, 1.0, seed};
  auto  rand_y   = random::uniform{-1.0, 1.0, seed};
  auto  rand_domain   = random::uniform{0.0, 1.0, seed};
  auto  g0       = grid{5, 5};
  auto  g1       = grid{5, 5};
  auto& vectors0 = g0.insert_vec2_vertex_property("vectors");
  auto& vectors1 = g1.insert_vec2_vertex_property("vectors");
  auto v0 = vectors0.linear_sampler();
  auto v1 = vectors1.linear_sampler();

  g0.vertices().iterate_indices([&](auto const... is) {
    vectors0(is...) = normalize(vec2{rand_x(), rand_y()});
  });
  g1.vertices().iterate_indices([&](auto const... is) {
    vectors1(is...) = normalize(vec2{rand_x(), rand_y()});
  });
  size_t cnt = 0;
  auto   ode_solver = ode::vclibs::rungekutta43<real_t, 2>{};
  for (auto const tg : linspace{0.0, 1.0, 100})
  //auto tg = 0.0;
  {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(4) << cnt;
    auto  g       = grid{5, 5};
    auto& vectors = g.insert_vec2_vertex_property("vectors");
    auto& param_g = g.insert_scalar_vertex_property("parameterization");
    g.vertices().iterate_indices([&](auto const... is) {
      vectors(is...) =
          normalize(vectors0(is...) * (1 - tg) + vectors1(is...) * tg);
      param_g(is...) = tg;
    });

    gpu::lic(vectors, {1000, 1000}, 31, 0.001).write_png("lic." + ss.str() + ".png");
    g.write_vtk("random_vectorfield." + std::to_string(cnt) + ".vtk");
    auto         integral_curves = std::vector<line2>{};
    auto         v               = vectors.linear_sampler();
    size_t const num_lines       = 20;
    for (size_t i = 0; i < num_lines; ++i) {
      auto       streamline = line2{};
      auto&      param      = streamline.parameterization();
      auto&      tangents   = streamline.tangents();
      auto const x0         = vec2{rand_domain(), rand_domain()};
      ode_solver.solve(v, x0, 0, -100,
                       [&](auto const& x, auto const t, auto const& dx) {
                         auto const v = streamline.push_front(x);
                         param[v]     = t;
                         tangents[v]  = dx;
                       });
      ode_solver.solve(v, x0, 0, 100,
                       [&](auto const& x, auto const t, auto const& dx) {
                         auto const v = streamline.push_back(x);
                         param[v]     = t;
                         tangents[v]  = dx;
                       });
      streamline.write_vtk("streamline." + std::to_string(i) + ".vtk");
      streamline.compute_chordal_parameterization();
      auto resampled = streamline.resample<interpolation::linear>(100);
      for (auto v : resampled.vertices()) {
        resampled.parameterization()[v] = tg;
      }

      resampled.write_vtk("streamline_resampled." + std::to_string(i) +
                          ".vtk");
    }
    ++cnt;
  }
  auto unsteady = [&](auto const& x,
                      auto const  t) -> ode::vclibs::maybe_vec_t<real_t, 2> {
    if (x(0) < 0 || x(0) > 1 || x(1) < 0 || x(1) > 1 || t < 0 || t > 1) {
      return ode::vclibs::out_of_domain;
    }
    return v0(x) * (1 - t) + v1(x) * t;
  };
  for (size_t i = 0; i < 20; ++i) {
    auto       pathline = line2{};
    auto&      param    = pathline.parameterization();
    auto&      tangents = pathline.tangents();
    auto const x0       = vec2{rand_domain(), rand_domain()};
    ode_solver.solve(unsteady, x0, 0.5, -0.5,
                     [&](auto const& x, auto const t, auto const& dx) {
                       auto const v = pathline.push_front(x);
                       param[v]     = t;
                       tangents[v]  = dx;
                     });
    ode_solver.solve(unsteady, x0, 0.5, 0.5,
                     [&](auto const& x, auto const t, auto const& dx) {
                       auto const v = pathline.push_back(x);
                       param[v]     = t;
                       tangents[v]  = dx;
                     });
    pathline.write_vtk("pathline" + std::to_string(i) + ".vtk");
  }
  gpu::lic(vectors0, {1000, 1000}, 31, 0.001).write_png("lic.png");
}
