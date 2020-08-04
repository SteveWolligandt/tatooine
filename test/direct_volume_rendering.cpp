#include <tatooine/lagrangian_Q_field.h>
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/color_scales/magma.h>
#include <tatooine/direct_volume_rendering.h>
#include <tatooine/perspective_camera.h>
#include <tatooine/spacetime_field.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("direct_volume_rendering_doublegyre",
          "[direct_volume_rendering][doublegyre]") {
  analytical::fields::numerical::doublegyre v;
  spacetime_field                           stv{v};
  color_scales::magma                       color_scale;
  constexpr auto                            alpha_scale = [](double const t) {
    //return t - 0.2;
    //return (std::exp(std::max(0.0, t - 0.5)) - 1) / (std::exp(1) - 1);
    return t * t - 0.2;
  };

  boundingbox bb{vec{0.0, 0.0, 0.0}, vec{2.0, 1.0, 10.0}};
  auto        mag = length(stv);
  auto        Qf  = lagrangian_Q(stv, 0, 5);
  REQUIRE(mag(vec{0.1, 0.1, 0.1}, 0) == length(stv(vec{0.1, 0.1, 0.1}, 0)));
  size_t const               width = 1000, height = 600;
  perspective_camera<double> cam{vec{-1, 2, -3}, vec{0.5, 1, 0.0}, 60, width,
                                 height};
  auto                       mag_grid =
      direct_volume_rendering(cam, bb, mag, 0, 1, 1.1, 0.01, color_scale,
                              alpha_scale, vec<double, 3>::ones());
  write_png("direct_volume_stdg_mag.png",
            mag_grid.vertex_property<vec<double, 3>>("rendering"), width,
            height);
  auto Q_grid = direct_volume_rendering(
      cam, bb, Qf, 0, 0.0, 1.0, 0.001, color_scale,
      [](auto const t) -> double {
        auto const border = 0.5;
        if (t < border) {
          return 0;
        } else {
          return (t - border) * (t - border) / ((1 - border) * (1 - border)) *
                 0.7;
        }
      },
      vec<double, 3>::ones());
  write_png("direct_volume_stdg_Q.png",
            Q_grid.vertex_property<vec<double, 3>>("rendering"), width, height);
}
//==============================================================================
TEST_CASE("direct_volume_rendering_doublegyre_2",
          "[direct_volume_rendering][doublegyre]") {
  analytical::fields::numerical::doublegyre v;
  v.set_infinite_domain(true);
  spacetime_field     stv{v};
  color_scales::magma color_scale;
  auto                alpha_scale = [](double const t) {
    return t * t - 0.2;
  };

  boundingbox<double, 3>     bb{vec{30, 10, 0}, vec{50, 30, 8}};
  auto const                 Qf    = Q(stv);
  size_t const               width = 1000, height = 1000;
  vec<double, 3> const       top_left_front{bb.min(0), bb.center(1), bb.min(2)};
  auto const                 ctr      = bb.center();
  auto                       view_dir = top_left_front - ctr;
  perspective_camera<double> cam{top_left_front + view_dir*1.5,
                                 ctr,
                                 vec<double, 3>{0, 0, -1},
                                 60,
                                 width,
                                 height};
  auto Q_grid = direct_volume_rendering(cam, bb, Qf, 0, 0.1, 1.1, 0.01,
                                        color_scale, alpha_scale, vec<double, 3>::ones());
  write_png("direct_volume_stdg_Q_wide.png",
            Q_grid.vertex_property<vec<double, 3>>("rendering"), width, height);
}
//==============================================================================
TEST_CASE("direct_volume_rendering_abc_magnitude",
          "[direct_volume_rendering][abc][magnitude]") {
  analytical::fields::numerical::abcflow                     v;
  grid<linspace<double>, linspace<double>, linspace<double>> g{
      linspace{-1.0, 1.0, 200}, linspace{-1.0, 1.0, 200},
      linspace{-1.0, 1.0, 200}};
  color_scales::magma color_scale;
  constexpr auto                            alpha_scale = [](double const t) {
    return t - 0.2;
    //return (std::exp(t) - 1) / (std::exp(1) - 1);
  };
  auto&               mag =
      g.add_contiguous_vertex_property<double, x_fastest, interpolation::linear,
                                       interpolation::linear,
                                       interpolation::linear>("mag");
  double min = std::numeric_limits<double>::max(),
         max = -std::numeric_limits<double>::max();
  g.loop_over_vertex_indices([&](auto const... is) {
    mag.container().at(is...) = length(v(g.vertex_at(is...), 0));
    min                       = std::min(mag.container().at(is...), min);
    max                       = std::max(mag.container().at(is...), max);
  });
  size_t const               width = 1000, height = 1000;
  perspective_camera<double> cam{vec{40, 50, 50}, vec{0.0, 0.0, 0.0}, 30, width,
                                 height};
  std::cerr << "max: " << max << '\n';
  auto rendered_grid = direct_volume_rendering(
      cam, mag, 1, max, 0.01, color_scale, alpha_scale, vec<double, 3>::ones());
  write_png("direct_volume_abc_mag.png",
            rendered_grid.vertex_property<vec<double, 3>>("rendering"), width,
            height);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
