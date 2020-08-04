#include <tatooine/Q_field.h>
#include <tatooine/color_scales/magma.h>
#include <tatooine/direct_volume_rendering.h>
#include <tatooine/lagrangian_Q_field.h>
#include <tatooine/perspective_camera.h>

#include <filesystem>

#include "eddy_props.h"
#include "ensemble_file_paths.h"
#include "ensemble_member.h"
#include "integrate_pathline.h"
#include "monitor.h"
#include "positions_in_domain.h"
//==============================================================================
using namespace tatooine;
using namespace tatooine::scivis_contest_2020;
namespace fs = std::filesystem;
using V      = ensemble_member<interpolation::cubic>;
//==============================================================================
int main() {
  V                   v{ensemble_file_paths[0]};
  color_scales::magma color_scale;
  auto                alpha_scale = [](auto const t) -> double {
    auto const border = 0.5;
    if (t < border) {
      return 0;
    } else {
      return (t - border) * (t - border) / ((1 - border) * (1 - border)) * 0.7;
    }
  };

  boundingbox bb{
      vec{v.xc_axis.front(), v.yc_axis.front(), v.z_axis.front() * 0.0025},
      vec{v.xc_axis.back(), v.yc_axis.back(), v.z_axis.back() * 0.0025}};
  auto Qf = Q(v);

  size_t const               width = 1600, height = 800;
  vec<double, 3> const       top_left_front{bb.min(0), bb.center(1), bb.min(2)};
  auto const                 ctr      = bb.center();
  auto                       view_dir = top_left_front - ctr;
  perspective_camera<double> cam{top_left_front + view_dir * 1.5,
                                 ctr,
                                 vec<double, 3>{0, 0, -1},
                                 60,
                                 width,
                                 height};

  linspace times{front(v.t_axis), back(v.t_axis), (size(v.t_axis) - 1) * 6 + 1};
  size_t   ti = 0;
  if (!fs::exists("direct_volume")) {
    fs::create_directory("direct_volume");
  }
  double const tau = 24 * 5;
  for (auto const t : times) {
    monitor(
        [&] {
          double const max_ftau = v.t_axis.back() - t;
          auto const   ftau     = std::min<double>(tau, max_ftau);
          // double const min_btau = v.t_axis.front() - t;
          auto const btau = 0;  // std::max<double>(-tau, min_btau);
          auto const lQf  = lagrangian_Q(v, btau, ftau);
          auto       g    = direct_volume_rendering(
              cam, bb,
              [&](auto const& pos) {
                return lQf(vec{pos(0), pos(1), pos(2) / 0.0025}, t);
              },
              [&](auto const& pos) {
                return lQf.in_domain(vec{pos(0), pos(1), pos(2) / 0.0025}, t);
              },
              0.0, 1.0, 0.1, color_scale, alpha_scale, vec<double, 3>::ones());
          write_png(
              "direct_volume/direct_volume." + std::to_string(ti++) + ".png",
              g.vertex_property<vec<double, 3>>("rendering"), width, height);
        },
        "ti: " + std::to_string(ti));
  }
}
