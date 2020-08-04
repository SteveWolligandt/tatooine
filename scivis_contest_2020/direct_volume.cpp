#include <filesystem>

#include "eddy_props.h"
#include <tatooine/direct_volume_rendering.h>
#include <tatooine/okubo_weiss_field.h>
#include <tatooine/perspective_camera.h>
#include "ensemble_file_paths.h"
#include "ensemble_member.h"
#include "integrate_pathline.h"
#include "positions_in_domain.h"
//==============================================================================
using namespace tatooine;
using namespace tatooine::scivis_contest_2020;
using V = ensemble_member<interpolation::cubic>;
//==============================================================================
int main() {
  //V v{ensemble_file_paths[0]};
  V v{mean_file_path};

  boundingbox bb{
      vec{v.xc_axis.front(), v.yc_axis.front(), v.z_axis.front() * 0.0025},
      vec{v.xc_axis.back(), v.yc_axis.back(), v.z_axis.back() * 0.0025}};
  okubo_weiss_field Q{v};
  size_t const width = 50, height = 50;
  perspective_camera<double> cam{
      vec{49.897247128382894, -28.941337984209365, 22.768734542663655},
      vec{40.00709915161134, 20.007100105285634, -3.974999999999997},
      vec{-0.07992041070574998, 0.46545025369005494, 0.881458331001805},
      60,
      width,
      height};
  double const t = v.t_axis.front();
  auto         Q_grid = direct_volume_rendering(
      cam, bb,
      [&](auto const& pos) {
        return Q(vec{pos(0), pos(1), pos(2) / 0.0025}, t);
      },
      [&](auto const& pos) {
        return Q.in_domain(vec{pos(0), pos(1), pos(2) / 0.0025}, t);
      },
      0, 0.1, 10);
  write_png("direct_volume_stdg_Q.png",
            Q_grid.vertex_property<double>("rendering"), width, height);
}
