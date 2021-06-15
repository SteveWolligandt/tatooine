#include <tatooine/color_scales/magma.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/rendering/direct_volume.h>
#include <tatooine/grid.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  hdf5::file axis0_file{"/home/vcuser/channel_flow/axis0.h5"};
  hdf5::file axis1_file{"/home/vcuser/channel_flow/axis1.h5"};
  hdf5::file axis2_file{"/home/vcuser/channel_flow/axis2.h5"};
  auto const axis0 =
      axis0_file.dataset<double>("CartGrid/axis0").read_as_vector();
  auto const axis1 =
      axis1_file.dataset<double>("CartGrid/axis1").read_as_vector();
  auto const axis2 =
      axis2_file.dataset<double>("CartGrid/axis2").read_as_vector();

  grid full_domain{axis0, axis1, axis2};
  full_domain.set_chunk_size_for_lazy_properties(2);
  std::cerr << "full_domain:\n" << full_domain << '\n';

  // generate the pod domain
  std::vector<double> pod_domain_x(begin(axis0),
                                          begin(axis0) + 512);
  std::vector<double> pod_domain_y(begin(axis1),
                                   begin(axis1) + 1024);
  std::vector<double> pod_domain_z(begin(axis2),
                                          begin(axis2) + 256);
  grid pod_domain{pod_domain_x, pod_domain_y, pod_domain_z};
  std::cerr << "pod_domain:\n" << pod_domain << '\n';

  // open hdf5 files
  hdf5::file channelflow_154_file{
      "/home/vcuser/channel_flow/dino_res_154000.h5"};
  hdf5::file pod_file{"/home/vcuser/channel_flow/pod_0.h5"};

  auto& vely_154 = full_domain.insert_vertex_property(
      channelflow_154_file.dataset<double>("velocity/yvel"), "Vy_154");

  auto vely_154_field = vely_154.linear_sampler();

  color_scales::viridis color_scale;

  auto         pod_boundingbox       = pod_domain.bounding_box();
  size_t const width = 300, height = 150;

  auto const full_domain_eye =
      vec3{0.7940901239835871, 0.04097490152128994, 0.5004262802265552};
  auto const full_domain_lookat =
      vec3{-0.7384532106212904, 0.7745404345929863, -0.4576538576946477};
  auto const full_domain_up =
      vec3{-0.35221800146747856, 0.3807796045093859, 0.8549557720911246};
  rendering::perspective_camera<double> full_domain_cam{full_domain_eye,
                                                             full_domain_lookat,
                                                             full_domain_up,
                                                             60,
                                                             0.01,
                                                             1000,
                                                             width,
                                                             height};

  auto const pod_eye =
      vec3{0.17436402903670775, -0.029368613711112865, 0.11376422220314832};
  auto const pod_lookat =
      vec3{0.03328671241261789, 0.0723694723172821, 0.033031680721043566};
  auto const pod_up =
      vec3{-0.35434985513228934, 0.2282347045784469, 0.9068324540915563};
  rendering::perspective_camera pod_cam{pod_eye, pod_lookat, pod_up, 60,
                                        0.01,    1000,       width,  height};
  auto alpha = [](auto const t) -> double {
    auto const min = 0.0;
    auto const max = 0.2;
    if (t < 0) {
      return min;
    } else if (t > 1) {
      return max + min;
    } else {
      return t * t * (max - min) + min;
    }
  };
  ;

  auto const min             = 0;
  auto const max             = 100;
  auto const distance_on_ray = 0.001;
  auto       aabb            = axis_aligned_bounding_box{
      vec3{full_domain.dimension<0>()[0],
           full_domain.dimension<1>()[0],
           full_domain.dimension<2>()[0]},
      vec3{full_domain.dimension<0>()[511],
           full_domain.dimension<1>()[2047],
           full_domain.dimension<2>()[511]}};
  write_png("channelflow_direct_volume_streamwise_velocity.png",
            rendering::direct_volume(
                full_domain_cam, aabb, [](auto const&) { return true; },
                distance_on_ray,
                [&](auto const& x, auto const&) {
                  auto const t   = (vely_154_field(x) - min) / (max - min);
                  auto const rgb = color_scale(t);
                  return vec4{rgb(0), rgb(1), rgb(2), alpha(t)};
                })
                .vec3_vertex_property("rendering"));
}
