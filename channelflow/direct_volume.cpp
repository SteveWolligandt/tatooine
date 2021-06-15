#include <tatooine/color_scales/magma.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/grid.h>
#include <tatooine/rendering/direct_volume.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto       axis0_file = hdf5::file{"/home/vcuser/channel_flow/axis0.h5"};
  auto       axis1_file = hdf5::file{"/home/vcuser/channel_flow/axis1.h5"};
  auto       axis2_file = hdf5::file{"/home/vcuser/channel_flow/axis2.h5"};
  auto const axis0 =
      axis0_file.dataset<double>("CartGrid/axis0").read_as_vector();
  auto const axis1 =
      axis1_file.dataset<double>("CartGrid/axis1").read_as_vector();
  auto const axis2 =
      axis2_file.dataset<double>("CartGrid/axis2").read_as_vector();

  auto full_domain = grid{axis0, axis1, axis2};
  std::cerr << "full_domain:\n" << full_domain << '\n';

  // generate the pod domain
  auto pod_domain_x = std::vector<double>(begin(axis0), begin(axis0) + 512);
  auto pod_domain_y = std::vector<double>(begin(axis1), begin(axis1) + 1024);
  auto pod_domain_z = std::vector<double>(begin(axis2), begin(axis2) + 256);
  auto pod_domain   = grid{pod_domain_x, pod_domain_y, pod_domain_z};
  std::cerr << "pod_domain:\n" << pod_domain << '\n';

  // open hdf5 files
  auto channelflow_154_file =
      hdf5::file{"/home/vcuser/channel_flow/dino_res_154000.h5"};
  auto pod_file = hdf5::file{"/home/vcuser/channel_flow/pod_0.h5"};

  auto& discrete_vely_154 = full_domain.insert_vertex_property(
      channelflow_154_file.dataset<double>("velocity/yvel"), "Vy_154");

  auto vely_154_field = discrete_vely_154.linear_sampler();

  auto const color_scale = color_scales::viridis{};

  auto         pod_boundingbox = pod_domain.bounding_box();
  size_t const width = 20000, height = 10000;

  auto const full_domain_eye =
      vec3{0.7940901239835871, 0.04097490152128994, 0.5004262802265552};
  auto const full_domain_lookat =
      vec3{-0.7384532106212904, 0.7745404345929863, -0.4576538576946477};
  auto const full_domain_up =
      vec3{-0.35221800146747856, 0.3807796045093859, 0.8549557720911246};
  auto full_domain_cam =
      rendering::perspective_camera<double>{full_domain_eye,
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
  auto pod_cam = rendering::perspective_camera{
      pod_eye, pod_lookat, pod_up, 60, 0.01, 1000, width, height};
  auto alpha = [](auto const t) -> double {
    auto const min = 0.0;
    auto const max = 0.2;
    if (t < 0) {
      return min;
    } else if (t > 1) {
      return max + min;
    } else {
      return t * t * t * t * t * t * t * t * (max - min) + min;
    }
  };

  auto       min             = std::numeric_limits<double>::max();
  auto       max             = -std::numeric_limits<double>::max();

  full_domain.vertices().iterate_indices([&](auto const... is) {
    min = std::min(min, discrete_vely_154(is...));
    max = std::max(max, discrete_vely_154(is...));
  });
  min = (max - min) / 3;
  std::cerr << "data range: " << min << " - " << max << '\n';

  auto const distance_on_ray = 0.001;
  //auto const aabb            = axis_aligned_bounding_box{
  //    vec3{full_domain.dimension<0>()[0], full_domain.dimension<1>()[0],
  //         full_domain.dimension<2>()[0]},
  //    vec3{full_domain.dimension<0>()[511], full_domain.dimension<1>()[2047],
  //         full_domain.dimension<2>()[511]}};
  auto const aabb = full_domain.bounding_box();
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
