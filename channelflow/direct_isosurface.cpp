#include <tatooine/color_scales/magma.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/field.h>
#include <tatooine/line.h>
#include <tatooine/grid.h>
#include <tatooine/hdf5.h>
#include <tatooine/rendering/direct_isosurface.h>
#include <tatooine/rendering/perspective_camera.h>

#include <iomanip>
#include <sstream>
//==============================================================================
auto main() -> int {
  using namespace tatooine;
  // read full domain axes
  std::cerr << "loading axes ...";
  hdf5::file axis0_file{"/home/vcuser/channel_flow/axis0.h5"};
  hdf5::file axis1_file{"/home/vcuser/channel_flow/axis1.h5"};
  hdf5::file axis2_file{"/home/vcuser/channel_flow/axis2.h5"};
  auto const axis0 =
      axis0_file.dataset<double>("CartGrid/axis0").read_as_vector();
  auto const axis1 =
      axis1_file.dataset<double>("CartGrid/axis1").read_as_vector();
  auto const axis2 =
      axis2_file.dataset<double>("CartGrid/axis2").read_as_vector();
  std::cerr << "done!\n";

  std::cerr << "creating grids ...";
  grid full_domain{axis0, axis1, axis2};
  full_domain.set_chunk_size_for_lazy_properties(256);
  std::cerr << "full_domain:\n" << full_domain << '\n';

  auto axis0_Q = axis0;
  axis0_Q.pop_back();
  grid full_domain_Q{axis0_Q, axis1, axis2};
  full_domain_Q.set_chunk_size_for_lazy_properties(256);
  std::cerr << "full_domain_Q:\n" << full_domain_Q << '\n';
  std::cerr << "done!\n";

  std::cerr << "creating files ...";
  hdf5::file channelflow_122_full_file{
      "/home/vcuser/channel_flow/dino_res_122000_full.h5"};
  hdf5::file channelflow_154_full_file{
      "/home/vcuser/channel_flow/dino_res_154000.h5"};
  std::cerr << "done!\n";

  std::cerr << "loading data ...";
  // auto& velocity_x_122_full = full_domain.insert_lazy_vertex_property(
  //    channelflow_122_full_file.dataset<double>("velocity/xvel"),
  //    "velocity_x_122");
  // auto& velocity_y_122_full = full_domain.insert_lazy_vertex_property(
  //    channelflow_122_full_file.dataset<double>("velocity/yvel"),
  //    "velocity_y_122");
  // auto& velocity_z_122_full = full_domain.insert_lazy_vertex_property(
  //    channelflow_122_full_file.dataset<double>("velocity/zvel"),
  //    "velocity_z_122");
  auto& velocity_y_154_full = full_domain.insert_vertex_property(
      channelflow_154_full_file.dataset<double>("velocity/yvel"),
      "velocity_y_154");
  // auto& Q_122_full = full_domain_Q.insert_vertex_property(
  //    channelflow_122_full_file.dataset<double>("Q_pnorm"), "Q_122");
  // auto& velocity_magnitude_122_full = full_domain_Q.insert_vertex_property(
  //    channelflow_122_full_file.dataset<double>("velocity_magnitude"),
  //    "velocity_magnitude_122");
  // velocity_x_122_full.limit_num_chunks_loaded();
  // velocity_y_122_full.limit_num_chunks_loaded();
  // velocity_z_122_full.limit_num_chunks_loaded();
  // Q_122_full.limit_num_chunks_loaded();
  // velocity_x_122_full.set_max_num_chunks_loaded(30);
  // velocity_y_122_full.set_max_num_chunks_loaded(30);
  // velocity_z_122_full.set_max_num_chunks_loaded(30);
  // Q_122_full.set_max_num_chunks_loaded(30);
  std::cerr << "done!\n";

  std::cerr << "creating samplers ...";
  // auto velocity_x_122_full_sampler = velocity_x_122_full.linear_sampler();
  // auto velocity_y_122_full_sampler = velocity_y_122_full.linear_sampler();
  // auto velocity_z_122_full_sampler = velocity_z_122_full.linear_sampler();
  // auto vel_122_field = vectorfield{velocity_x_122_full_sampler,
  // velocity_y_122_full_sampler,
  //                                 velocity_z_122_full_sampler};
  // auto Q_122_full_sampler       = Q_122_full.linear_sampler();
  // auto velocity_magnitude_122_full_sampler =
  // velocity_magnitude_122_full.linear_sampler();
  auto velocity_y_154_full_sampler = velocity_y_154_full.linear_sampler();
  std::cerr << "done!\n";

  color_scales::viridis color_scale;

  std::cerr << "creating cameras ...";
  size_t const width = 2000, height = 1000;

  auto  full_domain_eye                  = line3{};
  auto& full_domain_eye_parameterization = full_domain_eye.parameterization();
  auto  const veye0                            = full_domain_eye.push_back(
      vec3{0.6, full_domain.front(1) - full_domain.extent(1) / 5, 0.75} * 2 /
      3);
  auto const veye1 = full_domain_eye.push_back(
      vec3{0.6, full_domain.back(1) + full_domain.extent(1) / 5, 0.75} * 2 / 3);
  auto  full_domain_eye_sampler                  = full_domain_eye.linear_sampler();
  full_domain_eye_parameterization[veye0] = 0;
  full_domain_eye_parameterization[veye1] = 1;
  auto full_domain_lookat                 = line3{};
  auto & full_domain_lookat_parameterization = full_domain_lookat.parameterization();
  auto const vla0 = full_domain_lookat.push_back(vec3{
      full_domain.center(0), full_domain.front(1) + full_domain.extent(1) / 6,
      full_domain.center(2)});
  auto const vla1 = full_domain_lookat.push_back(vec3{
      full_domain.center(0), full_domain.back(1) - full_domain.extent(1) / 6,
      full_domain.center(2)});
  full_domain_lookat_parameterization[vla0] = 0;
  full_domain_lookat_parameterization[vla1] = 1;
  auto  full_domain_lookat_sampler                  = full_domain_lookat.linear_sampler();
  //auto const full_domain_up =
  //    vec3{-0.35221800146747856, 0.3807796045093859, 0.8549557720911246};
  auto const full_domain_up = vec3{0, 0, 1};
  std::cerr << "done!\n";
  // real_t const min      = 13;
  // real_t const max      = 27;
  // real_t const isovalue_Q = 5e6;

  // auto mapped_velocity_magnitude_shader = [&](auto const& x_iso,
  //                                            auto const& gradient,
  //                                            auto const& view_dir) {
  //  auto const normal      = normalize(gradient);
  //  auto const diffuse     = std::abs(dot(view_dir, normal));
  //  auto const reflect_dir = reflect(-view_dir, normal);
  //  auto const spec_dot =
  //      std::max<real_t>(std::abs(dot(reflect_dir, view_dir)), 0);
  //  auto const specular = std::pow(spec_dot, 100);
  //  auto const scalar   = std::clamp<real_t>(
  //      (velocity_magnitude_122_full_sampler(x_iso) - min) / (max - min), 0,
  //      1);
  //  auto const albedo = color_scale(scalar);
  //  auto const col    = albedo * diffuse;
  //  return vec{col(0), col(1), col(2),
  //                  scalar * scalar * scalar * scalar * scalar};
  //};
  // auto const rendering_grid =
  //    rendering::direct_isosurface(full_domain_cam, Q_122_full_sampler,
  //                                 isovalue_Q,
  //                                 mapped_velocity_magnitude_shader);
  // write_png("channelflow_Q_5e6_with_velocity_magnitude.png",
  //          rendering_grid.vec3_vertex_property("rendered_isosurface"));
  std::cerr << "calculating min and max velocity y 154 ...";
  auto min_velocity_y_154 = std::numeric_limits<double>::max();
  auto max_velocity_y_154 = -std::numeric_limits<double>::max();

  full_domain.vertices().iterate_indices([&](auto const... is) {
    min_velocity_y_154 =
        std::min(min_velocity_y_154, velocity_y_154_full(is...));
    max_velocity_y_154 =
        std::max(max_velocity_y_154, velocity_y_154_full(is...));
  });
  std::cerr << "done!\n";
  std::cerr << "data range: " << min_velocity_y_154 << " - "
            << max_velocity_y_154 << '\n';
  size_t i = 0;
  size_t const num_frames = 100;
  for (auto const t : linspace{0.0, 1.0, num_frames}) {
    std::cerr << "rendering " << i+1 << " / " << num_frames << "...";
    auto full_domain_cam =
        rendering::perspective_camera<double>{full_domain_eye_sampler(t),
                                              full_domain_lookat_sampler(t),
                                              full_domain_up,
                                              60,
                                              0.01,
                                              1000,
                                              width,
                                              height};
    auto isovalues =
        std::vector{(max_velocity_y_154 - min_velocity_y_154) * 2 / 3,(max_velocity_y_154 - min_velocity_y_154) * 5 / 6};
    auto const rendering_grid = rendering::direct_isosurface(
        full_domain_cam, velocity_y_154_full_sampler, isovalues,
        [&](auto const /*x_iso*/, auto const isovalue, auto const& gradient,
            auto const& view_dir) {
          auto const normal  = normalize(gradient);
          auto const diffuse = std::abs(dot(view_dir, normal));
          auto const t       = (isovalue - min_velocity_y_154) /
                         (min_velocity_y_154 + max_velocity_y_154);
          auto const albedo = color_scale(t);
          auto const col    = albedo * diffuse * 0.8 + albedo * 0.2;
          return vec{col(0), col(1), col(2),
                     isovalue < isovalues.back() - 1e-5 ? 0.1 : 0.9};
        });
    std::cerr << "done!\n";
    std::cerr << "writing ...";
    std::stringstream str;
    str << std::setw(static_cast<size_t>(std::ceil(std::log10(num_frames))))
        << std::setfill('0') << i;
    write_png("channelflow_velocity_y_154." + str.str() + ".png",
              rendering_grid.vec3_vertex_property("rendered_isosurface"));
    std::cerr << "done!\n";
    ++i;
  }
}
