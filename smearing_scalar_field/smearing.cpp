#include <tatooine/geometry/sphere.h>
#include <tatooine/grid.h>

#include "io.h"
#include "parse_arguments.h"
//==============================================================================
using namespace tatooine;
//==============================================================================
/// Stores a smeared version of ping_field into pong_field.
/// For each point of a grid go in backward directions and sample field there.
/// Afterwards create the interpolation factor depending of position and time.
auto smear(auto& ping_field, auto& pong_field, geometry::sphere2 const& s,
           double const inner_radius, double const temporal_range,
           double const current_time, double const t0, vec2 const& dir) {
  // create a sampler of the ping_field
  auto sampler = ping_field.template sampler<interpolation::cubic>();

  ping_field.grid().parallel_loop_over_vertex_indices([&](auto const... is) {
    auto const current_pos  = ping_field.grid()(is...);
    auto const offset_pos = current_pos - dir;
    auto const sqr_distance_to_sphere_origin =
        sqr_distance(offset_pos, s.center());
    if (sqr_distance_to_sphere_origin < s.radius() * s.radius()) {
      auto const r   = std::sqrt(sqr_distance_to_sphere_origin);
      auto const s_x = [&]() -> double {
        if (r < inner_radius) {
          return 1;
        }
        if (r > s.radius()) {
          return 0;
        }
        return (r - s.radius()) / (inner_radius - s.radius());
      }();
      auto const lambda_s = s_x * s_x * s_x + 3 * s_x * s_x * (1 - s_x);
      auto const s_t      = [&]() -> double {
        if (auto const t_diff = std::abs(current_time - t0);
            t_diff < temporal_range) {
          return 1 - t_diff / temporal_range;
        }
        return 0;
      }();
      auto const lambda_t = s_t * s_t * s_t + 3 * s_t * s_t * (1 - s_t);
      assert(lambda_s >= 0 && lambda_s <= 1);
      auto const lambda = lambda_s * lambda_t;
      if (!ping_field.grid().bounding_box().is_inside(current_pos) ||
          !ping_field.grid().bounding_box().is_inside(offset_pos)) {
        pong_field(is...) = 0.0 / 0.0;
      } else {
        auto const sampled_current = sampler(current_pos);
        auto const sampled_smeared = sampler(offset_pos);
        pong_field(is...) =
            sampled_current * (1 - lambda) + sampled_smeared * lambda;
      }
    }
  });
}
//==============================================================================
auto main(int argc, char const** argv) -> int {
  using namespace tatooine::smearing;
  // parse arguments
  auto const args = parse_arguments(argc, argv);
  if (!args) {
    return 0;
  }
  auto const [input_file_path, output_file_path, sphere, inner_radius,
              end_point, temporal_range, t0, dir, num_steps, write_vtk,
              fields] = *args;

  // read file
  int    res_x{}, res_y{}, res_t{};
  double min_x{}, min_y{}, min_t{};
  double extent_x{}, extent_y{}, extent_t{};
  auto   grids = [&] {
    auto const ext = args->input_file_path.extension();
    if (ext != ".df" && ext != ".bin") {
      throw std::runtime_error{"File must have extension .df or .bin."};
    }
    if (ext == ".df") {
      return read_ascii(args->input_file_path, res_x, res_y, res_t, min_x,
                        min_y, min_t, extent_x, extent_y, extent_t);
    }
    return read_binary(args->input_file_path, res_x, res_y, res_t, min_x, min_y,
                       min_t, extent_x, extent_y, extent_t);
  }();
  linspace<double> times{min_t, min_t + extent_t, static_cast<size_t>(res_t)};

  // write originals to vtk files
  if (write_vtk) {
    for (size_t i = 0; i < size(grids); ++i) {
      grids[i].write_vtk("original_" + std::to_string(i) + ".vtk");
    }
  }

  // for each grid smear a scalar field
  for (auto const& scalar_field_name : fields) {
    for (size_t j = 0; j < size(times); ++j) {
      auto                   s = sphere;
      [[maybe_unused]] auto& ping_field =
          grids[j].vertex_property<double>(scalar_field_name);
      auto& pong_field = grids[j].add_vertex_property<double>("pong");
      grids[j].loop_over_vertex_indices(
          [&](auto const... is) { pong_field(is...) = ping_field(is...); });
      bool       ping         = true;
      auto const current_time = times[j];
      for (size_t i = 0; i < num_steps; ++i) {
        if (ping) {
          smear(ping_field, pong_field, s, inner_radius, temporal_range,
                current_time, t0, dir);
        } else {
          smear(pong_field, ping_field, s, inner_radius, temporal_range,
                current_time, t0, dir);
        }
        s.center() += dir;
        ping = !ping;
      }
      if (ping) {
        grids[j].remove_vertex_property(scalar_field_name);
        grids[j].rename_vertex_property("pong", scalar_field_name);
      } else {
        grids[j].remove_vertex_property("pong");
      }
    }
  }
  // write to vtk
  if (write_vtk) {
    for (size_t i = 0; i < size(grids); ++i) {
      grids[i].write_vtk("smeared_" + std::to_string(i) + ".vtk");
    }
  }
  // write to output file
  if (output_file_path.extension() == ".bin") {
    write_binary(output_file_path, grids, res_x, res_y, res_t, min_x, min_y,
                 min_t, extent_x, extent_y, extent_t);
  } else if (output_file_path.extension() == ".df") {
    write_ascii(output_file_path, grids, res_x, res_y, res_t, min_x, min_y,
                min_t, extent_x, extent_y, extent_t);
  }
}
