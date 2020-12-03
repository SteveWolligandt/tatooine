#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/grid.h>
#include <tatooine/ode/vclibs/rungekutta43.h>

#include "io.h"
#include "vf_grid_prop.h"
#include "isolines.h"
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
    if (std::abs(t0 - current_time) > temporal_range) {
      auto const sampled_current = ping_field(is...);
      pong_field(is...)          = sampled_current;
      return;
    }
    auto const current_pos = ping_field.grid()(is...);
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
              isolevels_specified, isolevel_a, isolevel_b, fields] = *args;

  vf_grid_time_dependent_prop<std::vector<double>, std::vector<double>, std::vector<double>> pipe{
      "/home/steve/flows/pipedcylinder2d.vtk"};
  std::cerr << pipe.grid().front<0>() << '\n';
  std::cerr << pipe.grid().front<1>() << '\n';
  std::cerr << pipe.grid().size<0>() << '\n';
  std::cerr << pipe.grid().size<1>() << '\n';
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
  if (isolevels_specified) {
    line<double, 2> pathline;
    auto const isolines_a =
        create_iso_lines_a(grids, isolevel_a);
    auto const isolines_b =
        create_iso_lines_b(grids, isolevel_b);
    size_t i = 0;
    for (auto const& ls : isolines_a) {
      tatooine::write_vtk(ls, "isolines_a_" + std::to_string(i++) + ".vtk");
    }
    i = 0;
    for (auto const& ls : isolines_b) {
      tatooine::write_vtk(ls, "isolines_b_" + std::to_string(i++) + ".vtk");
    }
    for (size_t i = 0; i < size(grids); ++i) {
      auto const is = intersections(isolines_a[i], isolines_b[i]);
      if (!is.empty()) {
        pathline.push_back(is.front());
      }
      pathline.write_vtk("pathline_" + std::to_string(i) + ".vtk");
    }

    line<double, 2> numerical_pathline;
    numerical_pathline.push_back(pathline.front_vertex());
    ode::vclibs::rungekutta43<double, 2> solver;
    analytical::fields::numerical::doublegyre v;
    for (size_t i = 1; i < size(grids); ++i) {
      solver.solve(v, numerical_pathline.back_vertex(), times[i],
                   times.spacing(), [&](auto const& x, auto const /*t*/) {
                     numerical_pathline.push_back(x);
                   });
    }
    numerical_pathline.write_vtk("numerical_pathline.vtk");
  }

  if (isolevels_specified) {
    // for each grid smear a scalar field
    for (size_t i = 0; i < num_steps; ++i) {
      for (auto const& scalar_field_name : fields) {
        for (size_t j = 0; j < size(times); ++j) {
          auto                   s = sphere;
          [[maybe_unused]] auto& ping_field =
              grids[j].vertex_property<double>(scalar_field_name);
          auto& pong_field = grids[j].add_vertex_property<double>("pong");
          grids[j].loop_over_vertex_indices(
              [&](auto const... is) { pong_field(is...) = ping_field(is...); });
          auto const current_time = times[j];
          smear(ping_field, pong_field, s, inner_radius, temporal_range,
                current_time, t0, dir);
          s.center() += dir;
          grids[j].remove_vertex_property(scalar_field_name);
          grids[j].rename_vertex_property("pong", scalar_field_name);
        }
      }
      line<double, 2> pathline;
      auto const      isolines_a =
          create_iso_lines_a(grids, isolevel_a);
      auto const isolines_b =
          create_iso_lines_b(grids, isolevel_b);
      for (size_t i = 0; i < size(grids); ++i) {
        auto const is = intersections(isolines_a[i], isolines_b[i]);
        if (size(is) > 1) {
          std::cerr << "more than one intersection found!\n";
        }
        if (!is.empty()) {
          pathline.push_back(is.front());
        }
      }
      pathline.write_vtk("pathline_smeared_" + std::to_string(i) + ".vtk");
      grids[grids.size()/2].write_vtk("smeared_field_" + std::to_string(i) + ".vtk");
      tatooine::write_vtk(isolines_a[grids.size()/2], "smeared_isolines_a_" + std::to_string(i) + ".vtk");
      tatooine::write_vtk(isolines_b[grids.size()/2], "smeared_isolines_b_" + std::to_string(i) + ".vtk");
    }
  } else {
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
