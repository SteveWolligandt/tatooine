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
struct rotating_flow : vectorfield<rotating_flow, double, 2> {
  using parent_t = vectorfield<rotating_flow, double, 2>;
  using parent_t::real_t;
  using parent_t::pos_t;
  using parent_t::tensor_t;

  constexpr auto evaluate(pos_t const& p, real_t const /*t*/) const
      -> tensor_t final {
    return {-p.y() * (1 - p.x() * p.x() - p.y() * p.y()),
             p.x() * (1 - p.x() * p.x() - p.y() * p.y())};
  }
  constexpr auto in_domain(pos_t const& p, real_t const /*t*/) const
      -> bool final {
    constexpr auto half = real_t(1) / real_t(2);
    return -half <= p.x() && p.x() <= half && -half <= p.y() && p.y() <= half;
  }
};
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
    auto const distance_to_sphere_origin =
        distance(current_pos, s.center());
    if (distance_to_sphere_origin < s.radius()) {
      auto const s_x = [&]() -> double {
        if (distance_to_sphere_origin <= inner_radius) {
          return 1;
        }
        if (distance_to_sphere_origin > s.radius()) {
          return 0;
        }
        return (distance_to_sphere_origin - s.radius()) /
               (inner_radius - s.radius());
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
  auto const outer_radius = sphere.radius();

  vf_grid_prop<std::vector<float>, std::vector<float>, std::vector<float>>
      cavity_flow_{
          "/home/steve/flows/2DCavity/Cavity2DTimeFilter3x3x7_100_bin.am"};
  vf_split cavity_flow{cavity_flow_};
  //vf_grid_time_dependent_prop<std::vector<float>, std::vector<float>,
  //                            std::vector<float>>
  //    pipe_flow{"/home/steve/flows/pipedcylinder2d.vtk", "u", "v"};
  //analytical::fields::numerical::doublegyre dg_flow;
  //rotating_flow                             rot_flow;

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
    //for (size_t i = 1; i < size(grids); ++i) {
    //  solver.solve(dg_flow, numerical_pathline.back_vertex(), times[i],
    //               times.spacing(), [&](auto const& x, auto const [>t<]) {
    //                 numerical_pathline.push_back(x);
    //               });
    //}
    //numerical_pathline.write_vtk("numerical_pathline_doublegyre.vtk");
    //for (size_t i = 1; i < size(grids); ++i) {
    //  solver.solve(rot_flow, numerical_pathline.back_vertex(), times[i],
    //               times.spacing(), [&](auto const& x, auto const [>t<]) {
    //                 numerical_pathline.push_back(x);
    //               });
    //}
    //numerical_pathline.write_vtk("numerical_pathline_rotating.vtk");

    line<double, 2> fnumerical_pathline;
    fnumerical_pathline.push_back(pathline.front_vertex());
    ode::vclibs::rungekutta43<float, 2> fsolver;
    fnumerical_pathline.push_back(fnumerical_pathline.back_vertex());
    //fsolver.solve(pipe_flow, fnumerical_pathline.back_vertex(), 2, 2,
    //              [&](auto const& x, auto const t) {
    //                fnumerical_pathline.push_back(x);
    //              });
    //fnumerical_pathline.write_vtk("numerical_pathline_pipe.vtk");
    fsolver.solve(cavity_flow, fnumerical_pathline.back_vertex(), 1, 3,
                  [&](auto const& x, auto const /*t*/) {
                    fnumerical_pathline.push_back(x);
                  });
    fnumerical_pathline.write_vtk("numerical_pathline_cavity.vtk");
  }

  if (isolevels_specified) {
    // for each grid smear a scalar field
    auto s = sphere;
    std::cerr << times[50] << '\n';
    {
 
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
      pathline.write_vtk("pathline_smeared_0.vtk");
      grids[50].write_vtk("smeared_field_0.vtk");
      tatooine::write_vtk(isolines_a[50], "smeared_isolines_a_0.vtk");
      tatooine::write_vtk(isolines_b[50], "smeared_isolines_b_0.vtk");
      discretize(s, 200).write_vtk("outer_sphere_0.vtk");
      s.radius() = inner_radius;
      discretize(s, 200).write_vtk("inner_sphere_0.vtk");
      s.radius() = outer_radius;
    }


    for (size_t i = 0; i < num_steps; ++i) {
      s.center() += dir;
      for (auto const& scalar_field_name : fields) {
        for (size_t j = 0; j < size(times); ++j) {
          [[maybe_unused]] auto& ping_field =
              grids[j].vertex_property<double>(scalar_field_name);
          auto& pong_field = grids[j].add_vertex_property<double>("pong");
          grids[j].loop_over_vertex_indices(
              [&](auto const... is) { pong_field(is...) = ping_field(is...); });
          auto const current_time = times[j];
          smear(ping_field, pong_field, s, inner_radius, temporal_range,
                current_time, t0, dir);
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
      pathline.write_vtk("pathline_smeared_" + std::to_string(i + 1) + ".vtk");
      grids[50].write_vtk("smeared_field_" + std::to_string(i + 1) + ".vtk");
      tatooine::write_vtk(isolines_a[50], "smeared_isolines_a_" +
                                              std::to_string(i + 1) + ".vtk");
      tatooine::write_vtk(isolines_b[50], "smeared_isolines_b_" +
                                              std::to_string(i + 1) + ".vtk");
      discretize(s, 200).write_vtk("outer_sphere_" + std::to_string(i+1) +
                                   ".vtk");
      s.radius() = inner_radius;
      discretize(s, 200).write_vtk("inner_sphere_" + std::to_string(i+1) +
                                   ".vtk");
      s.radius() = outer_radius;
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
