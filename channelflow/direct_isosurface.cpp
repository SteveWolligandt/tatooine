#include <tatooine/color_scales/cool_to_warm.h>
#include <tatooine/color_scales/jet.h>
#include <tatooine/color_scales/magma.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/field.h>
#include <tatooine/hdf5.h>
#include <tatooine/line.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/direct_isosurface.h>
#include <tatooine/rendering/orthographic_camera.h>
#include <tatooine/rendering/perspective_camera.h>
#include <tatooine/rendering/render_axis_aligned_bounding_box.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>
//==============================================================================
using namespace tatooine;
//==============================================================================
enum color_scale_enum { viridis, jet, cool_to_warm };
auto setup_eye_flight(line3& eye, auto const& domain) {
  auto& param = eye.parameterization();

  auto const v0 = eye.push_back(
      vec3{0.6, domain.front(1) - domain.extent(1) / 5, 0.75} * 2 / 3);
  param[v0] = 0;

  auto const v1 = eye.push_back(
      vec3{0.6, domain.back(1) + domain.extent(1) / 5, 0.75} * 2 / 3);
  param[v1] = 1;
}
//------------------------------------------------------------------------------
auto setup_lookat_flight(line3& lookat, auto const& domain) {
  auto&      param = lookat.parameterization();
  auto const v0    = lookat.push_back(vec3{domain.center(0),
                                        domain.front(1) + domain.extent(1) / 6,
                                        domain.center(2)});
  param[v0]        = 0;
  auto const v1    = lookat.push_back(vec3{domain.center(0),
                                        domain.back(1) - domain.extent(1) / 6,
                                        domain.center(2)});
  param[v1]        = 1;
}
//------------------------------------------------------------------------------
auto setup_up_flight(line3& up) {
  auto&      param = up.parameterization();
  auto const v0    = up.push_back(vec3{0, 0, 1});
  param[v0]        = 0;
  auto const v1    = up.push_back(vec3{0, 0, 1});
  param[v1]        = 1;
}
//==============================================================================
auto setup_eye_rotation(line3& eye, auto const& domain) {
  auto& param = eye.parameterization();
  for (auto const t : linspace{0.0, 2 * M_PI, 20}) {
    auto const v =
        eye.push_back(vec3{0.05 + std::cos(t) * 0.5,
                           (domain.front(1) - domain.extent(1) / 5) * 2 / 3,
                           0.05 + std::sin(t) * 0.5});
    param[v] = t / (2 * M_PI);
  }
}
//------------------------------------------------------------------------------
auto setup_lookat_rotation(line3& lookat, auto const& domain) {
  auto& param = lookat.parameterization();
  for (auto const t : linspace{0.0, 2 * M_PI, 20}) {
    auto const v =
        lookat.push_back(vec3{domain.center(0), domain.center(1),
                              // domain.front(1) + domain.extent(1) / 6,
                              domain.center(2)});
    param[v] = t / (2 * M_PI);
  }
}
//------------------------------------------------------------------------------
auto setup_up_rotation(line3& up) {
  auto& param = up.parameterization();
  for (auto const t : linspace{0.0, 2 * M_PI, 20}) {
    // auto const v = up.push_back(vec3{std::sin(t), 0, std::cos(t)});
    auto const v = up.push_back(vec3{1, 0, 0});
    param[v]     = t / (2 * M_PI);
  }
}
//==============================================================================
bool got_sigint = false;
//==============================================================================
auto sigint_handler(int s) -> void {
  got_sigint = true;
  std::cout << "\n>";
}
//==============================================================================
auto main() -> int {
  auto const path = filesystem::path{"channelflow_Q_streamwise_velocity"};
  std::cout << "creating files ...";
  auto channelflow_file =
      hdf5::file{"/home/vcuser/channel_flow/dino_res_154000.h5"};
  std::cout << "done!\n";

  std::cout << "loading axes ...";
  auto const axis0 =
      channelflow_file.dataset<real_t>("CartGrid/axis0").read_as_vector();
  auto const axis1 =
      channelflow_file.dataset<real_t>("CartGrid/axis1").read_as_vector();
  auto const axis2 =
      channelflow_file.dataset<real_t>("CartGrid/axis2").read_as_vector();
  std::cout << "done!\n";

  std::cout << "creating grids ...";
  auto discretized_domain = rectilinear_grid{axis0, axis1, axis2};
  discretized_domain.set_chunk_size_for_lazy_properties(256);
  std::cout << "discretized_domain:\n" << discretized_domain << '\n';
  std::cout << "done!\n";

  std::cout << "loading data ...";
  auto& scalar_field = discretized_domain.insert_vertex_property(
      channelflow_file.dataset<real_t>("Q_cheng"), "Q_cheng");
  auto& streamwise_velocity = discretized_domain.insert_vertex_property(
      channelflow_file.dataset<real_t>("velocity/yvel"), "velocity_y");
  std::cout << "done!\n";

  std::cout << "creating samplers ...";
  auto scalar_sampler              = scalar_field.linear_sampler();
  auto streamwise_velocity_sampler = streamwise_velocity.linear_sampler();
  std::cout << "done!\n";

  auto current_color_scale = color_scale_enum::jet;
  auto viridis             = color_scales::viridis{};
  auto jet                 = color_scales::jet{};
  auto cool_to_warm        = color_scales::cool_to_warm{};

  std::cout << "creating cameras ...";
  auto width = std::size_t(1000);
  auto height = std::size_t(1000);
  //real_t      ortho_scale = 2;

  auto animated_eye    = line3{};
  auto animated_lookat = line3{};
  auto animated_up     = line3{};

  setup_eye_rotation(animated_eye, discretized_domain);
  setup_lookat_rotation(animated_lookat, discretized_domain);
  setup_up_rotation(animated_up);
  // setup_eye_flight(eye, discretized_domain);
  // setup_lookat_flight(lookat, discretized_domain);
  // setup_up_flight(up);

  auto eye_sampler    = animated_eye.linear_sampler();
  auto up_sampler     = animated_up.linear_sampler();
  auto lookat_sampler = animated_lookat.linear_sampler();
  auto eye            = eye_sampler(0.75);
  auto lookat         = lookat_sampler(0.75);
  auto up             = up_sampler(0.75);
  auto fov            = real_t(30);
  auto animated       = false;
  std::cout << "done!\n";

  auto min_mapped = real_t(13);
  auto max_mapped = real_t(27);
  std::cout << "done!\n";
  auto num_frames = std::size_t(5);
  auto run        = true;
  auto n          = real_t(1);
  auto m          = real_t(1);
  auto k          = real_t(0);
  auto line_width = int(11);
  std::signal(SIGINT, sigint_handler);
  while (run) {
    std::cout << "> ";
    auto line = std::string{};
    std::getline(std::cin, line);
    auto const cmd = line.substr(0, line.find(" "));
    if (line == "quit" || line == "q" || line == "exit") {
      std::cout << "k bye.\n";
      run = false;
    } else if (cmd == "animation") {
      animated = true;
    } else if (cmd == "no_animation") {
      animated = false;
    } else if (cmd == "eye") {
      auto line_stream = std::stringstream{line};
      auto cmd         = std::string{};
      auto x           = real_t{};
      auto y           = real_t{};
      auto z           = real_t{};

      line_stream >> cmd >> x >> y >> z;
      eye = vec3{x, y, z};
    } else if (cmd == "up") {
      auto line_stream = std::stringstream{line};
      auto cmd         = std::string{};
      auto x           = real_t{};
      auto y           = real_t{};
      auto z           = real_t{};

      line_stream >> cmd >> x >> y >> z;
      up = vec3{x, y, z};
    } else if (cmd == "lookat" || cmd == "look_at") {
      auto line_stream = std::stringstream{line};
      auto cmd         = std::string{};
      auto x           = real_t{};
      auto y           = real_t{};
      auto z           = real_t{};

      line_stream >> cmd >> x >> y >> z;
      lookat = vec3{x, y, z};
    } else if (cmd == "color_scale") {
      auto line_stream = std::stringstream{line};
      auto cmd         = std::string{};
      auto scale_name  = std::string{};

      line_stream >> cmd >> scale_name;

      if (scale_name == "jet") {
        current_color_scale = color_scale_enum::jet;
        std::cout << "using jet scale\n";
      } else if (scale_name == "viridis") {
        current_color_scale = color_scale_enum::viridis;
        std::cout << "using viridis scale\n";
      } else if (scale_name == "cool_to_warm") {
        current_color_scale = color_scale_enum::cool_to_warm;
        std::cout << "using viridis scale\n";
      } else {
        std::cout << "unknown color scale \"" << scale_name << "\"\n";
      }
    } else if (line == "render" || line == "r") {
      auto r = [&](auto const eye, auto const lookat, auto const up) {
        auto cam =
            rendering::perspective_camera{eye, lookat, up, fov, width, height};
        auto rendered_lines_grid =
            rectilinear_grid{linspace<real_t>{0.0, width - 1, width},
                             linspace<real_t>{0.0, height - 1, height}};
        auto& rendered_lines_mask =
            rendered_lines_grid.vertex_property<int>("mask");
        auto& rendered_lines_pos =
            rendered_lines_grid.vec3_vertex_property("depth");

        rendered_lines_grid.vertices().iterate_indices(
            [&](auto const... is) { rendered_lines_mask(is...) = 0; });

        auto const aabb = discretized_domain.bounding_box();

        render(aabb, line_width, rendered_lines_grid, cam,
               [&](auto const t, auto const& x0, auto const& x1, auto const ix,
                   auto const iy) {
                 auto const x = x0.xyz() * (1 - t) + x1.xyz() * t;
                 if (rendered_lines_mask(ix, iy) == 0) {
                   rendered_lines_pos(ix, iy) = x.xyz();
                 } else {
                   auto const new_depth =
                       euclidean_distance(x.xyz(), eye_sampler(t));
                   auto const old_depth = euclidean_distance(
                       rendered_lines_pos(ix, iy), eye_sampler(t));
                   if (new_depth < old_depth) {
                     rendered_lines_pos(ix, iy) = x.xyz();
                   }
                 }
                 rendered_lines_mask(ix, iy) = 1;
               });

        auto isovalues      = std::vector{5e6};
        auto rendering_grid = rendering::direct_isosurface(
            cam, scalar_sampler, isovalues,
            [&](auto const x_iso, auto const isovalue, auto const& gradient,
                auto const& view_dir, auto const pixel_pos) {
              if (rendered_lines_mask(pixel_pos.x(), pixel_pos.y()) > 0) {
                auto const iso_depth  = euclidean_distance(x_iso, eye);
                auto const line_depth = euclidean_distance(
                    rendered_lines_pos(pixel_pos.x(), pixel_pos.y()), eye);
                if (line_depth - 1e-5 <= iso_depth) {
                  return vec{0.0, 0.0, 0.0, 1.0};
                }
                rendered_lines_mask(pixel_pos.x(), pixel_pos.y()) = 2;
              }

              auto const normal  = normalize(gradient);
              auto const diffuse = std::abs(dot(view_dir, normal));
              auto const vel     = streamwise_velocity_sampler(x_iso);
              auto const s       = std::clamp<real_t>(
                  (vel - min_mapped) / (max_mapped - min_mapped), 0, 1);
              auto const albedo = [&] {
                switch (current_color_scale) {
                  case color_scale_enum::jet:
                    return jet(s);
                  case color_scale_enum::cool_to_warm:
                    return cool_to_warm(s);
                  default:
                  case color_scale_enum::viridis:
                    return viridis(s);
                }
              }();
              auto const col = albedo * diffuse * 0.8 + albedo * 0.2;
              return vec{col(0), col(1), col(2),
                         std::clamp<real_t>(std::pow(s, n) * m + k, 0.0, 1.0)};
            });

        auto& rendered_isosurface =
            rendering_grid.vec3_vertex_property("rendered_isosurface");
        rendering_grid.vertices().iterate_indices([&](auto const... is) {
          if (rendered_lines_mask(is...) == 1) {
            rendered_isosurface(is...) = vec3::zeros();
          }
        });
        return rendering_grid;
      };
      if (filesystem::exists(path)) {
        filesystem::remove_all(path);
      }
      filesystem::create_directory(path);
      if (animated) {
        auto i = std::size_t(0);
        for (auto const t : linspace{0.0, 1.0, num_frames}) {
          if (got_sigint) {
            break;
          }
          auto rendering_grid =
              r(eye_sampler(t), lookat_sampler(t), up_sampler(t));
          std::cout << "rendering " << i + 1 << " / " << num_frames << "...\r";
          ++i;
          auto& rendered_isosurface =
              rendering_grid.vec3_vertex_property("rendered_isosurface");
          std::stringstream str;
          str << std::setw(static_cast<std::size_t>(
                     std::ceil(std::log10(num_frames))))
              << std::setfill('0') << i;
          write_png(path / ("direct_isosurface." + str.str() + ".png"),
                    rendered_isosurface);
        }
      } else {
        auto rendering_grid = r(eye, lookat, up);
        std::cout << "rendering...\r";
        auto& rendered_isosurface =
            rendering_grid.vec3_vertex_property("rendered_isosurface");
        write_png(path / "direct_isosurface.png", rendered_isosurface);
      }
      std::cout << "rendering done!                      \n";
    } else {
      auto line_stream = std::stringstream{line};
      auto cmd         = std::string{};
      auto number      = real_t{};
      line_stream >> cmd >> number;
      if (cmd == "min_scalar" || cmd == "min") {
        std::cout << "setting min scalar\n";
        min_mapped = number;
      } else if (cmd == "max_scalar" || cmd == "max") {
        std::cout << "setting max scalar\n";
        max_mapped = number;
      } else if (cmd == "num_frames") {
        std::cout << "setting number of frames\n";
        num_frames = number;
      } else if (cmd == "n") {
        std::cout << "setting n\n";
        n = number;
      } else if (cmd == "m") {
        std::cout << "setting m\n";
        m = number;
      } else if (cmd == "k") {
        std::cout << "setting k\n";
        k = number;
      } else if (cmd == "width") {
        std::cout << "setting width\n";
        width = number;
      } else if (cmd == "height") {
        std::cout << "setting height\n";
        height = number;
      } else if (cmd == "ortho_scale") {
        std::cout << "setting ortho_scale\n";
        ortho_scale = number;
      } else if (cmd == "linewidth" || cmd == "line_width") {
        std::cout << "setting line width\n";
        line_width = number;
      } else if (cmd == "fov") {
        std::cout << "setting fov\n";
        fov = number;
      } else {
        std::cout << "unknown command\n";
      }
    }
    got_sigint = false;
  }
}
