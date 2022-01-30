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
#include <tatooine/rendering/render_line.h>

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
  std::cerr << "creating files ...";
  auto channelflow_file =
      hdf5::file{"/home/vcuser/channel_flow/dino_res_154000.h5"};
  std::cerr << "done!\n";

  std::cerr << "loading axes ...";
  auto const axis0 =
      channelflow_file.dataset<double>("CartGrid/axis0").read_as_vector();
  auto const axis1 =
      channelflow_file.dataset<double>("CartGrid/axis1").read_as_vector();
  auto const axis2 =
      channelflow_file.dataset<double>("CartGrid/axis2").read_as_vector();
  std::cerr << "done!\n";

  std::cerr << "creating grids ...";
  auto discretized_domain = rectilinear_grid{axis0, axis1, axis2};
  discretized_domain.set_chunk_size_for_lazy_properties(256);
  std::cerr << "discretized_domain:\n" << discretized_domain << '\n';
  std::cerr << "done!\n";

  std::cerr << "loading data ...";
  auto& scalar_field = discretized_domain.insert_vertex_property(
      channelflow_file.dataset<double>("Q_cheng"), "Q_cheng");
  auto& streamwise_velocity = discretized_domain.insert_vertex_property(
      channelflow_file.dataset<double>("velocity/yvel"), "velocity_y");
  std::cerr << "done!\n";

  std::cerr << "creating samplers ...";
  auto scalar_sampler              = scalar_field.linear_sampler();
  auto streamwise_velocity_sampler = streamwise_velocity.linear_sampler();
  std::cerr << "done!\n";

  auto current_color_scale = color_scale_enum::viridis;
  auto viridis             = color_scales::viridis{};
  auto jet                 = color_scales::jet{};
  auto cool_to_warm        = color_scales::cool_to_warm{};

  std::cerr << "creating cameras ...";
  std::size_t width = 1000, height = 1000;
  real_t      ortho_scale = 2;

  auto eye    = line3{};
  auto lookat = line3{};
  auto up     = line3{};

  setup_eye_rotation(eye, discretized_domain);
  setup_lookat_rotation(lookat, discretized_domain);
  setup_up_rotation(up);
  // setup_eye_flight(eye, discretized_domain);
  // setup_lookat_flight(lookat, discretized_domain);
  // setup_up_flight(up);

  auto eye_sampler    = eye.linear_sampler();
  auto up_sampler     = up.linear_sampler();
  auto lookat_sampler = lookat.linear_sampler();
  std::cerr << "done!\n";

  // std::cerr << "calculating min and max scalars ...";
  // auto min_scalar = std::numeric_limits<double>::max();
  // auto max_scalar = -std::numeric_limits<double>::max();
  //
  // discretized_domain.vertices().iterate_indices([&](auto const... is) {
  //  min_scalar = std::min(min_scalar, streamwise_velocity(is...));
  //  max_scalar = std::max(max_scalar, streamwise_velocity(is...));
  //});
  //[[maybe_unused]] auto const medium_scalar = (max_scalar + min_scalar) / 2;
  auto min_mapped = real_t(13);
  auto max_mapped = real_t(27);
  std::cerr << "done!\n";
  // std::cerr << "data range: " << min_scalar << " - " << max_scalar << '\n';
  auto num_frames = std::size_t(5);
  auto run        = true;
  auto n          = real_t(1);
  auto m          = real_t(1);
  auto k          = real_t(0);
  std::signal(SIGINT, sigint_handler);
  while (run) {
    std::cout << "> ";
    auto line = std::string{};
    std::getline(std::cin, line);
    if (line == "quit" || line == "q" || line == "exit") {
      std::cout << "k bye.\n";
      run = false;
    } else if (line.substr(0, line.find(" ")) == "color_scale") {
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
      if (filesystem::exists(path)) {
        filesystem::remove_all(path);
      }
      filesystem::create_directory(path);
      auto i = std::size_t(0);
      // for (auto const t : linspace{0.0, 1.0, num_frames}) {
      auto t = 0.75;
      if (got_sigint) {
        break;
      }
      std::cerr << "rendering " << i + 1 << " / " << num_frames << "...\r";
      auto cam = rendering::perspective_camera{
          eye_sampler(t), lookat_sampler(t), up_sampler(t), 90, width, height};
      auto rendered_lines_grid =
          rectilinear_grid{linspace<double>{0.0, width - 1, width},
                           linspace<double>{0.0, height - 1, height}};
      auto& rendered_lines_mask =
          rendered_lines_grid.vertex_property<int>("mask");
      auto& rendered_lines_pos =
          rendered_lines_grid.vec3_vertex_property("depth");

      rendered_lines_grid.vertices().iterate_indices(
          [&](auto const... is) { rendered_lines_mask(is...) = 0; });
      auto render = [&](vec4 const& x0, vec4 const& x1) {
        auto pixels = rendering::render_line(
            cam.project(x0).xy(), cam.project(x1).xy(), rendered_lines_grid);

        auto const s_offset = double(1) / (size(pixels) - 1);
        auto       j        = std::size_t(0);
        for (auto const& ix : pixels) {
          auto const s = s_offset * j++;
          auto const x = x0 * (1 - s) + x1 * s;
          if (rendered_lines_mask(ix(0), ix(1)) == 0) {
            rendered_lines_pos(ix(0), ix(1)) = x.xyz();
          } else {
            auto const new_depth = euclidean_distance(x.xyz(), eye_sampler(t));
            auto const old_depth = euclidean_distance(
                rendered_lines_pos(ix(0), ix(1)), eye_sampler(t));
            if (new_depth < old_depth) {
              rendered_lines_pos(ix(0), ix(1)) = x.xyz();
            }
          }
          rendered_lines_mask(ix(0), ix(1)) = 1;
        }
      };

      auto const aabb = discretized_domain.bounding_box();
      render(vec4{aabb.min(0), aabb.min(1), aabb.min(2), 1},
             vec4{aabb.max(0), aabb.min(1), aabb.min(2), 1});
      render(vec4{aabb.min(0), aabb.max(1), aabb.min(2), 1},
             vec4{aabb.max(0), aabb.max(1), aabb.min(2), 1});
      render(vec4{aabb.min(0), aabb.min(1), aabb.max(2), 1},
             vec4{aabb.max(0), aabb.min(1), aabb.max(2), 1});
      render(vec4{aabb.min(0), aabb.max(1), aabb.max(2), 1},
             vec4{aabb.max(0), aabb.max(1), aabb.max(2), 1});

      render(vec4{aabb.min(0), aabb.min(1), aabb.min(2), 1},
             vec4{aabb.min(0), aabb.max(1), aabb.min(2), 1});
      render(vec4{aabb.max(0), aabb.min(1), aabb.min(2), 1},
             vec4{aabb.max(0), aabb.max(1), aabb.min(2), 1});
      render(vec4{aabb.min(0), aabb.min(1), aabb.max(2), 1},
             vec4{aabb.min(0), aabb.max(1), aabb.max(2), 1});
      render(vec4{aabb.max(0), aabb.min(1), aabb.max(2), 1},
             vec4{aabb.max(0), aabb.max(1), aabb.max(2), 1});

      render(vec4{aabb.min(0), aabb.min(1), aabb.min(2), 1},
             vec4{aabb.min(0), aabb.min(1), aabb.max(2), 1});
      render(vec4{aabb.max(0), aabb.min(1), aabb.min(2), 1},
             vec4{aabb.max(0), aabb.min(1), aabb.max(2), 1});
      render(vec4{aabb.min(0), aabb.max(1), aabb.min(2), 1},
             vec4{aabb.min(0), aabb.max(1), aabb.max(2), 1});
      render(vec4{aabb.max(0), aabb.max(1), aabb.min(2), 1},
             vec4{aabb.max(0), aabb.max(1), aabb.max(2), 1});

      auto isovalues      = std::vector{5e6};
      auto rendering_grid = rendering::direct_isosurface(
          cam, scalar_sampler, isovalues,
          [&](auto const x_iso, auto const isovalue, auto const& gradient,
              auto const& view_dir, auto const pixel_pos) {
            if (rendered_lines_mask(pixel_pos.x(), pixel_pos.y()) > 0) {
              auto const iso_depth  = euclidean_distance(x_iso, eye_sampler(t));
              auto const line_depth = euclidean_distance(
                  rendered_lines_pos(pixel_pos.x(), pixel_pos.y()),
                  eye_sampler(t));
              if (line_depth <= iso_depth) {
                return vec{0.0, 0.0, 0.0, 1.0};
              }
              static std::mutex m;
              auto              l = std::lock_guard{m};
              std::cout << "============\n";
              std::cout << "line_depth: " << line_depth << '\n';
              std::cout << "iso_depth: " << iso_depth << '\n';
              std::cout << "rendered_lines_pos: " << rendered_lines_pos(pixel_pos.x(), pixel_pos.y()) << '\n';
              std::cout << "diff: " << std::abs(line_depth - iso_depth) << '\n';
              rendered_lines_mask(pixel_pos.x(), pixel_pos.y()) = 2;
            }

            auto const normal  = normalize(gradient);
            auto const diffuse = std::abs(dot(view_dir, normal));
            auto const vel     = streamwise_velocity_sampler(x_iso);
            auto const s       = std::clamp<double>(
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
                       std::clamp<double>(std::pow(s, n) * m + k, 0.0, 1.0)};
          });
      auto& rendered_isosurface =
          rendering_grid.vec3_vertex_property("rendered_isosurface");
      rendering_grid.vertices().iterate_indices([&](auto const... is) {
        if (rendered_lines_mask(is...) == 1) {
          rendered_isosurface(is...) = vec3::zeros();
        }
      });
      std::stringstream str;
      str << std::setw(
                 static_cast<std::size_t>(std::ceil(std::log10(num_frames))))
          << std::setfill('0') << i;
      write_png(path / ("direct_isosurface." + str.str() + ".png"),
                rendered_isosurface);
      ++i;
      //}
      std::cerr << "rendering done!                      \n";
    } else {
      auto line_stream = std::stringstream{line};
      auto cmd         = std::string{};
      auto number      = double{};
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
      } else {
        std::cout << "unknown command\n";
      }
    }
    got_sigint = false;
  }
}
