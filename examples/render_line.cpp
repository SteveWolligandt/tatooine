#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/rendering/orthographic_camera.h>
#include <tatooine/rendering/perspective_camera.h>
#include <tatooine/rendering/render_line.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto const width  = std::size_t(500);
  auto const height = std::size_t(1000);
  auto       grid =
      uniform_rectilinear_grid2{linspace<real_t>{0, width - 1, width},
                                linspace<real_t>{0, height - 1, height}};
  auto& rasterized_line = grid.vec3_vertex_property("line");
  grid.vertices().iterate_indices(
      [&](auto const... is) { rasterized_line(is...) = vec3::ones(); },
      execution_policy::parallel);
  auto col = color_scales::viridis{};

  auto cam = rendering::perspective_camera{
      vec3{2, 4, 3}, vec3{0, 0, 0}, 90.0, 0.01, 100, width, height};
  // auto cam = rendering::orthographic_camera<double>{
  //     vec3{1, 1, 1}, vec3{0, 0, 0}, -1, 1, -1, 1, -10, 10, 1000, 1000};

  auto render     = [&](vec4 const& x0, vec4 const& x1) {
    rendering::render_line(
            cam.project(x0).xy(), cam.project(x1).xy(), 3, grid,
            [&](auto const t, auto const... is) { rasterized_line(is...) = col(t); });
  };

  auto render_aabb = [&](auto const& aabb) {
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
  };
  auto aabb0 = axis_aligned_bounding_box{vec3{-1, -1, -1}, vec3{1, 1, 1}};
  auto aabb1 = axis_aligned_bounding_box{vec3{1, -1, -1}, vec3{3, 1, 1}};
  render_aabb(aabb0);
  render_aabb(aabb1);
  rasterized_line.write_png("rasterized_line.png");
}
