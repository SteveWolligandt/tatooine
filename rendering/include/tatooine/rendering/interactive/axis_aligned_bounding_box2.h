#ifndef TATOOINE_RENDERING_INTERACTIVE_AXIS_ALIGNED_BOUNDING_BOX2_H
#define TATOOINE_RENDERING_INTERACTIVE_AXIS_ALIGNED_BOUNDING_BOX2_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/color_scale.h>
#include <tatooine/rendering/interactive/renderer.h>
#include <tatooine/rendering/interactive/shaders.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <floating_point Real>
struct renderer<tatooine::axis_aligned_bounding_box<Real, 2>> {
  using renderable_type = tatooine::axis_aligned_bounding_box<Real, 2>;
  //============================================================================
  struct geometry : gl::indexeddata<Vec2<GLfloat>> {
    static auto get() -> auto& {
      static auto instance = geometry{};
      return instance;
    }
    geometry() {
      vertexbuffer().resize(4);
      {
        auto vb_map = vertexbuffer().wmap();
        vb_map[0]   = Vec2<GLfloat>{0, 0};
        vb_map[1]   = Vec2<GLfloat>{1, 0};
        vb_map[2]   = Vec2<GLfloat>{1, 1};
        vb_map[3]   = Vec2<GLfloat>{0, 1};
      }
      indexbuffer().resize(4);
      {
        auto data = indexbuffer().wmap();
        data[0]   = 0;
        data[1]   = 1;
        data[2]   = 2;
        data[3]   = 3;
      }
    }
  };
  //============================================================================
  using line_shader = shaders::colored_pass_through_2d;
  //============================================================================
 private:
  int                                               line_width = 1;
  Vec4<GLfloat>                                     color      = {0, 0, 0, 1};

 public:
  //============================================================================
  renderer(renderable_type const& aabb) {}
  //----------------------------------------------------------------------------
  auto properties(renderable_type const& aabb) {
    ImGui::Text("Axis Aligned Bounding Box");
    ImGui::DragInt("Line width", &line_width, 1, 1, 20);
    ImGui::ColorEdit4("Color", color.data());
  }
  //============================================================================
  auto render() {
    auto& line_shader = line_shader::get();
    line_shader.bind();

    line_shader.set_color(color(0), color(1), color(2), color(3));
    gl::line_width(line_width);
    geometry::get().draw_line_loop();
  }
  //----------------------------------------------------------------------------
  auto update(auto const dt, renderable_type const& aabb,
              camera auto const& cam) {
    using CamReal = typename std::decay_t<decltype(cam)>::real_type;
    static auto constexpr cam_is_float = is_same<GLfloat, CamReal>;
    if constexpr (cam_is_float) {
      line_shader::get().set_projection_matrix(cam.projection_matrix());
    } else {
      line_shader::get().set_projection_matrix(
          Mat4<GLfloat>{cam.projection_matrix()});
    }

    if constexpr (cam_is_float) {
      line_shader::get().set_model_view_matrix(
          cam.view_matrix() *
          translation_matrix<GLfloat>(aabb.min(0), aabb.min(1), 0) *
          scale_matrix<GLfloat>(aabb.max(0) - aabb.min(0),
                                aabb.max(1) - aabb.min(1), 1));
    } else {
      line_shader::get().set_model_view_matrix(
          Mat4<GLfloat>{cam.view_matrix()} *
          translation_matrix<GLfloat>(aabb.min(0), aabb.min(1), 0) *
          scale_matrix<GLfloat>(aabb.max(0) - aabb.min(0),
                                aabb.max(1) - aabb.min(1), 1));
    }
  }
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
