#ifndef TATOOINE_RENDERING_INTERACTIVE_EDGESET2_H
#define TATOOINE_RENDERING_INTERACTIVE_EDGESET2_H
//==============================================================================
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/color_scale.h>
#include <tatooine/rendering/interactive/renderer.h>
#include <tatooine/rendering/interactive/shaders.h>
#include <tatooine/edgeset.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <floating_point Real>
struct renderer<tatooine::unstructured_simplicial_grid<Real, 2, 1>> {
  using renderable_type = tatooine::unstructured_simplicial_grid<Real, 2, 1>;
  //============================================================================
  using line_shader = shaders::colored_pass_through_2d;
  //============================================================================
 private:
  int           line_width = 1;
  Vec4<GLfloat> line_color = {0, 0, 0, 1};

  gl::vertexbuffer<Vec2<GLfloat>, GLfloat> m_geometry;
  gl::indexbuffer                          m_lines;

 public:
  //============================================================================
  renderer(renderable_type const& grid) {
    if (grid.vertices().size() > 0) {
      init_geometry(grid);
    }
  }
  //----------------------------------------------------------------------------
  auto init_geometry(renderable_type const& grid) {
    m_geometry.resize(grid.vertices().size());
    m_lines.resize(grid.simplices().size() * 2);
    {
      auto data = m_geometry.wmap();
      auto k    = std::size_t{};
      for (auto const v : grid.vertices()) {
        data[k++] = Vec2<GLfloat>{grid[v]};
      }
    }
    {
      auto data = m_lines.wmap();
      auto k    = std::size_t{};
      for (auto const s : grid.simplices()) {
        auto const [v0, v1] = grid[s];
        data[k++]               = v0.index();
        data[k++]               = v1.index();
      }
    }
  }
  //============================================================================
  auto render() {
    auto& line_shader = line_shader::get();
    line_shader.bind();

    line_shader.set_color(line_color(0), line_color(1), line_color(2),
                          line_color(3));
    gl::line_width(line_width);
    auto vao = gl::vertexarray{};
    vao.bind();
    m_geometry.bind();
    m_geometry.activate_attributes();
    m_lines.bind();
    vao.draw_lines(m_lines.size());
  }
  //----------------------------------------------------------------------------
  auto properties(renderable_type const& grid) {
    ImGui::PushID("##__");
    ImGui::Text("Edgeset");
    ImGui::DragInt("Line width", &line_width, 1, 1, 20);
    ImGui::ColorEdit4("Wireframe Color", line_color.data());
    ImGui::PopID();
  }
  //----------------------------------------------------------------------------
  auto update(auto const dt, renderable_type const& grid,
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
      line_shader::get().set_model_view_matrix(cam.view_matrix());
    } else {
      line_shader::get().set_model_view_matrix(
          Mat4<GLfloat>{cam.view_matrix()});
    }
  }
};
template <floating_point Real>
struct renderer<tatooine::edgeset<Real, 2>>
    : renderer<tatooine::unstructured_simplicial_grid<Real, 2, 1>> {
  using parent_type =
      renderer<tatooine::unstructured_simplicial_grid<Real, 2, 1>>;
  using parent_type::parent_type;
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
