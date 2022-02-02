#ifndef TATOOINE_RENDERING_INTERACTIVE_RECTILINEAR_GRID_H
#define TATOOINE_RENDERING_INTERACTIVE_RECTILINEAR_GRID_H
//==============================================================================
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/renderer.h>
//==============================================================================
namespace tatooine::rendering::detail::interactive {
//==============================================================================
template <typename Axis0, typename Axis1>
struct renderer<tatooine::rectilinear_grid<Axis0, Axis1>> {
  //==============================================================================
  struct shader : gl::shader {
    //------------------------------------------------------------------------------
    static constexpr std::string_view vertex_shader =
        "#version 330 core\n"
        "layout (location = 0) in vec2 position;\n"
        "uniform mat4 modelview_matrix;\n"
        "uniform mat4 projection_matrix;\n"
        "void main() {\n"
        "  gl_Position = projection_matrix *\n"
        "                modelview_matrix *\n"
        "                vec4(position, 0, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static constexpr std::string_view fragment_shader =
        "#version 330 core\n"
        "uniform vec4 color;\n"
        "out vec4 out_color;\n"
        "void main() {\n"
        "  out_color = color;"
        "}\n";
    //------------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = shader{};
      return s;
    }
    //------------------------------------------------------------------------------
   private:
    //------------------------------------------------------------------------------
    shader() {
      add_stage<gl::vertexshader>(gl::shadersource{vertex_shader});
      add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader});
      create();
      set_color(0, 0, 0);
      set_projection_matrix(Mat4<GLfloat>::eye());
      set_modelview_matrix(Mat4<GLfloat>::eye());
    }
    //------------------------------------------------------------------------------
   public:
    //------------------------------------------------------------------------------
    auto set_color(GLfloat const r, GLfloat const g, GLfloat const b,
                   GLfloat const a = 1) -> void {
      set_uniform("color", r, g, b, a);
    }
    //------------------------------------------------------------------------------
    auto set_projection_matrix(Mat4<GLfloat> const& P) -> void {
      set_uniform_mat4("projection_matrix", P.data().data());
    }
    //------------------------------------------------------------------------------
    auto set_modelview_matrix(Mat4<GLfloat> const& MV) -> void {
      set_uniform_mat4("modelview_matrix", MV.data().data());
    }
  };
  //==============================================================================
  struct render_data {
    int                            line_width = 1;
    Vec4<GLfloat>                  color      = {0, 0, 0, 1};
    gl::indexeddata<Vec2<GLfloat>> geometry;
  };
  //==============================================================================
  static auto init(tatooine::rectilinear_grid<Axis0, Axis1> const& grid) {
    auto       d            = render_data{};
    auto const num_vertices = grid.template size<0>() * 2 + grid.template size<1>() * 2;
    d.geometry.vertexbuffer().resize(num_vertices);
    d.geometry.indexbuffer().resize(num_vertices);
    {
      auto data = d.geometry.vertexbuffer().wmap();
      auto k    = std::size_t{};
      for (std::size_t i = 0; i < grid.template size<0>(); ++i) {
        data[k++] = Vec2<GLfloat>{grid.template dimension<0>()[i],
                                  grid.template dimension<1>().front()};
        data[k++] = Vec2<GLfloat>{grid.template dimension<0>()[i],
                                  grid.template dimension<1>().back()};
      }
      for (std::size_t i = 0; i < grid.template size<1>(); ++i) {
        data[k++] = Vec2<GLfloat>{grid.template dimension<0>().front(),
                                  grid.template dimension<1>()[i]};
        data[k++] = Vec2<GLfloat>{grid.template dimension<0>().back(),
                                  grid.template dimension<1>()[i]};
      }
    }
    {
      auto data = d.geometry.indexbuffer().wmap();
      for (std::size_t i = 0; i < num_vertices; ++i) {
        data[i] = i;
      }
    }
    return d;
  }
  //==============================================================================
  static auto properties(render_data& data) {
    ImGui::Text("Rectilinear Grid");
    ImGui::DragInt("Line width", &data.line_width, 1, 1, 20);
    ImGui::ColorEdit4("Color", data.color.data().data());
  }
  //==============================================================================
  static auto render(camera auto const&                              cam,
                     tatooine::rectilinear_grid<Axis0, Axis1> const& grid,
                     render_data&                                    data) {
    using CamReal = typename std::decay_t<decltype(cam)>::real_t;
    static auto constexpr cam_is_float = is_same<GLfloat, CamReal>;
    auto& shader = shader::get();
    shader.bind();
    if constexpr (cam_is_float) {
      shader.set_projection_matrix(cam.projection_matrix());
    } else {
      shader.set_projection_matrix(Mat4<GLfloat>{cam.projection_matrix()});
    }

    if constexpr (cam_is_float) {
      shader.set_modelview_matrix(cam.view_matrix());
    } else {
      shader.set_modelview_matrix(Mat4<GLfloat>{cam.view_matrix()});
    }

    shader.set_color(data.color(0), data.color(1), data.color(2),
                     data.color(3));
    gl::line_width(data.line_width);
    data.geometry.draw_lines();
  }
};
//==============================================================================
}  // namespace tatooine::rendering::detail::interactive
//==============================================================================
#endif
