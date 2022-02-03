#ifndef TATOOINE_RENDERING_INTERACTIVE_ELLIPSE_H
#define TATOOINE_RENDERING_INTERACTIVE_ELLIPSE_H
//==============================================================================
#include <tatooine/geometry/ellipse.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/linspace.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/renderer.h>
//==============================================================================
namespace tatooine::rendering::detail::interactive {
//==============================================================================
template <floating_point Real>
struct renderer<tatooine::geometry::ellipse<Real>> {
using renderable_type = tatooine::geometry::ellipse<Real>;
  struct geometry : gl::indexeddata<Vec2<GLfloat>> {
    static auto get() -> auto& {
      static auto instance = geometry{};
      return instance;
    }
    explicit geometry(std::size_t const num_vertices = 128) {
      vertexbuffer().resize(num_vertices);
      {
        auto ts = linspace<float>{0, 2 * M_PI, num_vertices + 1};
        ts.pop_back();
        auto vb_map = vertexbuffer().wmap();
        auto i      = std::size_t{};
        for (auto const t : ts) {
          vb_map[i++] = Vec2<GLfloat>{std::cos(t), std::sin(t)};
        }
      }
      indexbuffer().resize(num_vertices);
      {
        auto data = indexbuffer().wmap();
        for (std::size_t i = 0; i < num_vertices; ++i) {
          data[i] = i;
        }
      }
    }
  };
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
    int           line_width = 1;
    Vec4<GLfloat> color      = {0, 0, 0, 1};
  };
  //==============================================================================
  static auto init(renderable_type const& ell) {
    return render_data{};
  }
  //==============================================================================
  static auto properties(renderable_type const& /*ell*/, render_data& data) {
    ImGui::Text("Ellipse");
    ImGui::DragInt("Line width", &data.line_width, 1, 1, 20);
    ImGui::ColorEdit4("Color", data.color.data().data());
  }
  //==============================================================================
  static auto render(camera auto const&                       cam,
                     renderable_type const& ell,
                     render_data&                             data) {
    using CamReal = typename std::decay_t<decltype(cam)>::real_t;
    static auto constexpr ell_is_float = is_same<GLfloat, Real>;
    static auto constexpr cam_is_float = is_same<GLfloat, CamReal>;
    auto& shader                       = shader::get();
    shader.bind();
    if constexpr (cam_is_float) {
      shader.set_projection_matrix(cam.projection_matrix());
    } else {
      shader.set_projection_matrix(Mat4<GLfloat>{cam.projection_matrix()});
    }

    auto M = [&] {
      auto constexpr O = GLfloat(0);
      auto constexpr I = GLfloat(1);
      if constexpr (ell_is_float) {
        return Mat4<GLfloat>{{ell.S()(0, 0), ell.S()(0, 1), O, ell.center(0)},
                             {ell.S()(1, 0), ell.S()(1, 1), O, ell.center(1)},
                             {O, O, I, O},
                             {O, O, O, I}};
      } else {
        return Mat4<GLfloat>{{GLfloat(ell.S()(0, 0)), GLfloat(ell.S()(0, 1)), O,
                              GLfloat(ell.center(0))},
                             {GLfloat(ell.S()(1, 0)), GLfloat(ell.S()(1, 1)), O,
                              GLfloat(ell.center(1))},
                             {O, O, I, O},
                             {O, O, O, I}};
      }
    }();
    auto V = [&] {
      if constexpr (cam_is_float) {
        return cam.view_matrix();
      } else {
        return Mat4<GLfloat>{cam.view_matrix()};
      }
    }();
    shader.set_modelview_matrix(V * M);

    shader.set_color(data.color(0), data.color(1), data.color(2),
                     data.color(3));
    gl::line_width(data.line_width);
    geometry::get().draw_line_loop();
  }
};
//==============================================================================
}  // namespace tatooine::rendering::detail::interactive
//==============================================================================
#endif
