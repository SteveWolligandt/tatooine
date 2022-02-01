#ifndef TATOOINE_RENDERING_INTERACTIVE_ELLIPSE_H
#define TATOOINE_RENDERING_INTERACTIVE_ELLIPSE_H
//==============================================================================
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/linspace.h>
#include <tatooine/rendering/camera.h>
//==============================================================================
namespace tatooine::rendering::detail::interactive {
//==============================================================================
struct ellipse_geometry : gl::indexeddata<Vec2<GLfloat>> {
  static auto get() -> auto& {
    static auto instance = ellipse_geometry();
    return instance;
  }
  explicit ellipse_geometry(std::size_t const num_vertices = 32) {
    vertexbuffer().resize(num_vertices);
    {
      auto ts = linspace<float>{0, 2 * M_PI, num_vertices + 1};
      ts.pop_back();
      auto vb_map = vertexbuffer().rmap();
      auto i      = std::size_t{};
      for (auto const t : ts) {
        vb_map[i++] = Vec2<GLfloat>{std::cos(t), std::sin(t)};
        std::cout << vb_map[i-1] << '\n';
      }
    }
    indexbuffer().resize((num_vertices + 1) * 2);
    {
      auto ib_map = indexbuffer().rmap();
      auto i      = std::size_t{};
      auto j      = std::size_t{};
      for (; i < num_vertices - 1; ++i) {
        ib_map[j++] = i;
        ib_map[j++] = i + 1;
      }
      ib_map[j++] = i;
      ib_map[j++] = 0;
    }
  }
};
//==============================================================================
struct ellipse_shader : gl::shader {
  //------------------------------------------------------------------------------
  static constexpr std::string_view vertex_shader =
      "#version 330 core\n"
      "layout (location = 0) in vec2 position;\n"
      "uniform mat4 modelview_matrix;\n"
      "uniform mat4 projection_matrix;\n"
      "uniform mat2 S;\n"
      "uniform vec2 center;\n"
      "void main() {\n"
      "  gl_Position = projection_matrix *\n"
      //"                modelview_matrix *\n"
      "                vec4(position, 0, 1);\n"
      //"  gl_Position = vec4(position, 0, 1);\n"
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
    static auto s = ellipse_shader{};
    return s;
  }
  //------------------------------------------------------------------------------
 private:
  //------------------------------------------------------------------------------
  ellipse_shader() {
    add_stage<gl::vertexshader>(gl::shadersource{vertex_shader});
    add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader});
    create();
  }
  //------------------------------------------------------------------------------
 public:
  //------------------------------------------------------------------------------
  auto set_color(GLfloat const r, GLfloat const g, GLfloat const b,
                 GLfloat const a = 1) {
    set_uniform("color", r, g, b, a);
  }
  //------------------------------------------------------------------------------
  auto set_S(Mat2<GLfloat> const& S) { set_uniform_mat2("S", S.data().data()); }
  //------------------------------------------------------------------------------
  auto set_center(Vec2<GLfloat> const& c) {
    set_uniform_vec2("center", c.data().data());
  }
  //------------------------------------------------------------------------------
  auto set_projection_matrix(Mat4<GLfloat> const& P) {
    set_uniform_mat4("projection_matrix", P.data().data());
  }
  //------------------------------------------------------------------------------
  auto set_modelview_matrix(Mat4<GLfloat> const& MV) {
    set_uniform_mat4("modelview_matrix", MV.data().data());
  }
};
//==============================================================================
template <floating_point Real>
auto render(camera auto const&                      cam,
            geometry::hyper_ellipse<Real, 2> const& ell) {
  auto& shader = ellipse_shader::get();
  shader.bind();
  if constexpr (std::same_as<GLfloat,
                             typename std::decay_t<decltype(cam)>::real_t>) {
    shader.set_projection_matrix(cam.projection_matrix());
    shader.set_modelview_matrix(cam.view_matrix());
  } else {
    shader.set_projection_matrix(Mat4<GLfloat>{cam.projection_matrix()});
    shader.set_modelview_matrix(Mat4<GLfloat>{cam.view_matrix()});
  }

  if constexpr (std::same_as<GLfloat, Real>) {
    shader.set_S(ell.S());
    shader.set_center(ell.center());
  } else {
    shader.set_S(Mat2<GLfloat>{ell.S()});
    shader.set_center(Vec2<GLfloat>{ell.center()});
  }
  shader.set_color(0, 0, 0);

  gl::line_width(4);
  gl::point_size(4);
  ellipse_geometry::get().draw_points();
}
//==============================================================================
}  // namespace tatooine::rendering::detail::interactive
//==============================================================================
#endif
