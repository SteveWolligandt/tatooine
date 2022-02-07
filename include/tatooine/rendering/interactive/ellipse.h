#ifndef TATOOINE_RENDERING_INTERACTIVE_ELLIPSE_H
#define TATOOINE_RENDERING_INTERACTIVE_ELLIPSE_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/geometry/ellipse.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/linspace.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/renderer.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <typename Ellipse>
requires(is_derived_from_hyper_ellipse<Ellipse>&& Ellipse::num_dimensions() ==
         2) struct renderer<Ellipse> {
  using renderable_type = Ellipse;
  using real_type       = typename Ellipse::real_type;
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
    static constexpr std::string_view fragment_shader = "#version 330 core\n"
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
  int           line_width = 1;
  Vec4<GLfloat> color      = {0, 0, 0, 1};
  //==============================================================================
  renderer(renderable_type const& /*ell*/) {}
  //------------------------------------------------------------------------------
  static auto set_projection_matrix(Mat4<GLfloat> const& P) {
    shader::get().set_projection_matrix(P);
  }
  //------------------------------------------------------------------------------
  auto properties(renderable_type const& /*ell*/) {
    ImGui::Text("Ellipse");
    ImGui::DragInt("Line width", &line_width, 1, 1, 20);
    ImGui::ColorEdit4("Color", color.data().data());
  }
  //==============================================================================
  auto update(auto const dt, auto& ell, camera auto const& cam) {
    auto& shader        = shader::get();
    using cam_real_type = typename std::decay_t<decltype(cam)>::real_t;
    static auto constexpr ell_is_float = is_same<GLfloat, real_type>;
    static auto constexpr cam_is_float = is_same<GLfloat, cam_real_type>;

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
    shader.set_color(color(0), color(1), color(2), color(3));
  }
  //----------------------------------------------------------------------------
  auto render() {
    shader::get().bind();
    gl::line_width(line_width);
    geometry::get().draw_line_loop();
  }
};
//==============================================================================
template <typename Ellipse>
requires(is_derived_from_hyper_ellipse<Ellipse>&& Ellipse::num_dimensions() ==
         2) struct renderer<std::vector<Ellipse>> {
  using renderable_type = std::vector<Ellipse>;
  using real_type       = typename Ellipse::real_type;
  //==============================================================================
  using geometry =
      typename renderer<tatooine::geometry::ellipse<real_type>>::geometry;
  using shader =
      typename renderer<tatooine::geometry::ellipse<real_type>>::shader;
  //==============================================================================
  int           line_width = 1;
  Vec4<GLfloat> color      = {0, 0, 0, 1};
  //==============================================================================
  renderer(renderable_type const& ell) {}
  //------------------------------------------------------------------------------
  static auto set_projection_matrix(Mat4<GLfloat> const& P) {
    shader::get().set_projection_matrix(P);
  }
  //------------------------------------------------------------------------------
  auto properties(renderable_type const& ell) {
    ImGui::Text("Ellipse");
    ImGui::DragInt("Line width", &line_width, 1, 1, 20);
    ImGui::ColorEdit4("Color", color.data().data());
  }
  //==============================================================================
  auto render(auto const& ellipses, camera auto const& cam) {
    auto& shader = shader::get();
    shader.bind();
    using cam_real_type = typename std::decay_t<decltype(cam)>::real_t;
    static auto constexpr ell_is_float = is_same<GLfloat, real_type>;
    static auto constexpr cam_is_float = is_same<GLfloat, cam_real_type>;

    gl::line_width(line_width);
    shader.set_color(color(0), color(1), color(2), color(3));
    auto const V = [&] {
      if constexpr (cam_is_float) {
        return cam.view_matrix();
      } else {
        return Mat4<GLfloat>{cam.view_matrix()};
      }
    }();
    for (auto const& ell : ellipses) {
      auto M = [&] {
        auto constexpr O = GLfloat(0);
        auto constexpr I = GLfloat(1);
        if constexpr (ell_is_float) {
          return Mat4<GLfloat>{{ell.S()(0, 0), ell.S()(0, 1), O, ell.center(0)},
                               {ell.S()(1, 0), ell.S()(1, 1), O, ell.center(1)},
                               {O, O, I, O},
                               {O, O, O, I}};
        } else {
          return Mat4<GLfloat>{{GLfloat(ell.S()(0, 0)), GLfloat(ell.S()(0, 1)),
                                O, GLfloat(ell.center(0))},
                               {GLfloat(ell.S()(1, 0)), GLfloat(ell.S()(1, 1)),
                                O, GLfloat(ell.center(1))},
                               {O, O, I, O},
                               {O, O, O, I}};
        }
      }();
      shader.set_modelview_matrix(V * M);
      geometry::get().draw_line_loop();
    }
  }
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
