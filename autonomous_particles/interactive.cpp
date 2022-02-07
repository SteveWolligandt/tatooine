#include <tatooine/rendering/interactive.h>
#include <tatooine/analytical/fields/doublegyre.h>
using namespace tatooine;
struct foo {
  struct shader : gl::shader {
    //------------------------------------------------------------------------------
    static constexpr std::string_view vertex_shader =
        "#version 330 core\n"
        "layout (location = 0) in vec2 position;\n"
        "uniform mat4 view_matrix;\n"
        "uniform mat4 projection_matrix;\n"
        "void main() {\n"
        "  gl_Position = projection_matrix *\n"
        "                view_matrix *\n"
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
      set_view_matrix(Mat4<GLfloat>::eye());
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
    auto set_view_matrix(Mat4<GLfloat> const& MV) -> void {
      set_uniform_mat4("view_matrix", MV.data().data());
    }
  };
  //============================================================================
  private:
  gl::indexeddata<Vec2<GLfloat>> geometry;
  vec2d                          old_cursor_pos;
  vec2d                          cursor_pos;
  Vec2<GLfloat>                  x{0, 0};
  bool                           down = false;

 public:
  //============================================================================
  foo() {
    geometry.vertexbuffer().reserve(2);
    geometry.indexbuffer().reserve(2);
    geometry.vertexbuffer().push_back(x);
    geometry.vertexbuffer().push_back(Vec2<GLfloat>{1, 1});
    geometry.indexbuffer().push_back(0);
    geometry.indexbuffer().push_back(1);
  }
  auto update(rendering::camera auto const& cam, auto const dt) {
    shader::get().set_projection_matrix(cam.projection_matrix());
    shader::get().set_view_matrix(cam.view_matrix());

    auto const dist = euclidean_distance(
        cam.project(vec3f{x.x(), x.y(), 0}).xy(),
        vec2f{cursor_pos.x(), cam.plane_height() - 1 - cursor_pos.y()});
    std::cout << dist << '\n';
    if (dist < 20) {
      shader::get().set_color(1,0,0);
    } else {
      shader::get().set_color(0,0,0);
    }
    if (down) {

      auto const move_dir =
          cam.unproject(vec2f{old_cursor_pos.x(),
                              cam.plane_height() - 1 - old_cursor_pos.y()}).xy() -
          cam.unproject(
              vec2f{cursor_pos.x(), cam.plane_height() - 1 - cursor_pos.y()}).xy();
      x -= move_dir.xy();

      geometry.vertexbuffer()[0] = vec2f{x};
    }
  }
  auto render() {
    shader::get().bind();
    gl::line_width(3);
    geometry.draw_lines();
    gl::point_size(20);
    geometry.draw_points();
  }
  auto on_cursor_moved(double const x, double const y) {
    old_cursor_pos = cursor_pos;
    cursor_pos     = {x, y};
  }

  auto on_button_pressed(gl::button) { down = true; }
  auto on_button_released(gl::button) { down = false; }
};
auto main() -> int {
  auto ps = pointset2{};
  ps.insert_vertex(3, 3);
  ps.insert_vertex(4, 4);
  ps.insert_vertex(-1, -2);
  auto g = rectilinear_grid{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}};
  auto v = analytical::fields::numerical::doublegyre{};
  discretize(v, g, "velocity", execution_policy::parallel);

  rendering::interactive::pre_setup();
  rendering::interactive::render(
      foo{}
      //,ps, g, geometry::ellipse{vec2f{0, 0}, 0.5f, 1.0f}
      //, geometry::ellipse{vec2f{1, 0}, 0.5f, 1.0f}
      //, geometry::ellipse{vec2f{0, 2}, 0.5f, 1.0f}
      //, geometry::ellipse{vec2f{1, 2}, 0.5f, 1.0f}
  );
}
