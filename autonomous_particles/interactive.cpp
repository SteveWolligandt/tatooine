#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/rendering/interactive.h>
using namespace tatooine;
struct movable_line {
  struct line_shader : gl::shader {
    //------------------------------------------------------------------------------
    static constexpr std::string_view vertex_shader =
        "#version 330 core\n"
        "layout (location = 0) in vec2 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static constexpr std::string_view fragment_shader =
        "#version 330 core\n"
        "out vec4 out_color;\n"
        "void main() {\n"
        "  out_color = vec4(0, 0, 0, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = line_shader{};
      return s;
    }
    //------------------------------------------------------------------------------
   private:
    //------------------------------------------------------------------------------
    line_shader() {
      add_stage<gl::vertexshader>(gl::shadersource{vertex_shader});
      add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader});
      create();
    }
  };
  struct point_shader : gl::shader {
    //------------------------------------------------------------------------------
    static constexpr std::string_view vertex_shader =
        "#version 330 core\n"
        "layout (location = 0) in vec2 position;\n"
        "layout (location = 1) in int hovered;\n"
        "flat out int frag_hovered;\n"
        "uniform mat4 view_matrix;\n"
        "uniform mat4 projection_matrix;\n"
        "void main() {\n"
        "  frag_hovered = hovered;\n"
        "  gl_Position = projection_matrix *\n"
        "                view_matrix *\n"
        "                vec4(position, 0, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static constexpr std::string_view fragment_shader =
        "#version 330 core\n"
        "uniform vec4 color;\n"
        "out vec4 out_color;\n"
        "flat in int frag_hovered;\n"
        "void main() {\n"
        "  if (frag_hovered == 1) {\n"
        "    out_color = vec4(1,0,0,1);\n"
        "  } else {\n"
        "    out_color = vec4(0,0,0,1);\n"
        "  }\n"
        "  out_color = vec4(frag_hovered,0,0,1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = point_shader{};
      return s;
    }
    //------------------------------------------------------------------------------
   private:
    //------------------------------------------------------------------------------
    point_shader() {
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
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  gl::indexeddata<Vec2<GLfloat>, int>      geometry;
  vec2d                                    cursor_pos;
  std::vector<bool>                        hovered;
  int                                      point_size = 20;
  rendering::orthographic_camera<GLfloat>  cam;
  std::vector<autonomous_particle2> const& ps;

  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  movable_line(auto const& ps)
      : hovered(size(ps), false),
        cam{Vec3<GLfloat>{0, 0, 0},
            Vec3<GLfloat>{0, 0, -1},
            -0.5, 0.5,
            -0.5, 0.5,
            -1, 1,
            Vec4<std::size_t>{10, 10, 200, 200}},
        ps{ps} {
    geometry.vertexbuffer().resize(size(ps));
    geometry.indexbuffer().resize(size(ps));
    for (std::size_t i = 0; i < size(ps); ++i) {
      geometry.indexbuffer()[i] = i;
    }
  }
  //----------------------------------------------------------------------------
  auto late_render() {
    cam.set_gl_viewport();
    line_shader::get().bind();
    {
      auto outline = gl::indexeddata<Vec2<GLfloat>>{};
      outline.vertexbuffer().resize(4);
      outline.indexbuffer().resize(4);
      {
        auto data = outline.vertexbuffer().map();
        data[0]   = Vec2<GLfloat>{-0.999, -0.999};
        data[1]   = Vec2<GLfloat>{0.999, -0.999};
        data[2]   = Vec2<GLfloat>{0.999, 0.999};
        data[3]   = Vec2<GLfloat>{-0.999, 0.999};
      }
      {
        auto data = outline.indexbuffer().map();
        data[0]   = 0;
        data[1]   = 1;
        data[2]   = 2;
        data[3]   = 3;
      }
      gl::line_width(3);
      outline.draw_line_loop();
    }
    {
      auto axes = gl::indexeddata<Vec2<GLfloat>>{};
      axes.vertexbuffer().resize(4);
      axes.indexbuffer().resize(4);
      {
        auto data = axes.vertexbuffer().map();
        data[0]   = Vec2<GLfloat>{0, -1};
        data[1]   = Vec2<GLfloat>{0, 1};
        data[2]   = Vec2<GLfloat>{-1, 0};
        data[3]   = Vec2<GLfloat>{1, 0};
      }
      {
        auto data = axes.indexbuffer().map();
        data[0]   = 0;
        data[1]   = 1;
        data[2]   = 2;
        data[3]   = 3;
      }
      gl::line_width(1);
      axes.draw_lines();
    }

    point_shader::get().set_projection_matrix(cam.projection_matrix());
    point_shader::get().set_view_matrix(cam.view_matrix());
    point_shader::get().bind();
    gl::point_size(point_size);
    geometry.draw_points();
  }
  //----------------------------------------------------------------------------
  auto on_cursor_moved(double const cursor_x, double const cursor_y) {
    cursor_pos = {cursor_x, cursor_y};
  }
  //----------------------------------------------------------------------------
  auto on_button_pressed(gl::button /*b*/, rendering::camera auto const& cam) {
    auto const unprojected =
        vec2{cam.unproject(vec2f{cursor_pos.x(), cursor_pos.y()}).xy()};
    auto i = std::size_t{};
    auto map = geometry.vertexbuffer().wmap();
    for (auto const& p : ps) {
      auto       s               = p.sampler();
      auto const p1              = s.sample(unprojected, backward);
      auto const local           = s.opposite_center(backward) - p1;
      map[i] = {vec2f{local}, hovered[i] ? 1 : 0};
      ++i;
    }
  }
};
//------------------------------------------------------------------------------
auto main() -> int {
  auto g = rectilinear_grid{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}};
  auto v = analytical::fields::numerical::doublegyre{};
  discretize(v, g, "velocity", execution_policy::parallel);

  auto uuid_generator = std::atomic_uint64_t{};
  auto p = autonomous_particle2{vec2{1, 0.5}, 0, 0.1, uuid_generator};
  auto [ps, ss, es] =
      p.advect_with_three_splits(flowmap(v), 0.01, 3, uuid_generator);

  rendering::interactive::pre_setup();
  auto m = movable_line{ps};
  rendering::interactive::render(ps, g, m);
}
