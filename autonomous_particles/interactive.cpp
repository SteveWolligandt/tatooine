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
        "  out_color = vec4(0,0,0,1);\n"
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
        "layout (location = 1) in int active;\n"
        "flat out int frag_active;\n"
        "uniform mat4 view_matrix;\n"
        "uniform mat4 projection_matrix;\n"
        "void main() {\n"
        "  frag_active = active;\n"
        "  gl_Position = projection_matrix *\n"
        "                view_matrix *\n"
        "                vec4(position, 0, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static constexpr std::string_view fragment_shader =
        "#version 330 core\n"
        "uniform vec4 color;\n"
        "out vec4 out_color;\n"
        "flat in int frag_active;\n"
        "void main() {\n"
        "  if (frag_active == 1) {\n"
        "    out_color = vec4(1,0,0,1);\n"
        "  } else {\n"
        "    out_color = vec4(0,0,0,1);\n"
        "  }\n"
        "  out_color = vec4(frag_active,0,0,1);\n"
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
  gl::indexeddata<Vec2<GLfloat>, int>     geometry;
  vec2d                                   old_cursor_pos;
  vec2d                                   cursor_pos;
  std::vector<Vec2<GLfloat>>              xs = {{0, 0}, {1, 1}, {2, 1}, {2, 0}};
  bool                                    down = false;
  std::vector<bool>                       hovered;
  std::vector<bool>                       grabbed;
  int                                     point_size = 20;
  rendering::orthographic_camera<GLfloat> cam;

  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  movable_line()
      : hovered(size(xs), false),
        grabbed(size(xs), false),
        cam{Vec3<GLfloat>{0, 0, 0},
            Vec3<GLfloat>{0, 0, -1},
            -3,
            3,
            -3,
            3,
            -3,
            3,
            Vec4<std::size_t>{10, 10, 1000, 1000}} {
    geometry.vertexbuffer().reserve(size(xs));
    geometry.indexbuffer().reserve(size(xs));
    for (auto const& x : xs) {
      geometry.vertexbuffer().push_back(x, false);
    }
    for (std::size_t i = 0; i < size(xs); ++i) {
      geometry.indexbuffer().push_back(i);
    }
  }
  //----------------------------------------------------------------------------
  auto late_render() {
    cam.set_gl_viewport();
    auto outline = gl::indexeddata<Vec2<GLfloat>>{};
    outline.vertexbuffer().resize(4);
    outline.indexbuffer().resize(4);
    {
      auto data = outline.vertexbuffer().map();
      data[0]   = Vec2<GLfloat>{-0.999, -0.999};
      data[1]   = Vec2<GLfloat>{ 0.999, -0.999};
      data[2]   = Vec2<GLfloat>{ 0.999,  0.999};
      data[3]   = Vec2<GLfloat>{-0.999,  0.999};
    }
    {
      auto data = outline.indexbuffer().map();
      data[0]   = 0;
      data[1]   = 1;
      data[2]   = 2;
      data[3]   = 3;
    }
    line_shader::get().bind();
    gl::line_width(3);
    outline.draw_line_loop();

    point_shader::get().set_projection_matrix(cam.projection_matrix());
    point_shader::get().set_view_matrix(cam.view_matrix());
    point_shader::get().bind();
    gl::point_size(point_size);
    geometry.draw_points();
  }
  //----------------------------------------------------------------------------
  auto on_cursor_moved(double const cursor_x, double const cursor_y) {
    old_cursor_pos = cursor_pos;
    cursor_pos     = {cursor_x, cursor_y};

    auto const old_unprojected =
        cam.unproject(vec2f{old_cursor_pos.x(),
                            cam.plane_height() - 1 - old_cursor_pos.y()})
            .xy();
    auto const unprojected =
        cam.unproject(
               vec2f{cursor_pos.x(), cam.plane_height() - 1 - cursor_pos.y()})
            .xy();

    auto const move_dir = unprojected - old_unprojected;

    auto       i = std::size_t{};
    auto const one_is_grabbed =
        std::ranges::find(grabbed, true) != end(grabbed);
    for (auto& x : xs) {
      auto const dist = euclidean_distance(
          cam.project(vec3f{x.x(), x.y(), 0}).xy(),
          vec2f{cursor_pos.x(), cursor_pos.y()});
      hovered[i] = dist < point_size / 2;
      if (hovered[i] && down && !one_is_grabbed) {
        grabbed[i] = true;
      }
      if (grabbed[i]) {
        x += move_dir.xy();
      }
      geometry.vertexbuffer()[i] = {vec2f{x}, hovered[i] ? 1 : 0};
      ++i;
    }
  }
  //----------------------------------------------------------------------------
  auto on_button_pressed(gl::button /*b*/) { down = true; }
  //----------------------------------------------------------------------------
  auto on_button_released(gl::button /*b*/) {
    down = false;
    for (auto&& g : grabbed) {
      g = false;
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
  rendering::interactive::render(ps, g, movable_line{});
}
