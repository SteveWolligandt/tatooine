#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/analytical/fields/saddle.h>
#include <tatooine/rendering/interactive.h>
#include <tatooine/rendering/interactive/shaders.h>
using namespace tatooine;
struct movable_line {
  using pass_through =
      rendering::interactive::shaders::colored_pass_through_2d_without_matrices;
  struct hoverable_shader : gl::shader {
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
        "    discard;\n"
        "  }\n"
        "  out_color = vec4(frag_hovered,0,0,1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = hoverable_shader{};
      return s;
    }
    //------------------------------------------------------------------------------
   private:
    //------------------------------------------------------------------------------
    hoverable_shader() {
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
  struct only_hovered_shader : gl::shader {
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
        "  if (frag_hovered == 0) {\n"
        "    discard;\n"
        "  }\n"
        "  out_color = vec4(frag_hovered,0,0,1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = only_hovered_shader{};
      return s;
    }
    //------------------------------------------------------------------------------
   private:
    //------------------------------------------------------------------------------
    only_hovered_shader() {
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
  vec2d                                    cursor_pos;
  vec2d                                    current_point;
  std::vector<bool>                        hovered;
  int                                      point_size = 20;
  rendering::orthographic_camera<GLfloat>  cam;
  std::vector<autonomous_particle2> const& advected_particles;
  std::vector<vec2>                        locals;
  gl::indexeddata<Vec2<GLfloat>, int>      locals_gpu;
  bool                                     mouse_down = false;
  std::vector<std::size_t>                 hovered_indices;
  //static auto constexpr minimap_range = 10;
  static auto constexpr minimap_range = 0.25;

  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  movable_line(auto const& advected_particles)
      : hovered(size(advected_particles), false),
        cam{Vec3<GLfloat>{0, 0, 0},
            Vec3<GLfloat>{0, 0, -1},
            -minimap_range, minimap_range,
            -minimap_range, minimap_range,
            -1,
            1,
            Vec4<std::size_t>{10, 10, 500, 500}},
        advected_particles{advected_particles},
        locals(size(advected_particles), vec2::zeros()) {
    locals_gpu.vertexbuffer().resize(size(advected_particles));
    locals_gpu.indexbuffer().resize(size(advected_particles));
    for (std::size_t i = 0; i < size(advected_particles); ++i) {
      locals_gpu.indexbuffer()[i] = i;
    }
  }
  //----------------------------------------------------------------------------
  auto render(auto const& renderable, rendering::camera auto const& cam) {
    hoverable_shader::get().set_projection_matrix(cam.projection_matrix());
    hoverable_shader::get().set_view_matrix(cam.view_matrix());
    hoverable_shader::get().bind();
    gl::point_size(point_size);
    {
      auto p = gl::indexeddata<Vec2<GLfloat>, int>{};
      p.vertexbuffer().resize(1);
      p.indexbuffer().resize(1);
      {
        auto data = p.vertexbuffer().map();
        data[0]   = {Vec2<GLfloat>{current_point}, 0};
      }
      {
        auto data = p.indexbuffer().map();
        data[0]   = 0;
      }
      p.draw_points();
    }

    auto& ellipse_shader = rendering::interactive::renderer<
        geometry::ellipse<double>>::shader::get();
    auto& ellipse_geometry = rendering::interactive::renderer<
        geometry::ellipse<double>>::geometry::get();
    ellipse_shader.set_projection_matrix(cam.projection_matrix());
    ellipse_shader.bind();
    ellipse_shader.set_color(1, 0, 0);
    gl::line_width(3);
    for (std::size_t i = 0; i < size(hovered); ++i) {
      if (hovered[i]) {
        ellipse_shader.set_model_view_matrix(
            cam.view_matrix() *
            rendering::interactive::renderer<geometry::ellipse<double>>::
                construct_model_matrix(advected_particles[i].S(),
                                       advected_particles[i].center()));
        ellipse_geometry.draw_line_loop();
      }
    }
  }
  //----------------------------------------------------------------------------
  auto late_render() {
    cam.set_gl_viewport();
    pass_through::get().bind();
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

    only_hovered_shader::get().set_projection_matrix(cam.projection_matrix());
    only_hovered_shader::get().set_view_matrix(cam.view_matrix());
    only_hovered_shader::get().bind();
    gl::point_size(10);
    locals_gpu.draw_points();
    hoverable_shader::get().set_projection_matrix(cam.projection_matrix());
    hoverable_shader::get().set_view_matrix(cam.view_matrix());
    hoverable_shader::get().bind();
    gl::point_size(5);
    locals_gpu.draw_points();
  }
  //----------------------------------------------------------------------------
  auto update_points(rendering::camera auto const& cam) {
    current_point =
        vec2{cam.unproject(vec2f{cursor_pos.x(), cursor_pos.y()}).xy()};

    auto i   = std::size_t{};
    auto map = locals_gpu.vertexbuffer().wmap();
    for (auto const& p : advected_particles) {
      auto s    = p.sampler();
      locals[i] = s.nabla_phi_inv() * (current_point - s.ellipse1().center());
      //locals[i] = *inv(s.ellipse1().S()) * (current_point - s.ellipse1().center());
      map[i]    = {vec2f{locals[i]}, hovered[i] ? 1 : 0};
      ++i;
    }
  }
  //----------------------------------------------------------------------------
  auto on_cursor_moved(double const cursor_x, double const cursor_y,
                       rendering::camera auto const& cam) {
    cursor_pos = {cursor_x, cursor_y};
    if (mouse_down) {
      update_points(cam);
    }

    for (std::size_t i = 0; i < size(locals); ++i) {
      auto const proj  = this->cam.project(vec2f{locals[i]}).xy();
      auto const proj2 = vec2{cam.unproject(vec2f{cursor_pos}).xy()};
      hovered[i]       = euclidean_distance(proj, cursor_pos) < 10 ||
                   advected_particles[i].is_inside(proj2);
      locals_gpu.vertexbuffer()[i] = {vec2f{locals[i]}, hovered[i] ? 1 : 0};
    }
  }
  //----------------------------------------------------------------------------
  auto on_button_pressed(gl::button b, rendering::camera auto const& cam) {
    if (b == gl::button::left) {
      mouse_down = true;
      update_points(cam);
    }
  }
  auto on_button_released(gl::button b) {
    if (b == gl::button::left) {
      mouse_down = false;
    }
  }
};
//------------------------------------------------------------------------------
auto main() -> int {
  auto g = rectilinear_grid{linspace{-2.0, 2.0, 101}, linspace{-2.0, 2.0, 101}};
  auto dg = analytical::fields::numerical::doublegyre{};
  auto s = analytical::fields::numerical::saddle{};
  discretize(dg, g, "velocity_doublegyre", execution_policy::parallel);
  discretize(s, g, "velocity_saddle", execution_policy::parallel);

  auto uuid_generator = std::atomic_uint64_t{};
  auto p = autonomous_particle2{vec2{1, 0.5}, 0, 0.1, uuid_generator};
  auto initial_grid =
      rectilinear_grid{linspace{-0.1, 0.1, 3}, linspace{-0.1, 0.1, 3}};
  auto initial_particles =
      std::vector<autonomous_particle2>{
        {vec2{0.1, 0}, 0, 0.05, uuid_generator},
        {vec2{-0.1, 0}, 0, 0.05, uuid_generator},
        {vec2{0, 0.1}, 0, 0.05, uuid_generator},
        {vec2{0, -0.1}, 0, 0.05, uuid_generator}
      };
  auto const [advected_particles, advected_simple_particles, edges] =
      autonomous_particle2::advect_with_three_splits(flowmap(s), 0.01, 0, 1,
                                                     initial_grid);
  rendering::interactive::pre_setup();
  auto m = movable_line{advected_particles};
  rendering::interactive::render(m, advected_particles, g);
}
