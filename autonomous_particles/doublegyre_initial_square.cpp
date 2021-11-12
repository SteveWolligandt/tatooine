#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/gl/indexeddata.h>
//==============================================================================
using namespace tatooine;
using analytical::fields::numerical::doublegyre;
//==============================================================================
auto win             = std::unique_ptr<rendering::first_person_window>{};
//==============================================================================
struct listener_t : gl::window_listener {
  Vec2<size_t> mouse_pos;
  bool         left_down = false;
  auto         on_cursor_moved(double x, double y) -> void override {
    mouse_pos = {x, y};

    if (left_down) {
    }
  }
  auto on_button_released(gl::button b) -> void override {
    if (b == gl::button::left) {
      left_down = false;
    }
  }
  auto on_button_pressed(gl::button b) -> void override {
    if (b != gl::button::left) {
      return;
    }
    left_down = true;

    auto unprojected = win->camera_controller().unproject(
        vec4{mouse_pos(0), mouse_pos(1), 0, 1});
    auto q                 = vec2{unprojected(0), unprojected(1)};
  }
};
//==============================================================================
struct particle_shader_t : gl::shader {
  particle_shader_t() {
    add_stage<gl::vertexshader>(
        gl::shadersource{"#version 330\n"
                         "uniform mat4 projection_matrix;\n"
                         "uniform mat4 view_matrix;\n"
                         "uniform mat2 S;\n"
                         "uniform vec2 center;\n"
                         "layout(location = 0) in vec2 pos;\n"
                         "void main(){\n"
                         "  vec2 x = S * pos + center;\n"
                         "  gl_Position = projection_matrix * view_matrix * vec4(x, 0, 1);\n"
                         "}\n"});
    add_stage<gl::fragmentshader>(
        gl::shadersource{"#version 330\n"
                         "uniform vec4 color;\n"
                         "out vec4 frag_out;\n"
                         "void main(){\n"
                         "  frag_out = color;\n"
                         "}\n"});
    create();
  }
  auto set_center(vec2f const& center) -> void {
    set_uniform_vec2("center", center.data_ptr());
  }
  auto set_S(mat2f const& S) -> void {
    set_uniform_mat2("S", S.data_ptr());
  }
  auto set_projection_matrix(mat4f const& P) -> void {
    set_uniform_mat4("projection_matrix", P.data_ptr());
  }
  auto set_view_matrix(mat4f const& V) -> void {
    set_uniform_mat4("view_matrix", V.data_ptr());
  }
  auto set_color(float const r, float const g, float const b, float const a)
      -> void {
    set_uniform("color", r, g, b, a);
  }
};
//==============================================================================
auto ellipse_geom    = std::unique_ptr<gl::indexeddata<vec2f>>{};
auto particle_shader = std::unique_ptr<particle_shader_t>{};
auto listener        = std::unique_ptr<listener_t>{};
//==============================================================================
auto v = doublegyre{};
auto disc =
    std::unique_ptr<autonomous_particle_flowmap_discretization<real_t, 2>>{};
auto initial_particles = std::vector<autonomous_particle2>{};
auto center_of_square  = vec2{1.0, 0.5};
auto radius            = real_t(0.1);
//==============================================================================
auto render_ui() {
  ImGui::Begin("Controls");
  //ImGui::SliderDouble("radius", &radius, 0.0001, 0.1);
  ImGui::End();
}
//==============================================================================
auto render_loop(auto const dt) {
  gl::clear_color(255, 255, 255, 255);
  gl::clear_color_depth_buffer();

  particle_shader->bind();
  particle_shader->set_projection_matrix(
      win->camera_controller().projection_matrix());
  particle_shader->set_view_matrix(
      win->camera_controller().view_matrix());
  particle_shader->set_color(0.9, 0.9, 0.9, 1);
  for (auto const& p : initial_particles) {
    particle_shader->set_S(mat2f{p.S()});
    particle_shader->set_center(vec2f{p.center()});
    ellipse_geom->draw_line_strip();
  }
  particle_shader->set_color(0, 0, 0, 1);
  for (auto const& s : disc->samplers()) {
    particle_shader->set_S(mat2f{s.ellipse1().S()});
    particle_shader->set_center(vec2f{s.ellipse1().center()});
    ellipse_geom->draw_line_strip();
  }

  render_ui();
}
//==============================================================================
auto build_ellipse_geometry() {
  ellipse_geom = std ::make_unique<gl::indexeddata<vec2f>>();
  auto ts      = linspace{0.0, 2 * M_PI, 17+15};
  {
    ellipse_geom->vertexbuffer().resize(ts.size() - 1);
    auto map = ellipse_geom->vertexbuffer().wmap();
    auto map_it = begin(map);
    for (auto it = begin(ts); it != prev(end(ts)); ++it) {
      *(map_it++) = {std::cos(*it), std::sin(*it)};
    }
  }
  {
    ellipse_geom->indexbuffer().resize(ts.size());
    auto map = ellipse_geom->indexbuffer().wmap();
    std::iota(begin(map), end(map), 0);
    map.back() = 0;
  }
}
//==============================================================================
int main(){
  initial_particles.reserve(4);
  initial_particles.emplace_back(
      vec2{center_of_square(0) - radius, center_of_square(1) - radius}, 0.0,
      radius);
  initial_particles.emplace_back(
      vec2{center_of_square(0) + radius, center_of_square(1) - radius}, 0.0,
      radius);
  initial_particles.emplace_back(
      vec2{center_of_square(0) - radius, center_of_square(1) + radius}, 0.0,
      radius);
  initial_particles.emplace_back(
      vec2{center_of_square(0) + radius, center_of_square(1) + radius}, 0.0,
      radius);
  disc =
      std::make_unique<autonomous_particle_flowmap_discretization<real_t, 2>>(
          flowmap(v), 0.0, 2.0, 0.01, initial_particles);
  win = std::make_unique<rendering::first_person_window>();
  listener        = std::make_unique<listener_t>();

  win->add_listener(*listener);
  particle_shader = std::make_unique<particle_shader_t>();
  build_ellipse_geometry();
  win->camera_controller().use_orthographic_camera();
  win->camera_controller().use_orthographic_controller();
  win->render_loop([](auto const dt) { render_loop(dt); });
}
