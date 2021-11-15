#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/rendering/line_loop.h>
#include <tatooine/rendering/pointset.h>
//==============================================================================
namespace ImGui {
//==============================================================================
auto square_widget(const char* label, double& x, double& y) -> int {
  size_t const grab_radius = 6;

  const ImGuiStyle& Style    = GetStyle();
  ImDrawList*       DrawList = GetWindowDrawList();
  ImGuiWindow*      Window   = GetCurrentWindow();
  if (Window->SkipItems) {
    return false;
  }

  int changed = 0;

  // prepare canvas
  ImVec2 Canvas(100.f, 100.f);
  ImRect bb(Window->DC.CursorPos, Window->DC.CursorPos + Canvas);

  RenderFrame(bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBg, 1), true,
              Style.FrameRounding);

  // background grid
  for (float i = 0; i <= Canvas.x; i += (Canvas.x / 4)) {
    DrawList->AddLine(ImVec2(bb.Min.x + i, bb.Min.y),
                      ImVec2(bb.Min.x + i, bb.Max.y),
                      GetColorU32(ImGuiCol_TextDisabled));
  }
  for (float i = 0; i <= Canvas.y; i += (Canvas.y / 4)) {
    DrawList->AddLine(ImVec2(bb.Min.x, bb.Min.y + i),
                      ImVec2(bb.Max.x, bb.Min.y + i),
                      GetColorU32(ImGuiCol_TextDisabled));
  }

  ImVec2 pos = ImVec2(x, 1 - y) * (bb.Max - bb.Min) + bb.Min;
  SetCursorScreenPos(pos - ImVec2(grab_radius, grab_radius));
  InvisibleButton(label, ImVec2(2 * grab_radius, 2 * grab_radius));
  if (IsItemActive() || IsItemHovered()) {
    SetTooltip("(%4.3f, %4.3f)", x, y);
  }
  if (IsItemActive() && IsMouseDragging(0)) {
    x += GetIO().MouseDelta.x / Canvas.x;
    y -= GetIO().MouseDelta.y / Canvas.y;
    x       = std::clamp<double>(x, 0, 1);
    y       = std::clamp<double>(y, 0, 1);
    changed = true;
  }

  // draw lines and grabbers
  ImVec4 white(GetStyle().Colors[ImGuiCol_Text]);
  DrawList->AddCircleFilled(ImVec2(x, 1 - y) * (bb.Max - bb.Min) + bb.Min,
                            grab_radius, ImColor(white));

  // restore cursor pos
  SetCursorScreenPos(ImVec2(bb.Min.x, bb.Max.y + grab_radius));  // :P

  return changed;
}
//==============================================================================
}  // namespace ImGui
//==============================================================================
using namespace tatooine;
using analytical::fields::numerical::doublegyre;
//==============================================================================
auto win = std::unique_ptr<rendering::first_person_window>{};
//==============================================================================
auto ellipse_geom         = std::unique_ptr<rendering::line_loop<vec2f>>{};
auto square_t0_geom       = std::unique_ptr<rendering::line_loop<vec2f>>{};
auto square_t1_geom       = std::unique_ptr<rendering::line_loop<vec2f>>{};
auto domain_boundary_geom = std::unique_ptr<rendering::line_loop<vec2f>>{};
auto points_geom          = std::unique_ptr<rendering::pointset<vec2f>>{};
//==============================================================================
auto v = doublegyre{};
auto disc =
    std::unique_ptr<autonomous_particle_flowmap_discretization<real_t, 2>>{};
auto initial_particles   = std::vector<autonomous_particle2>{};
auto center_of_square    = vec2{1.0, 0.5};
auto local_pos_in_square = vec2{0.5, 0.5};
auto x0                  = center_of_square;
auto x1                  = x0;
auto radius              = real_t(0.01);
auto t0                  = real_t(0.0);
auto tau                 = real_t(3.0);
auto error               = 0.0;
auto physical_pos_in_square() {
  return (1 - local_pos_in_square(0)) * (1 - local_pos_in_square(1)) *
             initial_particles[0].center() +
         (local_pos_in_square(0)) * (1 - local_pos_in_square(1)) *
             initial_particles[1].center() +
         (local_pos_in_square(0)) * (local_pos_in_square(1)) *
             initial_particles[2].center() +
         (1 - local_pos_in_square(0)) * (local_pos_in_square(1)) *
             initial_particles[3].center();
}
//==============================================================================
auto update_initial_particles() -> void;
auto update_advected_particles() -> void;
//==============================================================================
struct listener_t : gl::window_listener {
  Vec2<size_t> mouse_pos;
  bool         left_down = false;

  bool grabbed_x1     = false;
  bool grabbed_center = false;

  virtual ~listener_t() = default;

  auto on_cursor_moved(double x, double y) -> void override {
    mouse_pos = {x, y};

    if (left_down) {
      auto unprojected4 = win->camera_controller().unproject(
          vec4{mouse_pos(0), mouse_pos(1), 0, 1});
      auto unprojected = vec{unprojected4.x(), unprojected4.y()};
      if (grabbed_x1) {
        x1(0) = unprojected(0);
        x1(1) = unprojected(1);
        x0    = flowmap(v)(x1, t0 + tau, -tau);
        error = euclidean_distance(x0, physical_pos_in_square());
        points_geom->vertexbuffer()[0] = x0;
        points_geom->vertexbuffer()[1] = x1;
      } else if (grabbed_center) {
        auto const newpos = vec{unprojected(0), unprojected(1)};
        auto const offset = newpos - center_of_square;
        center_of_square  = newpos;
        x0                = x0 + offset;
        x1                = flowmap(v)(x0, t0, tau);
        x0                = flowmap(v)(x1, t0 + tau, -tau);
        error             = euclidean_distance(x0, physical_pos_in_square());
        points_geom->vertexbuffer()[0] = x0;
        points_geom->vertexbuffer()[1] = x1;
        update_initial_particles();
        update_advected_particles();
      }
    }
  }
  auto on_button_released(gl::button b) -> void override {
    if (b == gl::button::left) {
      left_down      = false;
      grabbed_x1     = false;
      grabbed_center = false;
    }
  }
  auto on_button_pressed(gl::button b) -> void override {
    if (b == gl::button::left) {
      left_down         = true;
      auto unprojected4 = win->camera_controller().unproject(
          vec4{mouse_pos(0), mouse_pos(1), 0, 1});
      auto unprojected = vec{unprojected4.x(), unprojected4.y()};
      if (euclidean_distance(unprojected, x1) < 0.05) {
        grabbed_x1 = true;
      } else if (euclidean_distance(unprojected, center_of_square) < radius) {
        grabbed_center = true;
      }
    }
  }
};
auto listener = std::unique_ptr<listener_t>{};
//==============================================================================
struct line_shader_t : gl::shader {
  line_shader_t() {
    add_stage<gl::vertexshader>(gl::shadersource{
        "#version 330\n"
        "uniform mat4 projection_matrix;\n"
        "uniform mat4 view_matrix;\n"
        "layout(location = 0) in vec2 pos;\n"
        "void main(){\n"
        "  gl_Position = projection_matrix * view_matrix * vec4(pos, 0, 1);\n"
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
auto line_shader = std::unique_ptr<line_shader_t>{};
//==============================================================================
struct particle_shader_t : gl::shader {
  particle_shader_t() {
    add_stage<gl::vertexshader>(gl::shadersource{
        "#version 330\n"
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
  auto set_S(mat2f const& S) -> void { set_uniform_mat2("S", S.data_ptr()); }
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
auto particle_shader = std::unique_ptr<particle_shader_t>{};
//------------------------------------------------------------------------------
auto render_ui() {
  ImGui::Begin("Controls");
  bool need_update = false;
  if (ImGui::DragDouble("t0", &t0, 0.01, 0.0001, 10.0)) {
    need_update = true;
  }
  if (ImGui::DragDouble("tau", &tau, 0.01, 0.0001, 10.0)) {
    need_update = true;
  }
  if (ImGui::DragDouble("radius", &radius, 0.01, 0.0001, 0.5)) {
    x0          = physical_pos_in_square();
    need_update = true;
  }
  if (ImGui::Button("x0 to center of square")) {
    local_pos_in_square = {0.5, 0.5};
    x0                  = physical_pos_in_square();
    need_update         = true;
  }
  ImGui::Text("error: %e", error);
  if (ImGui::square_widget("foo", local_pos_in_square(0),
                           local_pos_in_square(1))) {
    x0          = physical_pos_in_square();
    need_update = true;
  }
  ImGui::End();
  if (need_update) {
    x1    = flowmap(v)(x0, t0, tau);
    x0    = flowmap(v)(x1, t0 + tau, -tau);
    error = euclidean_distance(x0, physical_pos_in_square());
    points_geom->vertexbuffer()[0] = x0;
    points_geom->vertexbuffer()[1] = x1;
    update_initial_particles();
    update_advected_particles();
  }
}
//==============================================================================
auto render_loop(auto const dt) {
  gl::clear_color(255, 255, 255, 255);
  gl::clear_color_depth_buffer();
  auto const P = win->camera_controller().projection_matrix();
  auto const V = win->camera_controller().view_matrix();
  particle_shader->set_projection_matrix(P);
  particle_shader->set_view_matrix(V);
  line_shader->set_projection_matrix(P);
  line_shader->set_view_matrix(V);

  particle_shader->bind();

  particle_shader->set_color(0.5, 0.5, 0.5, 1);
  for (auto const& p : initial_particles) {
    particle_shader->set_S(mat2f{p.S()});
    particle_shader->set_center(vec2f{p.center()});
    ellipse_geom->draw();
  }
  particle_shader->set_color(0, 0, 0, 1);
  for (auto const& s : disc->samplers()) {
    particle_shader->set_S(mat2f{s.ellipse1().S()});
    particle_shader->set_center(vec2f{s.ellipse1().center()});
    ellipse_geom->draw();
  }

  line_shader->bind();
  line_shader->set_color(0.9, 0.9, 0.9, 1);
  domain_boundary_geom->draw();
  line_shader->set_color(1, 0.5, 0.5, 1);
  square_t0_geom->draw();
  line_shader->set_color(1, 0, 0, 1);
  square_t1_geom->draw();

  gl::point_size(10);
  line_shader->set_color(0, 0, 0.5, 1);
  points_geom->draw();

  render_ui();
}
//==============================================================================
auto build_ellipse_geometry() {
  static constexpr auto num_vertices = 17 + 15;
  ellipse_geom = std::make_unique<rendering::line_loop<vec2f>>(num_vertices);
  auto ts      = linspace{0.0, 2 * M_PI, num_vertices + 1};
  {
    auto map    = ellipse_geom->vertexbuffer().wmap();
    auto map_it = begin(map);
    for (auto it = begin(ts); it != prev(end(ts)); ++it) {
      *(map_it++) = {std::cos(*it), std::sin(*it)};
    }
  }
}
//==============================================================================
auto build_square_geometry() {
  square_t0_geom       = std ::make_unique<rendering::line_loop<vec2f>>(4);
  square_t1_geom       = std ::make_unique<rendering::line_loop<vec2f>>(4);
  domain_boundary_geom = std ::make_unique<rendering::line_loop<vec2f>>(4);
  domain_boundary_geom->vertexbuffer()[0] = vec{0, 0};
  domain_boundary_geom->vertexbuffer()[1] = vec{2, 0};
  domain_boundary_geom->vertexbuffer()[2] = vec{2, 1};
  domain_boundary_geom->vertexbuffer()[3] = vec{0, 1};
}
//==============================================================================
auto update_initial_particles() -> void {
  initial_particles = {
      autonomous_particle2{
          vec2{center_of_square(0) - radius, center_of_square(1) - radius}, t0,
          radius},
      autonomous_particle2{
          vec2{center_of_square(0) + radius, center_of_square(1) - radius}, t0,
          radius},
      autonomous_particle2{
          vec2{center_of_square(0) + radius, center_of_square(1) + radius}, t0,
          radius},
      autonomous_particle2{
          vec2{center_of_square(0) - radius, center_of_square(1) + radius}, t0,
          radius}};
  square_t0_geom->vertexbuffer()[0] = initial_particles[0].center();
  square_t0_geom->vertexbuffer()[1] = initial_particles[1].center();
  square_t0_geom->vertexbuffer()[2] = initial_particles[2].center();
  square_t0_geom->vertexbuffer()[3] = initial_particles[3].center();
}
//------------------------------------------------------------------------------
auto update_advected_particles() -> void {
  disc =
      std::make_unique<autonomous_particle_flowmap_discretization<real_t, 2>>(
          flowmap(v), t0, tau, 0.01, initial_particles);
  auto phi = flowmap(v);
  square_t1_geom->vertexbuffer()[0] =
      phi(initial_particles[0].center(), t0, tau);
  square_t1_geom->vertexbuffer()[1] =
      phi(initial_particles[1].center(), t0, tau);
  square_t1_geom->vertexbuffer()[2] =
      phi(initial_particles[2].center(), t0, tau);
  square_t1_geom->vertexbuffer()[3] =
      phi(initial_particles[3].center(), t0, tau);
}
//------------------------------------------------------------------------------
int main() {
  win = std::make_unique<rendering::first_person_window>();
  win->camera_controller().use_orthographic_camera();
  win->camera_controller().use_orthographic_controller();
  listener = std::make_unique<listener_t>();
  win->add_listener(*listener);
  build_ellipse_geometry();
  particle_shader = std::make_unique<particle_shader_t>();
  line_shader     = std::make_unique<line_shader_t>();
  build_square_geometry();
  points_geom = std::make_unique<rendering::pointset<vec2f>>(2);
  update_initial_particles();
  update_advected_particles();
  x1    = flowmap(v)(x0, t0, tau);
  x0    = flowmap(v)(x1, t0 + tau, -tau);
  error = euclidean_distance(x0, physical_pos_in_square());
  points_geom->vertexbuffer()[0] = x0;
  points_geom->vertexbuffer()[1] = x1;

  win->render_loop([](auto const dt) { render_loop(dt); });
}
