#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/texture.h>
#include <tatooine/gpu/upload.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/first_person_window.h>

#include <cmath>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto v = analytical::fields::numerical::doublegyre{};
// auto v                = analytical::fields::numerical::saddle{};
auto particles        = autonomous_particle2::container_type{};
auto samplers         = std::vector<autonomous_particle2::sampler_type>{};
auto active_particles = std::vector<bool>{};
auto x0               = vec2{};
auto x1               = vec2{};
auto x1_integrated    = vec2{};
auto t0               = real_t{0};
auto tau              = real_t{0};
auto r0               = real_t{0.01};
//==============================================================================
auto render_ui() -> void;
auto render_particles() -> void;
auto render_pathline() -> void;
auto render_query_point() -> void;
auto render(std::chrono::duration<double> const& dt) -> void;
auto create_point_geometry() -> void;
auto create_particle_geometry() -> void;
auto create_pathline_geometry() -> void;
auto update_pathline_geometry() -> void;
//==============================================================================
struct point_shader_t : gl::shader {
  point_shader_t() {
    add_stage<gl::vertexshader>(gl::shadersource{
        "#version 330 core\n"
        "layout(location = 0) in vec2 pos;\n"
        "uniform mat4 projection_matrix;\n"
        "uniform mat4 modelview_matrix;\n"
        "void main() {\n"
        "  gl_Position = projection_matrix * modelview_matrix * vec4(pos, 0, "
        "1);\n"
        "}\n"});
    add_stage<gl::fragmentshader>(
        gl::shadersource{"#version 330 core\n"
                         "uniform vec4 color;\n"
                         "out vec4 frag_out;\n"
                         "void main() {\n"
                         "  frag_out = color;\n"
                         "}\n"});
    create();
  }
  auto set_modelview_matrix(mat4f const& MV) -> void {
    set_uniform_mat4("modelview_matrix", MV.data_ptr());
  }
  auto set_projection_matrix(mat4f const& P) -> void {
    set_uniform_mat4("projection_matrix", P.data_ptr());
  }
  auto set_color(float const r, float const g, float const b, float const a)
      -> void {
    set_uniform("color", r, g, b, a);
  }
};
//==============================================================================
struct particle_shader_t : gl::shader {
  particle_shader_t() {
    add_stage<gl::vertexshader>(gl::shadersource{
        "#version 330 core\n"
        "layout(location = 0) in vec2 pos;\n"
        "uniform mat4 projection_matrix;\n"
        "uniform mat4 modelview_matrix;\n"
        "//------------------------------------------------------------------\n"
        "void main() {\n"
        "  gl_Position = projection_matrix * modelview_matrix * vec4(pos, 0, "
        "1);\n"
        "}\n"});
    add_stage<gl::fragmentshader>(gl::shadersource{
        "#version 330 core\n"
        "uniform vec4 color;\n"
        "out vec4 frag_out;\n"
        "//------------------------------------------------------------------\n"
        "void main() {\n"
        "  frag_out = color;\n"
        "}\n"});
    create();
    set_color(0, 0, 0, 1);
  }
  auto set_modelview_matrix(mat4f const& MV) -> void {
    set_uniform_mat4("modelview_matrix", MV.data_ptr());
  }
  auto set_projection_matrix(mat4f const& P) -> void {
    set_uniform_mat4("projection_matrix", P.data_ptr());
  }
  auto set_color(float const r, float const g, float const b, float const a)
      -> void {
    set_uniform("color", r, g, b, a);
  }
};
//==============================================================================
struct pathline_shader_t : gl::shader {
  pathline_shader_t() {
    add_stage<gl::vertexshader>(gl::shadersource{
        "#version 330 core\n"
        "layout(location = 0) in vec2 pos;\n"
        "uniform mat4 projection_matrix;\n"
        "uniform mat4 modelview_matrix;\n"
        "//------------------------------------------------------------------\n"
        "void main() {\n"
        "  gl_Position = projection_matrix * modelview_matrix * vec4(pos, 0, "
        "1);\n"
        "}\n"});
    add_stage<gl::fragmentshader>(gl::shadersource{
        "#version 330 core\n"
        "out vec4 frag_out;\n"
        "//------------------------------------------------------------------\n"
        "void main() {\n"
        "  frag_out = vec4(0.1,0.1,0.1,1);\n"
        "}\n"});
    create();
  }
  auto set_modelview_matrix(mat4f const& MV) -> void {
    set_uniform_mat4("modelview_matrix", MV.data_ptr());
  }
  auto set_projection_matrix(mat4f const& P) -> void {
    set_uniform_mat4("projection_matrix", P.data_ptr());
  }
};
//==============================================================================
struct listener_t;
//==============================================================================
auto menu_open         = false;
auto particle_shader   = std::unique_ptr<particle_shader_t>{};
auto pathline_shader   = std::unique_ptr<pathline_shader_t>{};
auto point_shader      = std::unique_ptr<point_shader_t>{};
auto win               = std::unique_ptr<rendering::first_person_window>{};
auto listener          = std::unique_ptr<listener_t>{};
auto particle_geometry = std::unique_ptr<gl::indexeddata<vec2f>>{};
auto particle_center_geometry = std::unique_ptr<gl::indexeddata<vec2f>>{};
auto pathline_geometry        = std::unique_ptr<gl::indexeddata<vec2f>>{};
auto x0_geometry              = std::unique_ptr<gl::indexeddata<vec2f>>{};
auto x1_geometry              = std::unique_ptr<gl::indexeddata<vec2f>>{};
auto update_x0(Vec2<size_t> const& mouse_pos) -> void {
  auto q = win->camera_controller().unproject(
      vec2f{mouse_pos(0), win->height() - 1 - mouse_pos(1)}).xy();
  //auto active_it                 = begin(active_particles);
  active_particles               = std::vector<bool>(size(particles), false);
  x0                             = vec2{q};
  x0_geometry->vertexbuffer()[0] = q;
}
//------------------------------------------------------------------------------
auto update_x1() -> void {
  x1                = vec2::zeros();
  auto active_it    = begin(active_particles);
  auto sampler_it   = begin(samplers);
  auto active_count = size_t{0};
  for (; active_it != end(active_particles); ++active_it, ++sampler_it) {
    if (*active_it) {
      x1 += sampler_it->sample(x0, backward);
      ++active_count;
    }
  }
  if (active_count > 0) {
    x1 /= active_count;
  } else {
    x1 = vec2::fill(0.0 / 0.0);
  };
  x1_geometry->vertexbuffer()[0] = vec2f{x1};
}
//------------------------------------------------------------------------------
auto update_nearest() {
  active_particles       = std::vector<bool>(size(particles), false);
  auto shortest_distance = std::numeric_limits<real_t>::infinity();
  auto a                 = end(active_particles);
  auto active_it         = begin(active_particles);
  //autonomous_particle2::sampler_type const* nearest_sampler = nullptr;
  for (auto const& s : samplers) {
    if (auto const dist =
            s.ellipse1().squared_euclidean_distance_to_center(x0) *
            s.ellipse1().squared_local_euclidean_distance_to_center(x0);
        dist < shortest_distance) {
      shortest_distance = dist;
      //nearest_sampler   = &s;
      a                 = active_it;
    }
    ++active_it;
  }
  *a = true;
}
//==============================================================================
struct listener_t : gl::window_listener {
  Vec2<size_t> mouse_pos{};
  bool         left_down{false};

  auto on_cursor_moved(double x, double y) -> void override {
    mouse_pos = {x, y};

    if (left_down) {
      update_x0(mouse_pos);
      update_pathline_geometry();
      update_nearest();
      update_x1();
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

    auto q =
        win->camera_controller()
            .unproject(vec2f{mouse_pos(0), win->height() - 1 - mouse_pos(1)})
            .xy();
    auto active_it         = begin(active_particles);
    bool q_inside_particle = false;
    for (auto const& s : samplers) {
      if (s.ellipse1().is_inside(vec2{q})) {
        *active_it        = !*active_it;
        q_inside_particle = true;
      }
      ++active_it;
    }
    if (!q_inside_particle) {
      update_x0(mouse_pos);
      update_nearest();
      update_pathline_geometry();
    }
    update_x1();
  }
};
//==============================================================================
auto advect_particles() -> void {
  particles.clear();
  samplers.clear();
  auto const bottom_left = vec2{1.0, 0.5};
  auto       uuid_generator = std::atomic_uint64_t{};
  particles.emplace_back(bottom_left, t0, r0, uuid_generator);
  particles.emplace_back(bottom_left + vec2{2 * r0, 0}, t0, r0, uuid_generator);
  particles.emplace_back(bottom_left + vec2{0, 2 * r0}, t0, r0, uuid_generator);
  particles.emplace_back(bottom_left + vec2{2 * r0, 2 * r0}, t0, r0,
                         uuid_generator);
  // particles.emplace_back(vec2{0.2, 0.2}, t0, r0);
  // particles.emplace_back(vec2{0.4, 0.2}, t0, r0);
  // particles.emplace_back(vec2{0.2, 0.4}, t0, r0);
  // particles.emplace_back(vec2{0.4, 0.4}, t0, r0);
  particles           = std::get<0>(autonomous_particle2::advect<
                          autonomous_particle2::split_behaviors::three_splits>(
      flowmap(v), 0.01, tau, particles, uuid_generator));
  active_particles = std::vector<bool>(size(particles), false);
  std::transform(begin(particles), end(particles), std::back_inserter(samplers),
                 [](auto const& p) { return p.sampler(); });
  create_particle_geometry();
}
//==============================================================================
auto main() -> int {
  win             = std::make_unique<rendering::first_person_window>();
  particle_shader = std::make_unique<particle_shader_t>();
  pathline_shader = std::make_unique<pathline_shader_t>();
  point_shader    = std::make_unique<point_shader_t>();
  listener        = std::make_unique<listener_t>();

  win->add_listener(*listener);
  win->camera_controller().use_orthographic_camera();
  win->camera_controller().use_orthographic_controller();
  create_point_geometry();
  create_pathline_geometry();
  advect_particles();

  win->render_loop(render);
}
//------------------------------------------------------------------------------
auto render(std::chrono::duration<double> const& dt) -> void {
  gl::clear_color(1, 1, 1, 1);
  gl::clear_color_depth_buffer();
  render_pathline();
  render_particles();
  render_query_point();
  render_ui();
}
//------------------------------------------------------------------------------
auto render_query_point() -> void {
  point_shader->bind();
  point_shader->set_projection_matrix(
      win->camera_controller().projection_matrix());
  auto const V         = win->camera_controller().view_matrix();
  auto       M         = mat4f::zeros();
  //auto       active_it = begin(active_particles);
  point_shader->set_modelview_matrix(V);
  gl::point_size(5);
  point_shader->set_color(0, 0, 0, 1);
  x0_geometry->draw_points();
  point_shader->set_color(1, 0, 0, 1);
  x1_geometry->draw_points();
}
//------------------------------------------------------------------------------
auto render_particles() -> void {
  particle_shader->bind();
  particle_shader->set_projection_matrix(
      win->camera_controller().projection_matrix());
  auto const V   = win->camera_controller().view_matrix();
  auto       M   = mat4f::zeros();
  M(2, 2)        = static_cast<float>(1);
  M(3, 3)        = static_cast<float>(1);
  auto active_it = begin(active_particles);
  for (auto const& p : particles) {
    auto const is_active = *(active_it++);
    M(0, 0)              = static_cast<float>(p.S()(0, 0));
    M(1, 0)              = static_cast<float>(p.S()(1, 0));
    M(0, 1)              = static_cast<float>(p.S()(0, 1));
    M(1, 1)              = static_cast<float>(p.S()(1, 1));
    M(0, 3)              = static_cast<float>(p.center(0));
    M(1, 3)              = static_cast<float>(p.center(1));
    particle_shader->set_modelview_matrix(V * M);
    if (is_active) {
      gl::line_width(3);
      particle_shader->set_color(1, 0, 0, 1);
    } else {
      gl::line_width(1);
      particle_shader->set_color(0, 0, 0, 1);
    }
    particle_geometry->draw_line_strip();

    M(0, 0) = static_cast<float>(p.S0()(0, 0));
    M(1, 0) = static_cast<float>(p.S0()(1, 0));
    M(0, 1) = static_cast<float>(p.S0()(0, 1));
    M(1, 1) = static_cast<float>(p.S0()(1, 1));
    M(0, 3) = static_cast<float>(p.x0()(0));
    M(1, 3) = static_cast<float>(p.x0()(1));
    particle_shader->set_modelview_matrix(V * M);
    gl::line_width(1);
    if (is_active) {
      gl::line_width(3);
      particle_shader->set_color(1, 0.8, 0.8, 1);
    } else {
      gl::line_width(1);
      particle_shader->set_color(0.8, 0.8, 0.8, 1);
    }
    particle_geometry->draw_line_strip();
  }

  point_shader->bind();
  point_shader->set_projection_matrix(
      win->camera_controller().projection_matrix());
  point_shader->set_modelview_matrix(V);
  gl::point_size(2);
  point_shader->set_color(0.8, 0.8, 0.8, 1);
  particle_center_geometry->draw_points();
}
//------------------------------------------------------------------------------
auto render_pathline() -> void {
  pathline_shader->bind();
  pathline_shader->set_projection_matrix(
      win->camera_controller().projection_matrix());
  gl::line_width(1);
  pathline_shader->set_modelview_matrix(win->camera_controller().view_matrix());
  pathline_geometry->draw_line_strip();
}
//------------------------------------------------------------------------------
auto render_ui() -> void {
  if (menu_open) {
    float const width = std::min<float>(400, win->width() / 2);
    ImGui::SetNextWindowSize(ImVec2{width, float(win->height())});
    ImGui::SetNextWindowPos(ImVec2{win->width() - width, 0});
    ImGui::Begin("##", &menu_open,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoMove);

    if (ImGui::DragDouble("tau", &tau, 0.1, 0, 10)) {
      advect_particles();
      update_nearest();
      update_x1();
      update_pathline_geometry();
    }
    ImGui::SetWindowFontScale(3);
    ImGui::Text("offset: %f", euclidean_distance(x1_integrated, x1));
    ImGui::SetWindowFontScale(1);
  } else {
    ImGui::SetNextWindowSize(ImVec2{50, 50});
    ImGui::SetNextWindowPos(ImVec2{float(win->width() - 50), 0.0f});
    ImGui::Begin("##", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground);
    if (ImGui::Button("<", ImGui::GetWindowContentRegionMax() -
                               ImGui::GetWindowContentRegionMin())) {
      menu_open = true;
    }
  }
  ImGui::End();
}
//------------------------------------------------------------------------------
auto create_point_geometry() -> void {
  x0_geometry = std::make_unique<gl::indexeddata<vec2f>>();
  x0_geometry->vertexbuffer().resize(1);
  x0_geometry->vertexbuffer().front() = vec2f{x0};
  x0_geometry->indexbuffer().resize(1);
  x0_geometry->indexbuffer().front() = 0;
  x1_geometry = std::make_unique<gl::indexeddata<vec2f>>();
  x1_geometry->vertexbuffer().resize(1);
  x1_geometry->vertexbuffer().front() = vec2f{x1};
  x1_geometry->indexbuffer().resize(1);
  x1_geometry->indexbuffer().front() = 0;
}
//------------------------------------------------------------------------------
auto create_particle_geometry() -> void {
  auto const ellipse_num_points = 35;
  particle_geometry             = std::make_unique<gl::indexeddata<vec2f>>();
  particle_geometry->vertexbuffer().resize(ellipse_num_points);
  particle_geometry->indexbuffer().resize(ellipse_num_points + 1);
  {
    auto const t    = linspace{0.0, 2 * M_PI, ellipse_num_points + 1};
    auto       data = particle_geometry->vertexbuffer().wmap();
    size_t     i    = 0;
    for (auto t_it = begin(t); t_it != prev(end(t)); ++t_it) {
      data[i++] = vec2f{std::cos(*t_it), std::sin(*t_it)};
    }
  }
  {
    auto data = particle_geometry->indexbuffer().wmap();
    for (size_t i = 0; i < ellipse_num_points; ++i) {
      data[i] = i;
    }
    data[ellipse_num_points] = 0;
  }
  particle_center_geometry = std::make_unique<gl::indexeddata<vec2f>>();
  particle_center_geometry->vertexbuffer().resize(size(samplers));
  particle_center_geometry->indexbuffer().resize(size(samplers));
  {
    auto data = particle_center_geometry->vertexbuffer().wmap();
    auto it   = begin(data);
    for (auto const& s : samplers) {
      *(it++) = s.ellipse1().center();
    }
  }
  {
    auto data = particle_center_geometry->indexbuffer().wmap();
    for (size_t i = 0; i < size(samplers); ++i) {
      data[i] = i;
    }
    data[ellipse_num_points] = 0;
  }
}
//------------------------------------------------------------------------------
auto create_pathline_geometry() -> void {
  pathline_geometry = std::make_unique<gl::indexeddata<vec2f>>();
}
//------------------------------------------------------------------------------
auto update_pathline_geometry() -> void {
  pathline_geometry->clear();
  {
    auto integrator = ode::vclibs::rungekutta43<real_t, 2>{};
    pathline_geometry->vertexbuffer().push_back(vec2f{x0});
    integrator.solve(v, x0, t0 + tau, -tau, [&](auto const& x, auto const t) {
      pathline_geometry->vertexbuffer().push_back(vec2f{x});
      x1_integrated = x;
    });
  }
  {
    pathline_geometry->indexbuffer().resize(
        pathline_geometry->vertexbuffer().size());
    auto indexbuffer = pathline_geometry->indexbuffer().wmap();
    std::iota(indexbuffer.begin(), indexbuffer.end(), 0);
  }
}
