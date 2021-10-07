#include <tatooine/analytical/fields/doublegyre.h>
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
        "uniform int active;\n"
        "out vec4 frag_out;\n"
        "//------------------------------------------------------------------\n"
        "void main() {\n"
        "  if (active == 1) {\n"
        "    frag_out = vec4(1,0,0,1);\n"
        "  } else {\n"
        "    frag_out = vec4(0,0,0,1);\n"
        "  }\n"
        "}\n"});
    create();
    set_active(0);
  }
  auto set_modelview_matrix(mat4f const& MV) -> void {
    set_uniform_mat4("modelview_matrix", MV.data_ptr());
  }
  auto set_projection_matrix(mat4f const& P) -> void {
    set_uniform_mat4("projection_matrix", P.data_ptr());
  }
  auto set_active(int const active) -> void { set_uniform("active", active); }
};
//==============================================================================
  struct listener_t : gl::window_listener {
    void on_button_pressed(gl::button /*b*/) override {}
  };
//==============================================================================
auto menu_open = false;
auto particles = autonomous_particle2::container_t{};

auto win              = std::unique_ptr<rendering::first_person_window>{};
auto particle_shader  = std::unique_ptr<particle_shader_t>{};
auto listener  = std::unique_ptr<listener_t>{};
auto ellipse_geometry = std::unique_ptr<gl::indexeddata<vec2f>>{};
//==============================================================================
auto render_ui() -> void;
auto render_particles() -> void;
auto render(std::chrono::duration<double> const& dt) -> void;
auto create_ellipse_geometry() -> void;
//==============================================================================
auto main() -> int {
  auto v          = analytical::fields::numerical::doublegyre{};
  particles.emplace_back(vec2{1,0.5}, 0, 0.1);
  particles = autonomous_particle2::advect_with_3_splits(flowmap(v), 0.01, 6,
                                                         particles);

  win       = std::make_unique<rendering::first_person_window>();
  particle_shader = std::make_unique<particle_shader_t>();
  listener        = std::make_unique<listener_t>();

  win->add_listener(*listener);
  create_ellipse_geometry();
  win->render_loop(render);
}
//------------------------------------------------------------------------------
auto render(std::chrono::duration<double> const& dt) -> void {
  gl::clear_color(1, 1, 1, 1);
  gl::clear_color_depth_buffer();
  render_particles();
  render_ui();
}
//------------------------------------------------------------------------------
auto render_particles() -> void {
  particle_shader->bind();
  particle_shader->set_projection_matrix(
      win->camera_controller().projection_matrix());
  auto const V = win->camera_controller().view_matrix();
  auto       M = mat4f::zeros();
  for (auto const& p : particles) {
    M(0, 0) = static_cast<float>(p.S1()(0, 0));
    M(1, 0) = static_cast<float>(p.S1()(1, 0));
    M(0, 1) = static_cast<float>(p.S1()(0, 1));
    M(1, 1) = static_cast<float>(p.S1()(1, 1));
    M(2, 2) = static_cast<float>(1);
    M(3, 3) = static_cast<float>(1);
    M(0, 3) = static_cast<float>(p.x1()(0));
    M(1, 3) = static_cast<float>(p.x1()(1));
    particle_shader->set_modelview_matrix(V*M);
    ellipse_geometry->draw_line_strip();
  }
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

    if (ImGui::Button("Button")) {
    }
    ImGui::End();
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
    ImGui::End();
  }
}
//------------------------------------------------------------------------------
auto create_ellipse_geometry() -> void {
  auto const ellipse_num_points = 35;
  ellipse_geometry              = std::make_unique<gl::indexeddata<vec2f>>();
  ellipse_geometry->vertexbuffer().resize(ellipse_num_points);
  ellipse_geometry->indexbuffer().resize(ellipse_num_points + 1);
  {
    auto const t    = linspace{0.0, 2 * M_PI, ellipse_num_points+1};
    auto       data = ellipse_geometry->vertexbuffer().wmap();
    size_t     i    = 0;
    for (auto t_it = begin(t); t_it != prev(end(t)); ++t_it) {
      data[i++] = vec2f{std::cos(*t_it), std::sin(*t_it)};
    }
  }
  {
    auto data = ellipse_geometry->indexbuffer().wmap();
    for (size_t i = 0; i < ellipse_num_points; ++i) {
      data[i] = i;
    }
    data[ellipse_num_points] = 0;
  }
}
