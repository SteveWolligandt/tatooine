#include <tatooine/doublegyre.h>
#include <tatooine/gpu/first_person_window.h>
#include <tatooine/gpu/line_renderer.h>
#include <tatooine/gpu/line_shader.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/boundingbox.h>
template <typename F>
struct key_pressed_listener : yavin::window_listener {
  const F& f;
  key_pressed_listener(const F& _f) : f{_f} {}
  void on_key_pressed(yavin::key k) override { f(k); }
};
int main() {
  using namespace tatooine;
  first_person_window w;
  float               col[3]          = {1.0f, 0.0f, 0.0f};
  float               line_width      = 0.01f;
  float               contour_width   = 0.005f;
  float               ambient_factor  = 0.5f;
  float               diffuse_factor  = 0.5f;
  float               specular_factor = 1.0f;
  float               shininess       = 10.0f;
  bool                animate         = false;
  float               time            = 0.0f;
  auto                shader          = std::make_unique<gpu::line_shader>(
      col[0], col[1], col[2], line_width, contour_width, ambient_factor,
      diffuse_factor, specular_factor, shininess);
  using integrator_t =
      integration::vclibs::rungekutta43<double, 3, interpolation::linear>;
  integrator_t           integrator;
  numerical::doublegyre  v;
  spacetime_field        vst{v};
  boundingbox<double, 3> bb{vec{0, 0, 0}, vec{2, 1, 0}};

  std::vector<integrator_t::integral_t> lines;
  for (size_t i = 0; i < 100; ++i) {
    lines.push_back(integrator.integrate(vst, bb.random_point(), 0, 10));
  }
  const auto key_pressed = [&](auto k) {
    if (k == yavin::KEY_SPACE) {
      try {
        shader = std::make_unique<gpu::line_shader>(col[0], col[1], col[2],
                                                    line_width, contour_width);
      } catch (std::exception& e) { std::cerr << e.what() << '\n'; }
    }
  };
  key_pressed_listener l{key_pressed};
  w.add_listener(l);

  auto line_renderers = gpu::upload(lines);
  w.render_loop([&](const auto& dt) {
    auto ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dt).count());
    time += ms / 500;
    while (time > 11) { time -= 11; }
    yavin::gl::clear_color(255, 255, 255, 255);
    yavin::clear_color_depth_buffer();
      yavin::enable_blending();
      yavin::blend_func_alpha();
    shader->set_modelview_matrix(w.view_matrix());
    shader->set_projection_matrix(w.projection_matrix());
    shader->set_color(col[0], col[1], col[2]);
    shader->set_line_width(line_width);
    shader->set_contour_width(contour_width);
    shader->set_ambient_factor(ambient_factor);
    shader->set_diffuse_factor(diffuse_factor);
    shader->set_specular_factor(specular_factor);
    shader->set_shininess(shininess);
    shader->set_animate(animate);
    shader->set_time(time);
    shader->bind();
    for (auto& renderer : line_renderers) { renderer.draw_lines(); }
    ImGui::Text(std::to_string(time).c_str());
    ImGui::SliderFloat("line width", &line_width, 0.0f, 0.1f);
    ImGui::SliderFloat("contour width", &contour_width, 0.0f, line_width);
    ImGui::SliderFloat("ambient factor", &ambient_factor, 0.0f, 1.0f);
    ImGui::SliderFloat("diffuse factor", &diffuse_factor, 0.0f, 1.0f);
    ImGui::SliderFloat("specular factor", &specular_factor, 0.0f, 1.0f);
    ImGui::SliderFloat("shininess", &shininess, 1.0f, 50.0f);
    ImGui::Checkbox("animate", &animate);
    ImGui::ColorEdit3("line color", col);
  });
}
