#include <tatooine/doublegyre.h>
#include <tatooine/gpu/first_person_window.h>
#include <tatooine/gpu/line_renderer.h>
#include <tatooine/gpu/line_shader.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/boundingbox.h>
template <typename F>
struct listener : yavin::window_listener {
  F& f;
  listener(F _f) : f{std::forward<F>(_f)} {}
  void on_key_pressed(yavin::key k) override { f(k); }
};
int main() {
  using namespace tatooine;
  first_person_window w;
  float               col[3]        = {1.0f, 0.0f, 0.0f};
  float               line_width    = 0.01f;
  float               contour_width = 0.005f;
  gpu::line_shader    shader{col[0], col[1], col[2], line_width, contour_width};
  using integrator_t = integration::vclibs::rungekutta43<double, 3, interpolation::linear>;
  integrator_t                        integrator;
  numerical::doublegyre               v;
  spacetime_field                     vst{v};
  boundingbox<double, 3>                bb{vec{0, 0, 0}, vec{2, 1, 0}};

  std::vector<integrator_t::integral_t> lines;
  for (size_t i = 0; i < 100; ++i) {
    lines.push_back(integrator.integrate(vst,bb.random_point(), 0, 10));
  }
  auto key_pressed = [&](auto k) {
    if (k == yavin::KEY_SPACE) { shader = gpu::line_shader{col[0], col[1], col[2], line_width, contour_width}; }
  };
  listener l{key_pressed};
  w.add_listener(l);

  auto line_renderers = gpu::upload(lines);
  w.render_loop([&]() {
    shader.set_color(col[0], col[1], col[2]);
    shader.set_line_width(line_width);
    shader.set_contour_width(contour_width);
    yavin::gl::clear_color(255, 255, 255, 255);
    yavin::clear_color_depth_buffer();
    shader.bind();
    shader.set_modelview_matrix(w.view_matrix());
    shader.set_projection_matrix(w.projection_matrix());
    for (auto& renderer : line_renderers) { renderer.draw_lines(); }
    ImGui::SliderFloat("line width", &line_width, 0.0f, 0.1f);
    ImGui::SliderFloat("contour width", &contour_width, 0.0f, line_width);
    ImGui::ColorEdit3("line color", col);
  });
}
