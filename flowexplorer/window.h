#ifndef TATOOINE_FLOWEXPLORER_WINDOW_H
#define TATOOINE_FLOWEXPLORER_WINDOW_H
//==============================================================================
#include <tatooine/boundingbox.h>
#include <tatooine/gpu/line_shader.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/spacetime_field.h>

#include <tatooine/gpu/first_person_window.h>
#include <tatooine/gpu/line_renderer.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window : first_person_window {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using integrator_t =
      integration::vclibs::rungekutta43<double, 3, interpolation::linear>;
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  double                            btau, ftau;
  bool                              show_gui;
  float                             line_color[3];
  float                             contour_color[3];
  float                             line_width;
  float                             contour_width;
  float                             ambient_factor;
  float                             diffuse_factor;
  float                             specular_factor;
  float                             shininess;
  bool                              animate;
  bool                              play;
  float                             fade_length;
  float                             general_alpha;
  float                             animation_min_alpha;
  float                             time;
  float                             speed;
  std::unique_ptr<gpu::line_shader> shader;
  integrator_t                      integrator;
  std::vector<yavin::indexeddata<yavin::vec3, yavin::vec3, yavin::scalar>>
      line_renderers;

  std::vector<integrator_t::integral_t> lines;

  //----------------------------------------------------------------------------
  // ctor
  //----------------------------------------------------------------------------
  template <typename V, typename VReal, size_t N, typename BBReal>
  window(const vectorfield<V, VReal, N, N>& v,
                         const boundingbox<BBReal, N>&      seedarea,
                         size_t num_pathlines, double _btau, double _ftau)
      : btau{_btau},
        ftau{_ftau},
        show_gui{true},
        line_color{1.0f, 0.0f, 0.0f},
        contour_color{0.0f, 0.0f, 0.0f},
        line_width{0.02f},
        contour_width{0.005f},
        ambient_factor{0.1f},
        diffuse_factor{0.9f},
        specular_factor{1.0f},
        shininess{20.0f},
        animate{false},
        play{false},
        fade_length{1.0f},
        general_alpha{1.0f},
        animation_min_alpha{0.05f},
        time{0.0f},
        speed{1.0f},
        shader{std::make_unique<gpu::line_shader>(
            line_color[0], line_color[1], line_color[2], contour_color[0],
            contour_color[1], contour_color[2], line_width, contour_width,
            ambient_factor, diffuse_factor, specular_factor, shininess)} {
    for (size_t i = 0; i < num_pathlines; ++i) {
      lines.push_back(
          integrator.integrate(v, seedarea.random_point(), 0, btau, ftau));
    }
    add_key_pressed_event([&](auto k) {
      if (k == yavin::KEY_F1) {
        show_gui = !show_gui;
      } else if (k == yavin::KEY_SPACE) {
        try {
          shader = std::make_unique<gpu::line_shader>(
            line_color[0], line_color[1], line_color[2], contour_color[0],
            contour_color[1], contour_color[2], line_width, contour_width,
            ambient_factor, diffuse_factor, specular_factor, shininess);
        } catch (std::exception& e) { std::cerr << e.what() << '\n'; }
      }
    });

    line_renderers = gpu::upload(lines);
    start();
  }

  void start() {
    render_loop([&](const auto& dt) {
      if (shader->files_changed()) {
        try {
          shader = std::make_unique<gpu::line_shader>(
              line_color[0], line_color[1], line_color[2], contour_color[0],
              contour_color[1], contour_color[2], line_width, contour_width,
              ambient_factor, diffuse_factor, specular_factor, shininess);
        } catch (std::exception& e) { std::cerr << e.what() << '\n'; }
      }
      auto ms = static_cast<float>(
          std::chrono::duration_cast<std::chrono::milliseconds>(dt).count());
      if (animate) {
        if (play) {
          time += speed * ms / 1000;
          while (time > ftau + fade_length) { time = btau; }
        }
      } else {
        time = btau;
      }
      yavin::gl::clear_color(255, 255, 255, 255);
      yavin::clear_color_depth_buffer();
      if (animate || general_alpha < 1) {
        yavin::enable_blending();
        yavin::blend_func_alpha();
        yavin::disable_depth_test();
      } else {
        yavin::disable_blending();
        yavin::enable_depth_test();
      }
      update_shader();
      shader->bind();
      for (auto& renderer : line_renderers) { renderer.draw_lines(); }
      render_ui();
    });
  }
  void update_shader() {
    shader->set_modelview_matrix(view_matrix());
    shader->set_projection_matrix(projection_matrix());
    shader->set_line_color(line_color[0], line_color[1], line_color[2]);
    shader->set_contour_color(contour_color[0], contour_color[1],
                              contour_color[2]);
    shader->set_line_width(line_width);
    shader->set_contour_width(contour_width);
    shader->set_ambient_factor(ambient_factor);
    shader->set_diffuse_factor(diffuse_factor);
    shader->set_specular_factor(specular_factor);
    shader->set_shininess(shininess);
    shader->set_animate(animate);
    shader->set_general_alpha(general_alpha);
    shader->set_animation_min_alpha(animation_min_alpha);
    shader->set_fade_length(fade_length);
    shader->set_time(time);
  }
  void render_ui() {
    if (show_gui) {
      ImGui::Begin("settings", &show_gui);
      ImGui::SliderFloat("line width", &line_width, 0.0f, 0.1f);
      ImGui::SliderFloat("contour width", &contour_width, 0.0f, line_width / 2);
      ImGui::SliderFloat("ambient factor", &ambient_factor, 0.0f, 1.0f);
      ImGui::SliderFloat("diffuse factor", &diffuse_factor, 0.0f, 1.0f);
      ImGui::SliderFloat("specular factor", &specular_factor, 0.0f, 1.0f);
      ImGui::SliderFloat("shininess", &shininess, 1.0f, 80.0f);
      ImGui::ColorEdit3("line color", line_color);
      ImGui::ColorEdit3("contour color", contour_color);
      ImGui::Checkbox("animate", &animate);
      if (animate) {
        ImGui::Checkbox("play", &play);
        ImGui::SliderFloat("animation_min_alpha", &animation_min_alpha, 0.0f,
                           1.0f);
        ImGui::SliderFloat("fade_length", &fade_length, 0.1f, 10.0f);
        ImGui::SliderFloat("speed", &speed, 0.1f, 10.0f);
        ImGui::SliderFloat("time", &time, btau, ftau);
      } else {
        ImGui::SliderFloat("general_alpha", &general_alpha, 0.0f, 1.0f);
      }
      ImGui::End();
    }
  }
};
//==============================================================================
}  // namespace tatooine::gpu
//==============================================================================
#endif
