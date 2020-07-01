#ifndef TATOOINE_FLOWEXPLORER_WINDOW_H
#define TATOOINE_FLOWEXPLORER_WINDOW_H
//==============================================================================
#include <tatooine/boundingbox.h>
#include <tatooine/gpu/first_person_window.h>
#include <tatooine/interpolation.h>

#include "boundingbox.h"
#include "grid.h"
#include "pathlines_boundingbox.h"
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window : first_person_window {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  bool                                     show_gui;
  std::vector<std::unique_ptr<renderable>> m_renderables;

  int mouse_x, mouse_y;

  //----------------------------------------------------------------------------
  // ctor
  //----------------------------------------------------------------------------
  template <typename V, typename VReal, size_t N>
  window(const vectorfield<V, VReal, N, N>& v) : show_gui{true} {
    add_key_pressed_event([&](auto k) {
      if (k == yavin::KEY_F1) {
        show_gui = !show_gui;
      } else if (k == yavin::KEY_SPACE) {
        //try {
        //  shader = std::make_unique<gpu::line_shader>(
        //    line_color[0], line_color[1], line_color[2], contour_color[0],
        //    contour_color[1], contour_color[2], line_width, contour_width,
        //    ambient_factor, diffuse_factor, specular_factor, shininess);
        //} catch (std::exception& e) { std::cerr << e.what() << '\n'; }
      }
    });
    add_mouse_motion_event([&](int x, int y) {
      mouse_x = x;
      mouse_y = y;
    });
    add_button_released_event([&](auto b) {
      //if (b == yavin::BUTTON_LEFT) {
      //  auto       r  = cast_ray(mouse_x, mouse_y);
      //  const auto x0 = r(0.5);
      //  if (v.in_domain(x0, 0)) {
      //    lines.push_back(integrator.integrate(v, x0, 0, btau, ftau));
      //    line_renderers.push_back(gpu::upload(lines.back()));
      //  }
      //}
    });


    start(v);
  }

  template <typename V, typename VReal, size_t N>
  void start(const vectorfield<V, VReal, N, N>& v) {
    render_loop([&](const auto& dt) {
      //if (shader->files_changed()) {
      //  try {
      //    shader = std::make_unique<gpu::line_shader>(
      //        line_color[0], line_color[1], line_color[2], contour_color[0],
      //        contour_color[1], contour_color[2], line_width, contour_width,
      //        ambient_factor, diffuse_factor, specular_factor, shininess);
      //  } catch (std::exception& e) { std::cerr << e.what() << '\n'; }
      //}
      yavin::gl::clear_color(255, 255, 255, 255);
      yavin::clear_color_depth_buffer();
      for (auto& r : m_renderables) {
        if (r->is_active()) { r->update(dt); }
      }
      for (auto& r : m_renderables) {
        if (r->is_active()) { r->render(projection_matrix(), view_matrix()); }
      }
      render_ui(v);
    });
  }
  template <typename V, typename VReal, size_t N>
  void render_ui(const vectorfield<V, VReal, N, N>& v) {
    if (show_gui) {
      ImGui::Begin("GUI", &show_gui);
      if (ImGui::Button("add bounding box")){
        m_renderables.emplace_back(
            new boundingbox{vec{-1.0, -1.0, -1.0}, vec{1.0, 1.0, 1.0}});
      }
      if (ImGui::Button("add grid")){
        m_renderables.emplace_back(new grid{linspace{-1.0, 1.0, 3},
                                            linspace{-1.0, 1.0, 3},
                                            linspace{-1.0, 1.0, 3}});
      }
      if (ImGui::Button("add pathline bounding box")){
        m_renderables.emplace_back(new pathlines_boundingbox{
            v, vec{-1.0, -1.0, -1.0}, vec{1.0, 1.0, 1.0}});
      }

      size_t i = 0; 
      for (auto& r : m_renderables) {
        ImGui::PushID(++i);
        ImGui::BeginGroup();
        const std::string name = r->name();
        if (ImGui::CollapsingHeader(name.c_str())) { 
          ImGui::Checkbox("active", &r->is_active());
          r->draw_ui();

        }
        ImGui::EndGroup();
        ImGui::PopID();
      }
      ImGui::End();
    }
  }
};
//==============================================================================
}  // namespace tatooine::gpu
//==============================================================================
#endif
