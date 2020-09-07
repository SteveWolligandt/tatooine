#ifndef TATOOINE_FLOWEXPLORER_NODES_RANDOM_PATHLINES_H
#define TATOOINE_FLOWEXPLORER_NODES_RANDOM_PATHLINES_H
//==============================================================================
#include <tatooine/boundingbox.h>
#include <tatooine/gpu/line_renderer.h>
#include <tatooine/gpu/line_shader.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
#include "../renderable.h"

#include <mutex>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real, size_t N>
struct random_pathlines : renderable {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using vectorfield_t = parent::field<Real, N, N>;
  using integrator_t  = ode::vclibs::rungekutta43<Real, N>;
  //----------------------------------------------------------------------------

  vectorfield_t const*              m_v           = nullptr;
  boundingbox<Real, N>*             m_boundingbox = nullptr;
  integrator_t                      m_integrator;
  std::unique_ptr<gpu::line_shader> m_shader;
  yavin::indexeddata<vec<float, 3>, vec<float, 3>, float> m_gpu_data;
  double                                                  m_btau, m_ftau;
  int                                                     m_num_pathlines;
  float                                                   m_line_color[3];
  float                                                   m_contour_color[3];
  float                                                   m_line_width;
  float                                                   m_contour_width;
  float                                                   m_ambient_factor;
  float                                                   m_diffuse_factor;
  float                                                   m_specular_factor;
  float                                                   m_shininess;
  bool                                                    m_animate;
  bool                                                    m_play;
  float                                                   m_fade_length;
  float                                                   m_general_alpha;
  float                                                   m_animation_min_alpha;
  float                                                   m_time;
  float                                                   m_speed;
  bool m_integration_going_on = false;
  //----------------------------------------------------------------------------
  random_pathlines(struct window& w)
      : renderable{w, "Random Path Lines"},
        m_shader{std::make_unique<gpu::line_shader>(
            m_line_color[0], m_line_color[1], m_line_color[2],
            m_contour_color[0], m_contour_color[1], m_contour_color[2],
            m_line_width, m_contour_width, m_ambient_factor, m_diffuse_factor,
            m_specular_factor, m_shininess)},
        m_btau{-10},
        m_ftau{10},
        m_num_pathlines{100},
        m_line_color{1.0f, 0.0f, 0.0f},
        m_contour_color{0.0f, 0.0f, 0.0f},
        m_line_width{0.04f},
        m_contour_width{0.005f},
        m_ambient_factor{0.1f},
        m_diffuse_factor{0.9f},
        m_specular_factor{1.0f},
        m_shininess{20.0f},
        m_animate{false},
        m_play{false},
        m_fade_length{1.0f},
        m_general_alpha{1.0f},
        m_animation_min_alpha{0.05f},
        m_time{0.0f},
        m_speed{1.0f} {
    this->template insert_input_pin<vectorfield_t>("3D Vector Field");
    this->template insert_input_pin<boundingbox<Real, N>>("Bounding Box");
  }
  //----------------------------------------------------------------------------
  void render(mat<float, 4, 4> const& projection_matrix,
              mat<float, 4, 4> const& view_matrix) override {
    if (m_animate || m_general_alpha < 1) {
      yavin::enable_blending();
      yavin::blend_func_alpha();
      yavin::disable_depth_test();
    } else {
      yavin::disable_blending();
      yavin::enable_depth_test();
    }
    update_shader(projection_matrix, view_matrix);
    m_shader->bind();
    m_gpu_data.draw_lines();
  }
  //----------------------------------------------------------------------------
  void update(const std::chrono::duration<double>& dt) override {
    auto ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dt).count());
    if (m_animate) {
      if (m_play) {
        m_time += m_speed * ms / 1000;
        while (m_time > m_ftau + m_fade_length) { m_time = m_btau; }
      }
    } else {
      m_time = m_btau;
    }
  }
  //----------------------------------------------------------------------------
  void draw_ui() override {
    ui::node::draw_ui([this] {
      ImGui::DragInt("number of path lines", &m_num_pathlines, 1, 10, 1000);
      ImGui::DragDouble("backward tau", &m_btau, 0.1, -100, 0);
      ImGui::DragDouble("forward tau", &m_ftau, 0.1, 0, 100);
      ImGui::SliderFloat("line width", &m_line_width, 0.0f, 0.1f);
      ImGui::SliderFloat("contour width", &m_contour_width, 0.0f, m_line_width / 2);
      ImGui::SliderFloat("ambient factor", &m_ambient_factor, 0.0f, 1.0f);
      ImGui::SliderFloat("diffuse factor", &m_diffuse_factor, 0.0f, 1.0f);
      ImGui::SliderFloat("specular factor", &m_specular_factor, 0.0f, 1.0f);
      ImGui::SliderFloat("m_shininess", &m_shininess, 1.0f, 80.0f);
      ImGui::ColorEdit3("line color", m_line_color);
      ImGui::ColorEdit3("contour color", m_contour_color);
      ImGui::Checkbox("m_animate", &m_animate);
      if (m_animate) {
        ImGui::Checkbox("m_play", &m_play);
        ImGui::SliderFloat("m_animation_min_alpha", &m_animation_min_alpha, 0.0f,
                           1.0f);
        ImGui::SliderFloat("m_fade_length", &m_fade_length, 0.1f, 10.0f);
        ImGui::SliderFloat("m_speed", &m_speed, 0.1f, 10.0f);
        ImGui::SliderFloat("m_time", &m_time, m_btau, m_ftau);
      } else {
        ImGui::SliderFloat("m_general_alpha", &m_general_alpha, 0.0f, 1.0f);
      }
    });
  }
  //----------------------------------------------------------------------------
  void update_shader(mat<float, 4, 4> const& projection_matrix,
                     mat<float, 4, 4> const& view_matrix) {
    m_shader->set_modelview_matrix(view_matrix);
    m_shader->set_projection_matrix(projection_matrix);
    m_shader->set_line_color(m_line_color[0], m_line_color[1], m_line_color[2]);
    m_shader->set_contour_color(m_contour_color[0], m_contour_color[1],
                              m_contour_color[2]);
    m_shader->set_line_width(m_line_width);
    m_shader->set_contour_width(m_contour_width);
    m_shader->set_ambient_factor(m_ambient_factor);
    m_shader->set_diffuse_factor(m_diffuse_factor);
    m_shader->set_specular_factor(m_specular_factor);
    m_shader->set_shininess(m_shininess);
    m_shader->set_animate(m_animate);
    m_shader->set_general_alpha(m_general_alpha);
    m_shader->set_animation_min_alpha(m_animation_min_alpha);
    m_shader->set_fade_length(m_fade_length);
    m_shader->set_time(m_time);
  }
  //----------------------------------------------------------------------------
  void integrate_lines() {
    if (m_integration_going_on) {
      return;
    }
    m_integration_going_on = true;
    this->window().do_async([rp = this] {
      size_t index          = 0;
      bool   insert_segment = false;
      auto callback = [rp, &index, &insert_segment](auto const& y, auto const t,
                                                    auto const& dy) {
        //std::lock_guard lock{rp->m_gpu_data.mutex()};
        rp->m_gpu_data.vertexbuffer().push_back(
            vec<float, 3>{static_cast<float>(y(0)), static_cast<float>(y(1)),
                          static_cast<float>(y(2))},
            vec<float, 3>{static_cast<float>(dy(0)), static_cast<float>(dy(1)),
                          static_cast<float>(dy(2))},
            static_cast<float>(t));
        if (insert_segment) {
          rp->m_gpu_data.indexbuffer().push_back(index - 1);
          rp->m_gpu_data.indexbuffer().push_back(index);
        } else {
          insert_segment = true;
        }
        ++index;
      };
      rp->m_gpu_data.clear();
      for (size_t i = 0; i < rp->m_num_pathlines; ++i) {
        auto const x0 = rp->m_boundingbox->random_point();
        double const t0 = 0;
        insert_segment  = false;
        rp->m_integrator.solve(*rp->m_v, x0, t0, rp->m_btau, callback);
        insert_segment = false;
        rp->m_integrator.solve(*rp->m_v, x0, t0, rp->m_ftau, callback);
      }
      rp->m_integration_going_on = false;
    });
  }
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) override {
    if (other_pin.type() == typeid(boundingbox<Real, N>)) {
      m_boundingbox = dynamic_cast<boundingbox<Real, N>*>(&other_pin.node());
    } else if ((other_pin.type() == typeid(vectorfield_t))) {
      m_v = dynamic_cast<vectorfield_t*>(&other_pin.node());
    }
    if (m_boundingbox != nullptr && m_v != nullptr) {
      integrate_lines();
    }
  }
  //----------------------------------------------------------------------------
  void on_pin_disconnected(ui::pin& this_pin) override {
    m_gpu_data.clear();
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
