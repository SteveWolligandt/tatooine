#ifndef TATOOINE_FLOWEXPLORER_PATHLINES_BOUNDINGBOX_H
#define TATOOINE_FLOWEXPLORER_PATHLINES_BOUNDINGBOX_H
//==============================================================================
#include <tatooine/boundingbox.h>
#include <tatooine/gpu/line_renderer.h>
#include <tatooine/gpu/line_shader.h>
#include <tatooine/ode/vclibs/rungekutta43.h>

#include <mutex>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
template <typename Real, size_t N>
struct pathlines_boundingbox : boundingbox<Real, N> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using vectorfield_t = parent::field<Real, N, N>;
  using integrator_t  = ode::vclibs::rungekutta43<Real, N>;
  using parent_t = boundingbox<Real, N>;
  //----------------------------------------------------------------------------

  const vectorfield_t&                                        m_v;
  integrator_t                                                m_integrator;
  std::unique_ptr<gpu::line_shader>                           m_shader;
  yavin::indexeddata<yavin::vec3, yavin::vec3, yavin::scalar> m_gpu_data;
  bool                                                        hide_box = false;
  double                                                      btau, ftau;
  int                                                         num_pathlines;
  float                                                       line_color[3];
  float                                                       contour_color[3];
  float                                                       line_width;
  float                                                       contour_width;
  float                                                       ambient_factor;
  float                                                       diffuse_factor;
  float                                                       specular_factor;
  float                                                       shininess;
  bool                                                        animate;
  bool                                                        play;
  float                                                       fade_length;
  float                                                       general_alpha;
  float animation_min_alpha;
  float time;
  float speed;
  bool  m_integration_going_on = false;
  //----------------------------------------------------------------------------
  template <typename Real0, typename Real1>
  pathlines_boundingbox(struct window& w, const vectorfield_t& v, const vec<Real0, N>& min,
                        const vec<Real1, N>& max)
      : parent_t{w, min, max},
        m_v{v},
        m_shader{std::make_unique<gpu::line_shader>(
            line_color[0], line_color[1], line_color[2], contour_color[0],
            contour_color[1], contour_color[2], line_width, contour_width,
            ambient_factor, diffuse_factor, specular_factor, shininess)},
        btau{-10},
        ftau{10},
        num_pathlines{100},
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
        speed{1.0f} {
    this->m_color = {0.0f, 0.0f, 0.0f, 1.0f};
    this->min() = -vec{0.1, 0.1, 0.1};
    this->max() = vec{0.1, 0.1, 0.1};
  }
  //----------------------------------------------------------------------------
  void render(const yavin::mat4& projection_matrix,
              const yavin::mat4& view_matrix) override {
    if (!hide_box) { parent_t::render(projection_matrix, view_matrix); }
    if (animate || general_alpha < 1) {
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
    if (animate) {
      if (play) {
        time += speed * ms / 1000;
        while (time > ftau + fade_length) { time = btau; }
      }
    } else {
      time = btau;
    }
  }
  //----------------------------------------------------------------------------
  void draw_ui() override {
    if (m_integration_going_on) {
      ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    }
    if (ImGui::Button("integrate")) { integrate_lines(); }
    ImGui::SameLine(0);
    if (m_integration_going_on) {
      ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    }
    if (ImGui::Button("clear path lines")) {
      m_gpu_data.clear();
    }
    ImGui::Checkbox("hide box", &hide_box);
    parent_t::draw_ui_preferences();
    ImGui::DragInt("number of path lines", &num_pathlines, 1, 10, 1000);
    ImGui::DragDouble("backward tau", &btau, 0.1, -100, 0);
    ImGui::DragDouble("forward tau", &ftau, 0.1, 0, 100);
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
  }
  //----------------------------------------------------------------------------
  void update_shader(const yavin::mat4& projection_matrix,
                     const yavin::mat4& view_matrix) {
    m_shader->set_modelview_matrix(view_matrix);
    m_shader->set_projection_matrix(projection_matrix);
    m_shader->set_line_color(line_color[0], line_color[1], line_color[2]);
    m_shader->set_contour_color(contour_color[0], contour_color[1],
                              contour_color[2]);
    m_shader->set_line_width(line_width);
    m_shader->set_contour_width(contour_width);
    m_shader->set_ambient_factor(ambient_factor);
    m_shader->set_diffuse_factor(diffuse_factor);
    m_shader->set_specular_factor(specular_factor);
    m_shader->set_shininess(shininess);
    m_shader->set_animate(animate);
    m_shader->set_general_alpha(general_alpha);
    m_shader->set_animation_min_alpha(animation_min_alpha);
    m_shader->set_fade_length(fade_length);
    m_shader->set_time(time);
  }
  //----------------------------------------------------------------------------
  void integrate_lines() {
    if (m_integration_going_on) {
      return;
    }
    size_t index          = 0;
    bool insert_segment = false;
    auto callback = [this, &index, &insert_segment](auto const& y, auto const t,
                                                    auto const& dy) {
      std::lock_guard lock{m_gpu_data.mutex()};
      m_gpu_data.vertexbuffer().push_back(
          yavin::vec3{static_cast<float>(y(0)),
                      static_cast<float>(y(1)),
                      static_cast<float>(y(2))},
          yavin::vec3{static_cast<float>(dy(0)),
                      static_cast<float>(dy(1)),
                      static_cast<float>(dy(2))},
          static_cast<float>(t));
      if (insert_segment) {
        m_gpu_data.indexbuffer().push_back(index - 1);
        m_gpu_data.indexbuffer().push_back(index);
      } else {
        insert_segment = true;
      }
      ++index;
    };
    m_integration_going_on = true;
      m_gpu_data.vertexbuffer().resize(100);
      m_gpu_data.indexbuffer().resize(100);
    this->window().do_async([this, &index, &insert_segment, &callback] {
      m_gpu_data.clear();
      index = 0;
      for (size_t i = 0; i < num_pathlines; ++i) {
        auto const   x0 = this->random_point();
        double const t0 = 0;
        insert_segment  = false;
        m_integrator.solve(m_v, x0, t0, btau, callback);
        insert_segment = false;
        m_integrator.solve(m_v, x0, t0, ftau, callback);
      }
      m_integration_going_on = false;
    });
  }
  //----------------------------------------------------------------------------
  std::string name() const override { return "path lines"; }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
