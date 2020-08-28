#ifndef TATOOINE_RENDERING_FIRST_PERSON_WINDOW_H
#define TATOOINE_RENDERING_FIRST_PERSON_WINDOW_H
//==============================================================================
#include <tatooine/holder.h>
#include <tatooine/ray.h>
#include <tatooine/rendering/perspective_camera.h>
#include <yavin/glwrapper.h>
#include <yavin/window.h>

#include <chrono>
#include <cmath>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
struct first_person_window : yavin::window {
  using parent_t = yavin::window;
  bool                                 m_run;
  GLsizei                              m_width, m_height;
  vec<float, 3>                        m_eye, m_look_dir, m_up;
  float                                m_theta = M_PI, m_phi = M_PI / 2;
  perspective_camera<float>            m_cam;
  std::chrono::time_point<std::chrono::system_clock> m_time =
      std::chrono::system_clock::now();
  int                                       m_mouse_pos_x, m_mouse_pos_y;
  bool                                      m_middle_button_down = false;
  bool                                      m_w_down             = false;
  bool                                      m_s_down             = false;
  bool                                      m_a_down             = false;
  bool                                      m_d_down             = false;
  bool                                      m_q_down             = false;
  bool                                      m_e_down             = false;
  //============================================================================
  first_person_window(size_t width = 800, size_t height = 600)
      : yavin::window{"tatooine first person window", width, height},
        m_run{true},
        m_width{width},
        m_height{height},
        m_cam{vec{0.0f, 0.0f, 0.0f},
              vec{0.0f, 0.0f, -1.0f},
              vec{0.0f, 1.0f, 0.0f},
              60.0f,
              width,
              height},
        m_time{std::chrono::system_clock::now()} {
    yavin::enable_depth_test();
  }
  //============================================================================
  auto width() const {
    return m_width;
  }
  //----------------------------------------------------------------------------
  auto height() const {
    return m_height;
  }
  //----------------------------------------------------------------------------
  template <typename Event>
  void render_loop(Event&& event) {
    m_time = std::chrono::system_clock::now();
    while (m_run) {
      refresh();
      yavin::gl::viewport(0, 0, m_width, m_height);
      update(std::forward<Event>(event),
             std::chrono::system_clock::now() - m_time);
      render_imgui();
      swap_buffers();
    }
  }
  //----------------------------------------------------------------------------
  template <typename F>
  void update(F&& f, std::chrono::duration<double> const& dt) {
    auto ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dt).count());

    m_time = std::chrono::system_clock::now();
    if (m_w_down) {
      m_eye += m_look_dir / ms;
    }
    if (m_s_down) {
      m_eye -= m_look_dir / ms;
    }
    if (m_q_down) {
      m_eye(1) += 1 / ms;
    }
    if (m_e_down) {
      m_eye(1) -= 1 / ms;
    }
    if (m_a_down) {
      auto const right = cross(yavin::vec3{0, 1, 0}, -m_look_dir);
      m_eye -= right / ms;
    }
    if (m_d_down) {
      auto const right = cross(yavin::vec3{0, 1, 0}, -m_look_dir);
      m_eye += right / ms;
    }
    m_cam.look_at(m_eye, m_eye + m_look_dir + 0.1);
    f(dt);
  }
  //----------------------------------------------------------------------------
  auto projection_matrix() const {
    return m_cam.projection_matrix();
  }
  //============================================================================
  void on_key_pressed(yavin::key k) override {
    parent_t::on_key_pressed(k);
    if (k == yavin::KEY_W) {
      m_w_down = true;
    } else if (k == yavin::KEY_S) {
      m_s_down = true;
    } else if (k == yavin::KEY_A) {
      m_a_down = true;
    } else if (k == yavin::KEY_D) {
      m_d_down = true;
    } else if (k == yavin::KEY_Q) {
      m_q_down = true;
    } else if (k == yavin::KEY_E) {
      m_e_down = true;
    }
  }
  //----------------------------------------------------------------------------
  void on_key_released(yavin::key k) override {
    parent_t::on_key_released(k);
    if (k == yavin::KEY_W) {
      m_w_down = false;
    } else if (k == yavin::KEY_S) {
      m_s_down = false;
    } else if (k == yavin::KEY_A) {
      m_a_down = false;
    } else if (k == yavin::KEY_D) {
      m_d_down = false;
    } else if (k == yavin::KEY_Q) {
      m_q_down = false;
    } else if (k == yavin::KEY_E) {
      m_e_down = false;
    }
  }
  //----------------------------------------------------------------------------
  void on_button_pressed(yavin::button b) override {
    parent_t::on_button_pressed(b);
    if (b == yavin::BUTTON_MIDDLE) {
      m_middle_button_down = true;
    }
  }
  //----------------------------------------------------------------------------
  void on_button_released(yavin::button b) override {
    parent_t::on_button_released(b);
    if (b == yavin::BUTTON_MIDDLE) {
      m_middle_button_down = false;
    }
  }
  //----------------------------------------------------------------------------
  void on_mouse_motion(int x, int y) override {
    parent_t::on_mouse_motion(x, y);
    if (m_middle_button_down) {
      int offset_x = x - m_mouse_pos_x;
      int offset_y = y - m_mouse_pos_y;
      m_theta -= offset_x * 0.01f;
      m_phi      = std::min<float>(M_PI - 0.3f,
                              std::max(0.3f, m_phi + offset_y * 0.01f));
      m_look_dir = yavin::vec3{
          std::sin(m_phi) * std::sin(m_theta),
          std::cos(m_phi),
          std::sin(m_phi) * std::cos(m_theta),
      };
    }
    m_mouse_pos_x = x;
    m_mouse_pos_y = y;
  }
  //----------------------------------------------------------------------------
  void on_resize(int w, int h) override {
    parent_t::on_resize(w, h);
    m_width  = w;
    m_height = h;
    m_cam.set_projection(60, (float)(w) / (float)(h), 0.1f, 1000.0f, w, h);
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
