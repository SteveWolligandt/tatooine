#ifndef TATOOINE_FIRST_PERSON_WINDOW_H
#define TATOOINE_FIRST_PERSON_WINDOW_H
//==============================================================================
#include <yavin/perspectivecamera.h>
#include <yavin/vec.h>
#include <yavin/glwrapper.h>
#include <yavin/window.h>

#include <chrono>
#include <cmath>
//==============================================================================
namespace tatooine {
//==============================================================================
struct first_person_window : yavin::window, yavin::window_listener {
  bool         m_run;
  GLsizei m_width, m_height;
  yavin::vec3                                        m_eye, m_look_at, m_up;
  float                                              m_theta = 0, m_phi = M_PI/2;
  yavin::perspectivecamera m_cam;
  std::chrono::time_point<std::chrono::system_clock> m_time =
      std::chrono::system_clock::now();
  int  m_mouse_pos_x, m_mouse_pos_y;
  bool m_mouse_down = false;
  bool m_w_down = false;
  bool m_s_down = false;
  bool m_a_down = false;
  bool m_d_down = false;
  bool m_q_down = false;
  bool m_e_down = false;
  //============================================================================
  first_person_window(GLsizei width = 800, GLsizei height = 600)
      : yavin::window{"tatooine first person window", width, height},
        m_run{true},
        m_width{width},
        m_height{height},
        m_eye{0, 0, 0},
        m_look_at{0, 0, -1},
        m_up{0, 1, 0},
        m_cam{90,    (float)(width) / (float)(height), 0.001f, 1000.0f, width,
              height},
        m_time{std::chrono::system_clock::now()} {
    add_listener(*this);
    yavin::enable_depth_test();
  }
  //============================================================================
  template <typename F>
  void render_loop(F&&f) {
    m_time = std::chrono::system_clock::now();
    while (m_run) {
      refresh();
      yavin::gl::viewport(m_cam);
      update(std::forward<F>(f), std::chrono::system_clock::now() - m_time);
      render_imgui();
      swap_buffers();
    }
  }
  //----------------------------------------------------------------------------
  template <typename F>
  void update(F&& f, const std::chrono::duration<double>& dt) {
    auto ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dt).count());

    m_time = std::chrono::system_clock::now();
    if (m_w_down) { m_eye += m_look_at / ms; }
    if (m_s_down) { m_eye -= m_look_at / ms; }
    if (m_q_down) { m_eye(1) += 1 / ms; }
    if (m_e_down) { m_eye(1) -= 1 / ms; }
    if (m_a_down) {
      const auto right = cross(yavin::vec3{0, 1, 0}, -m_look_at);
      m_eye -= right / ms;
    }
    if (m_d_down) {
      const auto right = cross(yavin::vec3{0, 1, 0}, -m_look_at);
      m_eye += right / ms;
    }
    m_cam.look_at(m_eye, m_eye + m_look_at + 0.1);
    f(dt);
  }
  //----------------------------------------------------------------------------
  auto projection_matrix() {
    return m_cam.projection_matrix();
  }
  //----------------------------------------------------------------------------
  auto view_matrix() {
    return *inverse(look_at_matrix(m_eye, m_eye + m_look_at));
  }
  //============================================================================
  void on_key_pressed(yavin::key k) override {
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
    } else if (k == yavin::KEY_ESCAPE) {
      m_run = false;
    }
  }
  //----------------------------------------------------------------------------
  void on_key_released(yavin::key k) override {
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
    m_mouse_down = true;
  }
  //----------------------------------------------------------------------------
  void on_button_released(yavin::button b) override {
    m_mouse_down = false;
  }
  //----------------------------------------------------------------------------
  void on_mouse_motion(int x, int y) override {
    if (m_mouse_down) {
      int offset_x = x - m_mouse_pos_x;
      int offset_y = y - m_mouse_pos_y;
      m_theta -= offset_x * 0.01f;
      m_phi += offset_y * 0.01f;
      m_look_at = yavin::vec3{
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
    m_width  = w;
    m_height = h;
    m_cam.set_projection(90, (float)(w) / (float)(h), 0.001f, 1000.0f, w, h);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
