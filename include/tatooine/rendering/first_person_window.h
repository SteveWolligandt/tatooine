#ifndef TATOOINE_RENDERING_FIRST_PERSON_WINDOW_H
#define TATOOINE_RENDERING_FIRST_PERSON_WINDOW_H
//==============================================================================
#include <tatooine/holder.h>
#include <tatooine/ray.h>
#include <tatooine/rendering/camera_controller.h>
#include <yavin/glwrapper.h>
#include <yavin/window.h>

#include <chrono>
#include <cmath>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
struct first_person_window : yavin::window {
  using parent_t = yavin::window;
  size_t                    m_width, m_height;
  camera_controller<float>                           m_cam;
  std::chrono::time_point<std::chrono::system_clock> m_time =
      std::chrono::system_clock::now();
  //============================================================================
  first_person_window(size_t width = 800, size_t height = 600)
      : yavin::window{"tatooine first person window", width, height},
        m_width{width},
        m_height{height},
        m_cam{width, height},
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
    while (!should_close()) {
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
    m_cam.update(dt);
    f(dt);
  }
  //----------------------------------------------------------------------------
  auto projection_matrix() const {
    return m_cam.projection_matrix();
  }
  //----------------------------------------------------------------------------
  auto view_matrix() const {
    return m_cam.view_matrix();
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
      m_look_dir = vec{
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
    m_cam.set_resolution(w, h);
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
